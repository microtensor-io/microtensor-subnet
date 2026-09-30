from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microtensor.core.basemodels import size_floor_bytes
from microtensor.core.protocol import (
    Evaluation,
    Fault,
    GateFailure,
    GateResult,
    MeasuredEnvelope,
    Role,
    TaskOutcome,
    evaluate_gate,
)
from microtensor.core.tracks import DECIDE, HardwareClass, Track, get_class, get_track
from microtensor.envelope.device import POLICY_ENV
from microtensor.envelope.profiler import plan_for, plan_payload, run_profile
from microtensor.harness.cascade import CascadeResult, Leg, run_cascade
from microtensor.harness.contract import Response
from microtensor.harness.execute import run_tasks
from microtensor.harness.jail import run_jailed
from microtensor.harness.limits import Limits
from microtensor.harness.output import check as check_output
from microtensor.harness.registry import EngineUnavailable, available, load_builtin
from microtensor.registry.fetch import ArtifactMismatch, Unfetchable
from microtensor.registry.fetch import materialise as fetch_artifact
from microtensor.scoring.calibration import partition_report
from microtensor.scoring.execution import (
    ExecutionUnavailable,
    execute_module_rate,
    extract_code,
    has_module_tests,
    screen_solution,
)
from microtensor.scoring.metrics import combine_partitions, partition_scores, score_task
from microtensor.tasks.corpus import FIXED, NOVEL, ROTATING, Task
from microtensor.tasks.selection import RoundTasks, partition_of, to_requests
from microtensor.validator.context import ValidatorContext
from microtensor.validator.discover import Participant

log = logging.getLogger("microtensor.validator.evaluate")

REJECTED = GateResult(admitted=False, failures=())
BUDGET_EXHAUSTED = "exhausted its cpu budget"


INFRASTRUCTURE_ATTEMPTS = 3

_stopping = False


def stopping(flag: bool = True) -> None:
    global _stopping
    _stopping = flag


class Abstain(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CompetitionResult:
    track: str
    hardware_class: str
    corpus_version: str
    evaluations: tuple[Evaluation, ...]

    @property
    def admitted(self) -> tuple[Evaluation, ...]:
        return tuple(e for e in self.evaluations if e.gate.admitted)

    def __len__(self) -> int:
        return len(self.evaluations)


def _expected_ms(cascade: CascadeResult | None, measured: MeasuredEnvelope | None) -> float:
    """What one query costs, measured rather than assumed.

    A front-only system runs no cascade, so its cost is the total latency the
    profiler measured for the front itself, the same clock a cascade's legs
    use. Leaving it at zero would be a claim of free inference; falling back
    to the reference ceiling would put
    every such system at the worst cost on the grid, where its exclusive
    hypervolume is zero and it could never earn. Both misreport a quantity the
    network did in fact measure.
    """
    if cascade is not None:
        return cascade.expected_ms
    if measured is not None:
        return float(measured.total_p95_ms)
    return 0.0


def _verdict(gate: GateResult, failure: str) -> GateResult:
    if failure == BUDGET_EXHAUSTED:
        return GateResult(admitted=False, failures=(GateFailure.BUDGET_CEILING,))
    return gate


def _evaluation(
    participant: Participant,
    tasks: RoundTasks,
    *,
    gate: GateResult = REJECTED,
    measured: MeasuredEnvelope | None = None,
    rotating: float = 0.0,
    fixed: float = 0.0,
    novel: float = 0.0,
    n_rotating: int = 0,
    n_fixed: int = 0,
    n_novel: int = 0,
    cascade: CascadeResult | None = None,
    front_only: float = 0.0,
    calibration: dict[str, Any] | None = None,
    expected_ms: float | None = None,
) -> Evaluation:
    track, hardware_class = participant.competition
    return Evaluation(
        hotkey=participant.hotkey,
        track=track,
        hardware_class=hardware_class,
        artifact_digest=participant.manifest.artifact_digest,
        gate=gate,
        measured=measured,
        score_rotating=rotating,
        score_fixed=fixed,
        score_novel=novel,
        score_combined=(
            combine_partitions(rotating, fixed, novel, n_rotating, n_fixed, n_novel)
            if gate.admitted
            else 0.0
        ),
        n_rotating=n_rotating,
        n_fixed=n_fixed,
        n_novel=n_novel,
        corpus_version=tasks.corpus_version,
        resolve_rate=cascade.resolve_rate if cascade else 1.0,
        expected_ms=_expected_ms(cascade, measured) if expected_ms is None else expected_ms,
        front_only_score=front_only,
        system_digest=participant.manifest.system_digest,
        calibration=dict(calibration or {}),
    )


def _calibration(tasks: RoundTasks, by_ref: Mapping[str, Response]) -> dict[str, Any]:
    def pairs(bucket: Sequence[Task]) -> list[tuple[Any, Any, Any]]:
        found: list[tuple[Any, Any, Any]] = []
        for task in bucket:
            response = by_ref.get(task.ref)
            output = response.output if response and response.ok else None
            found.append((output, task.gold, task.inputs.get("decision")))
        return found

    return partition_report(
        {ROTATING: pairs(tasks.rotating), FIXED: pairs(tasks.fixed), NOVEL: pairs(tasks.novel)},
        NOVEL,
    )


def _limits(hardware: HardwareClass, seconds: int) -> Limits:
    return Limits.for_class(hardware, max(2, seconds))


def _cpu_budget(context: ValidatorContext, cpu_seconds: int) -> int:
    """The arena's budget when the anchored config carried one.

    Zero means no arena budget reached this worker, which happens on the
    standalone and loopback paths. The configured default stands in there
    rather than in the coordinated path, where disagreeing with peers about
    the budget would mean scoring a different competition from them.
    """
    return cpu_seconds or context.config.cpu_seconds_per_artifact


def materialise(context: ValidatorContext, participant: Participant) -> Path:
    try:
        return fetch_artifact(
            participant.manifest,
            context.cache,
            workdir=context.config.work_dir,
            key=participant.key,
            fallback_source=participant.commitment.source,
        )
    except Unfetchable as exc:
        raise Abstain(f"{participant.hotkey}: artifact unfetchable — {exc}") from exc


def profile(
    context: ValidatorContext,
    participant: Participant,
    artifact: Path,
    hardware: HardwareClass,
    seed: str,
) -> tuple[MeasuredEnvelope | None, str]:
    policy = context.certifications.get(hardware.id)
    if policy:
        os.environ[POLICY_ENV] = json.dumps(policy, sort_keys=True)
    else:
        os.environ.pop(POLICY_ENV, None)

    max_input = dict(participant.manifest.load.max_input)
    plan = plan_for(
        seed,
        max_input,
        participant.manifest.track,
        duration_seconds=context.config.profile_seconds,
    )
    result = run_jailed(
        run_profile,
        str(artifact),
        participant.manifest.load.to_dict(),
        hardware.id,
        plan_payload(plan),
        limits=_limits(hardware, context.config.profile_seconds * 2),
        allow_unsandboxed=context.config.allow_unsandboxed,
    )

    if result.ok:
        return result.value.envelope, ""
    if result.fault is Fault.INFRASTRUCTURE:
        raise Abstain(f"{participant.hotkey}: profiling infrastructure failed — {result.error}")
    return None, f"profiling failed: {result.error}"


def _outcome(
    task: Task,
    response: Response | None,
    metric: str,
    partition: str,
    track: Track | None = None,
) -> TaskOutcome:
    if response is None or not response.ok:
        return TaskOutcome(
            task_ref=task.ref,
            score=0.0,
            completed=False,
            partition=partition,
            error=response.error if response else "engine returned no response",
            fault=Fault.ARTIFACT,
        )
    if metric == "execution_pass_rate":
        reason = screen_solution(extract_code(str(response.output)))
        if reason:
            return TaskOutcome(
                task_ref=task.ref,
                score=0.0,
                completed=False,
                partition=partition,
                error=f"solution screened: {reason}",
                fault=Fault.ARTIFACT,
            )
    invalid = check_output(track, response) if track is not None else ""
    if invalid:
        return TaskOutcome(
            task_ref=task.ref,
            score=0.0,
            completed=True,
            partition=partition,
            error=f"invalid output: {invalid}",
            latency_ms=response.total_ms,
        )
    if has_module_tests(task.gold):
        value = execute_module_rate(str(response.output), task.gold["tests"])
    else:
        value = score_task(metric, response.output, task.gold)
    return TaskOutcome(
        task_ref=task.ref,
        score=value,
        completed=True,
        partition=partition,
        latency_ms=response.total_ms,
    )


def run_system(
    context: ValidatorContext,
    participant: Participant,
    artifact: Path,
    hardware: HardwareClass,
    tasks: RoundTasks,
    *,
    cpu_seconds: int = 0,
) -> tuple[CascadeResult | None, str]:
    """Execute the whole system, front then router then escalation."""
    system = participant.system
    load = participant.manifest.load.to_dict()
    requests = to_requests(
        tasks.all, tasks.seed, tasks.track, participant.manifest.artifact_digest
    )

    front_path = artifact / system.locate(Role.FRONT) if not system.degenerate else artifact
    router_path = str(artifact / system.locate(Role.ROUTER)) if system.router else ""
    specialist_path = (
        str(artifact / system.locate(Role.SPECIALIST)) if system.specialist else ""
    )

    result = run_jailed(
        run_cascade,
        str(front_path),
        load,
        requests,
        router_path,
        list(system.router_features),
        specialist_path,
        load if specialist_path else None,
        limits=_limits(hardware, _cpu_budget(context, cpu_seconds)),
        allow_unsandboxed=context.config.allow_unsandboxed,
    )

    if not result.ok:
        if result.fault is Fault.INFRASTRUCTURE:
            raise Abstain(
                f"{participant.hotkey}: execution infrastructure failed — {result.error}"
            )
        if not result.partial:
            return None, f"execution failed: {result.error}"
        log.info(
            "%s exhausted its cpu budget after %d of %d tasks",
            participant.hotkey,
            len(result.partial),
            len(tasks.all),
        )
        return None, BUDGET_EXHAUSTED

    return CascadeResult(legs=tuple(result.value)), ""


def outcomes_from(
    legs: Sequence[Leg], tasks: RoundTasks, metric: str
) -> tuple[tuple[TaskOutcome, ...], tuple[TaskOutcome, ...]]:
    """End-to-end outcomes, and the front's own for diagnostics.

    Only the first is ranked. A cascade is bought as a whole, so the quality
    that decides emission is the quality of what the system finally answered.
    """
    by_ref = {leg.task_ref: leg for leg in legs}
    track = get_track(tasks.track)

    end_to_end: list[TaskOutcome] = []
    front_only: list[TaskOutcome] = []
    for task in tasks.all:
        leg = by_ref.get(task.ref)
        partition = partition_of(tasks, task.ref)
        end_to_end.append(
            _outcome(task, leg.response if leg else None, metric, partition, track)
        )
        front_only.append(
            _outcome(task, leg.front_response if leg else None, metric, partition, track)
        )
    return tuple(end_to_end), tuple(front_only)


def score(
    context: ValidatorContext,
    participant: Participant,
    artifact: Path,
    hardware: HardwareClass,
    tasks: RoundTasks,
    *,
    cpu_seconds: int = 0,
) -> tuple[tuple[TaskOutcome, ...], dict[str, Response], str]:
    track = get_track(tasks.track)
    metric = track.metric
    requests = to_requests(
        tasks.all, tasks.seed, tasks.track, participant.manifest.artifact_digest
    )
    result = run_jailed(
        run_tasks,
        str(artifact),
        participant.manifest.load.to_dict(),
        requests,
        limits=_limits(hardware, _cpu_budget(context, cpu_seconds)),
        allow_unsandboxed=context.config.allow_unsandboxed,
    )

    if not result.ok:
        if result.fault is Fault.INFRASTRUCTURE:
            raise Abstain(f"{participant.hotkey}: execution infrastructure failed — {result.error}")
        if not result.partial:
            return (), {}, f"execution failed: {result.error}"
        log.info(
            "%s exhausted its cpu budget after %d of %d tasks",
            participant.hotkey,
            len(result.partial),
            len(tasks.all),
        )
        return (), {}, BUDGET_EXHAUSTED

    # A task with no response scores as a failure further down, so the tasks
    # the worker never reached are forfeit without any special case here.
    answered: Sequence[Response] = result.value if result.ok else result.partial
    by_ref: dict[str, Response] = {r.task_ref: r for r in answered}
    try:
        outcomes = tuple(
            _outcome(
                task,
                by_ref.get(task.ref),
                metric,
                partition_of(tasks, task.ref),
                track,
            )
            for task in tasks.all
        )
    except ExecutionUnavailable as exc:
        raise Abstain(
            f"{participant.hotkey}: the execution sandbox is unavailable — {exc}"
        ) from exc
    return outcomes, by_ref, ""


def _detection_partition_scores(
    tasks: RoundTasks, by_ref: dict[str, Response]
) -> tuple[float, float, float, int, int, int]:
    """COCO mAP per partition over the whole image set, not an average.

    A shared category list across both partitions keeps the class set stable,
    so rotating and fixed mAP are computed against the same definition of the
    problem rather than each inventing its own from the classes it happened to
    see.
    """
    from microtensor.scoring.detection import (
        Detection,
        category_ids,
        coco_map,
        parse_detections,
    )

    buckets: dict[str, tuple[list[list[Detection]], list[object]]] = {
        ROTATING: ([], []),
        FIXED: ([], []),
        NOVEL: ([], []),
    }
    for task in tasks.all:
        response = by_ref.get(task.ref)
        preds = parse_detections(response.output) if response and response.ok else []
        bucket = buckets[partition_of(tasks, task.ref)]
        bucket[0].append(preds)
        bucket[1].append(task.gold)

    cats = category_ids([g for _, gold in buckets.values() for g in gold])
    return (
        coco_map(*buckets[ROTATING], categories=cats),
        coco_map(*buckets[FIXED], categories=cats),
        coco_map(*buckets[NOVEL], categories=cats),
        len(buckets[ROTATING][1]),
        len(buckets[FIXED][1]),
        len(buckets[NOVEL][1]),
    )


def _extraction_partition_scores(
    tasks: RoundTasks, by_ref: dict[str, Response]
) -> tuple[float, float, float, int, int, int]:
    """Entity micro-F1 per partition, aggregated over the whole document set."""
    from microtensor.scoring.extraction import gold_entities, micro_f1, parse_entities

    buckets: dict[
        str, tuple[list[set[tuple[str, str]] | None], list[set[tuple[str, str]]]]
    ] = {ROTATING: ([], []), FIXED: ([], []), NOVEL: ([], [])}
    for task in tasks.all:
        response = by_ref.get(task.ref)
        preds = parse_entities(response.output) if response and response.ok else None
        bucket = buckets[partition_of(tasks, task.ref)]
        bucket[0].append(preds)
        bucket[1].append(gold_entities(task.gold))

    return (
        micro_f1(*buckets[ROTATING]),
        micro_f1(*buckets[FIXED]),
        micro_f1(*buckets[NOVEL]),
        len(buckets[ROTATING][1]),
        len(buckets[FIXED][1]),
        len(buckets[NOVEL][1]),
    )



def _decision_partition_scores(
    tasks: RoundTasks, by_ref: dict[str, Response]
) -> tuple[float, float, float, int, int, int]:
    from microtensor.scoring.metrics import decision_skill

    buckets: dict[str, list[tuple[Any, Any, Any]]] = {ROTATING: [], FIXED: [], NOVEL: []}
    for task in tasks.all:
        response = by_ref.get(task.ref)
        output = response.output if response and response.ok else None
        spec = task.inputs.get("decision")
        buckets[partition_of(tasks, task.ref)].append((output, task.gold, spec))
    return (
        decision_skill(buckets[ROTATING]),
        decision_skill(buckets[FIXED]),
        decision_skill(buckets[NOVEL]),
        len(buckets[ROTATING]),
        len(buckets[FIXED]),
        len(buckets[NOVEL]),
    )


# Metrics whose ranked quality is aggregated over the whole document set rather
# than averaged per task. Registering here keeps the dispatch in one place.
_DATASET_METRICS = {
    "map_at_iou": _detection_partition_scores,
    "entity_micro_f1": _extraction_partition_scores,
    "decision_brier": _decision_partition_scores,
}


def evaluate_participant(
    context: ValidatorContext,
    participant: Participant,
    tasks: RoundTasks,
    *,
    cpu_seconds: int = 0,
    hardware: HardwareClass | None = None,
    escalations: Mapping[str, Any] | None = None,
) -> Evaluation:
    hardware = hardware or get_class(participant.competition[1])
    artifact = materialise(context, participant)

    if participant.system.full:
        from microtensor.harness.package import package_reason

        reason = package_reason(artifact, participant.system)
        if reason:
            log.info("%s rejected: %s", participant.hotkey, reason)
            return _evaluation(participant, tasks)

    measured, failure = profile(context, participant, artifact, hardware, tasks.seed)
    if measured is None:
        log.info("%s scored zero: %s", participant.hotkey, failure)
        return _evaluation(participant, tasks)

    gate = evaluate_gate(
        measured,
        participant.manifest.declared,
        hardware,
        size_floor_bytes(participant.manifest.load.base_model),
    )
    if not gate.admitted:
        log.info("%s inadmissible: %s", participant.hotkey, gate.reason)
        return _evaluation(participant, tasks, gate=gate, measured=measured)

    if participant.system.full:
        return _evaluate_full(
            context,
            participant,
            artifact,
            hardware,
            tasks,
            gate,
            measured,
            dict(escalations or {}),
            _cpu_budget(context, cpu_seconds),
        )

    cascade: CascadeResult | None = None
    front_only_score = 0.0

    if not participant.system.degenerate:
        cascade, failure = run_system(
            context, participant, artifact, hardware, tasks, cpu_seconds=cpu_seconds
        )
        if failure or cascade is None:
            log.info("%s scored zero: %s", participant.hotkey, failure)
            return _evaluation(
                participant, tasks, gate=_verdict(gate, failure), measured=measured
            )
        metric = get_track(tasks.track).metric
        outcomes, front_outcomes = outcomes_from(cascade.legs, tasks, metric)
        front_only_score = combine_partitions(*partition_scores(front_outcomes))
        by_ref = {r.task_ref: r for r in cascade.responses()}
    else:
        outcomes, by_ref, failure = score(
            context, participant, artifact, hardware, tasks, cpu_seconds=cpu_seconds
        )
        if failure:
            log.info("%s scored zero: %s", participant.hotkey, failure)
            return _evaluation(
                participant, tasks, gate=_verdict(gate, failure), measured=measured
            )
        metric = get_track(tasks.track).metric

    dataset_scorer = _DATASET_METRICS.get(metric)
    if dataset_scorer is not None:
        rotating, fixed, novel, n_rotating, n_fixed, n_novel = dataset_scorer(tasks, by_ref)
    else:
        rotating, fixed, novel, n_rotating, n_fixed, n_novel = partition_scores(outcomes)
    calibration = (
        _calibration(tasks, by_ref) if get_track(tasks.track).answer_mode == DECIDE else None
    )
    return _evaluation(
        participant,
        tasks,
        gate=gate,
        measured=measured,
        rotating=rotating,
        fixed=fixed,
        novel=novel,
        n_rotating=n_rotating,
        n_fixed=n_fixed,
        n_novel=n_novel,
        cascade=cascade,
        front_only=front_only_score,
        calibration=calibration,
    )


LIVE_ROWS = 500


def _evaluate_full(
    context: ValidatorContext,
    participant: Participant,
    artifact: Path,
    hardware: HardwareClass,
    tasks: RoundTasks,
    gate: GateResult,
    measured: MeasuredEnvelope,
    escalations: dict[str, Any],
    cpu_seconds: int,
) -> Evaluation:
    from microtensor.scoring.metrics import score_task
    from microtensor.scoring.system import COST_UNITS_PER_USD, score_system
    from microtensor.validator.live import GatewaySystemClient, chain_verifier, run_live
    from microtensor.validator.verify_system import jailed_verify

    config = context.config
    system = participant.system
    if not config.gateway_url or not config.gateway_secret or system.endpoint is None:
        raise Abstain(
            "full systems are tested live through the gateway; "
            "set MT_GATEWAY_URL and MT_GATEWAY_SECRET"
        )
    client = GatewaySystemClient(
        config.gateway_url, config.gateway_secret, participant.hotkey, system.endpoint.worker
    )
    live = run_live(
        client,
        system,
        hotkey=participant.hotkey,
        round_index=tasks.round_index,
        tasks=tasks.all,
        verify=chain_verifier(),
    )
    for ref, reason in live.failures[:5]:
        log.info("%s %s: %s", participant.hotkey, ref, reason)
    if not live.traces:
        log.info("%s scored zero: no task produced a verified trace", participant.hotkey)
        return _evaluation(participant, tasks, measured=measured)

    track = get_track(tasks.track)
    result = run_jailed(
        jailed_verify,
        str(artifact),
        system.body(),
        str(artifact),
        participant.manifest.load.to_dict(),
        [trace.to_dict() for trace in live.traces],
        {task.ref: (task.prompt, dict(task.inputs)) for task in tasks.all},
        tasks.seed,
        [model.to_dict() for model in escalations.values()],
        track.chat,
        limits=_limits(hardware, cpu_seconds),
        allow_unsandboxed=config.allow_unsandboxed,
    )
    if not result.ok:
        if result.fault is Fault.INFRASTRUCTURE:
            raise Abstain(
                f"{participant.hotkey}: verification infrastructure failed: {result.error}"
            )
        log.info("%s not certified: verification failed: %s", participant.hotkey, result.error)
        return _evaluation(participant, tasks, measured=measured)
    verdict = dict(result.value)
    if not verdict.get("certified"):
        log.info("%s not certified: %s", participant.hotkey, "; ".join(verdict.get("reasons", [])))
        return _evaluation(participant, tasks, measured=measured)

    score = score_system(
        live.traces,
        {task.ref: task.gold for task in tasks.all},
        track.metric,
        escalations,
        small_ms=_expected_ms(None, measured),
        profiles={task.ref: task.profile for task in tasks.all},
    )
    answered = live.answered
    triggers = dict(verdict.get("triggers") or {})
    rows = []
    for task in tasks.all[:LIVE_ROWS]:
        trace = answered.get(task.ref)
        if trace is None:
            rows.append({"task_ref": task.ref, "answered": False, "profile": task.profile})
            continue
        price = (
            escalations.get(f"{trace.escalation.model}@{trace.escalation.revision}")
            if trace.escalation is not None
            else None
        )
        rows.append(
            {
                "task_ref": task.ref,
                "answered": True,
                "profile": task.profile,
                "escalated": trace.escalated,
                "decided_at_ms": round(trace.router.at_ms, 1),
                "total_ms": round(trace.total_ms, 1),
                "trigger": triggers.get(task.ref, ""),
                "features": {k: round(v, 4) for k, v in trace.router.features.items()},
                "confidence": round(trace.small.confidence, 4),
                "small_usd": score.small_usd,
                "escalation_tokens": (
                    trace.escalation.prompt_tokens + trace.escalation.completion_tokens
                    if trace.escalation is not None
                    else 0
                ),
                "escalation_usd": round(
                    price.cost_usd(
                        trace.escalation.prompt_tokens, trace.escalation.completion_tokens
                    )
                    if price is not None and trace.escalation is not None
                    else 0.0,
                    9,
                ),
                "score": round(score_task(track.metric, trace.final, task.gold), 4),
                "small_score": round(score_task(track.metric, trace.small.output, task.gold), 4),
            }
        )
    outcomes = tuple(
        TaskOutcome(
            task_ref=task.ref,
            score=score_task(track.metric, answered[task.ref].final, task.gold)
            if task.ref in answered
            else 0.0,
            completed=task.ref in answered,
            partition=partition_of(tasks, task.ref),
            latency_ms=answered[task.ref].total_ms if task.ref in answered else 0.0,
        )
        for task in tasks.all
    )
    rotating, fixed, novel, n_rotating, n_fixed, n_novel = partition_scores(outcomes)
    log.info(
        "%s full system: quality %.4f, small alone %.4f, escalated %.1f%%, waste %.1f%%, "
        "misses %.1f%%, cost $%.6f per task",
        participant.hotkey,
        score.quality,
        score.small_quality,
        score.escalation_rate * 100,
        score.waste * 100,
        score.misses * 100,
        score.cost_usd,
    )
    return _evaluation(
        participant,
        tasks,
        gate=gate,
        measured=measured,
        rotating=rotating,
        fixed=fixed,
        novel=novel,
        n_rotating=n_rotating,
        n_fixed=n_fixed,
        n_novel=n_novel,
        front_only=score.small_quality,
        calibration={**score.to_dict(), "live": rows},
        expected_ms=score.cost_usd * COST_UNITS_PER_USD,
    )


def require_engines() -> None:
    load_builtin()
    if not available():
        raise Abstain("no execution engine is available; the validator cannot score this round")


def evaluate_competition(
    context: ValidatorContext,
    participants: tuple[Participant, ...],
    tasks: RoundTasks,
    *,
    cpu_seconds: int = 0,
    hardware: HardwareClass | None = None,
    on_evaluated: Callable[[Evaluation, Participant], None] | None = None,
    escalations: Mapping[str, Any] | None = None,
) -> CompetitionResult:
    evaluations: list[Evaluation] = []

    for participant in participants:
        attempt = 0
        while True:
            attempt += 1
            try:
                evaluation = evaluate_participant(
                    context,
                    participant,
                    tasks,
                    cpu_seconds=cpu_seconds,
                    hardware=hardware,
                    escalations=escalations,
                )
                break
            except ArtifactMismatch as exc:
                log.info("%s scored zero: %s", participant.hotkey, exc)
                evaluation = _evaluation(participant, tasks)
                break
            except EngineUnavailable as exc:
                raise Abstain(str(exc)) from exc
            except Abstain as exc:
                if _stopping or attempt >= INFRASTRUCTURE_ATTEMPTS:
                    raise
                log.warning(
                    "%s hit an infrastructure fault (%d/%d), measuring it again: %s",
                    participant.hotkey,
                    attempt,
                    INFRASTRUCTURE_ATTEMPTS,
                    exc,
                )

        evaluations.append(evaluation)
        context.state.record_evaluation(tasks.round_index, evaluation)
        if on_evaluated is not None:
            try:
                on_evaluated(evaluation, participant)
            except Exception as exc:
                log.warning("could not publish %s as it finished: %s", participant.hotkey, exc)
        context.state.observe(
            tasks.track,
            tasks.hardware_class,
            participant.hotkey,
            participant.manifest.artifact_digest,
            tasks.round_index,
        )

    log.info(
        "%s/%s: %d evaluated, %d admitted",
        tasks.track,
        tasks.hardware_class,
        len(evaluations),
        sum(1 for e in evaluations if e.gate.admitted),
    )
    return CompetitionResult(
        track=tasks.track,
        hardware_class=tasks.hardware_class,
        corpus_version=tasks.corpus_version,
        evaluations=tuple(evaluations),
    )
