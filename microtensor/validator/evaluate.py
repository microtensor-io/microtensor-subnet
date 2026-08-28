from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

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
from microtensor.core.tracks import HardwareClass, get_class, get_track
from microtensor.envelope.device import POLICY_ENV
from microtensor.envelope.probe import max_input_prompt
from microtensor.envelope.profiler import ProfilePlan, run_profile
from microtensor.harness.cascade import CascadeResult, Leg, run_cascade
from microtensor.harness.contract import Response
from microtensor.harness.execute import run_tasks
from microtensor.harness.jail import run_jailed
from microtensor.harness.limits import Limits
from microtensor.harness.registry import EngineUnavailable, available, load_builtin
from microtensor.registry.fetch import ArtifactMismatch, Unfetchable
from microtensor.registry.fetch import materialise as fetch_artifact
from microtensor.scoring.execution import ExecutionUnavailable
from microtensor.scoring.metrics import combine_partitions, partition_scores, score_task
from microtensor.tasks.corpus import Task
from microtensor.tasks.selection import RoundTasks, to_requests
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

    A front-only system runs no cascade, so its cost is the latency the
    profiler measured for the front itself. Leaving it at zero would be a
    claim of free inference; falling back to the reference ceiling would put
    every such system at the worst cost on the grid, where its exclusive
    hypervolume is zero and it could never earn. Both misreport a quantity the
    network did in fact measure.
    """
    if cascade is not None:
        return cascade.expected_ms
    if measured is not None:
        return float(measured.ttft_p95_ms)
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
    n_rotating: int = 0,
    n_fixed: int = 0,
    cascade: CascadeResult | None = None,
    front_only: float = 0.0,
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
        score_combined=combine_partitions(rotating, fixed) if gate.admitted else 0.0,
        n_rotating=n_rotating,
        n_fixed=n_fixed,
        corpus_version=tasks.corpus_version,
        resolve_rate=cascade.resolve_rate if cascade else 1.0,
        expected_ms=_expected_ms(cascade, measured),
        front_only_score=front_only,
        system_digest=participant.manifest.system_digest,
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
    plan = ProfilePlan(
        prompt=max_input_prompt(seed, max_input),
        max_input=max_input,
        duration_seconds=context.config.profile_seconds,
    )
    result = run_jailed(
        run_profile,
        str(artifact),
        participant.manifest.load.to_dict(),
        hardware.id,
        {
            "prompt": plan.prompt,
            "max_input": plan.max_input,
            "duration_seconds": plan.duration_seconds,
            "max_requests": plan.max_requests,
            "sample_interval_ms": plan.sample_interval_ms,
            "max_output_tokens": plan.max_output_tokens,
        },
        limits=_limits(hardware, context.config.profile_seconds * 2),
        allow_unsandboxed=context.config.allow_unsandboxed,
    )

    if result.ok:
        return result.value.envelope, ""
    if result.fault is Fault.INFRASTRUCTURE:
        raise Abstain(f"{participant.hotkey}: profiling infrastructure failed — {result.error}")
    return None, f"profiling failed: {result.error}"


def _outcome(task: Task, response: Response | None, metric: str, partition: str) -> TaskOutcome:
    if response is None or not response.ok:
        return TaskOutcome(
            task_ref=task.ref,
            score=0.0,
            completed=False,
            partition=partition,
            error=response.error if response else "engine returned no response",
            fault=Fault.ARTIFACT,
        )
    return TaskOutcome(
        task_ref=task.ref,
        score=score_task(metric, response.output, task.gold),
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
    rotating = {t.ref for t in tasks.rotating}

    end_to_end: list[TaskOutcome] = []
    front_only: list[TaskOutcome] = []
    for task in tasks.all:
        leg = by_ref.get(task.ref)
        partition = "rotating" if task.ref in rotating else "fixed"
        end_to_end.append(_outcome(task, leg.response if leg else None, metric, partition))
        front_only.append(
            _outcome(task, leg.front_response if leg else None, metric, partition)
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
    metric = get_track(tasks.track).metric
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
    rotating = {t.ref for t in tasks.rotating}
    try:
        outcomes = tuple(
            _outcome(
                task,
                by_ref.get(task.ref),
                metric,
                "rotating" if task.ref in rotating else "fixed",
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
) -> tuple[float, float, int, int]:
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

    rotating = {t.ref for t in tasks.rotating}
    rot_preds: list[list[Detection]] = []
    rot_gold: list[object] = []
    fix_preds: list[list[Detection]] = []
    fix_gold: list[object] = []
    for task in tasks.all:
        response = by_ref.get(task.ref)
        preds = parse_detections(response.output) if response and response.ok else []
        if task.ref in rotating:
            rot_preds.append(preds)
            rot_gold.append(task.gold)
        else:
            fix_preds.append(preds)
            fix_gold.append(task.gold)

    cats = category_ids([*rot_gold, *fix_gold])
    return (
        coco_map(rot_preds, rot_gold, categories=cats),
        coco_map(fix_preds, fix_gold, categories=cats),
        len(rot_gold),
        len(fix_gold),
    )


def _extraction_partition_scores(
    tasks: RoundTasks, by_ref: dict[str, Response]
) -> tuple[float, float, int, int]:
    """Entity micro-F1 per partition, aggregated over the whole document set."""
    from microtensor.scoring.extraction import gold_entities, micro_f1, parse_entities

    rotating = {t.ref for t in tasks.rotating}
    rot_preds: list[set[tuple[str, str]] | None] = []
    rot_gold: list[set[tuple[str, str]]] = []
    fix_preds: list[set[tuple[str, str]] | None] = []
    fix_gold: list[set[tuple[str, str]]] = []
    for task in tasks.all:
        response = by_ref.get(task.ref)
        preds = parse_entities(response.output) if response and response.ok else None
        gold = gold_entities(task.gold)
        if task.ref in rotating:
            rot_preds.append(preds)
            rot_gold.append(gold)
        else:
            fix_preds.append(preds)
            fix_gold.append(gold)

    return (
        micro_f1(rot_preds, rot_gold),
        micro_f1(fix_preds, fix_gold),
        len(rot_gold),
        len(fix_gold),
    )


# Metrics whose ranked quality is aggregated over the whole document set rather
# than averaged per task. Registering here keeps the dispatch in one place.
_DATASET_METRICS = {
    "map_at_iou": _detection_partition_scores,
    "entity_micro_f1": _extraction_partition_scores,
}


def evaluate_participant(
    context: ValidatorContext,
    participant: Participant,
    tasks: RoundTasks,
    *,
    cpu_seconds: int = 0,
    hardware: HardwareClass | None = None,
) -> Evaluation:
    hardware = hardware or get_class(participant.competition[1])
    artifact = materialise(context, participant)

    measured, failure = profile(context, participant, artifact, hardware, tasks.seed)
    if measured is None:
        log.info("%s scored zero: %s", participant.hotkey, failure)
        return _evaluation(participant, tasks)

    gate = evaluate_gate(measured, participant.manifest.declared, hardware)
    if not gate.admitted:
        log.info("%s inadmissible: %s", participant.hotkey, gate.reason)
        return _evaluation(participant, tasks, gate=gate, measured=measured)

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
        front_only_score = combine_partitions(*partition_scores(front_outcomes)[:2])
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
        rotating, fixed, n_rotating, n_fixed = dataset_scorer(tasks, by_ref)
    else:
        rotating, fixed, n_rotating, n_fixed = partition_scores(outcomes)
    return _evaluation(
        participant,
        tasks,
        gate=gate,
        measured=measured,
        rotating=rotating,
        fixed=fixed,
        n_rotating=n_rotating,
        n_fixed=n_fixed,
        cascade=cascade,
        front_only=front_only_score,
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
) -> CompetitionResult:
    evaluations: list[Evaluation] = []

    for participant in participants:
        attempt = 0
        while True:
            attempt += 1
            try:
                evaluation = evaluate_participant(
                    context, participant, tasks, cpu_seconds=cpu_seconds, hardware=hardware
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
