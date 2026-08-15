from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from microtensor.core.protocol import (
    Evaluation,
    Fault,
    GateResult,
    MeasuredEnvelope,
    TaskOutcome,
    evaluate_gate,
)
from microtensor.core.tracks import HardwareClass, get_class, get_track
from microtensor.envelope.probe import max_input_prompt
from microtensor.envelope.profiler import ProfilePlan, run_profile
from microtensor.harness.contract import Response
from microtensor.harness.execute import run_tasks
from microtensor.harness.jail import run_jailed
from microtensor.harness.limits import Limits
from microtensor.harness.registry import EngineUnavailable, available, load_builtin
from microtensor.registry.fetch import ArtifactMismatch, Unfetchable
from microtensor.registry.fetch import materialise as fetch_artifact
from microtensor.scoring.metrics import combine_partitions, partition_scores, score_task
from microtensor.tasks.corpus import Task
from microtensor.tasks.selection import RoundTasks, to_requests
from microtensor.validator.context import ValidatorContext
from microtensor.validator.discover import Participant

log = logging.getLogger("microtensor.validator.evaluate")

REJECTED = GateResult(admitted=False, failures=())


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
    )


def _limits(hardware: HardwareClass, seconds: int) -> Limits:
    return Limits.for_class(hardware, max(2, seconds))


def materialise(context: ValidatorContext, participant: Participant) -> Path:
    try:
        return fetch_artifact(
            participant.manifest, context.cache, workdir=context.config.work_dir
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


def score(
    context: ValidatorContext,
    participant: Participant,
    artifact: Path,
    hardware: HardwareClass,
    tasks: RoundTasks,
) -> tuple[tuple[TaskOutcome, ...], str]:
    metric = get_track(tasks.track).metric
    requests = to_requests(
        tasks.all, tasks.seed, tasks.track, participant.manifest.artifact_digest
    )
    result = run_jailed(
        run_tasks,
        str(artifact),
        participant.manifest.load.to_dict(),
        requests,
        limits=_limits(hardware, context.config.cpu_seconds_per_artifact),
        allow_unsandboxed=context.config.allow_unsandboxed,
    )

    if not result.ok:
        if result.fault is Fault.INFRASTRUCTURE:
            raise Abstain(f"{participant.hotkey}: execution infrastructure failed — {result.error}")
        return (), f"execution failed: {result.error}"

    by_ref: dict[str, Response] = {r.task_ref: r for r in result.value}
    rotating = {t.ref for t in tasks.rotating}
    outcomes = tuple(
        _outcome(
            task,
            by_ref.get(task.ref),
            metric,
            "rotating" if task.ref in rotating else "fixed",
        )
        for task in tasks.all
    )
    return outcomes, ""


def evaluate_participant(
    context: ValidatorContext, participant: Participant, tasks: RoundTasks
) -> Evaluation:
    hardware = get_class(participant.competition[1])
    artifact = materialise(context, participant)

    measured, failure = profile(context, participant, artifact, hardware, tasks.seed)
    if measured is None:
        log.info("%s scored zero: %s", participant.hotkey, failure)
        return _evaluation(participant, tasks)

    gate = evaluate_gate(measured, participant.manifest.declared, hardware)
    if not gate.admitted:
        log.info("%s inadmissible: %s", participant.hotkey, gate.reason)
        return _evaluation(participant, tasks, gate=gate, measured=measured)

    outcomes, failure = score(context, participant, artifact, hardware, tasks)
    if failure:
        log.info("%s scored zero: %s", participant.hotkey, failure)
        return _evaluation(participant, tasks, gate=gate, measured=measured)

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
    )


def require_engines() -> None:
    load_builtin()
    if not available():
        raise Abstain("no execution engine is available; the validator cannot score this round")


def evaluate_competition(
    context: ValidatorContext, participants: tuple[Participant, ...], tasks: RoundTasks
) -> CompetitionResult:
    evaluations: list[Evaluation] = []

    for participant in participants:
        try:
            evaluation = evaluate_participant(context, participant, tasks)
        except ArtifactMismatch as exc:
            log.info("%s scored zero: %s", participant.hotkey, exc)
            evaluation = _evaluation(participant, tasks)
        except EngineUnavailable as exc:
            raise Abstain(str(exc)) from exc

        evaluations.append(evaluation)
        context.state.record_evaluation(tasks.round_index, evaluation)
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
