from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from microtensor.core.constants import REFERENCE_COST_MS
from microtensor.core.protocol import LoadManifest, Role
from microtensor.core.system import SystemManifest
from microtensor.core.tracks import DECIDE, get_track
from microtensor.harness.cascade import CascadeResult, run_cascade
from microtensor.harness.engines.router import RouterError, load_router
from microtensor.scoring.calibration import generation_summary, sequence_confidence, summarise
from microtensor.scoring.frontier import quantise_point
from microtensor.scoring.metrics import score_task
from microtensor.tasks.corpus import Task
from microtensor.tasks.selection import to_requests


class SimulationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Simulation:
    tasks: int
    resolve_rate: float
    expected_ms: float
    end_to_end: float
    front_only: float
    quantised: tuple[int, int]
    front_calibration: dict[str, Any] = field(default_factory=dict)
    system_calibration: dict[str, Any] = field(default_factory=dict)

    @property
    def uplift(self) -> float:
        return self.end_to_end - self.front_only

    def report(self) -> str:
        lines = [
            f"tasks              {self.tasks}",
            f"resolve rate       {self.resolve_rate:.1%}",
            f"expected cost      {self.expected_ms:.1f} ms per query",
            f"end-to-end quality {self.end_to_end:.4f}",
            f"front alone        {self.front_only:.4f}",
            f"escalation uplift  {self.uplift:+.4f}",
            f"frontier point     quality={self.quantised[0]} cost={self.quantised[1]}",
        ]
        if self.uplift <= 0.0:
            lines.append("")
            lines.append(
                "the specialist is not earning its cost here; a router that never "
                "escalates would score the same and cost less"
            )
        if self.system_calibration:
            lines.extend(self._calibration_lines())
        return "\n".join(lines)

    def _calibration_lines(self) -> list[str]:
        front, system = self.front_calibration, self.system_calibration
        lines = [
            "",
            f"{'calibration':<20}{'front':>10}{'system':>10}",
            f"{'  accuracy':<20}{front['accuracy']:>10.4f}{system['accuracy']:>10.4f}",
            f"{'  brier quality':<20}{front['brier_quality']:>10.4f}"
            f"{system['brier_quality']:>10.4f}",
            f"{'  calibration error':<20}{front['ece']:>10.4f}{system['ece']:>10.4f}",
        ]
        if system.get("level_mae") is not None and front.get("level_mae") is not None:
            lines.append(
                f"{'  level error':<20}{front['level_mae']:>10.4f}{system['level_mae']:>10.4f}"
            )
        lines.append("")
        lines.append("reliability (system)")
        lines.append(f"  {'stated':<11}{'questions':>10}{'mean stated':>13}{'correct':>9}")
        for row in system["reliability"]:
            span = f"{row['lower']:.1f}-{row['upper']:.1f}"
            if not row["count"]:
                lines.append(f"  {span:<11}{0:>10}")
                continue
            lines.append(
                f"  {span:<11}{row['count']:>10}{row['confidence']:>13.3f}{row['accuracy']:>9.3f}"
            )
        return lines


def check_system(system: SystemManifest, artifact: Path, hardware_class: str) -> None:
    """Refuse locally what discovery would refuse on chain."""
    fits, reason = system.fits_class(hardware_class)
    if not fits:
        raise SimulationError(reason)

    if system.router is None:
        return

    router_path = artifact / system.locate(Role.ROUTER)
    try:
        load_router(router_path, system.router_features)
    except RouterError as exc:
        raise SimulationError(str(exc)) from exc

    specialist = artifact / system.locate(Role.SPECIALIST)
    if not specialist.exists():
        raise SimulationError(f"the specialist is missing from {specialist}")


def simulate(
    artifact: Path,
    load: LoadManifest,
    system: SystemManifest,
    tasks: Sequence[Task],
    hardware_class: str,
    *,
    metric: str,
    track: str,
    limit: int = 0,
    seed: str = "local-simulation",
) -> Simulation:
    """Run the cascade over the public training split.

    A participant tunes router thresholds against these numbers rather than
    against a validator's, which is the difference between one submission and
    a blind guess per epoch.
    """
    check_system(system, artifact, hardware_class)

    if limit > 0:
        tasks = tasks[:limit]
    if not tasks:
        raise SimulationError(
            "no training tasks to simulate against; the public train split is what "
            "this command reads, never the scored partitions"
        )

    front = artifact / system.locate(Role.FRONT) if not system.degenerate else artifact
    router_path = str(artifact / system.locate(Role.ROUTER)) if system.router else ""
    specialist_path = str(artifact / system.locate(Role.SPECIALIST)) if system.specialist else ""
    payload = load.to_dict()

    legs = run_cascade(
        str(front),
        payload,
        to_requests(list(tasks), seed, track, "local"),
        router_path,
        list(system.router_features),
        specialist_path,
        payload if specialist_path else None,
    )
    result = CascadeResult(tuple(legs))

    from microtensor.scoring.execution import execute_module_rate, has_module_tests

    def scored(output: object, gold: object) -> float:
        if has_module_tests(gold):
            return execute_module_rate(str(output), gold["tests"])  # type: ignore[index]
        return score_task(metric, output, gold)

    by_ref = {leg.task_ref: leg for leg in legs}
    end_to_end = 0.0
    front_only = 0.0
    for task in tasks:
        leg = by_ref.get(task.ref)
        if leg is None:
            continue
        if leg.response.ok:
            end_to_end += scored(leg.response.output, task.gold)
        if leg.front_response.ok:
            front_only += scored(leg.front_response.output, task.gold)

    count = len(tasks)
    quality = end_to_end / count
    front_calibration: dict[str, Any] = {}
    system_calibration: dict[str, Any] = {}
    if get_track(track).answer_mode == DECIDE:
        answered = [(task, by_ref.get(task.ref)) for task in tasks]
        front_calibration = summarise(
            (leg.front_response.output if leg and leg.front_response.ok else None, task.gold)
            for task, leg in answered
        )
        system_calibration = summarise(
            (leg.response.output if leg and leg.response.ok else None, task.gold)
            for task, leg in answered
        )
    else:
        answered = [(task, by_ref.get(task.ref)) for task in tasks]
        front_calibration = generation_summary(
            (
                sequence_confidence(leg.front_response.logprobs) if leg else 0.0,
                scored(leg.front_response.output, task.gold)
                if leg and leg.front_response.ok
                else 0.0,
            )
            for task, leg in answered
        )
        system_calibration = generation_summary(
            (
                sequence_confidence(leg.response.logprobs) if leg else 0.0,
                scored(leg.response.output, task.gold) if leg and leg.response.ok else 0.0,
            )
            for task, leg in answered
        )
    return Simulation(
        tasks=count,
        resolve_rate=result.resolve_rate,
        expected_ms=result.expected_ms,
        end_to_end=quality,
        front_only=front_only / count,
        quantised=quantise_point(quality, result.expected_ms, REFERENCE_COST_MS),
        front_calibration=front_calibration,
        system_calibration=system_calibration,
    )
