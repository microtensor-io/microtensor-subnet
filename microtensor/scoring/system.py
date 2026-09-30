from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from microtensor.core.escalation import EscalationModel
from microtensor.core.trace import Trace
from microtensor.scoring.calibration import expected_calibration_error, reliability
from microtensor.scoring.metrics import score_task

REFERENCE_CPU_USD_PER_HOUR: Final[float] = 0.04
COST_UNITS_PER_USD: Final[float] = 1_000_000.0
CORRECT_AT: Final[float] = 0.5
DIGITS: Final[int] = 6


@dataclass(frozen=True, slots=True)
class SystemScore:
    tasks: int
    quality: float
    small_quality: float
    escalation_rate: float
    waste: float
    misses: float
    small_usd: float
    escalation_usd: float
    calibration: dict[str, Any] = field(default_factory=dict)
    escalation_by_profile: dict[str, float] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return self.small_usd + self.escalation_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": self.tasks,
            "quality": self.quality,
            "small_quality": self.small_quality,
            "escalation_rate": self.escalation_rate,
            "waste": self.waste,
            "misses": self.misses,
            "small_usd": self.small_usd,
            "escalation_usd": self.escalation_usd,
            "cost_usd": round(self.cost_usd, 9),
            "calibration": dict(self.calibration),
            "escalation_by_profile": dict(self.escalation_by_profile),
        }


def small_usd(ms: float, cpu_usd_per_hour: float = REFERENCE_CPU_USD_PER_HOUR) -> float:
    return max(0.0, ms) / 3_600_000.0 * cpu_usd_per_hour


def score_system(
    traces: Sequence[Trace],
    golds: Mapping[str, Any],
    metric: str,
    allowlist: Mapping[str, EscalationModel],
    *,
    small_ms: float | None = None,
    cpu_usd_per_hour: float = REFERENCE_CPU_USD_PER_HOUR,
    profiles: Mapping[str, str] | None = None,
) -> SystemScore:
    count = len(golds)
    if count == 0:
        return SystemScore(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    by_ref = {trace.task_ref: trace for trace in traces if trace.task_ref in golds}
    final_total = small_total = escalated = waste = misses = 0.0
    escalation_cost = 0.0
    small_times: list[float] = []
    judged: list[tuple[float, bool]] = []
    for ref, gold in golds.items():
        trace = by_ref.get(ref)
        if trace is None:
            misses += 1
            judged.append((0.0, False))
            continue
        final = score_task(metric, trace.final, gold)
        small = score_task(metric, trace.small.output, gold)
        final_total += final
        small_total += small
        small_times.append(trace.small.ms)
        judged.append((trace.small.confidence, small >= CORRECT_AT))
        if trace.escalation is not None:
            escalated += 1
            if small >= CORRECT_AT:
                waste += 1
            price = allowlist.get(f"{trace.escalation.model}@{trace.escalation.revision}")
            if price is not None:
                escalation_cost += price.cost_usd(
                    trace.escalation.prompt_tokens, trace.escalation.completion_tokens
                )
        elif small < CORRECT_AT:
            misses += 1
    by_profile: dict[str, list[bool]] = {}
    for ref, profile in (profiles or {}).items():
        trace = by_ref.get(ref)
        if profile and trace is not None:
            by_profile.setdefault(profile, []).append(trace.escalated)
    measured = math.fsum(small_times) / max(1, len(small_times))
    per_task_ms = small_ms if small_ms is not None else measured
    correct = [ok for _, ok in judged]
    return SystemScore(
        tasks=count,
        quality=round(final_total / count, DIGITS),
        small_quality=round(small_total / count, DIGITS),
        escalation_rate=round(escalated / count, DIGITS),
        waste=round(waste / count, DIGITS),
        misses=round(misses / count, DIGITS),
        small_usd=round(small_usd(per_task_ms, cpu_usd_per_hour), 12),
        escalation_usd=round(escalation_cost / count, 12),
        calibration={
            "accuracy": round(sum(correct) / len(correct), DIGITS) if correct else 0.0,
            "ece": expected_calibration_error(judged),
            "reliability": reliability(judged),
        },
        escalation_by_profile={
            profile: round(sum(flags) / len(flags), DIGITS)
            for profile, flags in sorted(by_profile.items())
        },
    )
