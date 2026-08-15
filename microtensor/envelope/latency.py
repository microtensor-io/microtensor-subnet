from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def quantile(samples: Sequence[float], q: float) -> float:
    if not samples:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")

    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]

    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def median(samples: Sequence[float]) -> float:
    return quantile(samples, 0.5)


@dataclass(frozen=True, slots=True)
class Distribution:
    count: int
    p50: float
    p95: float
    p99: float
    maximum: float
    mean: float

    @property
    def is_empty(self) -> bool:
        return self.count == 0

    @property
    def tail_ratio(self) -> float:
        return self.p95 / self.p50 if self.p50 > 0.0 else 0.0


def summarise(samples: Sequence[float]) -> Distribution:
    if not samples:
        return Distribution(count=0, p50=0.0, p95=0.0, p99=0.0, maximum=0.0, mean=0.0)
    return Distribution(
        count=len(samples),
        p50=quantile(samples, 0.50),
        p95=quantile(samples, 0.95),
        p99=quantile(samples, 0.99),
        maximum=max(samples),
        mean=sum(samples) / len(samples),
    )


def aggregate_median(reports: Sequence[float]) -> float:
    return median(reports)
