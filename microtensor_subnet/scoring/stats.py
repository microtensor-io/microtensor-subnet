from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

Z_ONE_SIDED_95: Final[float] = 1.645
BOOTSTRAP_RESAMPLES: Final[int] = 1000
BOOTSTRAP_PERCENTILE: Final[float] = 0.05
DIGITS: Final[int] = 6


def wilson_lb(successes: int, trials: int, z: float = Z_ONE_SIDED_95) -> float:
    if trials <= 0:
        return 0.0
    if successes < 0 or successes > trials:
        raise ValueError("successes must lie between zero and trials")
    p = successes / trials
    z2 = z * z
    centre = p + z2 / (2 * trials)
    spread = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))
    bound = (centre - spread) / (1 + z2 / trials)
    return round(max(0.0, min(1.0, bound)), DIGITS)


@dataclass(frozen=True, slots=True)
class Paired:
    delta: float
    standard_error: float
    lower_bound: float
    n: int
    families: int


def paired_lb(
    a: Sequence[float],
    b: Sequence[float],
    families: Sequence[str] | None = None,
    z: float = Z_ONE_SIDED_95,
) -> Paired:
    if len(a) != len(b):
        raise ValueError("paired conditions must cover the same items")
    n = len(a)
    if n == 0:
        return Paired(0.0, 0.0, 0.0, 0, 0)
    groups = list(families) if families is not None else [str(i) for i in range(n)]
    if len(groups) != n:
        raise ValueError("every item needs a family")
    deltas = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    mean = math.fsum(deltas) / n
    by_family: dict[str, float] = {}
    for group, d in zip(groups, deltas, strict=True):
        by_family[group] = by_family.get(group, 0.0) + (d - mean)
    se = math.sqrt(math.fsum(total * total for total in by_family.values())) / n
    return Paired(
        delta=round(mean, DIGITS),
        standard_error=round(se, DIGITS),
        lower_bound=round(mean - z * se, DIGITS),
        n=n,
        families=len(by_family),
    )


def bootstrap_seed(epoch_seed: str, label: str) -> int:
    digest = hashlib.sha256(f"{epoch_seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_lb(
    by_family: Mapping[str, Sequence[float]],
    metric: Callable[[Sequence[float]], float],
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    percentile: float = BOOTSTRAP_PERCENTILE,
) -> float:
    names = sorted(by_family)
    if not names:
        return 0.0
    rng = random.Random(seed)  # noqa: S311
    draws: list[float] = []
    for _ in range(resamples):
        chosen = [names[rng.randrange(len(names))] for _ in names]
        pooled = [value for name in chosen for value in by_family[name]]
        draws.append(metric(pooled) if pooled else 0.0)
    draws.sort()
    position = min(len(draws) - 1, max(0, math.floor(percentile * len(draws))))
    return round(draws[position], DIGITS)


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def lower_bound(
    outcomes: Sequence[float],
    *,
    pass_rate: bool,
    by_family: Mapping[str, Sequence[float]] | None = None,
    seed: int = 0,
) -> float:
    if pass_rate:
        return wilson_lb(sum(1 for value in outcomes if value >= 1.0), len(outcomes))
    grouped = by_family if by_family is not None else {str(i): [v] for i, v in enumerate(outcomes)}
    return bootstrap_lb(grouped, mean, seed)


def ladder(previous_best: float | None, new_score: float, step: float) -> float:
    if previous_best is None:
        return round(new_score, DIGITS)
    return round(new_score if new_score > previous_best + step else previous_best, DIGITS)


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = fraction * (len(ordered) - 1)
    low = math.floor(rank)
    high = min(len(ordered) - 1, low + 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight
