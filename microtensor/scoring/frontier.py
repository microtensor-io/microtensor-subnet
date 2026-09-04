from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from microtensor.core.constants import (
    EPS_COST,
    EPS_QUALITY,
    HV_QUANT_C,
    HV_QUANT_Q,
    INCUMBENT_DECAY,
    MIN_ROUNDS_OBSERVED,
    POSITION_SHARES,
    REFERENCE_COST_MS,
    TRACK_THRESHOLD,
)


@dataclass(frozen=True, slots=True)
class Point:
    key: str
    qi: int
    ci: int
    committed_at: int = 0
    stale_rounds: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.qi <= HV_QUANT_Q:
            raise ValueError(f"quality index {self.qi} is off the grid")
        if not 0 <= self.ci <= HV_QUANT_C:
            raise ValueError(f"cost index {self.ci} is off the grid")


@dataclass(frozen=True, slots=True)
class Entrant:
    key: str
    quality: float
    cost: float
    committed_at: int = 0
    rounds_observed: int = MIN_ROUNDS_OBSERVED
    stale_rounds: int = 0


def quantise_point(quality: float, cost: float, cost_ceiling: float) -> tuple[int, int]:
    """Map a measured (quality, cost) pair onto an integer grid.

    Every downstream comparison runs on these integers. Two validators that
    measure the same system to within a grid cell therefore agree exactly on
    frontier membership, which floating point comparison could not guarantee.
    """
    if cost_ceiling <= 0.0:
        raise ValueError("the cost ceiling must be positive")
    qi = round(max(0.0, min(quality, 1.0)) * HV_QUANT_Q)
    ci = round(min(max(cost, 0.0) / cost_ceiling, 1.0) * HV_QUANT_C)
    return int(qi), int(ci)


def to_points(
    entrants: Sequence[Entrant], cost_ceiling: float = REFERENCE_COST_MS
) -> list[Point]:
    points: list[Point] = []
    for entrant in entrants:
        qi, ci = quantise_point(entrant.quality, entrant.cost, cost_ceiling)
        points.append(
            Point(
                key=entrant.key,
                qi=qi,
                ci=ci,
                committed_at=entrant.committed_at,
                stale_rounds=entrant.stale_rounds,
            )
        )
    return points


def eligible(
    entrants: Sequence[Entrant],
    threshold: float = TRACK_THRESHOLD,
    min_rounds: int = MIN_ROUNDS_OBSERVED,
) -> list[Entrant]:
    return [
        e
        for e in entrants
        if e.quality >= threshold and e.quality > 0.0 and e.rounds_observed >= min_rounds
    ]


def epsilon_margins() -> tuple[int, int]:
    return round(EPS_QUALITY * HV_QUANT_Q), round(EPS_COST * HV_QUANT_C)


def dominates(a: Point, b: Point, eq: int | None = None, ec: int | None = None) -> bool:
    """a eps-dominates b: better on both axes by at least the stated margins."""
    if eq is None or ec is None:
        default_q, default_c = epsilon_margins()
        eq = default_q if eq is None else eq
        ec = default_c if ec is None else ec
    return a.qi >= b.qi + eq and a.ci <= b.ci - ec


def near(a: Point, b: Point, eq: int, ec: int) -> bool:
    """Whether two points sit inside one eps-neighbourhood on both axes."""
    return abs(a.qi - b.qi) < eq and abs(a.ci - b.ci) < ec


def frontier(points: Sequence[Point]) -> list[Point]:
    """The eps-non-dominated set, collapsed to one point per neighbourhood.

    Two systems within (eq, ec) of each other do not eps-dominate each other,
    so dominance alone leaves both on the frontier. Under leave-one-out each
    would then measure near-zero exclusive area, because neither covers much
    the other does not. As a competition matures and the frontier clusters,
    that collapses total exclusive hypervolume and makes emission noisy, while
    a mediocre but isolated system out-earns two good ones standing together.

    So the exact-tie rule extends to the whole neighbourhood: the earliest
    commitment holds the position and the later is excluded. That is also the
    right answer on its own terms, since the later submission added nothing the
    earlier one had not already provided.
    """
    eq, ec = epsilon_margins()
    survivors = [
        p for p in points if not any(dominates(o, p, eq, ec) for o in points if o is not p)
    ]

    kept: list[Point] = []
    for point in sorted(survivors, key=lambda p: (p.committed_at, p.key)):
        if any(near(point, held, eq, ec) for held in kept):
            continue
        kept.append(point)

    return sorted(kept, key=lambda p: (p.ci, -p.qi, p.key))


def staircase(points: Sequence[Point]) -> list[Point]:
    """The attained set, ascending in cost and strictly ascending in quality.

    A point the staircase drops is covered entirely by a cheaper one of equal
    or better quality, so it contributes nothing and earns nothing.
    """
    ordered = sorted(points, key=lambda p: (p.ci, -p.qi, p.key))
    kept: list[Point] = []
    for point in ordered:
        if kept and point.qi <= kept[-1].qi:
            continue
        kept.append(point)
    return kept


def hypervolume(points: Sequence[Point]) -> int:
    """Total attained hypervolume against the reference (qi=0, ci=HV_QUANT_C)."""
    total = 0
    previous_quality = 0
    for point in staircase(points):
        total += (point.qi - previous_quality) * (HV_QUANT_C - point.ci)
        previous_quality = point.qi
    return total


def exclusive_hypervolume(points: Sequence[Point]) -> dict[str, int]:
    """Each point's leave-one-out contribution, HV(F) - HV(F minus that point).

    With two objectives the attained region is a staircase, so the region a
    point alone covers is the rectangle between its own quality step and the
    cost of the next cheapest step up. That makes the whole sweep O(n log n)
    rather than requiring the general hypervolume algorithm.

    Note this is not the quality band (q_i - q_{i-1}) * (C_max - c_i): that
    band is the point's slice of the *total* hypervolume, and counts area its
    neighbours also cover. Emission follows what a system uniquely adds, so
    removing it is the measurement that matters.
    """
    if not points:
        return {}

    kept = staircase(points)
    areas = {p.key: 0 for p in points}

    previous_quality = 0
    for index, point in enumerate(kept):
        next_cost = kept[index + 1].ci if index + 1 < len(kept) else HV_QUANT_C
        areas[point.key] = (point.qi - previous_quality) * (next_cost - point.ci)
        previous_quality = point.qi

    return areas


def apply_incumbent_decay(
    areas: dict[str, int],
    points: Sequence[Point],
    decay: float = INCUMBENT_DECAY,
) -> dict[str, float]:
    stale = {p.key: p.stale_rounds for p in points}
    decayed: dict[str, float] = {}
    freed = 0.0

    for key, area in areas.items():
        rounds = stale.get(key, 0)
        if rounds <= 0:
            decayed[key] = float(area)
            continue
        retained = area * (1.0 - decay) ** rounds
        freed += area - retained
        decayed[key] = retained

    if freed <= 0.0:
        return decayed

    receivers = [k for k in decayed if stale.get(k, 0) <= 0 and decayed[k] > 0.0]
    if not receivers:
        return decayed

    basis = sum(decayed[k] for k in receivers)
    for key in receivers:
        decayed[key] += freed * decayed[key] / basis
    return decayed


def standing(
    entrants: Sequence[Entrant],
    areas: Mapping[str, float],
    cost_ceiling: float = REFERENCE_COST_MS,
) -> list[str]:
    best: dict[str, tuple[int, float]] = {}
    for entrant in entrants:
        area = float(areas.get(entrant.key, 0.0))
        place = (1, area) if area > 0.0 else (0, entrant.quality * (cost_ceiling - entrant.cost))
        if entrant.key not in best or place > best[entrant.key]:
            best[entrant.key] = place
    return [
        key
        for key, _ in sorted(best.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))
    ]


def allocate(
    entrants: Sequence[Entrant],
    *,
    cost_ceiling: float = REFERENCE_COST_MS,
    threshold: float = TRACK_THRESHOLD,
    min_rounds: int = MIN_ROUNDS_OBSERVED,
    incumbent_decay: float = INCUMBENT_DECAY,
    shares: Sequence[float] = POSITION_SHARES,
) -> dict[str, float]:
    """Emission shares by rank on the cost-quality frontier, paid down a ladder."""
    survivors = eligible(entrants, threshold, min_rounds)
    if not survivors:
        return {}

    members = frontier(to_points(survivors, cost_ceiling))
    areas = apply_incumbent_decay(exclusive_hypervolume(members), members, incumbent_decay)
    order = standing(survivors, areas, cost_ceiling)[: len(shares)]
    if not order:
        return {}

    paid = {key: float(shares[index]) for index, key in enumerate(order)}
    total = sum(paid.values())
    if total <= 0.0:
        return {}

    return {key: value / total for key, value in paid.items()}
