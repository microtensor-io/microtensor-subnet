from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

SLM: Final[str] = "slm"
HARNESS: Final[str] = "harness"
ROUTER: Final[str] = "router"
ESCALATION: Final[str] = "escalation"
SYSTEM: Final[str] = "system"
POOLS: Final[tuple[str, ...]] = (SLM, HARNESS, ROUTER, ESCALATION, SYSTEM)

POOL_SHARES: Final[dict[str, float]] = dict.fromkeys(POOLS, 0.2)
RANK_PAYOUT: Final[tuple[float, ...]] = (0.40, 0.25, 0.15, 0.12, 0.08)
LADDER_STEP: Final[float] = 0.01
DIGITS: Final[int] = 9


@dataclass(frozen=True, slots=True)
class Standing:
    hotkey: str
    score: float
    eligible: bool
    commit_block: int


def payout_schedule(eligible: int, schedule: Sequence[float] = RANK_PAYOUT) -> tuple[float, ...]:
    paid = list(schedule[: max(0, eligible)])
    total = math.fsum(paid)
    if not paid or total <= 0:
        return ()
    return tuple(share / total for share in paid)


def rank(standings: Sequence[Standing]) -> list[Standing]:
    eligible = [s for s in standings if s.eligible and s.score > 0]
    return sorted(eligible, key=lambda s: (-s.score, s.commit_block, s.hotkey))


def pool_payout(
    standings: Sequence[Standing], schedule: Sequence[float] = RANK_PAYOUT
) -> dict[str, float]:
    ordered = rank(standings)[: len(schedule)]
    shares = payout_schedule(len(ordered), schedule)
    return {standing.hotkey: share for standing, share in zip(ordered, shares, strict=True)}


def combine(
    pools: Mapping[str, Sequence[Standing]],
    shares: Mapping[str, float] = POOL_SHARES,
    schedule: Sequence[float] = RANK_PAYOUT,
) -> tuple[dict[str, float], dict[str, float]]:
    weights: dict[str, float] = {}
    burned: dict[str, float] = {}
    for pool, share in shares.items():
        paid = pool_payout(pools.get(pool, ()), schedule)
        if not paid:
            burned[pool] = share
            continue
        for hotkey, fraction in paid.items():
            weights[hotkey] = weights.get(hotkey, 0.0) + share * fraction
    total = math.fsum(weights.values())
    if total <= 0:
        return {}, burned
    return {hotkey: round(value / total, DIGITS) for hotkey, value in weights.items()}, burned
