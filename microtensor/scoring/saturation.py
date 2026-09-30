from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

SATURATION_CEILING: Final[float] = 0.95
SATURATION_ROUNDS: Final[int] = 3
STALL_EPSILON: Final[float] = 0.005


@dataclass(frozen=True, slots=True)
class Bar:
    rounds: int
    best: float
    gained: float
    saturated: bool
    stalled: bool

    @property
    def verdict(self) -> str:
        if self.saturated:
            return "saturated: retire the fixed set and publish a new corpus version"
        if self.stalled:
            return "stalled: the best system has not moved; consider fresh withheld tasks"
        return "moving"


def assess(
    best: Sequence[float],
    *,
    ceiling: float = SATURATION_CEILING,
    rounds: int = SATURATION_ROUNDS,
) -> Bar:
    recent = [float(value) for value in best][-rounds:]
    if not recent:
        return Bar(rounds=0, best=0.0, gained=0.0, saturated=False, stalled=False)
    full = len(recent) >= rounds
    gained = recent[-1] - recent[0]
    return Bar(
        rounds=len(recent),
        best=recent[-1],
        gained=gained,
        saturated=full and min(recent) >= ceiling,
        stalled=full and abs(gained) < STALL_EPSILON,
    )
