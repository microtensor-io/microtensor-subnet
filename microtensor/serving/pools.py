from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final

from microtensor.serving.settle import Settled

DEFAULT_RESERVE: Final[Fraction] = Fraction(0)
DEFAULT_CAP: Final[Fraction] = Fraction(1, 2)
DEFAULT_REBATE: Final[Fraction] = Fraction(1, 20)
DEFAULT_REVENUE_SHARE: Final[Fraction] = Fraction(9, 10)


class PoolError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Dials:
    reserve: Fraction = DEFAULT_RESERVE
    cap: Fraction = DEFAULT_CAP
    rebate: Fraction = DEFAULT_REBATE
    revenue_share: Fraction = DEFAULT_REVENUE_SHARE

    def __post_init__(self) -> None:
        if not 0 <= self.reserve < 1:
            raise PoolError("the reserve must be at least zero and below the whole emission")
        if not 0 <= self.cap <= 1:
            raise PoolError("the serving cap must be a fraction of the available emission")
        if self.rebate < 0:
            raise PoolError("the rebate cannot be negative")
        if not 0 <= self.revenue_share <= 1:
            raise PoolError("the revenue share must be a fraction")
        if self.rebate >= 1 - self.revenue_share:
            raise PoolError(
                f"a rebate of {self.rebate} does not exclude manufactured traffic against a "
                f"revenue share of {self.revenue_share}; it must stay below "
                f"{1 - self.revenue_share}"
            )


@dataclass(frozen=True, slots=True)
class Split:
    emission: int
    available: int
    serving: int
    modelling: int
    total_score: int
    capped: bool
    shares: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "emission": self.emission,
            "available": self.available,
            "serving": self.serving,
            "modelling": self.modelling,
            "total_score": self.total_score,
            "capped": self.capped,
            "shares": dict(self.shares),
        }


def identity(row: Settled) -> str:
    return row.operator


def divide(emission: int, settled: Sequence[Settled], dials: Dials | None = None) -> Split:
    if emission < 0:
        raise PoolError("emission cannot be negative")
    rules = dials or Dials()

    available = int(Fraction(emission) * (1 - rules.reserve))
    earned = [row for row in settled if row.score > 0]
    total_score = sum(row.score for row in earned)

    rebated = int(Fraction(total_score) * rules.rebate)
    ceiling = int(Fraction(available) * rules.cap)
    serving = min(rebated, ceiling)
    capped = rebated > ceiling

    shares = _shares(serving, earned)
    return Split(
        emission=emission,
        available=available,
        serving=serving,
        modelling=available - serving,
        total_score=total_score,
        capped=capped,
        shares=shares,
    )


def _shares(serving: int, earned: Sequence[Settled]) -> dict[str, int]:
    if serving <= 0 or not earned:
        return {}
    totals: dict[str, int] = {}
    for row in earned:
        key = identity(row)
        totals[key] = totals.get(key, 0) + row.score
    whole = sum(totals.values())
    if whole <= 0:
        return {}

    shares = {key: (serving * value) // whole for key, value in totals.items()}
    remainder = serving - sum(shares.values())
    if remainder:
        largest = max(totals, key=lambda key: (totals[key], key))
        shares[largest] += remainder
    return shares


def weights(split: Split, uid_by_operator: Mapping[str, int]) -> dict[int, int]:
    found: dict[int, int] = {}
    for operator, share in split.shares.items():
        uid = uid_by_operator.get(operator)
        if uid is None or share <= 0:
            continue
        found[int(uid)] = found.get(int(uid), 0) + share
    return found
