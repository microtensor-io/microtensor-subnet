"""Division of the miner share between modelling and serving.

A reserve fraction is withheld, leaving the available emission. The serving
pool is what the epoch's settlement earns at the rebate rate, capped at a
fraction of the available emission; modelling receives the remainder.

    available = (1 - reserve) * emission
    serving   = min( rebate * total_score, cap * available )
    modelling = available - serving

Three properties follow, and each is a test rather than a claim. With no
billed traffic the serving pool is empty and the whole of the available
emission funds modelling, so emission never pays for unbought service.
Serving can never exceed its cap, so modelling always receives at least one
minus the cap and can never be starved by a layer that sells what modelling
produced. And an operator buying its own traffic loses money, provided the
rebate obeys the bound below.

The bound. An operator paying for its own tokens recovers its revenue share
and gains the rebate on the reference value. Since the reference rate is set
no higher than the lowest price the model sells at, choosing

    rebate < 1 - revenue_share

makes the round trip lossy for every model, from the first billed request,
and the check is enforced here rather than left to whoever edits a constant.

All arithmetic is exact. Emission is an integer in the chain's smallest unit
and every ratio is a `Fraction`, so two validators cannot disagree by a
rounding. What a division cannot split evenly is handed to the largest share,
which is decided by score and then by identity, so the remainder is placed by
the data and not by dictionary order.
"""

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
    """The epoch's split, and what each operator takes of the serving pool."""
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
    """The serving pool by score, with the remainder to the largest earner.

    Summing per operator first matters: an operator serving several models is
    one identity on the weight vector, and rounding each of its models
    separately would cost it a unit per model rather than a unit in total.
    """
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
    """The serving side of the weight vector, keyed by uid.

    An operator the metagraph does not name is dropped rather than carried,
    because a weight cannot be set for a uid that does not exist. Its share is
    not redistributed: it was earned by an identity this network cannot pay,
    and moving it would pay somebody who did not earn it.
    """
    found: dict[int, int] = {}
    for operator, share in split.shares.items():
        uid = uid_by_operator.get(operator)
        if uid is None or share <= 0:
            continue
        found[int(uid)] = found.get(int(uid), 0) + share
    return found
