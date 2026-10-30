from __future__ import annotations

import datetime as dt
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Final

from microtensor_subnet.serving.settle import Settled

DEFAULT_RESERVE: Final[Fraction] = Fraction(0)
DEFAULT_CAP: Final[Fraction] = Fraction(1, 10)
DEFAULT_REBATE: Final[Fraction] = Fraction(1, 25)
DEFAULT_REVENUE_SHARE: Final[Fraction] = Fraction(19, 20)
MAX_FLOOR_SHARE: Final[Fraction] = Fraction(1, 2)
MODEL_PREFIX: Final[str] = "mt/"


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
    burned: int = 0
    floors: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "emission": self.emission,
            "available": self.available,
            "serving": self.serving,
            "modelling": self.modelling,
            "total_score": self.total_score,
            "capped": self.capped,
            "shares": dict(self.shares),
            "burned": self.burned,
            "floors": dict(self.floors),
        }


def fraction_of(value: Any) -> Fraction:
    try:
        found = Fraction(str(value)) if isinstance(value, float) else Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise PoolError(f"{value!r} is not a number") from exc
    if found < 0:
        raise PoolError(f"{value!r} is negative")
    return found


def arena_of(model: str) -> str:
    name = model[len(MODEL_PREFIX) :] if model.startswith(MODEL_PREFIX) else model
    head, sep, tail = name.rpartition("-")
    if sep and tail.endswith("g") and tail[:-1].isdigit():
        return head
    return name


def cap_for(
    schedule: Sequence[Mapping[str, Any]] | None,
    billing_usd: Any,
    default: Fraction = DEFAULT_CAP,
) -> Fraction:
    bands = [dict(band) for band in schedule or []]
    if not bands:
        return default
    billing = fraction_of(billing_usd)
    previous: Fraction | None = None
    for index, band in enumerate(bands):
        cap = fraction_of(band.get("cap"))
        if cap > 1:
            raise PoolError("a cap in the schedule must be a fraction of the emission")
        bound = band.get("billing_usd_below")
        if bound is None:
            if index != len(bands) - 1:
                raise PoolError("the band without billing_usd_below must be the last one")
            return cap
        limit = fraction_of(bound)
        if previous is not None and limit <= previous:
            raise PoolError("cap schedule bands must rise in billing_usd_below")
        previous = limit
        if billing < limit:
            return cap
    raise PoolError("the cap schedule needs a last band without billing_usd_below")


def dials_for(serving: Mapping[str, Any], billing_usd: Any) -> Dials:
    return Dials(cap=cap_for(serving.get("cap_schedule"), billing_usd))


def floors_for(
    serving: Mapping[str, Any],
    operators: Sequence[Mapping[str, Any]],
    *,
    now: dt.datetime,
    epoch_seconds: float,
    models: Collection[str] | None = None,
) -> dict[str, Fraction]:
    epochs = int(serving.get("floor_epochs") or 0)
    share = fraction_of(serving.get("floor_share") or 0)
    if epochs <= 0 or share <= 0:
        return {}
    if share > MAX_FLOOR_SHARE:
        raise PoolError(f"a floor share of {share} would hand most of the pool to newcomers")
    found: dict[str, Fraction] = {}
    for operator in operators:
        hotkey = str(operator.get("hotkey") or "")
        admitted = str(operator.get("admitted_at") or "")
        if not hotkey or not admitted:
            continue
        served = [str(m) for m in operator.get("models") or []]
        if models is not None and not any(m in models for m in served):
            continue
        try:
            moment = dt.datetime.fromisoformat(admitted)
        except ValueError:
            continue
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=dt.timezone.utc)
        age = (now - moment).total_seconds()
        if 0 <= age < epochs * epoch_seconds:
            found[hotkey] = share
    return found


def identity(row: Settled) -> str:
    return row.operator


def divide(
    emission: int,
    settled: Sequence[Settled],
    dials: Dials | None = None,
    *,
    floors: Mapping[str, Fraction] | None = None,
) -> Split:
    if emission < 0:
        raise PoolError("emission cannot be negative")
    rules = dials or Dials()

    available = int(Fraction(emission) * (1 - rules.reserve))
    earned = [row for row in settled if row.score > 0]
    total_score = sum(row.score for row in earned)

    rebated = int(Fraction(total_score) * rules.rebate)
    ceiling = int(Fraction(available) * rules.cap)
    paid = min(rebated, ceiling)

    shares = _shares(paid, earned)
    granted = _floors(ceiling, shares, floors or {})
    return Split(
        emission=emission,
        available=available,
        serving=ceiling,
        modelling=available - ceiling,
        total_score=total_score,
        capped=rebated > ceiling,
        shares=shares,
        burned=ceiling - sum(shares.values()),
        floors=granted,
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


def _floors(ceiling: int, shares: dict[str, int], floors: Mapping[str, Fraction]) -> dict[str, int]:
    granted: dict[str, int] = {}
    remaining = ceiling - sum(shares.values())
    for operator, share in sorted(floors.items()):
        target = int(Fraction(ceiling) * share)
        gap = target - shares.get(operator, 0)
        if gap <= 0 or remaining <= 0:
            continue
        topup = min(gap, remaining)
        shares[operator] = shares.get(operator, 0) + topup
        granted[operator] = topup
        remaining -= topup
    return granted


def weights(split: Split, uid_by_operator: Mapping[str, int]) -> dict[int, int]:
    found: dict[int, int] = {}
    for operator, share in split.shares.items():
        uid = uid_by_operator.get(operator)
        if uid is None or share <= 0:
            continue
        found[int(uid)] = found.get(int(uid), 0) + share
    return found
