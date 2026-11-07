from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final

SCALE: Final[int] = 1000
OUTSIDE_WEIGHT: Final[Fraction] = Fraction(1, 2)
TOKENS_PER_RATE_UNIT: Final[int] = 1000


class SettlementError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Settled:
    operator: str
    model: str
    gpu: str
    inside: int
    outside: int
    probes: int
    passes: int
    tokens: int
    billing: int
    score: int
    coldkey: str = ""
    team: bool = False
    first_live: tuple[str, ...] = ()

    @property
    def served(self) -> int:
        return self.inside + self.outside

    @property
    def pass_rate(self) -> Fraction:
        return pass_rate(self.probes, self.passes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "model": self.model,
            "gpu": self.gpu,
            "inside": self.inside,
            "outside": self.outside,
            "probes": self.probes,
            "passes": self.passes,
            "tokens": self.tokens,
            "billing": self.billing,
            "score": self.score,
            "coldkey": self.coldkey,
            "team": self.team,
            "first_live": list(self.first_live),
        }


def _count(row: Mapping[str, Any], name: str) -> int:
    try:
        found = int(row.get(name, 0) or 0)
    except (TypeError, ValueError) as exc:
        raise SettlementError(f"{name} must be a count") from exc
    if found < 0:
        raise SettlementError(f"{name} cannot be negative")
    return found


def pass_rate(probes: int, passes: int) -> Fraction:
    if probes < 0 or passes < 0 or passes > probes:
        raise SettlementError(f"{passes} passes out of {probes} probes is not a rate")
    return Fraction(1) if probes == 0 else Fraction(passes, probes)


def score_of(inside: int, outside: int, probes: int, passes: int) -> int:
    if inside < 0 or outside < 0:
        raise SettlementError("served requests are counts and cannot be negative")
    weighted = Fraction(inside) + OUTSIDE_WEIGHT * Fraction(outside)
    return int(weighted * pass_rate(probes, passes) * SCALE)


def billing_of(tokens: int, rate: int) -> int:
    if tokens < 0 or rate < 0:
        raise SettlementError("tokens and rate are counts and cannot be negative")
    return (tokens * rate) // TOKENS_PER_RATE_UNIT


def settle_row(row: Mapping[str, Any], *, rate: int) -> Settled:
    inside, outside = _count(row, "inside"), _count(row, "outside")
    probes, passes = _count(row, "probes"), _count(row, "passes")
    tokens = _count(row, "tokens")
    return Settled(
        operator=str(row.get("operator", "")),
        model=str(row.get("model", "")),
        gpu=str(row.get("gpu", "") or ""),
        inside=inside,
        outside=outside,
        probes=probes,
        passes=passes,
        tokens=tokens,
        billing=billing_of(tokens, rate),
        score=score_of(inside, outside, probes, passes),
        coldkey=str(row.get("coldkey", "") or ""),
        team=bool(row.get("team", False)),
        first_live=tuple(str(d) for d in row.get("first_live") or ()),
    )


def settle_epoch(rows: Sequence[Mapping[str, Any]], *, rates: Mapping[str, int]) -> list[Settled]:
    settled: list[Settled] = []
    for row in rows:
        model = str(row.get("model", ""))
        if model not in rates:
            raise SettlementError(f"no reference rate published for model {model!r}")
        settled.append(settle_row(row, rate=int(rates[model])))
    settled.sort(key=lambda s: (s.operator, s.model, s.gpu))
    return settled
