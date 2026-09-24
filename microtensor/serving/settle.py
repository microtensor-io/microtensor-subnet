from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Final

SUCCESS: Final[str] = "success_rate"
TTFT: Final[str] = "ttft_p99"
TPOT: Final[str] = "tpot_p99"
CHEAT_SHARE: Final[str] = "cheat_share"

GATES: Final[tuple[str, ...]] = (SUCCESS, TTFT, TPOT, CHEAT_SHARE)

WILSON_Z: Final[Fraction] = Fraction(23263, 10000)

TOKENS_PER_RATE_UNIT: Final[int] = 1000


class SettlementError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Observation:
    trials: int
    failures: int

    def __post_init__(self) -> None:
        if self.trials < 0 or self.failures < 0:
            raise SettlementError("a gate cannot observe a negative count")
        if self.failures > self.trials:
            raise SettlementError(
                f"{self.failures} failures out of {self.trials} trials is not a rate"
            )


@dataclass(frozen=True, slots=True)
class Verdict:
    name: str
    passed: bool
    trials: int
    failures: int
    bound: Fraction
    threshold: Fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.name,
            "passed": self.passed,
            "trials": self.trials,
            "failures": self.failures,
            "lower_bound": [self.bound.numerator, self.bound.denominator],
            "threshold": [self.threshold.numerator, self.threshold.denominator],
        }


@dataclass(frozen=True, slots=True)
class Settled:
    operator: str
    model: str
    tokens: int
    requests: int
    score: int
    gate_reason: str
    verdicts: tuple[Verdict, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "model": self.model,
            "tokens": self.tokens,
            "requests": self.requests,
            "score": self.score,
            "gate_reason": self.gate_reason,
            "gates": [v.to_dict() for v in self.verdicts],
        }


def wilson_lower_bound(failures: int, trials: int, z: Fraction = WILSON_Z) -> Fraction:
    if trials <= 0:
        return Fraction(0)
    k, n = Fraction(failures), Fraction(trials)
    z2 = z * z
    inner = k * (n - k) / n + z2 / 4
    root = _sqrt_floor(inner)
    bound = (k + z2 / 2 - z * root) / (n + z2)
    return bound if bound > 0 else Fraction(0)


def _sqrt_floor(value: Fraction, places: int = 12) -> Fraction:
    if value <= 0:
        return Fraction(0)
    scale = 10**places
    target = value * scale * scale
    root = math.isqrt(target.numerator // target.denominator)
    return Fraction(root, scale)


def gate(name: str, observed: Observation, threshold: Fraction, z: Fraction = WILSON_Z) -> Verdict:
    n, k = observed.trials, observed.failures
    if n <= 0:
        return Verdict(name, True, n, k, Fraction(0), threshold)
    z2 = z * z
    a = Fraction(k) + z2 / 2 - threshold * (Fraction(n) + z2)
    if a <= 0:
        failed = False
    else:
        b = Fraction(k) * Fraction(n - k) / Fraction(n) + z2 / 4
        failed = a * a > z2 * b
    return Verdict(name, not failed, n, k, wilson_lower_bound(k, n, z), threshold)


def score_of(tokens: int, rate: int) -> int:
    if tokens < 0 or rate < 0:
        raise SettlementError("tokens and rate are counts and cannot be negative")
    return (tokens * rate) // TOKENS_PER_RATE_UNIT


def settle_pair(
    operator: str,
    model: str,
    *,
    tokens: int,
    requests: int,
    rate: int,
    observations: Mapping[str, Observation],
    thresholds: Mapping[str, Fraction],
    z: Fraction = WILSON_Z,
) -> Settled:
    verdicts: list[Verdict] = []
    reason = ""
    for name in GATES:
        found = observations.get(name, Observation(0, 0))
        threshold = thresholds.get(name)
        if threshold is None:
            raise SettlementError(f"no threshold published for gate {name!r}")
        verdict = gate(name, found, threshold, z)
        verdicts.append(verdict)
        if not verdict.passed and not reason:
            reason = name
    return Settled(
        operator=operator,
        model=model,
        tokens=tokens,
        requests=requests,
        score=0 if reason else score_of(tokens, rate),
        gate_reason=reason,
        verdicts=tuple(verdicts),
    )


def settle_epoch(
    rows: Sequence[Mapping[str, Any]],
    *,
    rates: Mapping[str, int],
    thresholds: Mapping[str, Fraction],
    z: Fraction = WILSON_Z,
) -> list[Settled]:
    settled: list[Settled] = []
    for row in rows:
        model = str(row.get("model", ""))
        if model not in rates:
            raise SettlementError(f"no reference rate published for model {model!r}")
        observations = {
            name: Observation(*row.get("gates", {}).get(name, (0, 0))) for name in GATES
        }
        settled.append(
            settle_pair(
                str(row.get("operator", "")),
                model,
                tokens=int(row.get("tokens", 0) or 0),
                requests=int(row.get("requests", 0) or 0),
                rate=rates[model],
                observations=observations,
                thresholds=thresholds,
                z=z,
            )
        )
    settled.sort(key=lambda s: (s.operator, s.model))
    return settled
