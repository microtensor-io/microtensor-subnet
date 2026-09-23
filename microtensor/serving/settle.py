"""Settlement of a serving epoch, in exact arithmetic.

Every validator computes this vector independently and must reach the same
integer, so nothing here uses a float. Ratios are `Fraction`, which is a pair
of integers, and the only division that reaches a result is a floor. A change
to the arithmetic is a change to consensus, not an implementation detail.

Settlement is per operator and model over billed requests only:

    score = floor( sum over billed requests of tokens * rate ) if every gate
            passes, else zero

The rate is a per-model reference value fixed by the network rather than the
client-facing price, so identical work scores identically whatever discount a
client received, and it is set no higher than the lowest price at which the
model can be bought, which is what makes manufactured traffic unprofitable
once paired with the rebate bound in `pools`.

Four gates apply, and each fails only when the lower confidence bound of the
observed failure rate exceeds its threshold. A gate evaluated on few requests
therefore passes unless the evidence of failure is statistically clear, which
removes the arbitrary minimum sample count that the same job is usually done
with. A failed gate forfeits the epoch and excludes nobody.
"""

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

# One-sided 99% normal quantile, as an exact ratio. Every validator squares
# the same pair of integers, so no two of them can disagree by a rounding.
WILSON_Z: Final[Fraction] = Fraction(23263, 10000)

TOKENS_PER_RATE_UNIT: Final[int] = 1000


class SettlementError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Observation:
    """One gate's evidence: how many trials, and how many of them failed."""

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
    """The lower confidence bound on a failure rate, exactly.

    Scaling the usual expression by the number of trials clears the inner
    division and leaves

        L = (k + z^2/2 - z * sqrt(k(n-k)/n + z^2/4)) / (n + z^2)

    whose only irrational part is one square root. The caller compares the
    bound against a threshold rather than reading it, so `gate` avoids the root
    entirely; this function exists to report the number, and rounds the root
    down so the bound it returns is never optimistic about failure.
    """
    if trials <= 0:
        return Fraction(0)
    k, n = Fraction(failures), Fraction(trials)
    z2 = z * z
    inner = k * (n - k) / n + z2 / 4
    root = _sqrt_floor(inner)
    bound = (k + z2 / 2 - z * root) / (n + z2)
    return bound if bound > 0 else Fraction(0)


def _sqrt_floor(value: Fraction, places: int = 12) -> Fraction:
    """A rational at or below the true square root, to the given decimal places."""
    if value <= 0:
        return Fraction(0)
    scale = 10**places
    target = value * scale * scale
    root = math.isqrt(target.numerator // target.denominator)
    return Fraction(root, scale)


def gate(name: str, observed: Observation, threshold: Fraction, z: Fraction = WILSON_Z) -> Verdict:
    """Whether the evidence of failure is statistically clear.

    The comparison is made without taking a square root. Writing the bound as

        L > t  <=>  A > z * sqrt(B),  A = k + z^2/2 - t(n + z^2),  B = k(n-k)/n + z^2/4

    a non-positive A settles it immediately, since the right side is never
    negative, and otherwise both sides are squared. Every term is a ratio of
    integers, so the decision is exact and identical on every validator.
    """
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
    """Billed tokens at the network's reference rate, floored to an integer."""
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
    """One operator and model over one epoch.

    Gates are evaluated in a fixed order and every verdict is kept, so a
    settlement records what was observed rather than only what it concluded.
    The first gate to fail names the reason.
    """
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
    """Every operator and model in one epoch, ordered so the result is stable.

    A model with no published reference rate is refused rather than scored at
    zero: a missing rate is an operator error on our side, and silently paying
    nobody for real work would hide it.
    """
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
