from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

PASS: Final[str] = "pass"
CHEAT: Final[str] = "cheat"
UNPROVEN: Final[str] = "unproven"

GREEDY: Final[str] = "greedy"
SAMPLED: Final[str] = "sampled"

TIE_TOLERANCE: Final[float] = 1e-3


class VerificationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Statistic:
    mode: str
    positions: int
    disagreements: int
    disagreement_rate: float
    margin_mean: float
    margin_max: float
    nll: float

    @property
    def empty(self) -> bool:
        return self.positions == 0

    def value(self, aggregate: str) -> float:
        if self.mode == SAMPLED:
            return self.nll
        if aggregate == "rate":
            return self.disagreement_rate
        if aggregate == "margin":
            return self.margin_mean
        raise VerificationError(f"unknown aggregate {aggregate!r}; known: rate, margin")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "positions": self.positions,
            "disagreements": self.disagreements,
            "disagreement_rate": self.disagreement_rate,
            "margin_mean": self.margin_mean,
            "margin_max": self.margin_max,
            "nll": self.nll,
        }


@dataclass(frozen=True, slots=True)
class Calibration:
    mode: str
    aggregate: str
    threshold: float
    alpha: float
    honest_samples: int
    separated: bool
    substituted_min: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "aggregate": self.aggregate,
            "threshold": self.threshold,
            "alpha": self.alpha,
            "honest_samples": self.honest_samples,
            "separated": self.separated,
            "substituted_min": self.substituted_min,
        }


@dataclass(frozen=True, slots=True)
class Judgement:
    verdict: str
    score: float
    threshold: float
    reason: str
    statistic: Statistic | None = None

    def to_dict(self) -> dict[str, Any]:
        found: dict[str, Any] = {
            "verdict": self.verdict,
            "score": self.score,
            "threshold": self.threshold,
            "reason": self.reason,
        }
        if self.statistic is not None:
            found["statistic"] = self.statistic.to_dict()
        return found


def margins(logits: Sequence[Sequence[float]], tokens: Sequence[int]) -> list[float]:
    if len(logits) != len(tokens):
        raise VerificationError(
            f"{len(logits)} logit rows against {len(tokens)} tokens; the caller must align them"
        )
    found: list[float] = []
    for row, token in zip(logits, tokens, strict=True):
        if not row:
            raise VerificationError("a logit row is empty")
        if not 0 <= token < len(row):
            raise VerificationError(f"token {token} is outside a vocabulary of {len(row)}")
        gap = max(row) - row[token]
        found.append(gap if gap > 0.0 else 0.0)
    return found


def token_nll(
    logits: Sequence[Sequence[float]], tokens: Sequence[int], temperature: float = 1.0
) -> float:
    if temperature <= 0.0:
        raise VerificationError("temperature must be positive; greedy has its own statistic")
    if len(logits) != len(tokens):
        raise VerificationError(
            f"{len(logits)} logit rows against {len(tokens)} tokens; the caller must align them"
        )
    if not tokens:
        return 0.0
    total = 0.0
    for row, token in zip(logits, tokens, strict=True):
        if not 0 <= token < len(row):
            raise VerificationError(f"token {token} is outside a vocabulary of {len(row)}")
        scaled = [value / temperature for value in row]
        top = max(scaled)
        denominator = sum(math.exp(value - top) for value in scaled)
        total += -(scaled[token] - top - math.log(denominator))
    return total / len(tokens)


def statistic(
    logits: Sequence[Sequence[float]],
    tokens: Sequence[int],
    *,
    mode: str = GREEDY,
    temperature: float = 1.0,
    tolerance: float = TIE_TOLERANCE,
) -> Statistic:
    if mode not in (GREEDY, SAMPLED):
        raise VerificationError(f"unknown mode {mode!r}; known: {GREEDY}, {SAMPLED}")
    gaps = margins(logits, tokens)
    over = [gap for gap in gaps if gap > tolerance]
    positions = len(gaps)
    return Statistic(
        mode=mode,
        positions=positions,
        disagreements=len(over),
        disagreement_rate=(len(over) / positions) if positions else 0.0,
        margin_mean=(sum(over) / positions) if positions else 0.0,
        margin_max=max(over) if over else 0.0,
        nll=token_nll(logits, tokens, temperature) if mode == SAMPLED else 0.0,
    )


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise VerificationError("no honest samples to calibrate against")
    if len(values) == 1:
        return float(values[0])
    position = q * (len(values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(values[low])
    return float(values[low] + (values[high] - values[low]) * (position - low))


def calibrate(
    honest: Sequence[Statistic],
    substituted: Sequence[Statistic] = (),
    *,
    mode: str = GREEDY,
    aggregate: str = "margin",
    alpha: float = 0.01,
) -> Calibration:
    if not 0.0 < alpha < 1.0:
        raise VerificationError("alpha must lie strictly between zero and one")
    scores = sorted(row.value(aggregate) for row in honest if not row.empty)
    if not scores:
        raise VerificationError("every honest sample was empty")
    threshold = _quantile(scores, 1.0 - alpha)
    against = [row.value(aggregate) for row in substituted if not row.empty]
    return Calibration(
        mode=mode,
        aggregate=aggregate,
        threshold=threshold,
        alpha=alpha,
        honest_samples=len(scores),
        separated=bool(against) and min(against) > threshold,
        substituted_min=min(against) if against else 0.0,
    )


def separable(calibration: Calibration) -> bool:
    return calibration.separated


def judge(found: Statistic, calibration: Calibration) -> Judgement:
    if found.mode != calibration.mode:
        return Judgement(
            verdict=UNPROVEN,
            score=0.0,
            threshold=calibration.threshold,
            reason=f"statistic is {found.mode} against a {calibration.mode} threshold",
            statistic=found,
        )
    if found.empty:
        return Judgement(
            verdict=UNPROVEN,
            score=0.0,
            threshold=calibration.threshold,
            reason="the response carried no generated position to check",
            statistic=found,
        )
    if not calibration.separated:
        return Judgement(
            verdict=UNPROVEN,
            score=found.value(calibration.aggregate),
            threshold=calibration.threshold,
            reason="the threshold was never shown to separate a substitution for this artifact",
            statistic=found,
        )
    score = found.value(calibration.aggregate)
    if score > calibration.threshold:
        return Judgement(
            verdict=CHEAT,
            score=score,
            threshold=calibration.threshold,
            reason=(
                f"{calibration.aggregate} {score:.6g} over the "
                f"calibrated {calibration.threshold:.6g}"
            ),
            statistic=found,
        )
    return Judgement(
        verdict=PASS,
        score=score,
        threshold=calibration.threshold,
        reason="",
        statistic=found,
    )
