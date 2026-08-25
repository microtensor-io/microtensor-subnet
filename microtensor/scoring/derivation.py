"""Detecting a copy of an already-published artifact, without flagging honest work.

Every miner distils the same pinned base toward the same metric, so artifacts
converge in weight space whether or not one copied another. A naive similarity
threshold flags the field. Three signals that fail in different ways, with a
flag only on the conjunction of two, is what makes the test usable here.

Signal 1, displacement cosine: compare (A - base) against (B - base). Two
independent distillations move far from the base in different directions;
subtracting the shared origin removes exactly the convergence that produces
false positives.

Signal 2, attention profile: the per-layer standard-deviation profile of
attention projection matrices is stable under continued training, so it
survives the retraining that drifts a raw displacement.

Signal 3, behavioural: Jensen-Shannon divergence between output distributions
on a validator-held probe set, the only signal that works when weights cannot
be read at all.

Nothing here scores. It records distances. Thresholds are set from the measured
distribution in this network, not imported, which is why the enforcing switch
stays off until a report-only period has run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Report-only defaults. Deliberately not the literature's numbers: those were
# calibrated on unrelated model families, and importing them would flag honest
# miners on a field of same-base distillations. Replace from measured spread.
DISPLACEMENT_FLAG = 0.97
ATTENTION_FLAG = 0.985
BEHAVIOUR_FLAG = 0.02
SIGNALS_TO_FLAG = 2


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


def displacement_cosine(
    artifact_a: list[float], artifact_b: list[float], base: list[float]
) -> float:
    """Cosine between the two artifacts' displacement from the shared base.

    Near one means the two moved the base in near-identical directions, which
    independent training does not produce. The base is subtracted first, which
    is the whole point: it removes the term the artifacts share by construction.
    """
    delta_a = [x - m for x, m in zip(artifact_a, base, strict=True)]
    delta_b = [x - m for x, m in zip(artifact_b, base, strict=True)]
    return cosine(delta_a, delta_b)


def attention_profile_similarity(profile_a: list[float], profile_b: list[float]) -> float:
    """Cosine between two per-layer attention std-deviation profiles.

    A profile is one number per layer: the standard deviation of that layer's
    attention projection weights. The shape of that curve across depth is a
    fingerprint that continued training leaves largely intact.
    """
    return cosine(profile_a, profile_b)


def _js_divergence(p: list[float], q: list[float]) -> float:
    total_p = sum(p) or 1.0
    total_q = sum(q) or 1.0
    p = [x / total_p for x in p]
    q = [x / total_q for x in q]
    m = [(pi + qi) / 2 for pi, qi in zip(p, q, strict=True)]

    def _kl(a: list[float], b: list[float]) -> float:
        out = 0.0
        for ai, bi in zip(a, b, strict=True):
            if ai > 0.0 and bi > 0.0:
                out += ai * math.log(ai / bi)
        return out

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def behaviour_divergence(outputs_a: list[list[float]], outputs_b: list[list[float]]) -> float:
    """Mean Jensen-Shannon divergence across a probe set.

    Zero means the two artifacts answer the probes identically, which a copy
    does and independent training does not. Returned as a divergence, so the
    flag is a value below the threshold, unlike the two cosines above.
    """
    if not outputs_a or len(outputs_a) != len(outputs_b):
        return math.inf
    divergences = [_js_divergence(a, b) for a, b in zip(outputs_a, outputs_b, strict=True)]
    return sum(divergences) / len(divergences)


@dataclass(frozen=True, slots=True)
class Signal:
    name: str
    value: float | None
    fired: bool
    available: bool


@dataclass(frozen=True, slots=True)
class Comparison:
    earlier: str
    later: str
    signals: tuple[Signal, ...]
    flagged: bool
    report_only: bool

    def trace(self) -> dict[str, object]:
        return {
            "earlier": self.earlier,
            "later": self.later,
            "flagged": self.flagged,
            "report_only": self.report_only,
            "signals": {
                s.name: {"value": s.value, "fired": s.fired, "available": s.available}
                for s in self.signals
            },
        }


@dataclass(frozen=True, slots=True)
class Thresholds:
    displacement: float = DISPLACEMENT_FLAG
    attention: float = ATTENTION_FLAG
    behaviour: float = BEHAVIOUR_FLAG
    needed: int = SIGNALS_TO_FLAG


@dataclass
class Evidence:
    """What is known about two artifacts, filled in as far as it can be.

    A signal whose inputs are missing records available=False and never fires,
    so an artifact whose weights cannot be read is judged on whatever signals
    remain rather than being flagged or cleared by default.
    """

    displacement: float | None = None
    attention: float | None = None
    behaviour: float | None = None
    notes: list[str] = field(default_factory=list)


def compare(
    earlier: str,
    later: str,
    evidence: Evidence,
    thresholds: Thresholds | None = None,
    *,
    report_only: bool = True,
) -> Comparison:
    """Weigh the evidence into signals and a flag.

    A flag needs at least `needed` signals to fire, never one. On a field of
    same-base distillations a single fired signal is the expected false
    positive, so the conjunction is what separates a copy from convergence.
    """
    t = thresholds or Thresholds()

    disp = Signal(
        "displacement",
        evidence.displacement,
        evidence.displacement is not None and evidence.displacement >= t.displacement,
        evidence.displacement is not None,
    )
    attn = Signal(
        "attention",
        evidence.attention,
        evidence.attention is not None and evidence.attention >= t.attention,
        evidence.attention is not None,
    )
    behav = Signal(
        "behaviour",
        evidence.behaviour,
        evidence.behaviour is not None and evidence.behaviour <= t.behaviour,
        evidence.behaviour is not None,
    )

    signals = (disp, attn, behav)
    fired = sum(1 for s in signals if s.fired)
    flagged = (not report_only) and fired >= t.needed

    return Comparison(
        earlier=earlier,
        later=later,
        signals=signals,
        flagged=flagged,
        report_only=report_only,
    )
