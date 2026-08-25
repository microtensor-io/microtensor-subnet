"""Running derivation detection over a round, off the scoring path.

A copy of an already-published artifact is a fresh commitment nothing else
identifies as derivative. This pass compares each newly measured artifact
against a corpus of earlier ones and records the evidence. In report-only mode,
which is the default and the only mode until thresholds are calibrated, it
changes no score: it exists to gather the distribution of distances between
artifacts known to be independent in this network, on these base models.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from microtensor.core.constants import DERIVATION_ENFORCING
from microtensor.scoring import weightscan
from microtensor.scoring.derivation import (
    Comparison,
    Evidence,
    Thresholds,
    attention_profile_similarity,
    behaviour_divergence,
    compare,
    displacement_cosine,
)

log = logging.getLogger("microtensor.validator.derive")


@dataclass(frozen=True, slots=True)
class Subject:
    """One artifact to compare, with what has been read from it."""

    key: str
    committed_at: int
    path: Path
    base_sample: list[float] | None = None
    weight_sample: list[float] | None = None
    attention: list[float] | None = None
    probe_outputs: list[list[float]] | None = None


def read_subject(
    key: str, committed_at: int, path: Path, base_sample: list[float] | None
) -> Subject:
    return Subject(
        key=key,
        committed_at=committed_at,
        path=path,
        base_sample=base_sample,
        weight_sample=weightscan.weight_sample(path),
        attention=weightscan.attention_profile(path),
    )


def _evidence(earlier: Subject, later: Subject) -> Evidence:
    evidence = Evidence()

    base = later.base_sample or earlier.base_sample
    if (
        base is not None
        and earlier.weight_sample is not None
        and later.weight_sample is not None
        and len(base) == len(earlier.weight_sample) == len(later.weight_sample)
    ):
        evidence.displacement = displacement_cosine(
            earlier.weight_sample, later.weight_sample, base
        )
    else:
        evidence.notes.append("displacement: weight samples or base unavailable")

    if (
        earlier.attention is not None
        and later.attention is not None
        and len(earlier.attention) == len(later.attention)
    ):
        evidence.attention = attention_profile_similarity(earlier.attention, later.attention)
    else:
        evidence.notes.append("attention: profiles unavailable or shape mismatch")

    if earlier.probe_outputs is not None and later.probe_outputs is not None:
        evidence.behaviour = behaviour_divergence(earlier.probe_outputs, later.probe_outputs)
    else:
        evidence.notes.append("behaviour: probe outputs not collected")

    return evidence


def screen(
    subjects: Sequence[Subject],
    prior: Sequence[Subject],
    thresholds: Thresholds | None = None,
    *,
    report_only: bool | None = None,
) -> list[Comparison]:
    """Compare each subject against every earlier artifact and record verdicts.

    "Earlier" is by commitment block: on a flag the earlier certified artifact
    holds and the later is the derivative, so the corpus of prior artifacts and
    any subject that committed before another are both candidates to have been
    copied from.
    """
    only = DERIVATION_ENFORCING is False if report_only is None else report_only
    corpus = list(prior) + list(subjects)
    out: list[Comparison] = []

    for later in subjects:
        for earlier in corpus:
            if earlier.key == later.key:
                continue
            if earlier.committed_at >= later.committed_at:
                continue
            comparison = compare(
                earlier.key,
                later.key,
                _evidence(earlier, later),
                thresholds,
                report_only=only,
            )
            fired = [s.name for s in comparison.signals if s.fired]
            if fired:
                log.info(
                    "derivation %s: %s vs earlier %s fired [%s]%s",
                    "FLAG" if comparison.flagged else "signal",
                    later.key[:12],
                    earlier.key[:12],
                    ", ".join(fired),
                    "" if not comparison.report_only else " (report-only)",
                )
            out.append(comparison)
    return out


def flagged_keys(comparisons: Sequence[Comparison]) -> set[str]:
    """The later artifact of every comparison that flagged; empty in report-only."""
    return {c.later for c in comparisons if c.flagged}
