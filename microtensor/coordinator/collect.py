from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from microtensor.coordinator.report import Report

log = logging.getLogger("microtensor.coordinator")

QUALITY_QUANTUM = 4

VERSION_MISMATCH = "engine or corpus version does not match the round"
CORPUS_MISMATCH = "the worker's corpus does not hash to the one this round is measured against"
UNASSIGNED = "worker was not assigned this system"
DUPLICATE = "worker already reported this system this round"
MALFORMED = "report is malformed"
BAD_SIGNATURE = "signature does not verify against the worker hotkey"

DIVERGED_QUALITY = "quality"
NO_MAJORITY = "no-majority"


class ReportRejected(RuntimeError):
    """A report that must not enter reconciliation at all.

    Distinct from divergence. A rejected report is misconfigured or
    unauthorised and is dropped; a divergent report is well-formed and
    disagrees, which is a signal worth recording against the worker.
    """


@dataclass(frozen=True, slots=True)
class Reconciled:
    """The agreed measurement of one system, and who dissented."""

    system_digest: str
    quality: float | None
    resolve_rate: float
    expected_ms: float
    expected_j: float | None
    envelope: dict[str, Any]
    ablation: dict[str, float] | None
    agreed: tuple[str, ...] = ()
    diverged: tuple[str, ...] = ()
    conforming_reports: int = 0
    reason: str = ""

    @property
    def scored(self) -> bool:
        return self.quality is not None


@dataclass(slots=True)
class Divergence:
    system_digest: str
    worker_hotkey: str
    reported: float
    reconciled: float | None
    kind: str = DIVERGED_QUALITY


@dataclass(slots=True)
class Intake:
    reconciled: list[Reconciled] = field(default_factory=list)
    divergences: list[Divergence] = field(default_factory=list)
    unscored: list[str] = field(default_factory=list)


def quantise_quality(value: float) -> float:
    """Compare quality at the precision the mechanism itself publishes.

    Scoring is deterministic, so two honest workers agree exactly. Rounding to
    the published precision keeps a float printed and re-parsed in transit from
    being read as a disagreement, without softening a real one: any difference
    the network could act on survives this.
    """
    return round(value, QUALITY_QUANTUM)


def accept(
    report: Report,
    *,
    assigned: Sequence[str],
    engine_version: str,
    corpus_version: str,
    already: Sequence[str] = (),
    corpus_digest: str = "",
) -> None:
    """Whether a report may enter reconciliation. Raises with the reason.

    A version mismatch is a rejection, not a divergence. A worker on a
    different engine or corpus is misconfigured, and folding its numbers into
    the majority is how a subtly wrong answer becomes canonical.

    The digest is checked as well as the label, because the label is a claim and
    the digest is the thing. Two workers can both declare the same corpus
    version while holding different task files, and without this that
    disagreement arrives as a divergence between honest measurements.
    """
    if report.system_digest not in assigned:
        raise ReportRejected(UNASSIGNED)
    if report.worker_hotkey in already:
        raise ReportRejected(DUPLICATE)
    if engine_version and report.engine_version != engine_version:
        raise ReportRejected(VERSION_MISMATCH)
    if corpus_version and report.corpus_version != corpus_version:
        raise ReportRejected(VERSION_MISMATCH)
    if corpus_digest and report.corpus_digest != corpus_digest:
        raise ReportRejected(f"{CORPUS_MISMATCH}: {report.corpus_digest or 'none declared'}")


def _majority(values: list[float]) -> tuple[float | None, str]:
    counts: dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1

    best = max(counts.values())
    if best < 2:
        return None, NO_MAJORITY

    winners = sorted(v for v, n in counts.items() if n == best)
    if len(winners) > 1:
        return None, NO_MAJORITY
    return winners[0], ""


def reconcile(reports: Sequence[Report], advisory: Sequence[str] = ()) -> Reconciled:
    """One agreed measurement from several independent ones.

    Quality is deterministic, so a disagreement is a defect rather than noise
    and is never averaged: the majority value stands and the outliers are named.
    With no majority the system goes unscored for the round, because a
    coordinator that breaks a three-way tie by choosing is deciding the
    outcome rather than counting it.

    Envelope and energy are hardware-dependent and legitimately vary inside the
    conformance band, so those take the median across conforming workers only.
    """
    if not reports:
        raise ValueError("cannot reconcile an empty report set")

    digest = reports[0].system_digest
    deciding = [r for r in reports if r.worker_hotkey not in advisory] or list(reports)

    qualities = [quantise_quality(r.quality.combined) for r in deciding]
    agreed_value, reason = _majority(qualities) if len(deciding) > 1 else (qualities[0], "")

    agreed: list[str] = []
    diverged: list[str] = []
    for report in deciding:
        value = quantise_quality(report.quality.combined)
        (agreed if agreed_value is not None and value == agreed_value else diverged).append(
            report.worker_hotkey
        )

    conforming = [r for r in deciding if r.conforming]
    envelope_source = conforming or deciding

    energies = [r.cost.expected_j for r in envelope_source if r.cost.expected_j is not None]

    return Reconciled(
        system_digest=digest,
        quality=agreed_value,
        resolve_rate=median([r.resolve_rate for r in deciding]),
        expected_ms=median([r.cost.expected_ms for r in envelope_source]),
        expected_j=median(energies) if energies else None,
        envelope=_median_envelope(envelope_source),
        ablation=_median_ablation(deciding),
        agreed=tuple(agreed),
        diverged=tuple(diverged),
        conforming_reports=len(conforming),
        reason=reason,
    )


def _median_envelope(reports: Sequence[Report]) -> dict[str, Any]:
    roles = {role for r in reports for role in r.envelope}
    out: dict[str, Any] = {}

    for role in sorted(roles):
        blocks = [r.envelope[role] for r in reports if role in r.envelope]
        fields = {k for b in blocks for k, v in b.items() if isinstance(v, (int, float))}
        out[role] = {
            key: median([float(b[key]) for b in blocks if key in b]) for key in sorted(fields)
        }
    return out


def _median_ablation(reports: Sequence[Report]) -> dict[str, float] | None:
    contributed = [r.ablation for r in reports if r.ablation]
    if not contributed:
        return None
    roles = {role for a in contributed for role in a}
    return {
        role: median([a[role] for a in contributed if role in a]) for role in sorted(roles)
    }


def intake(
    by_system: dict[str, list[Report]], advisory: Sequence[str] = ()
) -> Intake:
    """Reconcile every system and collect what disagreed."""
    result = Intake()

    for digest in sorted(by_system):
        reports = by_system[digest]
        if not reports:
            continue
        agreed = reconcile(reports, advisory)

        if not agreed.scored:
            log.warning(
                "%s is unscored this round: %d reports, no majority on quality",
                digest,
                len(reports),
            )
            result.unscored.append(digest)

        for hotkey in agreed.diverged:
            reported = next(
                quantise_quality(r.quality.combined)
                for r in reports
                if r.worker_hotkey == hotkey
            )
            log.warning(
                "%s diverged on %s: reported %s against %s",
                hotkey,
                digest,
                reported,
                agreed.quality,
            )
            result.divergences.append(
                Divergence(
                    system_digest=digest,
                    worker_hotkey=hotkey,
                    reported=reported,
                    reconciled=agreed.quality,
                    kind=DIVERGED_QUALITY if agreed.scored else NO_MAJORITY,
                )
            )

        if agreed.scored:
            result.reconciled.append(agreed)

    return result


def quorum_reached(expected: int, received: int, fraction: float) -> bool:
    if expected <= 0:
        return False
    return received / expected >= fraction
