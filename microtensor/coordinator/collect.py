from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import fmean, median
from typing import Any

from microtensor.coordinator.report import Report

log = logging.getLogger("microtensor.coordinator")

QUALITY_QUANTUM = 4
QUALITY_SPREAD_WARNING = 0.01

VERSION_MISMATCH = "engine or corpus version does not match the round"
CORPUS_MISMATCH = "the worker's corpus does not hash to the one this round is measured against"
ENVIRONMENT_MISMATCH = "report was measured in a different environment than the arena pins"
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
    components: dict[str, str] = field(default_factory=dict)
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

    Two honest workers on different silicon can differ in the third decimal:
    float accumulation order shifts with the instruction set, greedy decoding
    flips the odd token, and generative scores move beneath the task's own
    noise floor. Agreement is judged at a hundredth, which no honest spread
    crosses and any score worth disputing does.
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
    environment_digest: str = "",
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
    if environment_digest and report.environment_digest != environment_digest:
        raise ReportRejected(ENVIRONMENT_MISMATCH)


def _publishable(values: Sequence[float]) -> list[int]:
    live = [i for i, value in enumerate(values) if value > 0.0]
    if not live or len(live) == len(values):
        return list(range(len(values)))
    return live


def reconcile(reports: Sequence[Report], advisory: Sequence[str] = ()) -> Reconciled:
    if not reports:
        raise ValueError("cannot reconcile an empty report set")

    digest = reports[0].system_digest
    deciding = [r for r in reports if r.worker_hotkey not in advisory] or list(reports)

    result_worker = os.environ.get("MT_RESULT_WORKER", "").strip()
    if result_worker:
        preferred = [r for r in deciding if r.worker_hotkey == result_worker]
        if preferred:
            deciding = preferred

    qualities = [quantise_quality(r.quality.combined) for r in deciding]
    keep = _publishable(qualities)
    kept = [deciding[i] for i in keep]
    dropped = [r for i, r in enumerate(deciding) if i not in set(keep)]

    for report in dropped:
        log.warning(
            "%s reported no measurement on %s while others measured it; discarded",
            report.worker_hotkey,
            digest,
        )

    agreed_value = quantise_quality(fmean([qualities[i] for i in keep])) if keep else None
    reason = "" if keep else NO_MAJORITY

    values = [qualities[i] for i in keep]
    if len(values) > 1 and max(values) - min(values) > QUALITY_SPREAD_WARNING:
        log.warning(
            "%s: workers spread %.4f on quality, wider than %.4f",
            digest,
            max(values) - min(values),
            QUALITY_SPREAD_WARNING,
        )

    conforming = [r for r in kept if r.conforming]
    envelope_source = conforming or kept or deciding

    energies = [r.cost.expected_j for r in envelope_source if r.cost.expected_j is not None]

    return Reconciled(
        system_digest=digest,
        quality=agreed_value,
        resolve_rate=median([r.resolve_rate for r in kept or deciding]),
        expected_ms=median([r.cost.expected_ms for r in envelope_source]),
        expected_j=median(energies) if energies else None,
        envelope=_median_envelope(envelope_source),
        components=next((dict(r.components) for r in kept or deciding if r.components), {}),
        ablation=_median_ablation(kept or deciding),
        agreed=tuple(r.worker_hotkey for r in kept),
        diverged=tuple(r.worker_hotkey for r in dropped),
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
    return {role: median([a[role] for a in contributed if role in a]) for role in sorted(roles)}


def intake(by_system: dict[str, list[Report]], advisory: Sequence[str] = ()) -> Intake:
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
                quantise_quality(r.quality.combined) for r in reports if r.worker_hotkey == hotkey
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
