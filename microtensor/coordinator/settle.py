from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from microtensor.coordinator.collect import Reconciled
from microtensor.scoring import frontier
from microtensor.scoring.weights import combine_competitions

log = logging.getLogger("microtensor.coordinator")

SETTLEMENT_VERSION = 1


@dataclass(frozen=True, slots=True)
class Entry:
    """One system as the settlement sees it."""

    system_digest: str
    miner_hotkey: str
    uid: int
    track: str
    hardware_class: str
    quality: float
    expected_ms: float
    committed_at: int = 0
    rounds_observed: int = 0
    stale_rounds: int = 0
    replication: int = 0
    contribution: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class Settlement:
    round_index: int
    config_hash: str
    corpus_version: str
    reports_root: str
    frontier: tuple[dict[str, Any], ...] = ()
    weights: dict[int, float] = field(default_factory=dict)
    unscored: tuple[str, ...] = ()
    under_replicated: tuple[str, ...] = ()
    signature: str = ""

    def body(self) -> dict[str, Any]:
        """The signed payload, excluding the signature itself."""
        return {
            "version": SETTLEMENT_VERSION,
            "round": self.round_index,
            "config_hash": self.config_hash,
            "corpus_version": self.corpus_version,
            "reports_root": self.reports_root,
            "frontier": list(self.frontier),
            "weights": {str(uid): value for uid, value in sorted(self.weights.items())},
            "unscored": list(self.unscored),
            "under_replicated": list(self.under_replicated),
        }

    def digest(self) -> str:
        canonical = json.dumps(self.body(), sort_keys=True, separators=(",", ":"))
        return f"sha256:{sha256(canonical.encode()).hexdigest()}"

    def signed_with(self, signature: str) -> Settlement:
        return Settlement(
            round_index=self.round_index,
            config_hash=self.config_hash,
            corpus_version=self.corpus_version,
            reports_root=self.reports_root,
            frontier=self.frontier,
            weights=self.weights,
            unscored=self.unscored,
            under_replicated=self.under_replicated,
            signature=signature,
        )


def merkle_root(leaves: Sequence[str]) -> str:
    """Commit to the exact report set the settlement was computed from.

    Publishing the reports alone lets someone recompute the result; publishing
    this alongside them proves the set was not edited afterwards to fit.
    """
    if not leaves:
        return "sha256:" + sha256(b"").hexdigest()

    level = [sha256(leaf.encode()).digest() for leaf in sorted(leaves)]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return f"sha256:{level[0].hex()}"


def to_entries(
    reconciled: Sequence[Reconciled],
    catalogue: dict[str, Entry],
) -> list[Entry]:
    """Attach each agreed measurement to what the round knows about the system."""
    entries: list[Entry] = []
    for agreed in reconciled:
        known = catalogue.get(agreed.system_digest)
        if known is None or agreed.quality is None:
            continue
        entries.append(
            Entry(
                system_digest=known.system_digest,
                miner_hotkey=known.miner_hotkey,
                uid=known.uid,
                track=known.track,
                hardware_class=known.hardware_class,
                quality=agreed.quality,
                expected_ms=agreed.expected_ms,
                committed_at=known.committed_at,
                rounds_observed=known.rounds_observed,
                stale_rounds=known.stale_rounds,
                replication=len(agreed.agreed) + len(agreed.diverged),
                contribution=agreed.ablation,
            )
        )
    return entries


def allocate(entries: Sequence[Entry]) -> dict[tuple[str, str], dict[str, float]]:
    """Emission share per competition, by exclusive hypervolume.

    The same frontier module every worker runs. The coordinator computes no
    mechanism of its own; it applies the published one to numbers other people
    measured, which is what makes the result reproducible by anyone.
    """
    per_competition: dict[tuple[str, str], dict[str, float]] = {}

    competitions = {(e.track, e.hardware_class) for e in entries}
    for competition in sorted(competitions):
        track, hardware_class = competition
        entrants = [
            frontier.Entrant(
                key=e.miner_hotkey,
                quality=e.quality,
                cost=e.expected_ms,
                committed_at=e.committed_at,
                rounds_observed=e.rounds_observed,
                stale_rounds=e.stale_rounds,
            )
            for e in entries
            if (e.track, e.hardware_class) == competition
        ]
        shares = frontier.allocate(entrants)
        if shares:
            per_competition[competition] = shares
        else:
            log.info("%s/%s: nobody cleared the track threshold", track, hardware_class)

    return per_competition


def build(
    round_index: int,
    *,
    config_hash: str,
    corpus_version: str,
    reconciled: Sequence[Reconciled],
    catalogue: dict[str, Entry],
    report_digests: Sequence[str],
    unscored: Sequence[str] = (),
    under_replicated: Sequence[str] = (),
) -> Settlement:
    """The canonical settlement for one round."""
    entries = to_entries(reconciled, catalogue)
    per_competition = allocate(entries)
    combined = combine_competitions(per_competition)

    uid_of = {e.miner_hotkey: e.uid for e in entries}
    weights = {uid_of[hotkey]: share for hotkey, share in combined.items() if hotkey in uid_of}

    by_digest = {e.system_digest: e for e in entries}
    published = tuple(
        {
            "system": digest,
            "miner": entry.miner_hotkey,
            "track": entry.track,
            "class": entry.hardware_class,
            "quality": entry.quality,
            "expected_ms": entry.expected_ms,
            "replication": entry.replication,
            "contribution": entry.contribution,
            "share": combined.get(entry.miner_hotkey, 0.0),
        }
        for digest, entry in sorted(by_digest.items())
    )

    return Settlement(
        round_index=round_index,
        config_hash=config_hash,
        corpus_version=corpus_version,
        reports_root=merkle_root(report_digests),
        frontier=published,
        weights=weights,
        unscored=tuple(sorted(unscored)),
        under_replicated=tuple(sorted(under_replicated)),
    )
