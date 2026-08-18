from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from microtensor.coordinator.collect import Reconciled
from microtensor.scoring import frontier
from microtensor.scoring.weights import (
    apply_concentration_cap,
    blend,
    combine_competitions,
    origin_group,
)

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
    catalogue: tuple[dict[str, Any], ...] = ()
    weights: dict[int, float] = field(default_factory=dict)
    unscored: tuple[str, ...] = ()
    under_replicated: tuple[str, ...] = ()
    advisory: tuple[str, ...] = ()
    capped: bool = False
    blended: dict[str, float] = field(default_factory=dict)
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
            "catalogue": list(self.catalogue),
            "weights": {str(uid): value for uid, value in sorted(self.weights.items())},
            "unscored": list(self.unscored),
            "under_replicated": list(self.under_replicated),
            "advisory": list(self.advisory),
            "capped": self.capped,
            "blended": dict(sorted(self.blended.items())),
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
            catalogue=self.catalogue,
            weights=self.weights,
            unscored=self.unscored,
            under_replicated=self.under_replicated,
            advisory=self.advisory,
            capped=self.capped,
            blended=self.blended,
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


def catalogue_payload(catalogue: dict[str, Entry]) -> tuple[dict[str, Any], ...]:
    """The eligibility inputs the weights were computed from.

    Published because a worker cannot otherwise recompute the settlement. It
    measured only its assigned subset, and `rounds_observed` is local state that
    legitimately differs between validators, so without these the recomputation
    would disagree for honest reasons and the check would be useless.

    This narrows what verification proves, and the narrowing is worth stating:
    it proves the weights are the correct output of the published mechanism
    applied to the published inputs. It does not by itself prove the inputs are
    honest. The uid and hotkey of every entry are checkable against the
    metagraph, which is what `cross_check` on the worker does; `rounds_observed`
    remains coordinator-asserted and is the reason reports are kept forever.
    """
    return tuple(
        {
            "system": e.system_digest,
            "miner": e.miner_hotkey,
            "uid": e.uid,
            "track": e.track,
            "class": e.hardware_class,
            "committed_at": e.committed_at,
            "rounds_observed": e.rounds_observed,
            "stale_rounds": e.stale_rounds,
        }
        for e in sorted(catalogue.values(), key=lambda x: x.system_digest)
    )


def catalogue_from(payload: Sequence[dict[str, Any]]) -> dict[str, Entry]:
    """Rebuild the catalogue a settlement was computed against."""
    return {
        str(row["system"]): Entry(
            system_digest=str(row["system"]),
            miner_hotkey=str(row["miner"]),
            uid=int(row["uid"]),
            track=str(row["track"]),
            hardware_class=str(row["class"]),
            quality=0.0,
            expected_ms=0.0,
            committed_at=int(row.get("committed_at", 0)),
            rounds_observed=int(row.get("rounds_observed", 0)),
            stale_rounds=int(row.get("stale_rounds", 0)),
        )
        for row in payload
    }


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
    advisory: Sequence[str] = (),
    coldkeys: Mapping[str, str] | None = None,
    previous: Mapping[str, float] | None = None,
) -> Settlement:
    """The canonical settlement for one round."""
    entries = to_entries(reconciled, catalogue)
    per_competition = allocate(entries)
    combined = combine_competitions(per_competition)

    capped = combined
    if coldkeys:
        origins = {h: origin_group(coldkeys[h]) for h in combined if h in coldkeys}
        capped = apply_concentration_cap(combined, origins)
        if len(capped) < len(combined):
            log.info(
                "concentration cap removed %d of %d positions",
                len(combined) - len(capped),
                len(combined),
            )
    elif combined:
        log.warning(
            "no coldkeys supplied, so the concentration cap was not applied; "
            "the settlement records this rather than implying the cap held"
        )

    final = blend(capped, dict(previous or {}))

    uid_of = {e.miner_hotkey: e.uid for e in entries}
    weights = {uid_of[hotkey]: share for hotkey, share in final.items() if hotkey in uid_of}

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
        catalogue=catalogue_payload(catalogue),
        weights=weights,
        unscored=tuple(sorted(unscored)),
        under_replicated=tuple(sorted(under_replicated)),
        advisory=tuple(sorted(advisory)),
        capped=bool(coldkeys),
        blended=dict(final),
    )
