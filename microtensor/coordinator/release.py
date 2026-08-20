from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from microtensor.chain.rounds import release_index, release_version

log = logging.getLogger("microtensor.coordinator")

RELEASE_VERSION = 1

ALREADY_PUBLISHED = "a release already exists for this index and competition"
NOT_A_BOUNDARY = "this round does not end a release cycle"
NO_FRONTIER = "no system was on the frontier at the cutoff"


class ReleaseError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Milestone:
    """The stated target for a cycle.

    Descriptive. Nothing gates on it and nothing pays for it. It exists so a
    miner has something concrete to aim at beyond being on the frontier, and so
    a cycle has a result that can be stated in one sentence.
    """

    target_quality: float
    target_cost: float
    met_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_quality": self.target_quality,
            "target_cost": self.target_cost,
            "met_by": self.met_by,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Milestone:
        met = raw.get("met_by")
        return cls(
            target_quality=float(raw["target_quality"]),
            target_cost=float(raw["target_cost"]),
            met_by=str(met) if met else None,
        )

    def met_by_any(self, entries: Sequence[ReleaseEntry]) -> Milestone:
        """The same milestone, with the first system that cleared it named.

        Ordered by cost so the cheapest qualifying system is the one credited.
        A milestone is a quality floor under a cost ceiling, and the system that
        reaches it for less is the better answer to the same question.
        """
        for entry in sorted(entries, key=lambda e: (e.cost, e.system_digest)):
            if entry.quality >= self.target_quality and entry.cost <= self.target_cost:
                return Milestone(self.target_quality, self.target_cost, entry.system_digest)
        return Milestone(self.target_quality, self.target_cost, None)


@dataclass(frozen=True, slots=True)
class ReleaseEntry:
    system_digest: str
    certificate: str = ""
    quality: float = 0.0
    cost: float = 0.0
    hv_exclusive: int = 0
    rank_by_cost: int = 0
    components: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_digest": self.system_digest,
            "certificate": self.certificate,
            "quality": self.quality,
            "cost": self.cost,
            "hv_exclusive": self.hv_exclusive,
            "rank_by_cost": self.rank_by_cost,
            "components": [dict(c) for c in self.components],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ReleaseEntry:
        return cls(
            system_digest=str(raw["system_digest"]),
            certificate=str(raw.get("certificate", "")),
            quality=float(raw.get("quality", 0.0)),
            cost=float(raw.get("cost", 0.0)),
            hv_exclusive=int(raw.get("hv_exclusive", 0)),
            rank_by_cost=int(raw.get("rank_by_cost", 0)),
            components=tuple(dict(c) for c in raw.get("components", ())),
        )


@dataclass(frozen=True, slots=True)
class Release:
    """One frozen frontier, named and signed.

    Immutable once published. A release is a historical record of what shipped,
    so a correction is a new release rather than an edit: a customer who pinned
    v1 must keep receiving the v1 they pinned.
    """

    version: str
    release_index: int
    competition: tuple[str, str]
    final_round: int
    cutoff_block: int
    corpus_version: str = ""
    frontier: tuple[ReleaseEntry, ...] = ()
    milestone: Milestone | None = None
    published_at: int = 0
    signature: str = ""

    @property
    def track(self) -> str:
        return self.competition[0]

    @property
    def hardware_class(self) -> str:
        return self.competition[1]

    def body(self) -> dict[str, Any]:
        """The signed payload, excluding the signature itself."""
        return {
            "version": RELEASE_VERSION,
            "name": self.version,
            "release_index": self.release_index,
            "track": self.track,
            "class": self.hardware_class,
            "final_round": self.final_round,
            "cutoff_block": self.cutoff_block,
            "corpus_version": self.corpus_version,
            "frontier": [entry.to_dict() for entry in self.frontier],
            "milestone": self.milestone.to_dict() if self.milestone else None,
            "published_at": self.published_at,
        }

    def digest(self) -> str:
        canonical = json.dumps(self.body(), sort_keys=True, separators=(",", ":"))
        return f"sha256:{sha256(canonical.encode()).hexdigest()}"

    def signed_with(self, signature: str) -> Release:
        return Release(
            version=self.version,
            release_index=self.release_index,
            competition=self.competition,
            final_round=self.final_round,
            cutoff_block=self.cutoff_block,
            corpus_version=self.corpus_version,
            frontier=self.frontier,
            milestone=self.milestone,
            published_at=self.published_at,
            signature=signature,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Release:
        milestone = raw.get("milestone")
        return cls(
            version=str(raw["name"]),
            release_index=int(raw["release_index"]),
            competition=(str(raw["track"]), str(raw["class"])),
            final_round=int(raw["final_round"]),
            cutoff_block=int(raw["cutoff_block"]),
            corpus_version=str(raw.get("corpus_version", "")),
            frontier=tuple(ReleaseEntry.from_dict(e) for e in raw.get("frontier", ())),
            milestone=Milestone.from_dict(milestone) if milestone else None,
            published_at=int(raw.get("published_at", 0)),
            signature=str(raw.get("signature", "")),
        )


def entries_from(
    frontier: Sequence[Mapping[str, Any]],
    track: str,
    hardware_class: str,
    *,
    certificates: Mapping[str, str] | None = None,
    components: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[ReleaseEntry, ...]:
    """The release rows for one competition, taken from a settled frontier.

    Only systems that earned a share are carried. A system measured in the
    final round but off the frontier when it settled is not in the release, so
    what ships is what the mechanism actually selected rather than everything
    that happened to be running.

    Ranked by cost, cheapest first, because that is the order a customer reads a
    frontier in: the question is always what the least expensive system that
    clears my quality bar is.
    """
    held = [
        row
        for row in frontier
        if str(row.get("track", "")) == track
        and str(row.get("class", "")) == hardware_class
        and float(row.get("share", 0.0)) > 0.0
    ]
    ordered = sorted(held, key=lambda r: (float(r.get("expected_ms", 0.0)), str(r.get("system"))))

    certs = certificates or {}
    parts = components or {}
    return tuple(
        ReleaseEntry(
            system_digest=str(row["system"]),
            certificate=str(certs.get(str(row["system"]), "")),
            quality=float(row.get("quality", 0.0)),
            cost=float(row.get("expected_ms", 0.0)),
            hv_exclusive=round(float(row.get("share", 0.0)) * 65535),
            rank_by_cost=position,
            components=tuple(dict(c) for c in parts.get(str(row["system"]), ())),
        )
        for position, row in enumerate(ordered, start=1)
    )


def build(
    settlement: Mapping[str, Any],
    *,
    track: str,
    hardware_class: str,
    cutoff_block: int,
    published_at: int,
    every: int | None = None,
    milestone: Milestone | None = None,
    certificates: Mapping[str, str] | None = None,
    components: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> Release:
    """Freeze one competition's frontier from a settled round.

    Reads the settlement rather than recomputing anything. Settlement already
    ran and already decided who is on the frontier; a release that recomputed it
    could disagree with the round that paid, and then the thing customers
    receive would not be the thing the subnet actually selected.
    """
    final_round = int(settlement["round"])
    index = (
        release_index(final_round) if every is None else release_index(final_round, every=every)
    )

    entries = entries_from(
        settlement.get("frontier", ()),
        track,
        hardware_class,
        certificates=certificates,
        components=components,
    )
    if not entries:
        raise ReleaseError(f"{NO_FRONTIER}: {track}/{hardware_class} at round {final_round}")

    return Release(
        version=release_version(track, hardware_class, index),
        release_index=index,
        competition=(track, hardware_class),
        final_round=final_round,
        cutoff_block=cutoff_block,
        corpus_version=str(settlement.get("corpus_version", "")),
        frontier=entries,
        milestone=milestone.met_by_any(entries) if milestone else None,
        published_at=published_at,
    )
