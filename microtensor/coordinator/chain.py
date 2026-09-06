from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from microtensor.chain.commitment import Commitment, Reveal, commitment_hash, decode_all
from microtensor.chain.rounds import Round, accepts_commitment, round_for_block
from microtensor.chain.wallet import verify_payload
from microtensor.coordinator.assign import System, Worker
from microtensor.coordinator.settle import Entry
from microtensor.core.constants import ALSO_ACCEPT_ROUNDS, GENESIS_BLOCK, ROUND_BLOCKS
from microtensor.registry.fetch import ArtifactMismatch, fetch_manifest
from microtensor.registry.manifest import ArtifactManifest

log = logging.getLogger("microtensor.coordinator")


class RoundSource(Protocol):
    """Where a round and its participants come from.

    The coordinator reads them from chain today. A future control plane will
    hand them over instead, so everything downstream takes this shape rather
    than a chain client.
    """

    def open_round(self) -> Round: ...

    def systems(self, round_: Round) -> tuple[Sequence[System], dict[str, Entry]]: ...

    def workers(self) -> Sequence[Worker]: ...

    def coldkeys(self) -> dict[str, str]: ...

    def uids(self) -> dict[str, int]: ...

    def seed(self, round_: Round) -> str: ...


@dataclass(slots=True)
class ChainSource:
    client: Any
    round_blocks: int = ROUND_BLOCKS
    genesis_block: int = GENESIS_BLOCK
    work_dir: Path | None = None
    verify_signatures: bool = True
    fetch: Callable[..., ArtifactManifest] = fetch_manifest
    recorded: Mapping[str, tuple[str, str, str, str, bool]] = field(default_factory=dict)

    def _workdir(self) -> Path:
        if self.work_dir is None:
            self.work_dir = Path(tempfile.mkdtemp(prefix="mt-coordinator-"))
        return self.work_dir

    def _admission_reason(self, hotkey: str, commitment: Commitment) -> str:
        try:
            manifest = self.fetch(commitment, workdir=self._workdir())
        except ArtifactMismatch as exc:
            return f"manifest rejected: {exc}"
        except Exception as exc:
            log.warning(
                "%s: manifest not fetchable at discovery (%s); left for the validators",
                hotkey,
                exc,
            )
            return ""
        if manifest.hotkey != hotkey:
            return "manifest declares a different hotkey than the one that committed it"
        if self.verify_signatures:
            if not manifest.signature:
                return "manifest carries no signature"
            if not verify_payload(manifest.hotkey, manifest.body(), manifest.signature):
                return "manifest signature does not verify against the declaring hotkey"
        fits, reason = manifest.fits_class()
        if not fits:
            return reason
        return ""

    def open_round(self) -> Round:
        return round_for_block(
            self.client.block(), length=self.round_blocks, genesis=self.genesis_block
        )

    def seed(self, round_: Round) -> str:
        return str(self.client.block_hash(round_.seed_block))

    def systems(self, round_: Round) -> tuple[list[System], dict[str, Entry]]:
        """Which submissions exist this round, read from chain commitments.

        Keyed by manifest digest, which is what the commitment carries. The
        system digest lives inside the manifest and reaching it would mean
        fetching a miner artifact, which is the one thing this process must
        never do.
        """
        snapshot = self.client.snapshot(refresh=True)
        raw = self.client.commitments(list(snapshot.hotkeys))
        commitments = decode_all(raw)
        commitments.update(recover_revealed(raw, commitments, round_.index, self.recorded))
        uid_by_hotkey = snapshot.uid_by_hotkey

        blocks: dict[str, int] = {}
        reader = getattr(self.client, "commitment_blocks", None)
        if reader is not None:
            try:
                blocks = dict(reader(list(commitments))) or {}
            except Exception as exc:
                log.warning("commitment blocks unreadable: %s", exc)

        found: list[System] = []
        catalogue: dict[str, Entry] = {}

        for hotkey, commitment in sorted(commitments.items()):
            if not accepts_commitment(
                round_.index, commitment.round_index, ALSO_ACCEPT_ROUNDS
            ):
                continue
            uid = uid_by_hotkey.get(hotkey)
            if uid is None:
                continue

            reason = self._admission_reason(hotkey, commitment)
            if reason:
                log.warning(
                    "round %d: %s excluded at discovery, never catalogued: %s",
                    round_.index,
                    hotkey,
                    reason,
                )
                continue

            track, hardware_class = commitment.competition
            digest = commitment.manifest_digest
            found.append(
                System(
                    digest=digest,
                    track=track,
                    hardware_class=hardware_class,
                    miner_hotkey=hotkey,
                    source=commitment.source,
                )
            )
            catalogue[digest] = Entry(
                system_digest=digest,
                miner_hotkey=hotkey,
                uid=uid,
                track=track,
                hardware_class=hardware_class,
                quality=0.0,
                expected_ms=0.0,
                source=commitment.source,
                committed_at=blocks.get(hotkey, round_.close_block),
            )

        log.info("round %d: %d systems committed on chain", round_.index, len(found))
        return found, catalogue

    def workers(self) -> list[Worker]:
        snapshot = self.client.snapshot()
        return [
            Worker(hotkey=hotkey)
            for hotkey in sorted(snapshot.hotkeys)
            if snapshot.has_permit(hotkey)
        ]

    def coldkeys(self) -> dict[str, str]:
        return dict(self.client.snapshot().coldkeys())

    def uids(self) -> dict[str, int]:
        return dict(self.client.snapshot().uid_by_hotkey)


def recover_revealed(
    raw: Mapping[str, str],
    commitments: Mapping[str, Commitment],
    round_index: int,
    recorded: Mapping[str, tuple[str, str, str, str, bool]],
) -> dict[str, Commitment]:
    out: dict[str, Commitment] = {}
    for hotkey, payload in raw.items():
        if hotkey in commitments:
            continue
        reveal = Reveal.decode(payload)
        if reveal is None or reveal.round_index != round_index:
            continue
        found = recorded.get(hotkey)
        if found is None:
            log.warning(
                "round %d: %s holds a reveal but no pointer was recorded for it; "
                "the submission cannot be restored",
                round_index,
                hotkey,
            )
            continue
        track, hardware_class, digest, source, sealed = found
        if reveal.manifest_digest != digest:
            log.warning(
                "round %d: %s revealed %s but the recorded pointer is %s; not restored",
                round_index,
                hotkey,
                reveal.manifest_digest[:16],
                digest[:16],
            )
            continue
        try:
            candidate = Commitment(
                round_index=round_index,
                track=track,
                hardware_class=hardware_class,
                manifest_digest=digest,
                source=source,
                sealed=sealed,
            )
        except ValueError as exc:
            log.warning(
                "round %d: recorded pointer for %s is unusable: %s", round_index, hotkey, exc
            )
            continue
        if reveal.commitment_hash and reveal.commitment_hash != commitment_hash(candidate):
            log.warning(
                "round %d: %s revealed against a commitment that differs from the record; "
                "not restored",
                round_index,
                hotkey,
            )
            continue
        out[hotkey] = candidate
    if out:
        log.info(
            "round %d: %d pointer(s) restored from the record after being replaced by a reveal",
            round_index,
            len(out),
        )
    return out


def observed(catalogue: dict[str, Entry], history: dict[str, tuple[int, int]]) -> dict[str, Entry]:
    """Fold each system's observation counters into the catalogue."""
    out: dict[str, Entry] = {}
    for digest, entry in catalogue.items():
        rounds_observed, stale_rounds = history.get(entry.miner_hotkey, (1, 0))
        out[digest] = Entry(
            system_digest=entry.system_digest,
            miner_hotkey=entry.miner_hotkey,
            uid=entry.uid,
            track=entry.track,
            hardware_class=entry.hardware_class,
            quality=entry.quality,
            expected_ms=entry.expected_ms,
            committed_at=entry.committed_at,
            rounds_observed=rounds_observed,
            stale_rounds=stale_rounds,
        )
    return out
