from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from microtensor.chain.commitment import decode_all
from microtensor.chain.rounds import Round, accepts_commitment, round_for_block
from microtensor.coordinator.assign import System, Worker
from microtensor.coordinator.settle import Entry
from microtensor.core.constants import ALSO_ACCEPT_ROUNDS, GENESIS_BLOCK, ROUND_BLOCKS

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
        uid_by_hotkey = snapshot.uid_by_hotkey

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

            track, hardware_class = commitment.competition
            digest = commitment.manifest_digest
            found.append(
                System(
                    digest=digest,
                    track=track,
                    hardware_class=hardware_class,
                    miner_hotkey=hotkey,
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
                committed_at=round_.index,
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
