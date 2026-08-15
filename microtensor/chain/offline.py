from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from microtensor.chain.metagraph import MetagraphSnapshot
from microtensor.chain.weights import WeightVector


def deterministic_block_hash(block: int) -> str:
    return "0x" + hashlib.sha256(f"microtensor:block:{block}".encode()).hexdigest()


class OfflineClient:
    def __init__(
        self,
        snapshot: MetagraphSnapshot,
        *,
        block: int | None = None,
        commitments: Mapping[str, str] | None = None,
        signer: str = "",
    ) -> None:
        self._snapshot = snapshot
        self._block = snapshot.block if block is None else block
        self._commitments: dict[str, str] = dict(commitments or {})
        self._signer = signer
        self.published: list[str] = []
        self.submitted: list[WeightVector] = []
        self.closed = False

    @property
    def netuid(self) -> int:
        return self._snapshot.netuid

    def block(self) -> int:
        return self._block

    def block_hash(self, block: int) -> str:
        return deterministic_block_hash(block)

    def snapshot(self, *, refresh: bool = False) -> MetagraphSnapshot:
        return self._snapshot

    def commitments(self, hotkeys: Sequence[str]) -> dict[str, str]:
        return {h: self._commitments[h] for h in hotkeys if h in self._commitments}

    def publish(self, payload: str) -> bool:
        self.published.append(payload)
        if self._signer:
            self._commitments[self._signer] = payload
        return True

    def set_weights(self, vector: WeightVector) -> tuple[bool, str]:
        if vector.is_empty:
            return False, "refusing to submit an empty weight vector"
        self.submitted.append(vector)
        return True, "accepted"

    def is_registered(self, hotkey: str) -> bool:
        return self._snapshot.is_registered(hotkey)

    def close(self) -> None:
        self.closed = True

    def advance(self, blocks: int = 1) -> int:
        self._block += blocks
        return self._block

    def set_snapshot(self, snapshot: MetagraphSnapshot) -> None:
        self._snapshot = snapshot

    def set_commitment(self, hotkey: str, payload: str) -> None:
        self._commitments[hotkey] = payload

    def clear_commitments(self) -> None:
        self._commitments.clear()
