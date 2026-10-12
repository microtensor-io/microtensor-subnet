from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

from microtensor.core.constants import BLOCK_TIME_SECONDS, GENESIS_BLOCK

DEFAULT_EPOCH_BLOCKS: Final[int] = 50_400
DEFAULT_COMMIT_BLOCKS: Final[int] = 7_200
DEFAULT_SETTLE_MARGIN_BLOCKS: Final[int] = 300
SEED_HEX_CHARS: Final[int] = 64


@dataclass(frozen=True, slots=True)
class Epoch:
    index: int
    start_block: int
    length: int = DEFAULT_EPOCH_BLOCKS
    commit_blocks: int = DEFAULT_COMMIT_BLOCKS
    settle_margin: int = DEFAULT_SETTLE_MARGIN_BLOCKS

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("epoch index must not be negative")
        if self.length < 2:
            raise ValueError("an epoch spans at least two blocks")
        if not 0 < self.commit_blocks < self.length:
            raise ValueError("the commit window must fall inside the epoch")
        if not 0 <= self.settle_margin < self.length - self.commit_blocks:
            raise ValueError("the settle margin must leave room to evaluate")

    @property
    def end_block(self) -> int:
        return self.start_block + self.length - 1

    @property
    def open_block(self) -> int:
        return self.start_block + self.commit_blocks

    @property
    def close_block(self) -> int:
        return self.end_block - self.settle_margin

    def accepts_commitments(self, block: int) -> bool:
        return self.start_block <= block < self.open_block

    def window_open(self, block: int) -> bool:
        return self.open_block <= block <= self.close_block

    def settling(self, block: int) -> bool:
        return self.close_block < block <= self.end_block

    def contains(self, block: int) -> bool:
        return self.start_block <= block <= self.end_block

    def blocks_until_open(self, block: int) -> int:
        return max(0, self.open_block - block)

    def blocks_until_close(self, block: int) -> int:
        return max(0, self.close_block - block)

    def seconds_until_open(self, block: int) -> int:
        return self.blocks_until_open(block) * BLOCK_TIME_SECONDS

    def shift(self, offset: int) -> Epoch:
        if self.index + offset < 0:
            raise ValueError("no epoch precedes the first")
        return Epoch(
            index=self.index + offset,
            start_block=self.start_block + offset * self.length,
            length=self.length,
            commit_blocks=self.commit_blocks,
            settle_margin=self.settle_margin,
        )

    @property
    def next(self) -> Epoch:
        return self.shift(1)

    @property
    def previous(self) -> Epoch:
        return self.shift(-1)


def epoch_at(
    index: int,
    *,
    length: int = DEFAULT_EPOCH_BLOCKS,
    commit_blocks: int = DEFAULT_COMMIT_BLOCKS,
    settle_margin: int = DEFAULT_SETTLE_MARGIN_BLOCKS,
    genesis: int = GENESIS_BLOCK,
) -> Epoch:
    return Epoch(
        index=index,
        start_block=genesis + index * length,
        length=length,
        commit_blocks=commit_blocks,
        settle_margin=settle_margin,
    )


def epoch_for_block(
    block: int,
    *,
    length: int = DEFAULT_EPOCH_BLOCKS,
    commit_blocks: int = DEFAULT_COMMIT_BLOCKS,
    settle_margin: int = DEFAULT_SETTLE_MARGIN_BLOCKS,
    genesis: int = GENESIS_BLOCK,
) -> Epoch:
    if block < genesis:
        raise ValueError("block precedes the subnet genesis")
    return epoch_at(
        (block - genesis) // length,
        length=length,
        commit_blocks=commit_blocks,
        settle_margin=settle_margin,
        genesis=genesis,
    )


def epoch_seed(open_block_hash: str, arena_id: str) -> str:
    body = open_block_hash.strip().lower()
    body = body[2:] if body.startswith("0x") else body
    try:
        raw = bytes.fromhex(body)
    except ValueError as exc:
        raise ValueError("the open block hash must be hex") from exc
    if not raw:
        raise ValueError("the open block hash is empty")
    return hashlib.sha256(raw + arena_id.encode("utf-8")).hexdigest()


def assignment_seed(seed: str, test: str) -> int:
    if len(seed) != SEED_HEX_CHARS:
        raise ValueError("the epoch seed is a sha256 hex digest")
    digest = hashlib.sha256(f"{seed}:{test}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
