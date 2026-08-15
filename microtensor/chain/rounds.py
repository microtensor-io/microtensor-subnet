from __future__ import annotations

from dataclasses import dataclass

from microtensor.core.constants import (
    BLOCK_TIME_SECONDS,
    DEADLINE_MARGIN_BLOCKS,
    GENESIS_BLOCK,
    ROUND_BLOCKS,
    SUBMISSION_CLOSES_BEFORE_BLOCKS,
)


@dataclass(frozen=True, slots=True)
class Round:
    index: int
    start_block: int
    length: int = ROUND_BLOCKS
    close_margin: int = SUBMISSION_CLOSES_BEFORE_BLOCKS
    deadline_margin: int = DEADLINE_MARGIN_BLOCKS

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("round index must not be negative")
        if self.length < 1:
            raise ValueError("round length must be at least one block")
        if not 0 < self.close_margin < self.length:
            raise ValueError("close margin must fall inside the round")
        if not 0 <= self.deadline_margin < self.close_margin:
            raise ValueError("deadline margin must leave room to evaluate")

    @property
    def end_block(self) -> int:
        return self.start_block + self.length - 1

    @property
    def close_block(self) -> int:
        return self.start_block + self.length - self.close_margin

    @property
    def seed_block(self) -> int:
        return self.close_block

    @property
    def deadline_block(self) -> int:
        return self.end_block - self.deadline_margin

    @property
    def evaluation_blocks(self) -> int:
        return self.deadline_block - self.close_block + 1

    def contains(self, block: int) -> bool:
        return self.start_block <= block <= self.end_block

    def accepts_submissions(self, block: int) -> bool:
        return self.start_block <= block < self.close_block

    def is_sealed(self, block: int) -> bool:
        return block >= self.close_block

    def blocks_until_close(self, block: int) -> int:
        return max(0, self.close_block - block)

    def blocks_until_deadline(self, block: int) -> int:
        return max(0, self.deadline_block - block)

    def seconds_until_close(self, block: int) -> int:
        return self.blocks_until_close(block) * BLOCK_TIME_SECONDS

    def seconds_until_deadline(self, block: int) -> int:
        return self.blocks_until_deadline(block) * BLOCK_TIME_SECONDS

    def shift(self, offset: int) -> Round:
        if self.index + offset < 0:
            raise ValueError("no round precedes the first")
        return Round(
            index=self.index + offset,
            start_block=self.start_block + offset * self.length,
            length=self.length,
            close_margin=self.close_margin,
            deadline_margin=self.deadline_margin,
        )

    @property
    def next(self) -> Round:
        return self.shift(1)

    @property
    def previous(self) -> Round:
        return self.shift(-1)


def round_at(
    index: int,
    *,
    length: int = ROUND_BLOCKS,
    genesis: int = GENESIS_BLOCK,
    close_margin: int = SUBMISSION_CLOSES_BEFORE_BLOCKS,
    deadline_margin: int = DEADLINE_MARGIN_BLOCKS,
) -> Round:
    return Round(
        index=index,
        start_block=genesis + index * length,
        length=length,
        close_margin=close_margin,
        deadline_margin=deadline_margin,
    )


def round_for_block(
    block: int,
    *,
    length: int = ROUND_BLOCKS,
    genesis: int = GENESIS_BLOCK,
    close_margin: int = SUBMISSION_CLOSES_BEFORE_BLOCKS,
    deadline_margin: int = DEADLINE_MARGIN_BLOCKS,
) -> Round:
    if block < genesis:
        raise ValueError("block precedes the subnet genesis")
    return round_at(
        (block - genesis) // length,
        length=length,
        genesis=genesis,
        close_margin=close_margin,
        deadline_margin=deadline_margin,
    )
