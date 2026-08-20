from __future__ import annotations

from dataclasses import dataclass

from microtensor.core.constants import (
    BLOCK_TIME_SECONDS,
    DEADLINE_MARGIN_BLOCKS,
    GENESIS_BLOCK,
    RELEASE_ROUNDS,
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


def release_index(round_index: int, *, every: int = RELEASE_ROUNDS) -> int:
    """Which release cycle a round belongs to, counting from one.

    Derived rather than stored, like the round index itself, so every party
    computes the same answer from block height without asking anyone.
    """
    if round_index < 0:
        raise ValueError("round index must not be negative")
    if every < 1:
        raise ValueError("a release cycle must span at least one round")
    return round_index // every + 1


def is_release_boundary(round_index: int, *, every: int = RELEASE_ROUNDS) -> bool:
    """Whether this round is the last of its cycle.

    The last, not the first. A release freezes what a cycle produced, so it can
    only be cut once the final round of that cycle has settled.
    """
    if round_index < 0:
        raise ValueError("round index must not be negative")
    if every < 1:
        raise ValueError("a release cycle must span at least one round")
    return round_index % every == every - 1


def rounds_until_release(round_index: int, *, every: int = RELEASE_ROUNDS) -> int:
    """Rounds left in this cycle, counting the current one.

    Zero never appears. A miner reading this on the final round has one round
    left to submit into, not none, and the boundary round is still open when
    they read it.
    """
    if round_index < 0:
        raise ValueError("round index must not be negative")
    if every < 1:
        raise ValueError("a release cycle must span at least one round")
    return every - (round_index % every)


def release_cutoff_block(
    round_index: int,
    *,
    every: int = RELEASE_ROUNDS,
    length: int = ROUND_BLOCKS,
    genesis: int = GENESIS_BLOCK,
    close_margin: int = SUBMISSION_CLOSES_BEFORE_BLOCKS,
    deadline_margin: int = DEADLINE_MARGIN_BLOCKS,
) -> int:
    """The close block of the final round in this cycle.

    Reuses the round's own close block rather than introducing a second
    deadline. A system on the frontier when that round settles is in the
    release, so the cutoff a miner races is one they already understand.
    """
    final = (release_index(round_index, every=every) * every) - 1
    return round_at(
        final,
        length=length,
        genesis=genesis,
        close_margin=close_margin,
        deadline_margin=deadline_margin,
    ).close_block


def blocks_until_release_cutoff(
    block: int,
    *,
    every: int = RELEASE_ROUNDS,
    length: int = ROUND_BLOCKS,
    genesis: int = GENESIS_BLOCK,
    close_margin: int = SUBMISSION_CLOSES_BEFORE_BLOCKS,
    deadline_margin: int = DEADLINE_MARGIN_BLOCKS,
) -> int:
    current = round_for_block(
        block,
        length=length,
        genesis=genesis,
        close_margin=close_margin,
        deadline_margin=deadline_margin,
    )
    cutoff = release_cutoff_block(
        current.index,
        every=every,
        length=length,
        genesis=genesis,
        close_margin=close_margin,
        deadline_margin=deadline_margin,
    )
    return max(0, cutoff - block)


def release_version(track: str, hardware_class: str, index: int) -> str:
    """The name a customer reads off a deployment.

    Track and envelope are in the string so `microtensor-code-mt3g-v1` says what
    it is without a lookup. The class keeps its digits and loses its hyphen,
    because the hyphen is already the field separator.
    """
    return f"microtensor-{track}-{hardware_class.replace('-', '')}-v{index}"
