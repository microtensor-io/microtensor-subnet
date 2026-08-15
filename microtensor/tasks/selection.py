from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from microtensor.core.constants import (
    FIXED_FRACTION,
    ROTATING_FRACTION,
    TASKS_PER_ROUND,
)
from microtensor.core.hashing import round_seed, select_deterministic, task_nonce
from microtensor.core.tracks import get_track
from microtensor.harness.contract import Request
from microtensor.tasks.corpus import FIXED, ROTATING, Corpus, CorpusError, Task


@dataclass(frozen=True, slots=True)
class RoundTasks:
    track: str
    hardware_class: str
    seed: str
    corpus_version: str
    rotating: tuple[Task, ...]
    fixed: tuple[Task, ...]
    round_index: int = 0

    @property
    def all(self) -> tuple[Task, ...]:
        return self.rotating + self.fixed

    def __len__(self) -> int:
        return len(self.rotating) + len(self.fixed)

    @property
    def refs(self) -> tuple[str, ...]:
        return tuple(t.ref for t in self.all)


def partition_sizes(
    budget: int = TASKS_PER_ROUND,
    rotating_fraction: float = ROTATING_FRACTION,
    fixed_fraction: float = FIXED_FRACTION,
) -> tuple[int, int]:
    if budget < 2:
        raise CorpusError("a round needs at least one task from each partition")
    return math.ceil(rotating_fraction * budget), math.floor(fixed_fraction * budget)


def competition_seed(block_hash: str, track: str, hardware_class: str) -> str:
    return round_seed(block_hash, track, hardware_class)


def select(
    corpus: Corpus,
    seed: str,
    hardware_class: str,
    *,
    budget: int = TASKS_PER_ROUND,
    round_index: int = 0,
) -> RoundTasks:
    want_rotating, want_fixed = partition_sizes(budget)

    rotating_refs = select_deterministic(
        [t.ref for t in corpus.rotating], seed, min(want_rotating, len(corpus.rotating))
    )
    fixed_pool = corpus.fixed
    fixed_refs = sorted(t.ref for t in fixed_pool)[:want_fixed] if want_fixed else []

    if not rotating_refs and not fixed_refs:
        raise CorpusError(f"corpus for {corpus.track!r} yielded no tasks for this round")

    return RoundTasks(
        track=corpus.track,
        hardware_class=hardware_class,
        seed=seed,
        corpus_version=corpus.version,
        rotating=corpus.by_ref(rotating_refs),
        fixed=corpus.by_ref(fixed_refs),
        round_index=round_index,
    )


def to_requests(
    tasks: Sequence[Task],
    seed: str,
    track: str,
    artifact_digest: str = "",
) -> list[Request]:
    decoding = get_track(track).decoding
    return [
        Request(
            task_ref=task.ref,
            prompt=task.prompt,
            inputs=dict(task.inputs),
            max_output_tokens=task.max_output_tokens,
            decoding=decoding,
            seed=int(task_nonce(seed, task.ref)[:8], 16) if decoding.value == "seeded" else 0,
            nonce=task_nonce(seed, task.ref, artifact_digest),
        )
        for task in tasks
    ]


def partition_of(tasks: RoundTasks, ref: str) -> str:
    return ROTATING if ref in {t.ref for t in tasks.rotating} else FIXED
