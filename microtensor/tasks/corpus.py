from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from microtensor.core.constants import CORPUS_VERSION
from microtensor.core.tracks import get_track

ROTATING: Final[str] = "rotating"
FIXED: Final[str] = "fixed"
PARTITIONS: Final[frozenset[str]] = frozenset({ROTATING, FIXED})


class CorpusError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Task:
    ref: str
    prompt: str
    gold: Any
    partition: str
    inputs: dict[str, Any] = field(default_factory=dict)
    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        if not self.ref:
            raise CorpusError("every task needs a stable reference")
        if not self.prompt:
            raise CorpusError(f"task {self.ref!r} has no prompt")
        if self.partition not in PARTITIONS:
            raise CorpusError(f"task {self.ref!r} names unknown partition {self.partition!r}")
        if self.max_output_tokens < 1:
            raise CorpusError(f"task {self.ref!r} allows no output")


@dataclass(frozen=True, slots=True)
class Corpus:
    track: str
    version: str
    tasks: tuple[Task, ...]

    def __post_init__(self) -> None:
        get_track(self.track)
        if not self.tasks:
            raise CorpusError(f"corpus for {self.track!r} is empty")
        refs = [t.ref for t in self.tasks]
        if len(set(refs)) != len(refs):
            raise CorpusError(f"corpus for {self.track!r} repeats a task reference")
        if not self.fixed:
            raise CorpusError(f"corpus for {self.track!r} has no fixed partition to anchor it")

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    @property
    def rotating(self) -> tuple[Task, ...]:
        return tuple(t for t in self.tasks if t.partition == ROTATING)

    @property
    def fixed(self) -> tuple[Task, ...]:
        return tuple(t for t in self.tasks if t.partition == FIXED)

    @property
    def metric(self) -> str:
        return get_track(self.track).metric

    def refs(self, partition: str | None = None) -> tuple[str, ...]:
        return tuple(t.ref for t in self.tasks if partition is None or t.partition == partition)

    def by_ref(self, refs: Sequence[str]) -> tuple[Task, ...]:
        index = {t.ref: t for t in self.tasks}
        return tuple(index[r] for r in refs if r in index)


def _parse(line: str, number: int, path: Path) -> Task:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{path}:{number} is not valid json: {exc}") from exc
    if not isinstance(payload, dict):
        raise CorpusError(f"{path}:{number} is not a json object")

    try:
        return Task(
            ref=str(payload["ref"]),
            prompt=str(payload["prompt"]),
            gold=payload.get("gold"),
            partition=str(payload.get("partition", ROTATING)),
            inputs=dict(payload.get("inputs", {})),
            max_output_tokens=int(payload.get("max_output_tokens", 512)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CorpusError(f"{path}:{number} is malformed: {exc}") from exc


def load_corpus(path: Path, track: str, version: str = CORPUS_VERSION) -> Corpus:
    if not path.is_file():
        raise CorpusError(f"corpus file {path} is missing")

    tasks: list[Task] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if stripped:
                tasks.append(_parse(stripped, number, path))
    return Corpus(track=track, version=version, tasks=tuple(tasks))


def load_all(root: Path, version: str = CORPUS_VERSION) -> dict[str, Corpus]:
    corpora: dict[str, Corpus] = {}
    for path in sorted(root.glob("*.jsonl")):
        corpora[path.stem] = load_corpus(path, path.stem, version)
    if not corpora:
        raise CorpusError(f"no corpus files found under {root}")
    return corpora
