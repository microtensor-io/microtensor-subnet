from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
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

    def digest(self) -> str:
        """A content hash of the task set.

        The version string says which corpus a worker believes it holds. This
        says which one it actually holds. Two workers can agree on a label while
        holding different files, and a disagreement then arrives as a divergence
        between honest measurements rather than the misconfiguration it is.

        The gold answers are covered. A corpus with the same prompts and
        different expected outputs scores the same systems differently, so
        hashing the prompts alone would leave the useful half unchecked.
        """
        from microtensor.core.hashing import canonical_hash

        return canonical_hash(
            {
                "track": self.track,
                "version": self.version,
                "tasks": [
                    {
                        "ref": task.ref,
                        "prompt": task.prompt,
                        "gold": task.gold,
                        "partition": task.partition,
                        "max_output_tokens": task.max_output_tokens,
                    }
                    for task in self.tasks
                ],
            }
        )

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


SIDECAR_SUFFIXES = (".tests.jsonl", ".train.jsonl", ".reference.jsonl")

# What the control plane will accept on upload, mirrored here so `mt corpus
# check` reports against the authority that actually decides rather than
# against a number of its own. Kept in step with MIN_TASKS in the server's
# corpuscheck; a corpus below these cannot be attached to an arena at all.
ADMISSION_MINIMUMS: Final[dict[str, int]] = {"train": 10, "fixed": 20, "rotating": 40}


def load_tests(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise CorpusError(f"tests file {path} is missing")
    bundle: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
                ref = str(row["ref"])
                tests = list(row["tests"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise CorpusError(f"{path}:{number} is malformed: {exc}") from exc
            for case in tests:
                if not isinstance(case, dict) or (
                    "module" not in case and not ("args" in case and "expected" in case)
                ):
                    raise CorpusError(
                        f"{path}:{number} has a test case that is neither "
                        f"pair shaped nor module shaped"
                    )
            bundle[ref] = {"entry_point": str(row["entry_point"]), "tests": tests}
    return bundle


def attach_tests(corpus: Corpus, bundle: dict[str, dict[str, Any]]) -> Corpus:
    from microtensor.core.hashing import canonical_hash

    attached: list[Task] = []
    for task in corpus:
        declared = task.inputs.get("tests_digest")
        if not declared:
            attached.append(task)
            continue
        entry = bundle.get(task.ref)
        if entry is None:
            raise CorpusError(f"task {task.ref!r} declares hidden tests but none were found")
        actual = canonical_hash(
            {"ref": task.ref, "entry_point": entry["entry_point"], "tests": entry["tests"]}
        )
        if actual != declared:
            raise CorpusError(
                f"task {task.ref!r}: the tests file does not hash to the declared digest"
            )
        attached.append(replace(task, gold=dict(entry)))
    return Corpus(track=corpus.track, version=corpus.version, tasks=tuple(attached))


def corpus_digest(corpora: dict[str, Corpus]) -> str:
    """One digest over every track a worker serves.

    Ordered by track so two workers holding the same corpora agree regardless of
    the order they happened to load them in.
    """
    from microtensor.core.hashing import canonical_hash

    return canonical_hash({track: corpora[track].digest() for track in sorted(corpora)})


def load_all(root: Path, version: str = CORPUS_VERSION) -> dict[str, Corpus]:
    corpora: dict[str, Corpus] = {}
    for path in sorted(root.glob("*.jsonl")):
        if path.name.endswith(SIDECAR_SUFFIXES):
            continue
        corpus = load_corpus(path, path.stem, version)
        tests_path = path.with_name(f"{path.stem}.tests.jsonl")
        if tests_path.is_file():
            corpus = attach_tests(corpus, load_tests(tests_path))
        corpora[path.stem] = corpus
    if not corpora:
        raise CorpusError(f"no corpus files found under {root}")
    return corpora
