from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from microtensor.core.constants import (
    PROVENANCE_BACKOFF_SECONDS,
    PROVENANCE_ENTITY,
    PROVENANCE_PROJECT,
    PROVENANCE_RETRIES,
)

CONFIG_TRACK: Final[str] = "mt_track"
CONFIG_CLASS: Final[str] = "mt_class"
CONFIG_BASE_MODEL: Final[str] = "mt_base_model"
CONFIG_CORPUS: Final[str] = "mt_corpus_version"

SUMMARY_DIGEST: Final[str] = "mt_artifact_digest"
SUMMARY_FINISHED: Final[str] = "mt_finished_at"

REQUIRED_CONFIG: Final[tuple[str, ...]] = (CONFIG_TRACK, CONFIG_CLASS, CONFIG_BASE_MODEL)


class ProvenanceUnavailable(RuntimeError):
    """The run store could not be reached or authenticated. Abstain, do not score."""


@dataclass(frozen=True, slots=True)
class Run:
    hotkey: str
    run_id: str
    config: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    entity: str = PROVENANCE_ENTITY
    project: str = PROVENANCE_PROJECT

    @property
    def path(self) -> str:
        return f"{self.entity}/{self.project}/{self.hotkey}"

    @property
    def url(self) -> str:
        return f"https://wandb.ai/{self.entity}/{self.project}/runs/{self.run_id}"

    @property
    def digest(self) -> str:
        return str(self.summary.get(SUMMARY_DIGEST, ""))

    @property
    def finished_block(self) -> int:
        """The chain block height at which training finished.

        A height, never a timestamp. The two were once confused here — a
        close block multiplied into fake seconds against a wall clock — and
        the gate rejected every compliant miner (issue #1). Heights are exact,
        need no anchor, and are what the rest of the system already speaks.
        A run that still carries a Unix timestamp compares as an absurdly
        distant future block and fails closed.
        """
        try:
            return int(self.summary.get(SUMMARY_FINISHED, 0))
        except (TypeError, ValueError):
            return 0


class RunStore(Protocol):
    def fetch(self, hotkey: str) -> Run | None: ...


@dataclass(frozen=True, slots=True)
class Verdict:
    admissible: bool
    reason: str = ""
    run: Run | None = None

    def payload(self) -> dict[str, Any]:
        if self.run is None:
            return {}
        return {
            "run": self.run.path,
            "run_url": self.run.url,
            "digest_match": self.admissible,
            "finished_block": self.run.finished_block,
        }


NO_RUN: Final[str] = "no training run for this hotkey"
DIGEST_MISMATCH: Final[str] = "training run digest does not match the submitted artifact"
BASE_NOT_ALLOWED: Final[str] = "training run declares a base model not on the allowlist"
WRONG_COMPETITION: Final[str] = "training run declares a different track or class"
FINISHED_LATE: Final[str] = "training run finished after the round's commitment deadline block"
MISSING_FIELD: Final[str] = "training run is missing a required field"


def check(
    run: Run | None,
    *,
    hotkey: str,
    artifact_digest: str,
    track: str,
    hardware_class: str,
    commit_block: int = 0,
    allowed_base_models: frozenset[str] = frozenset(),
) -> Verdict:
    if run is None:
        return Verdict(False, NO_RUN)

    missing = [key for key in REQUIRED_CONFIG if not str(run.config.get(key, "")).strip()]
    if missing or not run.digest:
        return Verdict(False, MISSING_FIELD, run)

    if run.digest != artifact_digest:
        return Verdict(False, DIGEST_MISMATCH, run)

    if str(run.config[CONFIG_TRACK]) != track or str(run.config[CONFIG_CLASS]) != hardware_class:
        return Verdict(False, WRONG_COMPETITION, run)

    # An empty set here means the caller knows of no allowlist to check
    # against, not that every model passes: discovery enforces the arena's own
    # list separately and rejects when it is empty.
    base = str(run.config[CONFIG_BASE_MODEL])
    if allowed_base_models and base not in allowed_base_models:
        return Verdict(False, BASE_NOT_ALLOWED, run)

    if commit_block and run.finished_block and run.finished_block > commit_block:
        return Verdict(False, FINISHED_LATE, run)

    return Verdict(True, "", run)


class CachedStore:
    def __init__(
        self,
        store: RunStore,
        *,
        retries: int = PROVENANCE_RETRIES,
        backoff: float = PROVENANCE_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._store = store
        self._retries = max(1, retries)
        self._backoff = backoff
        self._sleep = sleep
        self._cache: dict[str, Run | None] = {}
        self.calls = 0

    def fetch(self, hotkey: str) -> Run | None:
        if hotkey in self._cache:
            return self._cache[hotkey]

        last: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                self.calls += 1
                run = self._store.fetch(hotkey)
            except ProvenanceUnavailable as exc:
                last = exc
                if attempt < self._retries:
                    self._sleep(self._backoff * attempt)
                continue
            self._cache[hotkey] = run
            return run

        raise ProvenanceUnavailable(
            f"the run store was unreachable after {self._retries} attempts: {last}"
        )

    def reset(self) -> None:
        self._cache.clear()
