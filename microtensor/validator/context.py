from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from microtensor.chain.client import ChainClient
from microtensor.chain.config import ChainConfig
from microtensor.core.constants import (
    ARTIFACT_CACHE_CAP_BYTES,
    CORPUS_VERSION,
    GENESIS_BLOCK,
    ROUND_BLOCKS,
    TASKS_PER_ROUND,
)
from microtensor.core.readiness import unenforced
from microtensor.core.tracks import competitions, validate_registry
from microtensor.provenance.record import CachedStore
from microtensor.registry.cache import ArtifactCache
from microtensor.scoring import execution
from microtensor.store.state import ValidatorState
from microtensor.tasks.corpus import Corpus, load_all

log = logging.getLogger("microtensor.validator")


@dataclass(frozen=True, slots=True)
class ValidatorConfig:
    chain: ChainConfig
    home: Path
    corpus_dir: Path
    corpus_version: str = CORPUS_VERSION
    round_blocks: int = ROUND_BLOCKS
    genesis_block: int = GENESIS_BLOCK
    tasks_per_round: int = TASKS_PER_ROUND
    cache_cap_bytes: int = ARTIFACT_CACHE_CAP_BYTES
    cpu_seconds_per_artifact: int = 900
    profile_seconds: int = 60
    allow_unsandboxed: bool = False
    dry_run: bool = False
    verify_signatures: bool = True
    degraded: bool = False
    coordinator_url: str = ""
    standalone: bool = False

    @property
    def coordinated(self) -> bool:
        """Whether this worker takes its work from a coordinator.

        `--standalone` wins over a configured URL so a full independent round is
        always reachable by config rather than only by failure, which is how the
        fallback gets tested without taking the coordinator down.
        """
        return bool(self.coordinator_url) and not self.standalone

    @property
    def state_path(self) -> Path:
        return self.home / "state" / "validator.sqlite"

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def work_dir(self) -> Path:
        return self.home / "work"


@dataclass(slots=True)
class ValidatorContext:
    config: ValidatorConfig
    client: ChainClient
    state: ValidatorState
    cache: ArtifactCache
    corpora: dict[str, Corpus]
    hotkey: str = ""
    wallet: Any = None
    competitions: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    certifications: dict[str, dict[str, Any]] = field(default_factory=dict)
    runs: CachedStore | None = None
    coordinator: Any = None

    @classmethod
    def build(
        cls,
        config: ValidatorConfig,
        client: ChainClient,
        *,
        wallet: Any = None,
        hotkey: str = "",
        runs: CachedStore | None = None,
        coordinator: Any = None,
    ) -> ValidatorContext:
        validate_registry()
        execution.configure(allow_unsandboxed=config.allow_unsandboxed)
        config.work_dir.mkdir(parents=True, exist_ok=True)

        for gate in unenforced():
            log.warning("%s is UNENFORCED: %s", gate.name, gate.detail)

        corpora = load_all(config.corpus_dir, config.corpus_version)
        open_competitions = tuple(
            (track, hardware) for track, hardware in competitions() if track in corpora
        )
        if not open_competitions:
            raise RuntimeError(
                f"no corpus under {config.corpus_dir} matches an enabled track; "
                f"found {sorted(corpora)}"
            )

        missing = {track for track, _ in competitions()} - set(corpora)
        if missing:
            log.warning("no corpus for enabled tracks %s; they will not be scored", sorted(missing))

        from microtensor.envelope.certify import load_policies

        return cls(
            config=config,
            client=client,
            state=ValidatorState(config.state_path),
            cache=ArtifactCache(config.cache_dir, config.cache_cap_bytes),
            corpora=corpora,
            hotkey=hotkey,
            wallet=wallet,
            competitions=open_competitions,
            certifications=load_policies(config.home),
            runs=runs,
            coordinator=coordinator,
        )

    @property
    def tracks(self) -> tuple[str, ...]:
        return tuple(sorted({track for track, _ in self.competitions}))

    def corpus(self, track: str) -> Corpus:
        return self.corpora[track]

    def close(self) -> None:
        self.state.close()
        self.client.close()
