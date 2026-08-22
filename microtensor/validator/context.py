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
    CPU_SECONDS_PER_ARTIFACT,
    GENESIS_BLOCK,
    PROVENANCE_REQUIRED,
    ROUND_BLOCKS,
    TASKS_PER_ROUND,
)
from microtensor.core.readiness import unenforced
from microtensor.core.tracks import competitions, validate_registry
from microtensor.provenance.record import CachedStore
from microtensor.registry.cache import ArtifactCache
from microtensor.scoring import execution
from microtensor.store.state import ValidatorState
from microtensor.tasks.corpus import Corpus, CorpusError, load_all

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
    cpu_seconds_per_artifact: int = CPU_SECONDS_PER_ARTIFACT
    profile_seconds: int = 60
    allow_unsandboxed: bool = False
    dry_run: bool = False
    loopback: bool = False
    # Loopback and the end-to-end harness stand up a synthetic arena, so they
    # carry their own allowlist rather than reaching for a constant. A real
    # deployment leaves this empty and takes the arena's list from the config
    # the chain anchored.
    allowlists: dict[tuple[str, str], frozenset[str]] = field(default_factory=dict)
    verify_signatures: bool = True
    # Defaults to the constant and is not exposed on the command line. Only
    # the synthetic modes turn it off, the same way they turn off signature
    # verification: loopback has no W&B and no real miners, so requiring a
    # training run there would test the absence of a network rather than the
    # gate. An operator flag here would let one validator admit what its peers
    # reject, which is the divergence the gate exists to prevent.
    require_provenance: bool = PROVENANCE_REQUIRED
    degraded: bool = False
    coordinator_url: str = ""

    @property
    def coordinated(self) -> bool:
        """Whether this worker takes its work from a coordinator.

        Without one it holds: rounds exist only on the server now, so there is
        no independent round to run, only a standing vector to keep flowing.
        """
        return bool(self.coordinator_url)

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

        corpora = _corpora_for(config, coordinator)
        missing = sorted({track for track, _ in competitions()} - set(corpora))

        # A coordinated worker with nothing yet is waiting, not misconfigured.
        # The coordinator serves a corpus once an arena is live, so before then
        # there is nothing to have, and refusing to start makes an operator
        # debug a validator that is behaving correctly. It holds each round and
        # looks again instead. Standing alone with no corpus is still fatal:
        # there is nothing to wait for.
        if missing and config.coordinated and not corpora:
            log.warning(
                "no corpus yet for %s; the coordinator serves one when an arena is "
                "live, so this validator will hold and look again each round",
                ", ".join(missing),
            )
        elif missing:
            raise RuntimeError(
                f"no corpus for enabled track(s) {missing}. "
                f"A validator that skips a track publishes a vector missing that "
                f"track's emission share while its peers include it, which is a "
                f"consensus divergence rather than a smaller sample, so this is "
                f"refused rather than logged. Found corpora for {sorted(corpora)}."
            )

        open_competitions = tuple(competitions())

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
    def corpus_digest(self) -> str:
        from microtensor.tasks.corpus import corpus_digest

        return corpus_digest(self.corpora) if self.corpora else ""

    @property
    def tracks(self) -> tuple[str, ...]:
        return tuple(sorted({track for track, _ in self.competitions}))

    def corpus(self, track: str) -> Corpus:
        return self.corpora[track]

    def refresh_corpora(self) -> bool:
        """Look again for a corpus this worker did not have at startup.

        Called at the top of a round rather than on a timer, because the round
        is when it matters and the coordinator publishes one when an arena
        goes live. Returns whether anything is held now, so the caller can say
        what it is waiting for rather than failing.
        """
        if self.corpora:
            return True
        found = _corpora_for(self.config, self.coordinator)
        if found:
            self.corpora.update(found)
            log.info("the coordinator now serves %s", ", ".join(sorted(found)))
        return bool(self.corpora)

    def close(self) -> None:
        self.state.close()
        self.client.close()


def _corpora_for(config: Any, coordinator: Any) -> dict[str, Corpus]:
    """The task set this validator will measure against.

    A coordinated worker takes it from the coordinator so every worker in the
    round measures the same tasks. Without that, two workers disagreeing about a
    system is unreadable: nobody can tell a wrong measurement from a different
    question, and the reconciliation majority counts configurations instead.

    Falling back to a local corpus is deliberate for loopback and for a
    coordinator that serves none yet. It is logged either way, because a
    validator quietly measuring its own tasks is the failure this exists to stop.
    """
    from microtensor.validator.corpus import fetch

    if coordinator is not None:
        try:
            served = fetch(coordinator)
        except Exception as exc:
            log.warning("the coordinator served no usable corpus (%s); using the local one", exc)
        else:
            if served:
                return served
            log.warning("the coordinator serves no corpus yet; using the local one")

    log.info("loading the corpus from %s", config.corpus_dir)
    try:
        return load_all(config.corpus_dir, config.corpus_version)
    except CorpusError:
        # Nothing served and nothing local. For a coordinated worker that is a
        # wait, not a fault: the coordinator publishes a corpus when an arena
        # goes live. Standing alone it is fatal, because nothing is coming.
        if not config.coordinated:
            raise
        log.warning(
            "no corpus from the coordinator and none at %s; holding until one is served",
            config.corpus_dir,
        )
        return {}
