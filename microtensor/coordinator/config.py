from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

from microtensor.core.constants import (
    ALSO_ACCEPT_ROUNDS,
    CLASS_WEIGHTS,
    COORDINATOR_REPLICATION,
    CORPUS_VERSION,
    CPU_SECONDS_PER_ARTIFACT,
    GENESIS_BLOCK,
    MECHANISM_VERSION,
    ROLE_BASELINES,
    ROUND_BLOCKS,
    SUBMISSION_CLOSES_BEFORE_BLOCKS,
    TASKS_PER_ROUND,
)
from microtensor.core.tracks import CLASSES, competitions, enabled_tracks

CONFIG_VERSION = 1

MISMATCH = "the served config does not match the hash anchored on chain for this round"


def _role_baselines(arenas: Mapping[str, Mapping[str, Any]] | None) -> dict[str, str]:
    """The arena's published baselines when it carries them, else the constants.

    Baselines are published through the control plane per arena; the constants
    are the pre-arena fallback. Without this, an arena with all three roles
    published still anchored a config saying none were.
    """
    merged = dict(ROLE_BASELINES)
    for value in (arenas or {}).values():
        found = value.get("role_baselines")
        if isinstance(found, Mapping):
            merged.update({str(k): str(v) for k, v in found.items() if v})
    return merged


def _arena_block(value: Mapping[str, Any]) -> dict[str, Any]:
    """One arena's rules, with every number written out.

    The budget is stamped here rather than left to each worker's constants.
    An absent value must not mean "whatever your build happens to think": the
    anchored document is what every worker measures against, so it carries a
    concrete number and workers on different builds still agree.

    These live with the arena because they describe the same model class the
    ceilings do. A second class opening, or a ceiling moving, changes them
    together, and none of it should need a release.
    """
    block: dict[str, Any] = {
        "allowed_base_models": sorted(value.get("allowed_base_models", [])),
        "cpu_seconds_per_artifact": int(
            value.get("cpu_seconds_per_artifact") or CPU_SECONDS_PER_ARTIFACT
        ),
        "tasks_per_round": int(value.get("tasks_per_round") or TASKS_PER_ROUND),
        "ceilings": {
            k: int(v)
            for k, v in dict(value.get("ceilings") or {}).items()
            if k in ("max_size_bytes", "max_rss_bytes", "max_p95_ms") and int(v) > 0
        },
    }
    environment = value.get("environment_digest")
    if environment:
        block["environment_digest"] = str(environment)
    return block


def served_config(
    corpus_version: str = CORPUS_VERSION,
    arenas: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Everything a worker needs to agree on before it measures anything.

    Serving this is fine. Serving it unanchored is not, because the rules of
    the competition could then change without anyone being able to prove they
    had. The hash of this document is committed on chain at round start and
    every worker checks the two match.

    `arenas` carries what the control plane holds per competition, keyed
    "track/class" — today the base-model allowlist. It rides in the anchored
    config so a worker measures against a list the chain was told about,
    rather than one served over HTTP after the fact.
    """
    return {
        "version": CONFIG_VERSION,
        "mechanism_version": MECHANISM_VERSION,
        "corpus_version": corpus_version,
        "also_accept_rounds": list(ALSO_ACCEPT_ROUNDS),
        "genesis_block": GENESIS_BLOCK,
        "round_blocks": ROUND_BLOCKS,
        "submission_closes_before_blocks": SUBMISSION_CLOSES_BEFORE_BLOCKS,
        "tasks_per_round": TASKS_PER_ROUND,
        "replication": COORDINATOR_REPLICATION,
        "competitions": [list(c) for c in competitions()],
        "tracks": {
            t.id: {"metric": t.metric, "emission_share": t.emission_share} for t in enabled_tracks()
        },
        "class_weights": dict(sorted(CLASS_WEIGHTS.items())),
        "classes": {
            c.id: {
                "max_size_bytes": c.max_size_bytes,
                "max_rss_bytes": c.max_rss_bytes,
                "max_p95_ms": c.max_p95_ms,
            }
            for c in CLASSES.values()
        },
        "arenas": {
            key: _arena_block(value) for key, value in sorted((arenas or {}).items())
        },
        "role_baselines": dict(sorted(_role_baselines(arenas).items())),
    }


def canonical(config: dict[str, Any]) -> bytes:
    return json.dumps(config, sort_keys=True, separators=(",", ":")).encode()


def config_hash(config: dict[str, Any] | None = None) -> str:
    return f"sha256:{sha256(canonical(config or served_config())).hexdigest()}"


def matches(config: dict[str, Any], anchored: str) -> bool:
    """Whether a served config is the one the chain was told about.

    An empty anchor means the coordinator has not committed yet, which is a
    mismatch rather than a pass: an unverifiable config is exactly the state
    this check exists to refuse.
    """
    return bool(anchored) and config_hash(config) == anchored
