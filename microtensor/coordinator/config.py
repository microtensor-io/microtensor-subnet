from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

from microtensor.core.constants import (
    CLASS_WEIGHTS,
    COORDINATOR_REPLICATION,
    CORPUS_VERSION,
    MECHANISM_VERSION,
    ROLE_BASELINES,
    ROUND_BLOCKS,
    SUBMISSION_CLOSES_BEFORE_BLOCKS,
    TASKS_PER_ROUND,
)
from microtensor.core.tracks import CLASSES, competitions, enabled_tracks

CONFIG_VERSION = 1

MISMATCH = "the served config does not match the hash anchored on chain for this round"


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
            key: {"allowed_base_models": sorted(value.get("allowed_base_models", []))}
            for key, value in sorted((arenas or {}).items())
        },
        "role_baselines": dict(sorted(ROLE_BASELINES.items())),
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
