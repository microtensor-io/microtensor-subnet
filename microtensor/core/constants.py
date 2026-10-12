from __future__ import annotations

import os
from typing import Final


def _blocks(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _ms(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# What is packaged, tagged and compared to decide whether a release is newer.
# Moves every release, including one that only fixes a bug.
RELEASE_VERSION: Final[str] = "0.5.0"

# What the rules are. Moves only when admission or scoring changes, because
# validators on different values here measure the same round differently, and
# an update that changes it has to land on every host at the same block.
#
# These were one constant. That made every release read as a rule change, so
# auto-update held all of them forever waiting for an activation block that a
# bug fix has no reason to declare.
MECHANISM_VERSION: Final[str] = "0.6.0"
DEFAULT_NETWORK: Final[str] = os.environ.get("MT_NETWORK", "").strip() or "test"
DEFAULT_NETUID: Final[int] = _blocks("MT_NETUID", 576)
GENESIS_BLOCK: Final[int] = _blocks("MT_GENESIS_BLOCK", -17805488)

BLOCK_TIME_SECONDS: Final[int] = 12
ROUND_BLOCKS: Final[int] = _blocks("MT_ROUND_BLOCKS", 21600)
RELEASE_ROUNDS: Final[int] = 30
SUBMISSION_CLOSES_BEFORE_BLOCKS: Final[int] = _blocks("MT_SUBMISSION_CLOSES_BEFORE_BLOCKS", 7200)


def _rounds(name: str) -> tuple[int, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    found = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            found.append(int(part))
    return tuple(sorted(set(found)))


ALSO_ACCEPT_ROUNDS: Final[tuple[int, ...]] = _rounds("MT_ALSO_ACCEPT_ROUNDS")

REQUIRE_SEALED_SUBMISSIONS: Final[bool] = (
    os.environ.get("MT_REQUIRE_SEALED_SUBMISSIONS", "").strip() == "1"
)
DEADLINE_MARGIN_BLOCKS: Final[int] = 40

MIN_VALIDATOR_STAKE: Final[float] = 1000.0
MAX_COMMITMENT_BYTES: Final[int] = 128
METAGRAPH_TTL_SECONDS: Final[int] = 300
CHAIN_ATTEMPTS: Final[int] = 4
CHAIN_BACKOFF_SECONDS: Final[float] = 2.0

ROTATING_FRACTION: Final[float] = 0.60
FIXED_FRACTION: Final[float] = 0.30
NOVEL_FRACTION: Final[float] = 0.10
TASKS_PER_ROUND: Final[int] = 200

CORPUS_VERSION: Final[str] = "2026.1"

CLASS_WEIGHTS: Final[dict[str, float]] = {"mt-4g": 1.0}

HOST_PROFILE: Final[str] = "mt-16g"

ALLOWED_ROUTER_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "seq_logprob",
        "seq_logprob_norm",
        "output_tokens",
        "mean_entropy",
        "max_entropy",
        "schema_valid",
        "input_tokens",
        "answer_prob",
        "answer_margin",
        "answer_entropy",
        "input_typicality",
        "harness_errors",
    }
)
ROUTER_MAX_BYTES: Final[int] = 4 * 1024**2
ROUTER_ALLOWED_OPS: Final[frozenset[str]] = frozenset(
    {"Gemm", "MatMul", "Add", "Relu", "Sigmoid", "Tanh"}
)

ROLE_BASELINES: Final[dict[str, str]] = {
    "front": "",
    "router": "",
    "specialist": "",
}

COORDINATOR_URL: Final[str] = os.environ.get(
    "MT_COORDINATOR_URL", "https://coordinator.microtensor.cloud"
).strip()
PUBLIC_SERVER_URL: Final[str] = "https://api.microtensor.cloud"
GATEWAY_URL: Final[str] = os.environ.get(
    "MT_GATEWAY_URL", "wss://api.microtensor.cloud/v1/operators/socket"
).strip()

WALL_BACKSTOP_FACTOR: Final[int] = 3
DECLARATION_TOLERANCE_SIZE: Final[float] = 0.02
DECLARATION_TOLERANCE_LATENCY: Final[float] = 0.10
DECLARATION_LATENCY_FLOOR_MS: Final[int] = 2_000

ACCURACY_DECIMALS: Final[int] = 4

POSITION_SHARES: Final[tuple[float, ...]] = (0.30, 0.20, 0.14, 0.11, 0.09, 0.07, 0.05, 0.04)


MAX_MANIFEST_BYTES: Final[int] = 64 * 1024

ARTIFACT_FETCH_RETRIES: Final[int] = 3
ARTIFACT_FETCH_TIMEOUT_SECONDS: Final[int] = 900
ARTIFACT_CACHE_CAP_BYTES: Final[int] = 200 * 1024**3

RELEASE_CHANNELS: Final[tuple[str, ...]] = ("stable", "prerelease")
# The public half of the release signing key. Public on purpose: a validator
# needs it to verify a release before installing it, and a key only it holds
# would verify nothing. The private seed lives in the repository's Actions
# secrets as MT_RELEASE_SIGNING_SEED and is never committed — anyone holding
# it can sign a release every auto-updating validator will run.
RELEASE_SIGNING_KEY: Final[str] = (
    "0x3d8ea239db66637d762ffedf71ad6c0c487c7bc73d5a50d9dd86a0fbc22bdb16"
)
