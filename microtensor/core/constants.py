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
RELEASE_VERSION: Final[str] = "0.1.14"

# What the rules are. Moves only when admission or scoring changes, because
# validators on different values here measure the same round differently, and
# an update that changes it has to land on every host at the same block.
#
# These were one constant. That made every release read as a rule change, so
# auto-update held all of them forever waiting for an activation block that a
# bug fix has no reason to declare.
MECHANISM_VERSION: Final[str] = "0.1.2"
DEFAULT_NETUID: Final[int] = 92
GENESIS_BLOCK: Final[int] = _blocks("MT_GENESIS_BLOCK", -17805488)

BLOCK_TIME_SECONDS: Final[int] = 12
ROUND_BLOCKS: Final[int] = _blocks("MT_ROUND_BLOCKS", 21600)
RELEASE_ROUNDS: Final[int] = 30
WEIGHT_INTERVAL_SECONDS: Final[int] = 1320
EPOCH_BLOCKS: Final[int] = 360
WEIGHT_REFRESH_BLOCKS: Final[int] = 300
TELEMETRY_HEARTBEAT_BLOCKS: Final[int] = 300
TELEMETRY_RETAIN_DAYS: Final[int] = 30
SUBMISSION_CLOSES_BEFORE_BLOCKS: Final[int] = _blocks(
    "MT_SUBMISSION_CLOSES_BEFORE_BLOCKS", 7200
)


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

REVEAL_WINDOW_BLOCKS: Final[int] = _blocks("MT_REVEAL_WINDOW_BLOCKS", 25)
REQUIRE_SEALED_SUBMISSIONS: Final[bool] = (
    os.environ.get("MT_REQUIRE_SEALED_SUBMISSIONS", "").strip() == "1"
)
DERIVATION_ENFORCING: Final[bool] = (
    os.environ.get("MT_DERIVATION_ENFORCING", "").strip() == "1"
)
DEADLINE_MARGIN_BLOCKS: Final[int] = 40

MIN_VALIDATOR_STAKE: Final[float] = 1000.0
MAX_COMMITMENT_BYTES: Final[int] = 128
METAGRAPH_TTL_SECONDS: Final[int] = 300
CHAIN_ATTEMPTS: Final[int] = 4
CHAIN_BACKOFF_SECONDS: Final[float] = 2.0

ROTATING_FRACTION: Final[float] = 0.70
FIXED_FRACTION: Final[float] = 0.30
TASKS_PER_ROUND: Final[int] = 200

# The default round budget, stamped into an arena that does not set its own.
#
# Not the binding value: cpu_seconds_per_artifact and tasks_per_round live in
# the arena record and ride in the anchored config, because they describe the
# same model class the latency ceilings do and have to move with them. This is
# what served_config writes when the control plane leaves them unset, so the
# anchored document always carries a concrete number and workers on different
# builds still agree.
#
# Derived rather than chosen: a round runs TASKS_PER_ROUND tasks and each is
# allowed up to its class's p95 ceiling, so a budget under the product forfeits
# work the ceilings permit. The widest ceiling is mt-16g at 30 s, and 200 x 30 s
# is this number. Threads are pinned to one, so cpu and wall time are the same
# quantity here and the two limits compare directly.
#
# The previous 900 gave each task 4.5 cpu-seconds against ceilings written for
# 10 to 30 — a fivefold contradiction between two numbers describing the same
# model on the same hardware, resolving in the direction that scores honest
# miners zero. test_budget_agrees_with_the_ceilings fails if they drift again.
CPU_SECONDS_PER_ARTIFACT: Final[int] = 6000
CORPUS_VERSION: Final[str] = "2026.1"

CLASS_WEIGHTS: Final[dict[str, float]] = {"mt-3g": 1.0}

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

COORDINATOR_REPLICATION: Final[int] = 3
COORDINATOR_QUORUM: Final[float] = 0.67
COORDINATOR_RETRIES: Final[int] = 4
COORDINATOR_BACKOFF_SECONDS: Final[float] = 5.0
COORDINATOR_TIMEOUT_SECONDS: Final[int] = 30
COORDINATOR_URL: Final[str] = ""
COORDINATOR_SERVER_URL: Final[str] = ""
# The hotkey whose on-chain commitment counts as the config anchor. Empty
# means this build cannot check that a served config is the one the chain was
# told about, so a validator refuses the round rather than measuring against
# ceilings nobody committed to.
COORDINATOR_HOTKEY: Final[str] = "5FeHbWK12HHMLY4AtnWkKk8jtQajhQZMLCenGd96Hhs4UJGc"
PUBLIC_SERVER_URL: Final[str] = "https://api.microtensor.cloud"
# The operator plane is bound to loopback on the server host, so the default
# is what an operator reaches over an ssh tunnel rather than a public name.
CONTROL_URL: Final[str] = "http://127.0.0.1:8081"
COORDINATOR_PORT: Final[int] = 8443
REPORT_MAX_BYTES: Final[int] = 256 * 1024
REPUTATION_FLOOR: Final[float] = 0.80
REPUTATION_RECOVERY_ROUNDS: Final[int] = 5
REPUTATION_MIN_ROUNDS: Final[int] = 10

PROFILE_DURATION_SECONDS: Final[int] = 60
PROFILE_SAMPLE_INTERVAL_MS: Final[int] = 50
LATENCY_SAMPLE_COUNT: Final[int] = 200
WALL_BACKSTOP_FACTOR: Final[int] = 3
DECLARATION_TOLERANCE_SIZE: Final[float] = 0.02
DECLARATION_TOLERANCE_RSS: Final[float] = 0.02
DECLARATION_TOLERANCE_LATENCY: Final[float] = 0.10
DECLARATION_LATENCY_FLOOR_MS: Final[int] = 15

ACCURACY_DECIMALS: Final[int] = 4
TRACK_THRESHOLD: Final[float] = 0.20
MIN_ROUNDS_OBSERVED: Final[int] = _blocks("MT_MIN_ROUNDS_OBSERVED", 2)

PAID_RANKS: Final[int] = 8
RANK_DECAY: Final[float] = 0.85
HYSTERESIS_EPSILON: Final[float] = 0.005
INCUMBENT_DECAY: Final[float] = 0.05
STALE_ROUNDS_BEFORE_EVICTION: Final[int] = 6

HV_QUANT_Q: Final[int] = 10_000
HV_QUANT_C: Final[int] = 10_000
EPS_QUALITY: Final[float] = 0.005
EPS_COST: Final[float] = 0.01
REFERENCE_COST_MS: Final[float] = _ms("MT_REFERENCE_COST_MS", 2_000.0)

EMA_ALPHA: Final[float] = 0.30
DECAY_RATE: Final[float] = 0.40
RECOVERY_RATE: Final[float] = 0.10
CONCENTRATION_CAP_FRACTION: Final[float] = 0.25

PARAM_DISTANCE_THRESHOLD: Final[float] = 0.02
BEHAVIOUR_DISTANCE_THRESHOLD: Final[float] = 0.05
PROBE_SET_SIZE: Final[int] = 256

PROVENANCE_ENTITY: Final[str] = "microtensor"
PROVENANCE_PROJECT: Final[str] = "training-runs"
# On. The microtensor entity is a W&B team and training-runs is set to Open
# visibility, so a miner writes from their own account without an invitation
# and a validator reads with any W&B key. Both directions were checked with an
# account outside the team before this was turned on: refused under Team
# visibility, accepted under Open.
#
# What this gate is: an audit trail. Every field it reads is written by the
# miner, so it makes a claim public and checkable against the artifact digest,
# the competition and the allowlist. It does not prove a training run happened.
# The store assumes exactly that, which is why every run bearing a hotkey is a
# candidate and the checks decide rather than authorship.
PROVENANCE_REQUIRED: Final[bool] = True
PROVENANCE_RETRIES: Final[int] = 3
PROVENANCE_BACKOFF_SECONDS: Final[float] = 3.0

EXEC_CPU_SECONDS: Final[int] = 2
EXEC_WALL_SECONDS: Final[int] = 6
EXEC_RSS_BYTES: Final[int] = 256 * 1024**2

SLOTS_PER_TRACK_CLASS: Final[int] = 40
SUBMISSION_COOLDOWN_SECONDS: Final[int] = 6 * 3600
MAX_MANIFEST_BYTES: Final[int] = 64 * 1024

ARTIFACT_FETCH_RETRIES: Final[int] = 3
ARTIFACT_FETCH_TIMEOUT_SECONDS: Final[int] = 900
ARTIFACT_CACHE_CAP_BYTES: Final[int] = 200 * 1024**3

RELEASE_REPO: Final[str] = "microtensor-io/microtensor-subnet"
RELEASE_CHANNELS: Final[tuple[str, ...]] = ("stable", "prerelease")
# The public half of the release signing key. Public on purpose: a validator
# needs it to verify a release before installing it, and a key only it holds
# would verify nothing. The private seed lives in the repository's Actions
# secrets as MT_RELEASE_SIGNING_SEED and is never committed — anyone holding
# it can sign a release every auto-updating validator will run.
RELEASE_SIGNING_KEY: Final[str] = (
    "0x3d8ea239db66637d762ffedf71ad6c0c487c7bc73d5a50d9dd86a0fbc22bdb16"
)
UPDATE_POLL_SECONDS: Final[int] = 300
UPDATE_HTTP_TIMEOUT_SECONDS: Final[int] = 60
UPDATE_MAX_ASSET_BYTES: Final[int] = 64 * 1024**2

POLL_INTERVAL_SECONDS: Final[int] = 30
WEIGHT_REFRESH_SECONDS: Final[int] = 1200
WEIGHT_RETRY_SECONDS: Final[int] = 180
MIN_SCORED_FRACTION: Final[float] = 0.50
