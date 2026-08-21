from __future__ import annotations

from typing import Final

MECHANISM_VERSION: Final[str] = "0.1.0"
DEFAULT_NETUID: Final[int] = 92
GENESIS_BLOCK: Final[int] = 0

BLOCK_TIME_SECONDS: Final[int] = 12
ROUND_BLOCKS: Final[int] = 7200
RELEASE_ROUNDS: Final[int] = 30
WEIGHT_INTERVAL_SECONDS: Final[int] = 1320
EPOCH_BLOCKS: Final[int] = 360
WEIGHT_REFRESH_BLOCKS: Final[int] = 300
TELEMETRY_HEARTBEAT_BLOCKS: Final[int] = 300
TELEMETRY_RETAIN_DAYS: Final[int] = 30
SUBMISSION_CLOSES_BEFORE_BLOCKS: Final[int] = 600
DEADLINE_MARGIN_BLOCKS: Final[int] = 40

MIN_VALIDATOR_STAKE: Final[float] = 1000.0
MAX_COMMITMENT_BYTES: Final[int] = 128
METAGRAPH_TTL_SECONDS: Final[int] = 300
CHAIN_ATTEMPTS: Final[int] = 4
CHAIN_BACKOFF_SECONDS: Final[float] = 2.0

ROTATING_FRACTION: Final[float] = 0.70
FIXED_FRACTION: Final[float] = 0.30
TASKS_PER_ROUND: Final[int] = 200
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
COORDINATOR_HOTKEY: Final[str] = ""
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
MIN_ROUNDS_OBSERVED: Final[int] = 2

PAID_RANKS: Final[int] = 8
RANK_DECAY: Final[float] = 0.85
HYSTERESIS_EPSILON: Final[float] = 0.005
INCUMBENT_DECAY: Final[float] = 0.05
STALE_ROUNDS_BEFORE_EVICTION: Final[int] = 6

HV_QUANT_Q: Final[int] = 10_000
HV_QUANT_C: Final[int] = 10_000
EPS_QUALITY: Final[float] = 0.005
EPS_COST: Final[float] = 0.01
REFERENCE_COST_MS: Final[float] = 2_000.0

EMA_ALPHA: Final[float] = 0.30
DECAY_RATE: Final[float] = 0.40
RECOVERY_RATE: Final[float] = 0.10
CONCENTRATION_CAP_FRACTION: Final[float] = 0.25

PARAM_DISTANCE_THRESHOLD: Final[float] = 0.02
BEHAVIOUR_DISTANCE_THRESHOLD: Final[float] = 0.05
PROBE_SET_SIZE: Final[int] = 256

PROVENANCE_ENTITY: Final[str] = "microtensor"
PROVENANCE_PROJECT: Final[str] = "training-runs"
# Off until the run store can accept runs from arbitrary miners. The
# microtensor W&B entity is a personal account, so an outside key cannot
# write to it, and requiring a run nobody can create rejects every
# submission. Turn back on once the entity is a team miners can write to.
PROVENANCE_REQUIRED: Final[bool] = False
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
UPDATE_POLL_SECONDS: Final[int] = 900
UPDATE_HTTP_TIMEOUT_SECONDS: Final[int] = 60
UPDATE_MAX_ASSET_BYTES: Final[int] = 64 * 1024**2

POLL_INTERVAL_SECONDS: Final[int] = 30
WEIGHT_REFRESH_SECONDS: Final[int] = 1200
WEIGHT_RETRY_SECONDS: Final[int] = 180
MIN_SCORED_FRACTION: Final[float] = 0.50
