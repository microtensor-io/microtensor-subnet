from __future__ import annotations

from typing import Final

MECHANISM_VERSION: Final[str] = "0.1.0"
DEFAULT_NETUID: Final[int] = 92
GENESIS_BLOCK: Final[int] = 0

BLOCK_TIME_SECONDS: Final[int] = 12
ROUND_BLOCKS: Final[int] = 7200
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

EMA_ALPHA: Final[float] = 0.30
DECAY_RATE: Final[float] = 0.40
RECOVERY_RATE: Final[float] = 0.10
CONCENTRATION_CAP_FRACTION: Final[float] = 0.25

PARAM_DISTANCE_THRESHOLD: Final[float] = 0.02
BEHAVIOUR_DISTANCE_THRESHOLD: Final[float] = 0.05
PROBE_SET_SIZE: Final[int] = 256

ALLOWED_BASE_MODELS: Final[frozenset[str]] = frozenset()

PROVENANCE_ENTITY: Final[str] = "microtensor"
PROVENANCE_PROJECT: Final[str] = "training-runs"
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
RELEASE_SIGNING_KEY: Final[str] = ""
UPDATE_POLL_SECONDS: Final[int] = 900
UPDATE_HTTP_TIMEOUT_SECONDS: Final[int] = 60
UPDATE_MAX_ASSET_BYTES: Final[int] = 64 * 1024**2

POLL_INTERVAL_SECONDS: Final[int] = 30
WEIGHT_REFRESH_SECONDS: Final[int] = 1200
WEIGHT_RETRY_SECONDS: Final[int] = 180
MIN_SCORED_FRACTION: Final[float] = 0.50
