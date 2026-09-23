from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path

from microtensor.rigs.validator.storage.keys import MASTER_SECRET_ENV

TRUE_VALUES = ("1", "true", "yes", "on")
DEFAULT_SERVER_URL = "https://api.microtensor.cloud"
DEFAULT_CHAIN_ENDPOINT = "wss://test.finney.opentensor.ai:443"
DEFAULT_STATE_DIR = "/var/lib/compute-validator"
DEFAULT_CHALLENGE_LIBRARY = "/usr/lib/libmtverify.so"
DEFAULT_AGENT_LIBRARY = "/usr/lib/libmtchallenge.so"


def _text(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUE_VALUES


def _optional_int(name: str) -> int | None:
    raw = _text(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _list(name: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.environ.get(name, "").split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    server_url: str = field(default_factory=lambda: _text("CV_SERVER_URL", DEFAULT_SERVER_URL))
    label: str = field(default_factory=lambda: _text("CV_LABEL"))
    hotkey_seed: str = field(default_factory=lambda: _text("CV_HOTKEY_SEED"))
    hotkey_mnemonic: str = field(default_factory=lambda: _text("CV_HOTKEY_MNEMONIC"))
    wallet_hotkey_file: str = field(default_factory=lambda: _text("CV_WALLET_HOTKEY_FILE"))
    network: str = field(default_factory=lambda: _text("CV_NETWORK", "test"))
    netuid: int = field(default_factory=lambda: _int("CV_NETUID", 92))
    chain_endpoint: str = field(
        default_factory=lambda: _text("CV_CHAIN_ENDPOINT", DEFAULT_CHAIN_ENDPOINT)
    )
    set_weights: bool = field(default_factory=lambda: _bool("CV_SET_WEIGHTS", False))
    weights_seconds: int = field(default_factory=lambda: _int("CV_WEIGHTS_SECONDS", 3600))
    max_inflight: int = field(default_factory=lambda: max(1, _int("CV_MAX_INFLIGHT", 4)))
    per_miner: int = field(default_factory=lambda: max(1, _int("CV_PER_MINER", 2)))
    ssh_timeout: int = field(default_factory=lambda: _int("CV_SSH_TIMEOUT", 60))
    check_timeout: int = field(default_factory=lambda: _int("CV_CHECK_TIMEOUT", 300))
    state_dir: Path = field(default_factory=lambda: Path(_text("CV_STATE_DIR", DEFAULT_STATE_DIR)))
    express_seconds: int = field(default_factory=lambda: max(5, _int("CV_EXPRESS_SECONDS", 30)))
    deep_seconds: int = field(default_factory=lambda: max(0, _int("CV_DEEP_SECONDS", 0)))
    challenge_library: Path = field(
        default_factory=lambda: Path(_text("CV_CHALLENGE_LIBRARY", DEFAULT_CHALLENGE_LIBRARY))
    )
    agent_library: Path = field(
        default_factory=lambda: Path(_text("CV_AGENT_LIBRARY", DEFAULT_AGENT_LIBRARY))
    )
    allow_reference_challenge: bool = field(
        default_factory=lambda: _bool("CV_ALLOW_REFERENCE_CHALLENGE", False)
    )
    challenge_max_ms: float = field(default_factory=lambda: _float("CV_CHALLENGE_MAX_MS", 15000.0))
    reserve_uid: int | None = field(default_factory=lambda: _optional_int("CV_RESERVE_UID"))
    held_share: float = field(
        default_factory=lambda: min(max(_float("CV_HELD_SHARE", 0.4), 0.0), 1.0)
    )
    min_driver_version: str = field(default_factory=lambda: _text("CV_MIN_DRIVER_VERSION"))
    driver_cutoff: str = field(default_factory=lambda: _text("CV_DRIVER_CUTOFF"))
    mechanism_id: int | None = field(default_factory=lambda: _optional_int("CV_MECHANISM_ID"))
    volume_secret_set: bool = field(default_factory=lambda: bool(_text(MASTER_SECRET_ENV)))
    image_allowlist: tuple[str, ...] = field(default_factory=lambda: _list("CV_IMAGE_ALLOWLIST"))
    signature_window_seconds: int = 120

    def driver_cutoff_at(self) -> dt.datetime | None:
        if not self.driver_cutoff:
            return None
        try:
            moment = dt.datetime.fromisoformat(self.driver_cutoff)
        except ValueError:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=dt.timezone.utc)
        return moment

    def driver_cutoff_passed(self, now: dt.datetime | None = None) -> bool:
        cutoff = self.driver_cutoff_at()
        if cutoff is None:
            return False
        moment = now or dt.datetime.now(dt.timezone.utc)
        return moment >= cutoff

    def path(self, name: str) -> Path:
        return self.state_dir / name


def settings() -> Settings:
    return Settings()
