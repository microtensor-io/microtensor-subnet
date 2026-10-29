from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from microtensor_subnet.chain.config import ChainConfig
from microtensor_subnet.core.constants import PUBLIC_SERVER_URL

VLLM_IMAGE: Final[str] = "vllm/vllm-openai:v0.10.1"
LLAMA_IMAGE: Final[str] = "ghcr.io/ggml-org/llama.cpp:server"
SANDBOX_IMAGE: Final[str] = "ghcr.io/microtensor-io/sandbox:1"
REPLAYER_IMAGE: Final[str] = "ghcr.io/microtensor-io/replayer:1"
GATEWAY_IMAGE: Final[str] = "ghcr.io/microtensor-io/validator:1"
CUDA_IMAGE: Final[str] = "nvidia/cuda:12.4.1-base-ubuntu22.04"
PROFILE_ENV: Final[str] = "MT_PROFILE"
CPU_PROFILE: Final[str] = "C"
PROFILES_MODULE: Final[str] = "microtensor_subnet.core.profiles"
CONCURRENCY: Final[int] = 3


def _profiles() -> Any:
    return importlib.import_module(PROFILES_MODULE)


def detect_profile(override: str = "", env: Mapping[str, str] | None = None) -> str:
    named = override.strip() or str((env if env is not None else os.environ).get(PROFILE_ENV, ""))
    found = _profiles().detect({PROFILE_ENV: named.strip()} if named.strip() else {})
    return str(found.profile)


def profile_memory_bytes(profile: str, document: Mapping[str, Any] | None = None) -> int:
    return int(_profiles().memory_bytes(profile, document))


def evaluation_mode(definition: Mapping[str, Any]) -> str:
    return str(_profiles().evaluation_mode(definition))


def miner_mode(definition: Mapping[str, Any]) -> bool:
    return evaluation_mode(definition) == "miner"


def gpu_profile(profile: str) -> bool:
    return bool(profile) and profile != CPU_PROFILE


SANDBOX_CPUS: Final[float] = 4.0
SANDBOX_MEMORY: Final[str] = "8g"
SANDBOX_PIDS: Final[int] = 256
REQUEST_SECONDS: Final[float] = 120.0
ARTIFACT_SECONDS: Final[int] = 4 * 3600
SLM_READY_SECONDS: Final[int] = 900


@dataclass(frozen=True, slots=True)
class EpochConfig:
    chain: ChainConfig
    home: Path
    arena_id: str
    server_url: str = PUBLIC_SERVER_URL
    relay_url: str = PUBLIC_SERVER_URL
    gpus: tuple[str, ...] = ()
    concurrency: int = CONCURRENCY
    vllm_image: str = VLLM_IMAGE
    llama_image: str = LLAMA_IMAGE
    sandbox_image: str = SANDBOX_IMAGE
    replayer_image: str = REPLAYER_IMAGE
    gateway_image: str = GATEWAY_IMAGE
    cuda_image: str = CUDA_IMAGE
    profile: str = ""
    sandbox_runtime: str = ""
    sandbox_cpus: float = SANDBOX_CPUS
    sandbox_memory: str = SANDBOX_MEMORY
    sandbox_pids: int = SANDBOX_PIDS
    request_seconds: float = REQUEST_SECONDS
    artifact_seconds: int = ARTIFACT_SECONDS
    slm_ready_seconds: int = SLM_READY_SECONDS
    host_services_port: int = 9701
    allow_unsandboxed: bool = False
    slm_memory_share: float = 0.0
    dry_run: bool = False
    item_limit: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def work_dir(self) -> Path:
        return self.home / "work" / self.arena_id

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def keys_dir(self) -> Path:
        return self.home / "keys"

    @property
    def mirror_dir(self) -> Path:
        return self.cache_dir / "mirror"

    def resolved_profile(self) -> str:
        return detect_profile(self.profile)
