from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from microtensor_subnet.chain.config import ChainConfig
from microtensor_subnet.core.constants import PUBLIC_SERVER_URL

VLLM_IMAGE: Final[str] = "vllm/vllm-openai:v0.10.1"
SANDBOX_IMAGE: Final[str] = "ghcr.io/microtensor-io/sandbox:1"
GATEWAY_IMAGE: Final[str] = "ghcr.io/microtensor-io/validator:1"
CONCURRENCY: Final[int] = 3
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
    sandbox_image: str = SANDBOX_IMAGE
    gateway_image: str = GATEWAY_IMAGE
    sandbox_runtime: str = "runsc"
    sandbox_cpus: float = SANDBOX_CPUS
    sandbox_memory: str = SANDBOX_MEMORY
    sandbox_pids: int = SANDBOX_PIDS
    request_seconds: float = REQUEST_SECONDS
    artifact_seconds: int = ARTIFACT_SECONDS
    slm_ready_seconds: int = SLM_READY_SECONDS
    host_services_port: int = 9701
    allow_unsandboxed: bool = False
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
