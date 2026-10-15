from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from microtensor_subnet.validator.config import (
    ARTIFACT_SECONDS,
    GATEWAY_IMAGE,
    REQUEST_SECONDS,
    SANDBOX_CPUS,
    SANDBOX_IMAGE,
    SANDBOX_MEMORY,
    SANDBOX_PIDS,
    SLM_READY_SECONDS,
    VLLM_IMAGE,
)

DEFAULT_GPU_GB: Final[float] = 24.0
REVIEW_SECONDS: Final[float] = 300.0


@dataclass(frozen=True, slots=True)
class HostConfig:
    home: Path
    gpus: tuple[str, ...] = ("0",)
    gpu_gb: float = DEFAULT_GPU_GB
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
    review_seconds: float = REVIEW_SECONDS

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def work_dir(self) -> Path:
        return self.home / "work" / "serving"
