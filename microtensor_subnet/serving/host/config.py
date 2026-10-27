from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from microtensor_subnet.serving.engines import LLAMA, LLAMA_IMAGE, SGLANG, SGLANG_IMAGE
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
MEM_FRACTION: Final[float] = 0.88
ENGINE_READY_SECONDS: Final[int] = 900


@dataclass(frozen=True, slots=True)
class HostConfig:
    home: Path
    gpus: tuple[str, ...] = ("0",)
    gpu_gb: float = DEFAULT_GPU_GB
    vllm_image: str = VLLM_IMAGE
    sandbox_image: str = SANDBOX_IMAGE
    gateway_image: str = GATEWAY_IMAGE
    sandbox_runtime: str = ""
    slm_memory_share: float = 0.0
    sandbox_cpus: float = SANDBOX_CPUS
    sandbox_memory: str = SANDBOX_MEMORY
    sandbox_pids: int = SANDBOX_PIDS
    request_seconds: float = REQUEST_SECONDS
    artifact_seconds: int = ARTIFACT_SECONDS
    slm_ready_seconds: int = SLM_READY_SECONDS
    host_services_port: int = 9701
    review_seconds: float = REVIEW_SECONDS
    sglang_image: str = SGLANG_IMAGE
    llama_image: str = LLAMA_IMAGE
    mem_fraction: float = MEM_FRACTION
    engine_ready_seconds: int = ENGINE_READY_SECONDS
    hf_token: str = field(default="", repr=False)

    @property
    def images(self) -> dict[str, str]:
        return {
            backend: image
            for backend, image in ((SGLANG, self.sglang_image), (LLAMA, self.llama_image))
            if image
        }

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def work_dir(self) -> Path:
        return self.home / "work" / "serving"
