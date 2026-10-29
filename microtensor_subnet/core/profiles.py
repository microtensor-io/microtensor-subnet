from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

PROFILES: Final[tuple[str, ...]] = ("P1", "P2", "P3", "K", "C")
CPU_PROFILE: Final[str] = "C"
ENV_PROFILE: Final[str] = "MT_PROFILE"
ARCH: Final[dict[str, str]] = {
    "P1": "sm89",
    "P2": "sm80",
    "P3": "sm90",
    "K": "sm75",
    "C": "cpu",
}
COMPUTE_CAP: Final[dict[str, str]] = {"8.9": "P1", "8.0": "P2", "9.0": "P3", "7.5": "K"}
PRECISION: Final[dict[str, str]] = {
    "P1": "bf16",
    "P2": "bf16",
    "P3": "bf16",
    "K": "fp16",
    "C": "gguf",
}
MEMORY_GB: Final[dict[str, int]] = {"P1": 24, "P2": 80, "P3": 80, "K": 32, "C": 0}
TENSOR_PARALLEL: Final[dict[str, int]] = {"K": 2}
BATCH_INVARIANT: Final[frozenset[str]] = frozenset({"P1", "P2", "P3"})
BATCHED_SEQS: Final[int] = 16
CPU_THREADS: Final[int] = 8
SINGLE_SEQ: Final[int] = 1
GPU_QUERY: Final[tuple[str, ...]] = (
    "nvidia-smi",
    "--query-gpu=compute_cap",
    "--format=csv,noheader",
)
QUERY_SECONDS: Final[float] = 20.0


class ProfileError(ValueError):
    pass


def is_profile(name: str) -> bool:
    return name in PROFILES


def require(name: str) -> str:
    if not is_profile(name):
        raise ProfileError(f"unknown hardware profile {name!r}; known: {', '.join(PROFILES)}")
    return name


def arch_of(profile: str) -> str:
    return ARCH[require(profile)]


def precision_of(profile: str) -> str:
    return PRECISION[require(profile)]


def dtype_of(profile: str) -> str:
    return "float16" if precision_of(profile) == "fp16" else "bfloat16"


def batch_invariant(profile: str) -> bool:
    return require(profile) in BATCH_INVARIANT


def max_num_seqs(profile: str) -> int:
    return BATCHED_SEQS if batch_invariant(profile) else SINGLE_SEQ


def tensor_parallel(profile: str) -> int:
    return TENSOR_PARALLEL.get(require(profile), 1)


def on_cpu(profile: str) -> bool:
    return require(profile) == CPU_PROFILE


def memory_bytes(profile: str, document: Mapping[str, Any] | None = None) -> int:
    found = MEMORY_GB[require(profile)]
    if document is not None:
        entry = (document.get("profiles") or {}).get(profile) or {}
        if isinstance(entry, Mapping) and entry.get("memory_gb") is not None:
            found = int(entry["memory_gb"])
    return found * 1024**3


def profile_for_compute_cap(cap: str) -> str:
    return COMPUTE_CAP.get(cap.strip(), "")


def compute_caps(run: Any = subprocess.run) -> list[str]:
    try:
        found = run(
            list(GPU_QUERY),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=QUERY_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if found.returncode != 0:
        return []
    return [line.strip() for line in str(found.stdout).splitlines() if line.strip()]


@dataclass(frozen=True, slots=True)
class Detection:
    profile: str
    compute_caps: tuple[str, ...]
    source: str

    @property
    def gpu(self) -> bool:
        return self.profile != CPU_PROFILE


def detect(env: Mapping[str, str] | None = None, run: Any = subprocess.run) -> Detection:
    named = (env if env is not None else os.environ).get(ENV_PROFILE, "").strip()
    if named:
        return Detection(require(named), (), ENV_PROFILE)
    caps = compute_caps(run)
    for cap in caps:
        profile = profile_for_compute_cap(cap)
        if profile:
            return Detection(profile, tuple(caps), "nvidia-smi")
    return Detection(CPU_PROFILE, tuple(caps), "nvidia-smi" if caps else "no gpu")


def allowed_profiles(definition: Mapping[str, Any]) -> tuple[str, ...]:
    block = definition.get("evaluation") or {}
    found = block.get("profiles") if isinstance(block, Mapping) else None
    return tuple(str(p) for p in (found or []) if is_profile(str(p)))


def evaluation_mode(definition: Mapping[str, Any]) -> str:
    block = definition.get("evaluation") or {}
    mode = block.get("mode") if isinstance(block, Mapping) else None
    return str(mode or "validator")


def miner_mode(definition: Mapping[str, Any]) -> bool:
    return evaluation_mode(definition) == "miner"
