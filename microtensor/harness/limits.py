from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Final

from microtensor.core.constants import WALL_BACKSTOP_FACTOR
from microtensor.core.tracks import HardwareClass

POSIX = os.name == "posix"


class UnsupportedPlatform(RuntimeError):
    pass


def _resource() -> Any:
    if not POSIX:
        raise UnsupportedPlatform(
            "resource limits require a POSIX host; validators must run on Linux"
        )
    import resource

    return resource


@dataclass(frozen=True, slots=True)
class Limits:
    cpu_seconds: int
    wall_seconds: int
    rss_bytes: int
    open_files: int = 256
    processes: int = 0
    core_dumps: bool = False

    def __post_init__(self) -> None:
        if self.cpu_seconds < 1:
            raise ValueError("cpu budget must be at least one second")
        if self.wall_seconds <= self.cpu_seconds:
            raise ValueError("the wall backstop must exceed the cpu budget")
        if self.rss_bytes < 1:
            raise ValueError("rss ceiling must be positive")

    @classmethod
    def for_class(
        cls,
        hardware: HardwareClass,
        cpu_seconds: int,
        *,
        backstop: int = WALL_BACKSTOP_FACTOR,
        headroom: float = 1.10,
    ) -> Limits:
        return cls(
            cpu_seconds=cpu_seconds,
            wall_seconds=cpu_seconds * max(2, backstop),
            rss_bytes=int(hardware.max_rss_bytes * headroom),
        )


def maxrss_to_bytes(maxrss: int) -> int:
    return int(maxrss) if sys.platform == "darwin" else int(maxrss) * 1024


DETERMINISTIC_ENV: Final[dict[str, str]] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


def pin_threads() -> None:
    os.environ.update(DETERMINISTIC_ENV)


def apply(limits: Limits) -> None:
    resource = _resource()

    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (limits.rss_bytes, limits.rss_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (limits.open_files, limits.open_files))

    if limits.processes > 0:
        resource.setrlimit(resource.RLIMIT_NPROC, (limits.processes, limits.processes))

    if not limits.core_dumps:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def cpu_seconds_used() -> float:
    resource = _resource()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def children_cpu_seconds() -> float:
    try:
        resource = _resource()
    except UnsupportedPlatform:
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


def peak_rss_bytes() -> int:
    if not POSIX:
        return 0
    resource = _resource()
    return maxrss_to_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def sandbox_available() -> bool:
    return POSIX
