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


# The kernel kills on the hard CPU limit with no warning. A soft limit one
# grace period below it delivers SIGXCPU first, which the worker turns into a
# named failure — the difference between "the cpu budget ran out" and a bare
# exit -9 that could be anything.
CPU_GRACE_SECONDS: Final[int] = 5

# Address space is not resident memory. A GGUF is mmapped, so its bytes count
# against RLIMIT_AS while never being resident, and charging an artifact for
# a mapping it never faults in would reject the memory-efficient path while
# admitting the wasteful one. The RSS ceiling is enforced by measurement.
ADDRESS_SLACK_BYTES: Final[int] = 2 * 1024**3


@dataclass(frozen=True, slots=True)
class Limits:
    cpu_seconds: int
    wall_seconds: int
    rss_bytes: int
    address_bytes: int = 0
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

    @property
    def address_ceiling(self) -> int:
        """What RLIMIT_AS gets: resident ceiling plus room to map the weights."""
        return self.address_bytes or (self.rss_bytes + ADDRESS_SLACK_BYTES)

    @classmethod
    def for_class(
        cls,
        hardware: HardwareClass,
        cpu_seconds: int,
        *,
        backstop: int = WALL_BACKSTOP_FACTOR,
        headroom: float = 1.10,
    ) -> Limits:
        rss = int(hardware.max_rss_bytes * headroom)
        return cls(
            cpu_seconds=cpu_seconds,
            wall_seconds=cpu_seconds * max(2, backstop),
            rss_bytes=rss,
            # The artifact may be mapped in full alongside what it makes
            # resident, so the address ceiling carries the size ceiling too.
            address_bytes=rss + int(hardware.max_size_bytes) + ADDRESS_SLACK_BYTES,
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


CLONE_NEWUSER: Final[int] = 0x10000000
CLONE_NEWNET: Final[int] = 0x40000000


def network_available() -> bool:
    return POSIX and sys.platform.startswith("linux")


def _unshare(flags: int) -> None:
    call = getattr(os, "unshare", None)
    if call is not None:
        call(flags)
        return

    import ctypes

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.unshare(flags) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def interfaces() -> list[str]:
    try:
        with open("/proc/net/dev", encoding="ascii") as handle:
            lines = handle.read().splitlines()[2:]
    except OSError:
        return []
    return [line.split(":", 1)[0].strip() for line in lines if ":" in line]


def drop_network() -> None:
    if not network_available():
        raise UnsupportedPlatform(
            "network isolation requires Linux namespaces; validators must run on Linux"
        )

    try:
        _unshare(CLONE_NEWUSER | CLONE_NEWNET)
    except OSError:
        _unshare(CLONE_NEWNET)

    remaining = [name for name in interfaces() if name != "lo"]
    if remaining:
        raise UnsupportedPlatform(
            f"network namespace still exposes {', '.join(remaining)}"
        )


def apply(limits: Limits) -> None:
    resource = _resource()

    resource.setrlimit(
        resource.RLIMIT_CPU,
        (limits.cpu_seconds, limits.cpu_seconds + CPU_GRACE_SECONDS),
    )
    resource.setrlimit(resource.RLIMIT_AS, (limits.address_ceiling, limits.address_ceiling))
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


def children_peak_rss() -> int:
    """Peak resident bytes across reaped children, or zero off POSIX.

    A killed worker sends nothing back, so this is the only record of how
    much memory it actually held.
    """
    try:
        resource = _resource()
    except UnsupportedPlatform:
        return 0
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return maxrss_to_bytes(usage.ru_maxrss)


def peak_rss_bytes() -> int:
    if not POSIX:
        return 0
    resource = _resource()
    return maxrss_to_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def sandbox_available() -> bool:
    return POSIX
