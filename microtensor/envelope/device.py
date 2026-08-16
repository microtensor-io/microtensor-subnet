from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from typing import Any

from microtensor.core.hashing import canonical_hash
from microtensor.core.tracks import HardwareClass

PROFILE_PREFIX = "dev:"


POLICY_ENV = "MT_DEVICE_POLICY"


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    machine: str
    system: str
    cpu_cores: int
    total_memory_bytes: int
    accelerator: str = ""
    accelerator_memory_bytes: int = 0
    cooling_mode: str = ""
    power_mode: str = ""
    warmup_policy: str = ""
    idle_seconds: int = 0

    @property
    def digest(self) -> str:
        return PROFILE_PREFIX + canonical_hash(asdict(self))[7:23]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def conforms_to(self, reference: DeviceProfile, *, memory_tolerance: float = 0.05) -> bool:
        if self.machine != reference.machine or self.system != reference.system:
            return False
        if self.cpu_cores < reference.cpu_cores:
            return False
        if self.accelerator != reference.accelerator:
            return False
        floor = reference.total_memory_bytes * (1.0 - memory_tolerance)
        return self.total_memory_bytes >= floor


def _total_memory_bytes() -> int:
    try:
        import psutil
    except ImportError:
        return _sysconf_memory_bytes()
    return int(psutil.virtual_memory().total)


def _sysconf_memory_bytes() -> int:
    names = getattr(os, "sysconf_names", {})
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None or "SC_PHYS_PAGES" not in names or "SC_PAGE_SIZE" not in names:
        return 0
    try:
        return int(sysconf("SC_PHYS_PAGES")) * int(sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return 0


def _accelerator() -> tuple[str, int]:
    import shutil
    import subprocess

    binary = shutil.which("nvidia-smi")
    if binary is None:
        return "", 0

    try:
        out = subprocess.run(  # noqa: S603
            [binary, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            shell=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return "", 0

    line = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
    if not line:
        return "", 0
    name, _, memory = line.partition(",")
    try:
        return name.strip(), int(float(memory.strip())) * 1024 * 1024
    except ValueError:
        return name.strip(), 0


def _declared_policy() -> dict[str, Any]:
    raw = os.environ.get(POLICY_ENV, "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def detect() -> DeviceProfile:
    accelerator, accelerator_memory = _accelerator()
    policy = _declared_policy()
    return DeviceProfile(
        machine=platform.machine().lower(),
        system=platform.system().lower(),
        cpu_cores=os.cpu_count() or 1,
        total_memory_bytes=_total_memory_bytes(),
        accelerator=accelerator,
        accelerator_memory_bytes=accelerator_memory,
        cooling_mode=str(policy.get("cooling_mode", "")),
        power_mode=str(policy.get("power_mode", "")),
        warmup_policy=str(policy.get("warmup_policy", "")),
        idle_seconds=int(policy.get("idle_seconds", 0) or 0),
    )


def reference_digest(hardware: HardwareClass) -> str:
    if hardware.device_profile:
        return hardware.device_profile
    return PROFILE_PREFIX + hashlib.sha256(hardware.reference.encode()).hexdigest()[:16]


def conforms(profile: DeviceProfile, hardware: HardwareClass) -> bool:
    expected = hardware.device_profile
    return not expected or profile.digest == expected
