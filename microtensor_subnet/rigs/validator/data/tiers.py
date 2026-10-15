from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    ENTRY = "entry"
    STANDARD = "standard"
    PROFESSIONAL = "professional"
    FLAGSHIP = "flagship"


TIER_ORDER: tuple[Tier, ...] = (Tier.ENTRY, Tier.STANDARD, Tier.PROFESSIONAL, Tier.FLAGSHIP)
TIER_MEMORY_GB: dict[Tier, tuple[int, int | None]] = {
    Tier.ENTRY: (8, 16),
    Tier.STANDARD: (20, 24),
    Tier.PROFESSIONAL: (32, 48),
    Tier.FLAGSHIP: (80, None),
}
FEE_MULTIPLIER: dict[Tier, int] = {
    Tier.ENTRY: 1,
    Tier.STANDARD: 2,
    Tier.PROFESSIONAL: 3,
    Tier.FLAGSHIP: 4,
}


def tier_for_memory_gb(nominal_gb: float) -> Tier | None:
    if nominal_gb >= 80:
        return Tier.FLAGSHIP
    if nominal_gb >= 32:
        return Tier.PROFESSIONAL
    if nominal_gb >= 20:
        return Tier.STANDARD
    if nominal_gb >= 8:
        return Tier.ENTRY
    return None


def tier_at_least(tier: Tier, floor: Tier) -> bool:
    return TIER_ORDER.index(tier) >= TIER_ORDER.index(floor)


class WorkKind(str, Enum):
    RENTAL = "rental"
    INFERENCE = "inference"
    SHAPING = "shaping"
    MINING = "mining"


PRIORITY: tuple[WorkKind, ...] = (
    WorkKind.RENTAL,
    WorkKind.INFERENCE,
    WorkKind.SHAPING,
    WorkKind.MINING,
)


@dataclass(frozen=True)
class Minimums:
    tier: Tier
    vram_gb: int
    ram_ratio: float
    disk_gb: int
    cpu_cores: int
    isolation: bool = False
    quotas: bool = False
    public_ipv4: bool = False
    port_range: bool = False
    bandwidth_mbps: int = 0


MINIMUMS: dict[WorkKind, Minimums] = {
    WorkKind.INFERENCE: Minimums(Tier.ENTRY, 8, 1.0, 180, 4),
    WorkKind.SHAPING: Minimums(Tier.STANDARD, 24, 1.5, 500, 8, bandwidth_mbps=200),
    WorkKind.RENTAL: Minimums(
        Tier.STANDARD,
        24,
        2.0,
        1000,
        8,
        isolation=True,
        quotas=True,
        public_ipv4=True,
        port_range=True,
        bandwidth_mbps=200,
    ),
    WorkKind.MINING: Minimums(Tier.ENTRY, 8, 1.0, 180, 4),
}

MIN_KERNEL: tuple[int, int] = (5, 19)
MIN_DISK_GB = 180
SUPPORTED_OS: tuple[tuple[str, str], ...] = (("ubuntu", "22.04"), ("ubuntu", "24.04"))
SUPPORTED_ARCH = "x86_64"


@dataclass(frozen=True)
class Profile:
    gpu_model: str
    gpu_count: int
    vram_gb: float
    tier: Tier | None
    ram_gb: float
    disk_gb: float
    disk_ssd: bool
    cpu_cores: int
    isolation_ok: bool
    quotas_ok: bool
    public_ipv4: bool
    port_range_ok: bool
    bandwidth_mbps: float
    kernel: tuple[int, int]
    arch: str
    os_id: str
    os_version: str
    driver_version: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def total_vram_gb(self) -> float:
        return self.vram_gb * max(self.gpu_count, 0)


def host_reasons(profile: Profile) -> list[str]:
    reasons: list[str] = []
    if (profile.os_id.lower(), profile.os_version) not in SUPPORTED_OS:
        reasons.append(
            f"operating system {profile.os_id} {profile.os_version} is not Ubuntu 22.04 or 24.04"
        )
    if profile.kernel < MIN_KERNEL:
        reasons.append(f"kernel {profile.kernel[0]}.{profile.kernel[1]} is below 5.19")
    if profile.arch != SUPPORTED_ARCH:
        reasons.append(f"architecture {profile.arch} is not x86_64")
    if profile.disk_gb < MIN_DISK_GB:
        reasons.append(f"disk {profile.disk_gb:.0f} GB is below {MIN_DISK_GB} GB")
    if not profile.disk_ssd:
        reasons.append("disk is not solid state")
    if not profile.public_ipv4:
        reasons.append("no public IPv4 address without carrier grade translation")
    if profile.tier is None:
        reasons.append(f"{profile.gpu_model} with {profile.vram_gb:.0f} GB is below the entry tier")
    return reasons


def work_reasons(profile: Profile, kind: WorkKind) -> list[str]:
    need = MINIMUMS[kind]
    reasons: list[str] = []
    if profile.tier is None or not tier_at_least(profile.tier, need.tier):
        reasons.append(f"{need.tier.value} tier or above required")
    if profile.vram_gb < need.vram_gb:
        reasons.append(f"{need.vram_gb} GB of video memory per card required")
    if profile.ram_gb < need.ram_ratio * profile.total_vram_gb:
        reasons.append(f"system memory must be at least {need.ram_ratio:g}x total video memory")
    if profile.disk_gb < need.disk_gb:
        reasons.append(f"{need.disk_gb} GB of solid state disk required")
    if profile.cpu_cores < need.cpu_cores:
        reasons.append(f"{need.cpu_cores} CPU cores required")
    if need.isolation and not profile.isolation_ok:
        reasons.append("container isolation not active")
    if need.quotas and not profile.quotas_ok:
        reasons.append("storage driver cannot enforce per container quotas")
    if need.public_ipv4 and not profile.public_ipv4:
        reasons.append("public routable address required")
    if need.port_range and not profile.port_range_ok:
        reasons.append("reserved port range required")
    if need.bandwidth_mbps and profile.bandwidth_mbps < need.bandwidth_mbps:
        reasons.append(f"sustained bandwidth of {need.bandwidth_mbps} Mbps required")
    return reasons


def eligibility(profile: Profile, rental_opted_out: bool = False) -> dict[WorkKind, list[str]]:
    result: dict[WorkKind, list[str]] = {}
    for kind in PRIORITY:
        reasons = work_reasons(profile, kind)
        if kind is WorkKind.RENTAL and rental_opted_out:
            reasons.append("rental declined by the owner")
        result[kind] = reasons
    return result


def eligible_kinds(profile: Profile, rental_opted_out: bool = False) -> list[WorkKind]:
    return [kind for kind, reasons in eligibility(profile, rental_opted_out).items() if not reasons]


def render(profile: Profile, rental_opted_out: bool = False) -> list[str]:
    lines: list[str] = []
    for kind, reasons in eligibility(profile, rental_opted_out).items():
        lines.append(f"{kind.value:<10} {'eligible' if not reasons else 'not eligible'}")
        lines.extend(f"           {reason}" for reason in reasons)
    return lines
