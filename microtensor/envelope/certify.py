from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from microtensor.envelope.device import POLICY_ENV, DeviceProfile, detect
from microtensor.envelope.latency import Distribution, summarise
from microtensor.envelope.sampler import ResidentSampler

WORKLOAD_VERSION: Final[str] = "1"
DEFAULT_REPETITIONS: Final[int] = 15
HASH_ROUNDS: Final[int] = 60_000

LAUNCH_CLASSES: Final[tuple[str, ...]] = ("mt-3g",)

BUFFER_BYTES: Final[dict[str, int]] = {
    "mt-16g": 384 * 1024**2,
    "mt-4g": 256 * 1024**2,
    "mt-3g": 192 * 1024**2,
    "mt-1g": 96 * 1024**2,
}

CERT_BANDS: Final[dict[str, dict[str, float]]] = {}

DEFAULT_POLICY: Final[dict[str, Any]] = {
    "cooling_mode": "active",
    "power_mode": "performance",
    "warmup_policy": "one-pass",
    "idle_seconds": 5,
}


class CertificationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Certification:
    class_id: str
    workload_version: str
    latency: Distribution
    peak_rss_bytes: int
    device: DeviceProfile
    policy: dict[str, Any]

    @property
    def digest(self) -> str:
        return self.device.digest

    def payload(self) -> dict[str, Any]:
        return {
            "class": self.class_id,
            "workload_version": self.workload_version,
            "p50_ms": self.latency.p50,
            "p95_ms": self.latency.p95,
            "peak_rss_bytes": self.peak_rss_bytes,
            "device": self.device.to_dict(),
            "device_profile": self.digest,
            "policy": self.policy,
        }


def _workload_pass(buffer_bytes: int) -> float:
    started = time.perf_counter()
    block = bytearray(buffer_bytes)
    stride = 4096
    digest = hashlib.sha256(b"microtensor-certify").digest()
    for offset in range(0, buffer_bytes, stride):
        block[offset] = digest[offset % len(digest)]
    for _ in range(HASH_ROUNDS):
        digest = hashlib.sha256(digest).digest()
    checksum = int.from_bytes(digest[:4], "big") ^ block[0]
    if checksum < 0:
        raise CertificationError("unreachable")
    return (time.perf_counter() - started) * 1000.0


def certify(
    class_id: str,
    policy: dict[str, Any] | None = None,
    *,
    repetitions: int = DEFAULT_REPETITIONS,
) -> Certification:
    if class_id not in LAUNCH_CLASSES:
        raise CertificationError(
            f"certification ships for {list(LAUNCH_CLASSES)} at launch, not {class_id!r}"
        )
    if repetitions < 3:
        raise CertificationError("certification needs at least three repetitions")

    declared = dict(DEFAULT_POLICY)
    declared.update(policy or {})
    os.environ[POLICY_ENV] = json.dumps(declared, sort_keys=True)

    device = detect()
    buffer_bytes = BUFFER_BYTES[class_id]

    sampler = ResidentSampler(interval_ms=20)
    sampler.start()
    try:
        _workload_pass(buffer_bytes)
        samples = [_workload_pass(buffer_bytes) for _ in range(repetitions)]
    finally:
        sampler.stop()

    return Certification(
        class_id=class_id,
        workload_version=WORKLOAD_VERSION,
        latency=summarise(samples),
        peak_rss_bytes=sampler.peak_bytes,
        device=device,
        policy=declared,
    )


BAND_SLACK: Final[float] = 0.20
BAND_RSS_SLACK: Final[float] = 0.15


@dataclass(frozen=True, slots=True)
class FittedBand:
    class_id: str
    runs: int
    p50_observed: tuple[float, float]
    p95_observed: tuple[float, float]
    rss_observed: int
    band: dict[str, float]

    @property
    def p50_spread(self) -> float:
        lo, hi = self.p50_observed
        return (hi - lo) / lo if lo > 0 else 0.0

    @property
    def p95_spread(self) -> float:
        lo, hi = self.p95_observed
        return (hi - lo) / lo if lo > 0 else 0.0

    def as_constant(self) -> str:
        rows = ", ".join(f'"{k}": {v:.6g}' for k, v in self.band.items())
        return f'    "{self.class_id}": {{{rows}}},'


def fit_band(
    class_id: str,
    policy: dict[str, Any] | None = None,
    *,
    runs: int = 10,
    repetitions: int = DEFAULT_REPETITIONS,
    slack: float = BAND_SLACK,
    rss_slack: float = BAND_RSS_SLACK,
) -> tuple[FittedBand, list[Certification]]:
    if runs < 3:
        raise CertificationError("fitting a band needs at least three runs")

    results = [certify(class_id, policy, repetitions=repetitions) for _ in range(runs)]
    profiles = {c.digest for c in results}
    if len(profiles) != 1:
        raise CertificationError(
            f"the device profile changed across runs ({sorted(profiles)}); the declared "
            "policy is not stable, so a band fitted here would not mean anything"
        )

    p50s = [c.latency.p50 for c in results]
    p95s = [c.latency.p95 for c in results]
    rss = max(c.peak_rss_bytes for c in results)

    band = {
        "p50_lo": min(p50s) * (1.0 - slack),
        "p50_hi": max(p50s) * (1.0 + slack),
        "p95_lo": min(p95s) * (1.0 - slack),
        "p95_hi": max(p95s) * (1.0 + slack),
        "rss_max": float(int(rss * (1.0 + rss_slack))),
    }

    return (
        FittedBand(
            class_id=class_id,
            runs=runs,
            p50_observed=(min(p50s), max(p50s)),
            p95_observed=(min(p95s), max(p95s)),
            rss_observed=rss,
            band=band,
        ),
        results,
    )


def band_verdict(certification: Certification) -> tuple[bool | None, str]:
    band = CERT_BANDS.get(certification.class_id)
    if band is None:
        return None, (
            "no published tolerance band for this class yet; measurements recorded "
            "for calibration"
        )
    ok = (
        band["p50_lo"] <= certification.latency.p50 <= band["p50_hi"]
        and band["p95_lo"] <= certification.latency.p95 <= band["p95_hi"]
        and certification.peak_rss_bytes <= band["rss_max"]
    )
    return ok, "within the published band" if ok else "outside the published band"


def certification_dir(home: Path) -> Path:
    return home / "certification"


def save(certification: Certification, home: Path) -> Path:
    directory = certification_dir(home)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{certification.class_id}.json"
    path.write_text(
        json.dumps(certification.payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_policies(home: Path) -> dict[str, dict[str, Any]]:
    directory = certification_dir(home)
    if not directory.is_dir():
        return {}
    policies: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            policies[str(payload["class"])] = dict(payload["policy"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return policies
