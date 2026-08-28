from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from microtensor.core.constants import (
    DECLARATION_LATENCY_FLOOR_MS,
    DECLARATION_TOLERANCE_LATENCY,
    DECLARATION_TOLERANCE_RSS,
    DECLARATION_TOLERANCE_SIZE,
)
from microtensor.core.tracks import HardwareClass


class ArtifactFormat(str, Enum):
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    GGUF = "gguf"


class Role(str, Enum):
    FRONT = "front"
    ROUTER = "router"
    SPECIALIST = "specialist"


class Fault(str, Enum):
    ARTIFACT = "artifact"
    INFRASTRUCTURE = "infrastructure"


class GateFailure(str, Enum):
    SIZE_CEILING = "exceeds class size ceiling"
    RSS_CEILING = "exceeds class memory ceiling"
    LATENCY_CEILING = "exceeds class latency ceiling"
    RSS_OVER_DECLARED = "measured memory exceeds its own declaration"
    LATENCY_OVER_DECLARED = "measured latency exceeds its own declaration"
    SIZE_OVER_DECLARED = "measured size exceeds its own declaration"
    BUDGET_CEILING = "exhausted its cpu budget before finishing the round"


@dataclass(frozen=True, slots=True)
class DeclaredEnvelope:
    size_bytes: int
    peak_rss_bytes: int
    p95_latency_ms: int

    def __post_init__(self) -> None:
        for name in ("size_bytes", "peak_rss_bytes", "p95_latency_ms"):
            if getattr(self, name) <= 0:
                raise ValueError(f"declared {name} must be positive")


@dataclass(frozen=True, slots=True)
class MeasuredEnvelope:
    size_bytes: int
    peak_rss_bytes: int
    input_at_peak: dict[str, Any]
    ttft_p50_ms: int
    ttft_p95_ms: int
    tokens_per_second: float
    cold_start_ms: int
    device_profile: str
    conforming: bool = True


@dataclass(frozen=True, slots=True)
class LoadManifest:
    format: ArtifactFormat
    quantization: str
    entrypoint: str
    max_input: dict[str, Any]
    preprocessing: dict[str, Any] = field(default_factory=dict)
    base_model: str = ""

    def __post_init__(self) -> None:
        if not self.entrypoint:
            raise ValueError("manifest must name an entrypoint")
        if not self.max_input:
            raise ValueError("manifest must declare max_input")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["format"] = self.format.value
        return d


@dataclass(frozen=True, slots=True)
class Submission:
    hotkey: str
    track: str
    hardware_class: str
    artifact_digest: str
    manifest: LoadManifest
    declared: DeclaredEnvelope
    signature: str = ""

    def signing_payload(self) -> dict[str, Any]:
        return {
            "hotkey": self.hotkey,
            "track": self.track,
            "hardware_class": self.hardware_class,
            "artifact_digest": self.artifact_digest,
            "manifest": self.manifest.to_dict(),
            "declared": asdict(self.declared),
        }


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    task_ref: str
    score: float
    completed: bool
    partition: str
    latency_ms: float = 0.0
    error: str = ""
    fault: Fault | None = None

    @property
    def abstains(self) -> bool:
        return self.fault is Fault.INFRASTRUCTURE


@dataclass(frozen=True, slots=True)
class GateResult:
    admitted: bool
    failures: tuple[GateFailure, ...] = ()

    @property
    def reason(self) -> str:
        return "; ".join(f.value for f in self.failures)


@dataclass(frozen=True, slots=True)
class Evaluation:
    hotkey: str
    track: str
    hardware_class: str
    artifact_digest: str
    gate: GateResult
    measured: MeasuredEnvelope | None
    score_rotating: float
    score_fixed: float
    score_combined: float
    n_rotating: int
    n_fixed: int
    corpus_version: str
    resolve_rate: float = 1.0
    expected_ms: float = 0.0
    expected_j: float = 0.0
    front_only_score: float = 0.0
    system_digest: str = ""

    @property
    def earns(self) -> bool:
        return self.gate.admitted and self.score_combined > 0.0


def latency_allowance(declared_p95_ms: int) -> float:
    return max(
        declared_p95_ms * (1.0 + DECLARATION_TOLERANCE_LATENCY),
        declared_p95_ms + DECLARATION_LATENCY_FLOOR_MS,
    )


def evaluate_gate(
    measured: MeasuredEnvelope,
    declared: DeclaredEnvelope,
    hardware: HardwareClass,
) -> GateResult:
    failures: list[GateFailure] = []

    if measured.size_bytes > hardware.max_size_bytes:
        failures.append(GateFailure.SIZE_CEILING)
    if measured.peak_rss_bytes > hardware.max_rss_bytes:
        failures.append(GateFailure.RSS_CEILING)
    if measured.ttft_p95_ms > hardware.max_p95_ms:
        failures.append(GateFailure.LATENCY_CEILING)

    if measured.conforming:
        if measured.size_bytes > declared.size_bytes * (1.0 + DECLARATION_TOLERANCE_SIZE):
            failures.append(GateFailure.SIZE_OVER_DECLARED)
        if measured.peak_rss_bytes > declared.peak_rss_bytes * (1.0 + DECLARATION_TOLERANCE_RSS):
            failures.append(GateFailure.RSS_OVER_DECLARED)

    return GateResult(admitted=not failures, failures=tuple(failures))
