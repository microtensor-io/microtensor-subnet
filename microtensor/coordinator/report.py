from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from microtensor.core.protocol import Fault

REPORT_VERSION = 2


def canonical(body: dict[str, Any]) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def body_hash(body: dict[str, Any]) -> str:
    return f"sha256:{sha256(canonical(body)).hexdigest()}"


@dataclass(frozen=True, slots=True)
class QualityBlock:
    rotating: float
    fixed: float
    combined: float
    novel: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rotating": self.rotating,
            "fixed": self.fixed,
            "novel": self.novel,
            "combined": self.combined,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> QualityBlock:
        return cls(
            rotating=float(raw.get("rotating", 0.0)),
            fixed=float(raw.get("fixed", 0.0)),
            novel=float(raw.get("novel", 0.0)),
            combined=float(raw.get("combined", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class CostBlock:
    """What one query cost, as measured rather than modelled.

    `expected_j` is None when nothing measured energy. Zero would claim a
    system drew no power, which is a different statement from having no
    instrument.
    """

    expected_ms: float
    expected_j: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"expected_ms": self.expected_ms, "expected_j": self.expected_j}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CostBlock:
        energy = raw.get("expected_j")
        return cls(
            expected_ms=float(raw.get("expected_ms", 0.0)),
            expected_j=None if energy is None else float(energy),
        )


@dataclass(frozen=True, slots=True)
class Report:
    """One worker's measurement of one system, signed by its hotkey."""

    round_index: int
    worker_hotkey: str
    system_digest: str
    quality: QualityBlock
    resolve_rate: float
    cost: CostBlock
    envelope: dict[str, dict[str, Any]] = field(default_factory=dict)
    components: dict[str, str] = field(default_factory=dict)
    ablation: dict[str, float] | None = None
    device_profile: str = ""
    conforming: bool = False
    engine_version: str = ""
    corpus_version: str = ""
    corpus_digest: str = ""
    environment_digest: str = ""
    fault: Fault | None = None
    fault_reason: str = ""
    signature: str = ""

    def body(self) -> dict[str, Any]:
        """The signed payload. The signature is never part of what it covers."""
        return {
            "version": REPORT_VERSION,
            "round_index": self.round_index,
            "worker_hotkey": self.worker_hotkey,
            "system_digest": self.system_digest,
            "quality": self.quality.to_dict(),
            "resolve_rate": self.resolve_rate,
            "cost": self.cost.to_dict(),
            "envelope": self.envelope,
            "components": self.components,
            "ablation": self.ablation,
            "device_profile": self.device_profile,
            "conforming": self.conforming,
            "engine_version": self.engine_version,
            "corpus_version": self.corpus_version,
            "corpus_digest": self.corpus_digest,
            "environment_digest": self.environment_digest,
            "fault": self.fault.value if self.fault else None,
            "fault_reason": self.fault_reason,
        }

    def digest(self) -> str:
        return body_hash(self.body())

    def signed_with(self, signature: str) -> Report:
        return Report(
            round_index=self.round_index,
            worker_hotkey=self.worker_hotkey,
            system_digest=self.system_digest,
            quality=self.quality,
            resolve_rate=self.resolve_rate,
            cost=self.cost,
            envelope=self.envelope,
            components=self.components,
            ablation=self.ablation,
            device_profile=self.device_profile,
            conforming=self.conforming,
            engine_version=self.engine_version,
            corpus_version=self.corpus_version,
            corpus_digest=self.corpus_digest,
            environment_digest=self.environment_digest,
            fault=self.fault,
            fault_reason=self.fault_reason,
            signature=signature,
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Report:
        fault = raw.get("fault")
        return cls(
            round_index=int(raw["round_index"]),
            worker_hotkey=str(raw["worker_hotkey"]),
            system_digest=str(raw["system_digest"]),
            quality=QualityBlock.from_dict(dict(raw.get("quality", {}))),
            resolve_rate=float(raw.get("resolve_rate", 1.0)),
            cost=CostBlock.from_dict(dict(raw.get("cost", {}))),
            envelope=dict(raw.get("envelope", {})),
            components={str(k): str(v) for k, v in dict(raw.get("components", {})).items()},
            ablation=raw.get("ablation"),
            device_profile=str(raw.get("device_profile", "")),
            conforming=bool(raw.get("conforming", False)),
            engine_version=str(raw.get("engine_version", "")),
            corpus_version=str(raw.get("corpus_version", "")),
            corpus_digest=str(raw.get("corpus_digest", "")),
            environment_digest=str(raw.get("environment_digest", "")),
            fault=Fault(fault) if fault else None,
            fault_reason=str(raw.get("fault_reason", "")),
            signature=str(raw.get("signature", "")),
        )
