from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

HOST_KEY_PREFIX = b"SSH_HOST_KEY:"
REPORT_DATA_BYTES = 64


def report_data(host_key: str, nonce_hex: str) -> bytes:
    identity = hashlib.sha256(HOST_KEY_PREFIX + host_key.strip().encode()).digest()
    try:
        freshness = bytes.fromhex(nonce_hex)
    except ValueError:
        freshness = b""
    if len(freshness) != 32:
        freshness = b"\x00" * 32
    return identity + freshness


def report_data_matches(data: bytes, host_key: str, nonce_hex: str) -> bool:
    expected = report_data(host_key, nonce_hex)
    return len(data) >= REPORT_DATA_BYTES and data[:REPORT_DATA_BYTES] == expected


@dataclass(frozen=True)
class GpuEvidence:
    uuid: str
    model: str
    memory_mb: int
    index: int = 0
    serial: str = ""
    pci_bus_id: str = ""
    mig_mode: str = ""
    virtualization: str = ""
    driver: str = ""
    cuda: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "model": self.model,
            "memory_mb": self.memory_mb,
            "index": self.index,
            "serial": self.serial,
            "pci_bus_id": self.pci_bus_id,
            "mig_mode": self.mig_mode,
            "virtualization": self.virtualization,
            "driver": self.driver,
            "cuda": self.cuda,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> GpuEvidence:
        def number(key: str) -> int:
            try:
                return int(payload.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0

        return cls(
            uuid=str(payload.get("uuid", "")),
            model=str(payload.get("model", "")),
            memory_mb=number("memory_mb"),
            index=number("index"),
            serial=str(payload.get("serial", "") or ""),
            pci_bus_id=str(payload.get("pci_bus_id", "") or ""),
            mig_mode=str(payload.get("mig_mode", "") or ""),
            virtualization=str(payload.get("virtualization", "") or ""),
            driver=str(payload.get("driver", "") or ""),
            cuda=str(payload.get("cuda", "") or ""),
        )


@dataclass(frozen=True)
class Attestation:
    nonce: str
    agent_version: str
    hostname: str
    gpus: list[GpuEvidence] = field(default_factory=list)
    driver: str = ""
    cuda: str = ""
    machine_id: str = ""
    boot_id: str = ""
    host_key: str = ""
    report_data_hex: str = ""
    tdx_quote: str = ""
    gpu_evidence: dict[str, Any] = field(default_factory=dict)
    generated_at: float = 0.0

    def payload(self) -> dict[str, Any]:
        return {
            "nonce": self.nonce,
            "agent_version": self.agent_version,
            "hostname": self.hostname,
            "gpus": [gpu.payload() for gpu in self.gpus],
            "driver": self.driver,
            "cuda": self.cuda,
            "machine_id": self.machine_id,
            "boot_id": self.boot_id,
            "host_key": self.host_key,
            "report_data": self.report_data_hex,
            "tdx_quote": self.tdx_quote,
            "gpu_evidence": dict(self.gpu_evidence),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Attestation:
        raw_gpus = payload.get("gpus")
        gpus = (
            [GpuEvidence.from_payload(g) for g in raw_gpus if isinstance(g, dict)]
            if isinstance(raw_gpus, list)
            else []
        )
        evidence = payload.get("gpu_evidence")
        try:
            generated_at = float(payload.get("generated_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            generated_at = 0.0
        return cls(
            nonce=str(payload.get("nonce", "")),
            agent_version=str(payload.get("agent_version", "")),
            hostname=str(payload.get("hostname", "")),
            gpus=gpus,
            driver=str(payload.get("driver", "") or ""),
            cuda=str(payload.get("cuda", "") or ""),
            machine_id=str(payload.get("machine_id", "") or ""),
            boot_id=str(payload.get("boot_id", "") or ""),
            host_key=str(payload.get("host_key", "") or ""),
            report_data_hex=str(payload.get("report_data", "") or ""),
            tdx_quote=str(payload.get("tdx_quote", "") or ""),
            gpu_evidence=dict(evidence) if isinstance(evidence, dict) else {},
            generated_at=generated_at,
        )

    def bound_to(self, nonce: str) -> bool:
        return bool(nonce) and self.nonce == nonce
