from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import re
import secrets
import shlex
from dataclasses import dataclass, field
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from microtensor.rigs.protocol.failures import Failure, FailureClass, Verdict
from microtensor.rigs.validator.session.client import Session
from microtensor.rigs.validator.session.ssh import SshError, classify
from microtensor.rigs.validator.verify.payloads import text

PAYLOAD = "scrape.py"
FIELD_LINE = re.compile(r'^(F_[A-Z_]+) = "([a-z_]+)"$', re.MULTILINE)
TOKEN_PREFIX = "gAAAAA"
SCAN_BYTES = 256 * 1024
MAX_CANDIDATES = 20
ERROR_LINE = re.compile(r'^\{"error":')
PRIVATE_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
)


@dataclass(frozen=True)
class Obfuscated:
    source: str
    names: dict[str, str]
    key: bytes


def obfuscate(original: str) -> Obfuscated:
    names: dict[str, str] = {}
    ordered: list[str] = []

    def replace(match: re.Match[str]) -> str:
        fresh = "_" + secrets.token_hex(8)
        names[fresh] = match.group(2)
        ordered.append(fresh)
        return f'{match.group(1)} = "{fresh}"'

    rewritten = FIELD_LINE.sub(replace, original)
    if not ordered:
        raise ValueError("the scrape payload declares no field names")
    key = base64.urlsafe_b64encode(hashlib.sha256("".join(ordered).encode("utf-8")).digest())
    return Obfuscated(rewritten, names, key)


def candidates(stdout: str) -> list[str]:
    window = stdout[-SCAN_BYTES:]
    found: list[str] = []
    for line in reversed(window.splitlines()):
        clean = line.strip()
        if clean.startswith(TOKEN_PREFIX):
            found.append(clean)
            if len(found) >= MAX_CANDIDATES:
                break
    return found


def error_line(stdout: str) -> str:
    for line in reversed(stdout[-SCAN_BYTES:].splitlines()):
        clean = line.strip()
        if ERROR_LINE.match(clean):
            try:
                parsed = json.loads(clean)
            except ValueError:
                continue
            if isinstance(parsed, dict) and "error" in parsed:
                return str(parsed.get("error", ""))
    return ""


def decrypt(key: bytes, tokens: list[str]) -> dict[str, Any] | None:
    cipher = Fernet(key)
    for token in tokens:
        try:
            raw = cipher.decrypt(token.encode("ascii"))
        except (InvalidToken, ValueError, TypeError):
            continue
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def restore(data: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    gpus = []
    for card in data.get("gpus") or []:
        if not isinstance(card, dict):
            continue
        gpus.append({names.get(key, key): value for key, value in card.items()})
    restored = dict(data)
    restored["gpus"] = gpus
    return restored


def is_public(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return ip.version == 4 and not any(ip in network for network in PRIVATE_NETWORKS)


def build_spec(data: dict[str, Any], address: str, port_range: str) -> dict[str, Any]:
    host = data.get("host") or {}
    nvml = data.get("nvml") or {}
    docker = data.get("docker") or {}
    probes = data.get("probes") or {}
    disk = host.get("disk") or {}
    volume = disk.get("docker_root") or disk.get("root") or {}
    gpus = []
    for card in data.get("gpus") or []:
        if not card.get("uuid"):
            continue
        gpus.append(
            {
                "model": str(card.get("name", "") or ""),
                "memory_mb": int(card.get("memory_total_mb") or 0),
                "uuid": str(card.get("uuid", "")),
            }
        )
    sysbox = bool((probes.get("sysbox") or {}).get("ok", False))
    idmapped = bool((probes.get("idmapped") or {}).get("idmapped", False))
    rotational = volume.get("rotational")
    return {
        "gpus": gpus,
        "cpu_cores": int((host.get("cpu") or {}).get("count") or 0),
        "ram_mb": int((host.get("memory") or {}).get("total_mb") or 0),
        "disk_gb": float(volume.get("total_gb") or 0.0),
        "disk_ssd": rotational is not True,
        "os": str((host.get("os") or {}).get("id", "") or ""),
        "os_version": str((host.get("os") or {}).get("version", "") or ""),
        "kernel": str(host.get("kernel", "") or ""),
        "arch": str(host.get("arch", "") or ""),
        "driver": str(nvml.get("driver", "") or ""),
        "cuda": str(nvml.get("cuda", "") or ""),
        "isolation": {"sysbox": sysbox, "idmapped": idmapped},
        "storage_quota": bool((probes.get("storage_quota") or {}).get("ok", False)),
        "hostname": str(host.get("hostname", "") or ""),
        "public_ipv4": is_public(address),
        "port_range": port_range,
        "docker_runtimes": list(docker.get("runtimes") or []),
    }


@dataclass
class ScrapeResult:
    verdict: Verdict
    data: dict[str, Any] | None = None
    spec: dict[str, Any] | None = None
    names: dict[str, str] = field(default_factory=dict)


async def run(session: Session, timeout: float, seed: str) -> ScrapeResult:
    prepared = obfuscate(text(PAYLOAD))
    command = f"{shlex.quote(session.python)} -I -"
    try:
        result = await session.run(command, timeout, stdin=prepared.source)
    except SshError as exc:
        return ScrapeResult(Verdict("scrape", failure=exc.failure(seed)))
    evidence: dict[str, Any] = {
        "command_ms": round(result.elapsed_ms, 1),
        "stdout_bytes": len(result.stdout),
    }
    if result.transport_failed:
        return ScrapeResult(
            Verdict(
                "scrape",
                failure=Failure(FailureClass.SSH_TRANSPORT, result.error, seed=seed),
                evidence=evidence,
            )
        )
    reported = error_line(result.stdout)
    if reported:
        return ScrapeResult(
            Verdict(
                "scrape",
                failure=Failure(
                    FailureClass.AGENT_CRASH, f"the scrape reported: {reported[:300]}", seed=seed
                ),
                evidence=evidence,
            )
        )
    tokens = candidates(result.stdout)
    data = decrypt(prepared.key, tokens) if tokens else None
    if data is None:
        crash = classify(result, seed)
        failure = crash or Failure(
            FailureClass.AGENT_CRASH,
            "the scrape produced no output the validator could decrypt"
            if not tokens
            else "no scrape token decrypted with the job key",
            seed=seed,
            evidence={"candidates": len(tokens), "stderr": result.stderr_tail(400)},
        )
        return ScrapeResult(
            Verdict("scrape", failure=failure, evidence=evidence), names=prepared.names
        )
    restored = restore(data, prepared.names)
    spec = build_spec(restored, session.address, session.grant.port_range)
    evidence.update(
        {
            "summary": f"{len(spec['gpus'])} cards, driver {spec['driver'] or 'unknown'}",
            "elapsed_ms": restored.get("elapsed_ms"),
            "candidates": len(tokens),
            "nvml_errors": list((restored.get("nvml") or {}).get("errors") or []),
        }
    )
    return ScrapeResult(
        Verdict("scrape", evidence=evidence, elapsed_ms=result.elapsed_ms),
        restored,
        spec,
        prepared.names,
    )
