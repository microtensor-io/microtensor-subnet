from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

NONCE_BYTES = 32
NONCE_TTL_SECONDS = 600
SIGNATURE_WINDOW_SECONDS = 120
RELEASE_SKEW_SECONDS = 600

HEADER_AGENT = "x-mt-agent"
HEADER_HOTKEY = "x-mt-hotkey"
HEADER_TIMESTAMP = "x-mt-timestamp"
HEADER_SIGNATURE = "x-mt-signature"

NONCE_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PUBLIC_KEY_PATTERN = re.compile(
    r"^(ssh-ed25519|ecdsa-sha2-nistp256|ecdsa-sha2-nistp384|ssh-rsa) [A-Za-z0-9+/]+=*( [^\r\n]*)?$"
)
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

SESSION_OPEN_PATH = "/session/open"
SESSION_CLOSE_PATH = "/session/close"
PING_PATH = "/ping"
UTILISATION_PATH = "/utilisation"
VERSION_PATH = "/version"
JOBS_PATH = "/jobs"
SOCKET_PATH = "/v1/compute/agents/socket"


def signing_bytes(method: str, path: str, timestamp: str, body: bytes = b"") -> bytes:
    return b"\n".join([method.upper().encode(), path.encode(), timestamp.encode(), body])


def grant_blob(public_key: str, nonce: str) -> bytes:
    return f"{public_key}\n{nonce}".encode()


def release_message(digest: str, timestamp: int) -> bytes:
    return f"{digest}\n{int(timestamp)}".encode()


def well_formed_nonce(nonce: str) -> bool:
    return bool(NONCE_PATTERN.match(nonce or ""))


def well_formed_public_key(public_key: str) -> bool:
    text = (public_key or "").strip()
    return "\n" not in text and bool(PUBLIC_KEY_PATTERN.match(text))


def well_formed_digest(digest: str) -> bool:
    return bool(DIGEST_PATTERN.match(digest or ""))


@dataclass(frozen=True)
class SessionOpen:
    public_key: str
    nonce: str
    signature: str

    def payload(self) -> dict[str, str]:
        return {"public_key": self.public_key, "nonce": self.nonce, "signature": self.signature}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SessionOpen:
        return cls(
            public_key=str(payload.get("public_key", "")).strip(),
            nonce=str(payload.get("nonce", "")).strip().lower(),
            signature=str(payload.get("signature", "")).strip(),
        )

    def problems(self) -> list[str]:
        found: list[str] = []
        if not well_formed_public_key(self.public_key):
            found.append("public_key must be a single OpenSSH public key line")
        if not well_formed_nonce(self.nonce):
            found.append("nonce must be 32 bytes as 64 lowercase hex characters")
        if not self.signature:
            found.append("signature is required")
        return found

    @property
    def blob(self) -> bytes:
        return grant_blob(self.public_key, self.nonce)


@dataclass(frozen=True)
class SessionGrant:
    ssh_username: str
    ssh_port: int
    ssh_host_key: str
    python_path: str
    root_dir: str
    port_range: str
    validator: str
    expires_at: str
    gpu_attestation: dict[str, Any] = field(default_factory=dict)
    tdx_quote: str = ""
    session_id: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "ssh_username": self.ssh_username,
            "ssh_port": self.ssh_port,
            "ssh_host_key": self.ssh_host_key,
            "python_path": self.python_path,
            "root_dir": self.root_dir,
            "port_range": self.port_range,
            "gpu_attestation": dict(self.gpu_attestation),
            "tdx_quote": self.tdx_quote,
            "validator": self.validator,
            "expires_at": self.expires_at,
            "session_id": self.session_id,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SessionGrant:
        attestation = payload.get("gpu_attestation")
        return cls(
            ssh_username=str(payload.get("ssh_username", "")),
            ssh_port=int(payload.get("ssh_port", 22) or 22),
            ssh_host_key=str(payload.get("ssh_host_key", "")),
            python_path=str(payload.get("python_path", "python3") or "python3"),
            root_dir=str(payload.get("root_dir", "/tmp") or "/tmp"),
            port_range=str(payload.get("port_range", "")),
            validator=str(payload.get("validator", "")),
            expires_at=str(payload.get("expires_at", "")),
            gpu_attestation=dict(attestation) if isinstance(attestation, dict) else {},
            tdx_quote=str(payload.get("tdx_quote", "") or ""),
            session_id=str(payload.get("session_id", "") or ""),
        )


@dataclass(frozen=True)
class SessionClose:
    public_key: str
    session_id: str = ""

    def payload(self) -> dict[str, str]:
        return {"public_key": self.public_key, "session_id": self.session_id}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SessionClose:
        return cls(
            public_key=str(payload.get("public_key", "")).strip(),
            session_id=str(payload.get("session_id", "") or ""),
        )


@dataclass(frozen=True)
class Release:
    digest: str
    timestamp: int
    signature: str
    validator: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "validator": self.validator,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Release:
        try:
            timestamp = int(payload.get("timestamp", 0) or 0)
        except (TypeError, ValueError):
            timestamp = 0
        return cls(
            digest=str(payload.get("digest", "")),
            timestamp=timestamp,
            signature=str(payload.get("signature", "")),
            validator=str(payload.get("validator", "") or ""),
        )

    @property
    def message(self) -> bytes:
        return release_message(self.digest, self.timestamp)


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    kind: str
    subnet: str = ""
    pid: int = 0
    gpu_uuid: str = ""
    container: str = ""
    started_at: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "subnet": self.subnet,
            "pid": self.pid,
            "gpu_uuid": self.gpu_uuid,
            "container": self.container,
            "started_at": self.started_at,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> JobRecord:
        try:
            pid = int(payload.get("pid", 0) or 0)
        except (TypeError, ValueError):
            pid = 0
        return cls(
            job_id=str(payload.get("job_id", "")),
            kind=str(payload.get("kind", "")),
            subnet=str(payload.get("subnet", "") or ""),
            pid=pid,
            gpu_uuid=str(payload.get("gpu_uuid", "") or ""),
            container=str(payload.get("container", "") or ""),
            started_at=str(payload.get("started_at", "") or ""),
        )
