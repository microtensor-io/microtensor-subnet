from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

PROTOCOL = 1
PING_SECONDS = 20
PONG_TIMEOUT_SECONDS = 40
STATE_SECONDS = 30
HELLO_TIMEOUT_SECONDS = 15
RECONNECT_MIN_SECONDS = 1.0
RECONNECT_MAX_SECONDS = 60.0

CLOSE_BAD_HELLO = 4400
CLOSE_BAD_SIGNATURE = 4401
CLOSE_UNKNOWN_RIG = 4404


class ServerMessage(str, Enum):
    WELCOME = "welcome"
    COMMAND = "command"
    PONG = "pong"
    ERROR = "error"


class AgentMessage(str, Enum):
    HELLO = "hello"
    PING = "ping"
    STATE = "state"
    RESULT = "result"
    EVENT = "event"


class CommandKind(str, Enum):
    SPEC = "SpecRequest"
    UTILISATION = "UtilisationRequest"
    CHALLENGE = "ChallengeRequest"
    ATTESTATION = "AttestationRequest"
    JOB_PLACED = "JobPlaced"
    LOG_STREAM = "LogStreamRequest"
    PREFETCH = "PrefetchRequest"
    DRAIN = "DrainRequest"
    CLAIM_PROMPT = "ClaimPrompt"


def stamp() -> float:
    return round(time.time(), 3)


@dataclass(frozen=True)
class Hello:
    agent: str
    timestamp: str
    signature: str

    def payload(self) -> dict[str, Any]:
        return {
            "type": AgentMessage.HELLO.value,
            "agent": self.agent,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "protocol": PROTOCOL,
        }


@dataclass(frozen=True)
class Welcome:
    rig_id: str
    ping_seconds: int = PING_SECONDS
    protocol: int = PROTOCOL

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Welcome:
        try:
            ping = int(payload.get("ping_seconds", PING_SECONDS) or PING_SECONDS)
        except (TypeError, ValueError):
            ping = PING_SECONDS
        try:
            protocol = int(payload.get("protocol", PROTOCOL) or PROTOCOL)
        except (TypeError, ValueError):
            protocol = PROTOCOL
        return cls(rig_id=str(payload.get("rig_id", "")), ping_seconds=ping, protocol=protocol)


@dataclass(frozen=True)
class Command:
    id: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    expires_at: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Command:
        body = payload.get("payload")
        try:
            identifier = int(payload.get("id", 0) or 0)
        except (TypeError, ValueError):
            identifier = 0
        return cls(
            id=identifier,
            kind=str(payload.get("kind", "")),
            payload=dict(body) if isinstance(body, dict) else {},
            expires_at=str(payload.get("expires_at", "") or ""),
        )

    @property
    def known(self) -> bool:
        return self.kind in {kind.value for kind in CommandKind}


@dataclass(frozen=True)
class Result:
    id: int
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "type": AgentMessage.RESULT.value,
            "id": self.id,
            "ok": self.ok,
            "result": dict(self.result),
            "sent_at": stamp(),
        }


@dataclass(frozen=True)
class State:
    utilisation: dict[str, Any]
    containers: list[dict[str, Any]]
    queue_depth: int
    agent_version: str
    draining: bool
    jobs: list[str]

    def payload(self) -> dict[str, Any]:
        return {
            "type": AgentMessage.STATE.value,
            "utilisation": dict(self.utilisation),
            "containers": list(self.containers),
            "queue_depth": self.queue_depth,
            "agent_version": self.agent_version,
            "draining": self.draining,
            "jobs": list(self.jobs),
            "sent_at": stamp(),
        }


@dataclass(frozen=True)
class Event:
    kind: str
    detail: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "type": AgentMessage.EVENT.value,
            "kind": self.kind[:40],
            "detail": dict(self.detail),
            "sent_at": stamp(),
        }


def ping() -> dict[str, Any]:
    return {"type": AgentMessage.PING.value, "sent_at": stamp()}


@dataclass(frozen=True)
class ChallengeRequest:
    seed: int
    cipher: int
    n: int = 256
    rounds: int = 2
    benchmark_n: int = 2048
    iterations: int = 32

    def payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "cipher": self.cipher,
            "n": self.n,
            "rounds": self.rounds,
            "benchmark_n": self.benchmark_n,
            "iterations": self.iterations,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ChallengeRequest:
        def number(key: str, default: int) -> int:
            try:
                return int(payload.get(key, default) or default)
            except (TypeError, ValueError):
                return default

        return cls(
            seed=number("seed", 0),
            cipher=number("cipher", 0),
            n=number("n", 256),
            rounds=number("rounds", 2),
            benchmark_n=number("benchmark_n", 2048),
            iterations=number("iterations", 32),
        )


@dataclass(frozen=True)
class ChallengeAnswer:
    digest: str
    elapsed_ms: float
    gops: float = 0.0
    gbps: float = 0.0
    benchmark_ms: float = 0.0
    build: str = ""
    version: int = 0
    error: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "gops": round(self.gops, 3),
            "gbps": round(self.gbps, 3),
            "benchmark_ms": round(self.benchmark_ms, 3),
            "build": self.build,
            "version": self.version,
            "error": self.error,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ChallengeAnswer:
        def real(key: str) -> float:
            try:
                return float(payload.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        try:
            version = int(payload.get("version", 0) or 0)
        except (TypeError, ValueError):
            version = 0
        return cls(
            digest=str(payload.get("digest", "") or ""),
            elapsed_ms=real("elapsed_ms"),
            gops=real("gops"),
            gbps=real("gbps"),
            benchmark_ms=real("benchmark_ms"),
            build=str(payload.get("build", "") or ""),
            version=version,
            error=str(payload.get("error", "") or ""),
        )


@dataclass(frozen=True)
class LogStreamRequest:
    container: str
    tail: int = 200
    seconds: int = 30

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LogStreamRequest:
        def number(key: str, default: int) -> int:
            try:
                return int(payload.get(key, default) or default)
            except (TypeError, ValueError):
                return default

        return cls(
            container=str(payload.get("container", "") or payload.get("name", "") or ""),
            tail=max(1, min(number("tail", 200), 500)),
            seconds=max(1, min(number("seconds", 30), 300)),
        )


@dataclass(frozen=True)
class PrefetchRequest:
    image: str
    digest: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PrefetchRequest:
        return cls(
            image=str(payload.get("image", "") or ""), digest=str(payload.get("digest", "") or "")
        )

    @property
    def reference(self) -> str:
        if self.digest and "@" not in self.image:
            return f"{self.image.split(':', 1)[0]}@{self.digest}"
        return self.image


@dataclass(frozen=True)
class DrainRequest:
    drain: bool = True
    reason: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DrainRequest:
        return cls(
            drain=bool(payload.get("drain", True)), reason=str(payload.get("reason", "") or "")
        )


@dataclass(frozen=True)
class ClaimPrompt:
    hotkey: str
    expires_at: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ClaimPrompt:
        return cls(
            hotkey=str(payload.get("hotkey", "") or ""),
            expires_at=str(payload.get("expires_at", "") or ""),
        )


@dataclass(frozen=True)
class AttestationRequest:
    nonce: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> AttestationRequest:
        return cls(nonce=str(payload.get("nonce", "") or "").strip().lower())
