from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from microtensor_subnet.core.hashing import canonical_json

VERSION: Final[int] = 6
CHALLENGE: Final[str] = "challenge"
HELLO: Final[str] = "hello"
WELCOME: Final[str] = "welcome"
READY: Final[str] = "ready"
ACCEPTED: Final[str] = "accepted"
REFUSED: Final[str] = "refused"
HEARTBEAT: Final[str] = "heartbeat"
ASSIGN: Final[str] = "assign"
REQUEST: Final[str] = "request"
CHUNK: Final[str] = "chunk"
DONE: Final[str] = "done"
ERROR: Final[str] = "error"
CANCEL: Final[str] = "cancel"
DRAIN: Final[str] = "drain"
DRAINED: Final[str] = "drained"
SYSTEM: Final[str] = "system"
MODEL: Final[str] = "model"
KINDS: Final[tuple[str, ...]] = (SYSTEM, MODEL)
CHAT: Final[str] = "chat"
PULLING: Final[str] = "pulling"
LOADING: Final[str] = "loading"
READY_PHASE: Final[str] = "ready"
DRAINING: Final[str] = "draining"
PHASES: Final[tuple[str, ...]] = (PULLING, LOADING, READY_PHASE, DRAINING)
HELLO_PREFIX: Final[bytes] = b"microtensor-operator-hello\n"

Signer = Callable[[bytes], str]


class WireError(RuntimeError):
    pass


def hello(
    hotkey: str,
    worker: str,
    *,
    nonce: str,
    gpus: Sequence[Mapping[str, Any]] = (),
    agent: Mapping[str, Any] | None = None,
    timestamp: int | None = None,
) -> dict[str, Any]:
    if not nonce:
        raise WireError("a hello answers the gateway's challenge; it needs the nonce")
    return {
        "type": HELLO,
        "version": VERSION,
        "hotkey": hotkey,
        "worker": worker,
        "nonce": nonce,
        "timestamp": int(time.time()) if timestamp is None else int(timestamp),
        "agent": dict(agent or {}),
        "gpus": [dict(g) for g in gpus],
    }


def hello_message(frame: Mapping[str, Any]) -> bytes:
    return HELLO_PREFIX + canonical_json({k: v for k, v in frame.items() if k != "signature"})


def signed(frame: Mapping[str, Any], signer: Signer) -> dict[str, Any]:
    body = {k: v for k, v in frame.items() if k != "signature"}
    return {**body, "signature": signer(hello_message(body))}


def ready(
    instance: str,
    *,
    model: str,
    kind: str,
    artifact_digest: str,
    image_digest: str = "",
    concurrency: int = 1,
    engine: str = "",
    context: int = 0,
    tool_calling: bool = False,
    vision: bool = False,
) -> dict[str, Any]:
    return {
        "type": READY,
        "instance": instance,
        "model": model,
        "kind": kind,
        "artifact_digest": artifact_digest,
        "image_digest": image_digest,
        "concurrency": int(concurrency),
        "engine": engine,
        "context": int(context),
        "tool_calling": bool(tool_calling),
        "vision": bool(vision),
    }


def instance_report(
    instance: str,
    *,
    phase: str,
    in_flight: int = 0,
    queue: int = 0,
    gpu_utilisation: float = 0.0,
    free_vram_mb: int = 0,
    memory_used_mb: int = 0,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise WireError(f"{phase!r} is not a phase; phases are {', '.join(PHASES)}")
    return {
        "instance": instance,
        "phase": phase,
        "in_flight": int(in_flight),
        "queue": int(queue),
        "gpu_utilisation": round(min(1.0, max(0.0, float(gpu_utilisation))), 4),
        "free_vram_mb": int(free_vram_mb),
        "memory_used_mb": int(memory_used_mb),
    }


def heartbeat(instances: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"type": HEARTBEAT, "instances": [dict(i) for i in instances]}


def done(
    request_id: str,
    *,
    instance: str,
    model: str,
    usage: Mapping[str, int] | None = None,
    ttft_ms: float = 0.0,
    total_ms: float = 0.0,
    **fields: Any,
) -> dict[str, Any]:
    counted = dict(usage or {})
    return {
        "type": DONE,
        "id": request_id,
        "instance": instance,
        "model": model,
        **fields,
        "usage": {
            "input_tokens": int(counted.get("input_tokens", counted.get("prompt_tokens", 0))),
            "output_tokens": int(counted.get("output_tokens", counted.get("completion_tokens", 0))),
        },
        "ttft_ms": round(float(ttft_ms), 1),
        "total_ms": round(float(total_ms), 1),
    }


def error(request_id: str, message: str) -> dict[str, Any]:
    return {"type": ERROR, "id": request_id, "error": str(message)[:300]}


def chunk(request_id: str, *, seq: int, delta: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": CHUNK, "id": request_id, "seq": int(seq), "delta": dict(delta)}


def drain(instance: str = "") -> dict[str, Any]:
    return {"type": DRAIN, "instance": instance}


def drained(instance: str = "") -> dict[str, Any]:
    return {"type": DRAINED, "instance": instance}


def decode(raw: str | bytes) -> dict[str, Any]:
    try:
        found = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise WireError(f"the gateway sent a frame that is not json: {exc}") from exc
    if not isinstance(found, dict):
        raise WireError("a frame must be an object")
    if not found.get("type"):
        raise WireError("a frame must name its type")
    return found


def encode(frame: Mapping[str, Any]) -> str:
    return json.dumps(frame, separators=(",", ":"))
