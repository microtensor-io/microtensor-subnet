from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from microtensor_subnet.core.hashing import canonical_json

VERSION: Final[int] = 5
SYSTEMS_ONLY_VERSION: Final[int] = 3
HELLO: Final[str] = "hello"
WELCOME: Final[str] = "welcome"
REQUEST: Final[str] = "request"
RESPONSE: Final[str] = "response"
HEARTBEAT: Final[str] = "heartbeat"
CANCEL: Final[str] = "cancel"
CHUNK: Final[str] = "chunk"
SYSTEM: Final[str] = "system"
MODEL: Final[str] = "model"
KINDS: Final[tuple[str, ...]] = (SYSTEM, MODEL)
CHAT: Final[str] = "chat"
HELLO_PREFIX: Final[bytes] = b"microtensor-operator-hello\n"

Signer = Callable[[bytes], str]


class WireError(RuntimeError):
    pass


def hello(
    hotkey: str,
    worker: str,
    models: Sequence[Mapping[str, Any]],
    concurrency: int,
    *,
    gpus: Sequence[Mapping[str, Any]] = (),
    agent: Mapping[str, Any] | None = None,
    timestamp: int | None = None,
    nonce: str = "",
) -> dict[str, Any]:
    return {
        "type": HELLO,
        "version": VERSION,
        "hotkey": hotkey,
        "worker": worker,
        "timestamp": int(time.time()) if timestamp is None else int(timestamp),
        "nonce": nonce or secrets.token_hex(8),
        "agent": dict(agent or {}),
        "gpus": [dict(g) for g in gpus],
        "concurrency": concurrency,
        "models": [dict(m) for m in models],
    }


def hello_message(frame: Mapping[str, Any]) -> bytes:
    return HELLO_PREFIX + canonical_json({k: v for k, v in frame.items() if k != "signature"})


def signed(frame: Mapping[str, Any], signer: Signer) -> dict[str, Any]:
    body = {k: v for k, v in frame.items() if k != "signature"}
    return {**body, "signature": signer(hello_message(body))}


def heartbeat(
    running: int,
    idle_seconds: float,
    models: Mapping[str, Any],
    *,
    gpus: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "type": HEARTBEAT,
        "running": running,
        "idle_seconds": round(idle_seconds, 3),
        "models": dict(models),
        "gpus": [dict(g) for g in gpus],
    }


def response(
    correlation: str,
    *,
    model: str,
    answer: Any = None,
    p: float = 0.0,
    floor_met: bool = False,
    verified: bool = False,
    trace: str = "",
    usage: Mapping[str, int] | None = None,
    latency_ms: float = 0.0,
    finish_reason: str = "stop",
    error: str = "",
) -> dict[str, Any]:
    if error:
        return {"type": RESPONSE, "correlation": correlation, "model": model, "error": error}
    counted = dict(usage or {})
    return {
        "type": RESPONSE,
        "correlation": correlation,
        "model": model,
        "answer": answer,
        "p": round(float(p), 6),
        "floor_met": bool(floor_met),
        "verified": bool(verified),
        "trace": trace[:4000],
        "usage": {
            "prompt_tokens": int(counted.get("prompt_tokens", 0)),
            "completion_tokens": int(counted.get("completion_tokens", 0)),
        },
        "latency_ms": round(float(latency_ms), 1),
        "finish_reason": finish_reason,
    }


def chunk(correlation: str, *, model: str, seq: int, delta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": CHUNK,
        "correlation": correlation,
        "model": model,
        "seq": int(seq),
        "delta": dict(delta),
    }


def chat_response(
    correlation: str,
    *,
    model: str,
    content: str = "",
    reasoning: str = "",
    tool_calls: Sequence[Mapping[str, Any]] = (),
    finish_reason: str = "stop",
    usage: Mapping[str, int] | None = None,
    tokens: Mapping[str, Sequence[int]] | None = None,
    artifact_digest: str = "",
    latency_ms: float = 0.0,
    ttft_ms: float = 0.0,
) -> dict[str, Any]:
    counted = dict(usage or {})
    ids = dict(tokens or {})
    return {
        "type": RESPONSE,
        "correlation": correlation,
        "model": model,
        "kind": CHAT,
        "content": content,
        "reasoning": reasoning,
        "tool_calls": [dict(call) for call in tool_calls],
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": int(counted.get("prompt_tokens", 0)),
            "completion_tokens": int(counted.get("completion_tokens", 0)),
        },
        "tokens": {
            "prompt": [int(t) for t in ids.get("prompt") or ()],
            "completion": [int(t) for t in ids.get("completion") or ()],
        },
        "artifact_digest": artifact_digest,
        "latency_ms": round(float(latency_ms), 1),
        "ttft_ms": round(float(ttft_ms), 1),
    }


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
