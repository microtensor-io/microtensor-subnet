from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

VERSION: Final[int] = 3
HELLO: Final[str] = "hello"
WELCOME: Final[str] = "welcome"
REQUEST: Final[str] = "request"
RESPONSE: Final[str] = "response"
HEARTBEAT: Final[str] = "heartbeat"
CANCEL: Final[str] = "cancel"


class WireError(RuntimeError):
    pass


def hello(
    hotkey: str, worker: str, models: Sequence[Mapping[str, Any]], concurrency: int
) -> dict[str, Any]:
    return {
        "type": HELLO,
        "version": VERSION,
        "hotkey": hotkey,
        "worker": worker,
        "concurrency": concurrency,
        "models": [dict(m) for m in models],
    }


def heartbeat(running: int, idle_seconds: float, models: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": HEARTBEAT,
        "running": running,
        "idle_seconds": round(idle_seconds, 3),
        "models": dict(models),
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
