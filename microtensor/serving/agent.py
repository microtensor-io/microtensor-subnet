"""The inference operator's agent.

It dials the gateway, and for every request routed to it: runs the generation
on a local engine the operator started themselves, and returns the completion
together with the token ids that produced it. Nothing else is asked of the
operator. There is no proof to build, no activations to capture, and no patch
to the engine, because verification happens entirely on the validator, which
re-derives the canonical logits for those tokens from the certified artifact.

Dialling out rather than listening is the operational decision that matters.
An operator behind NAT, on a home connection, or inside a network it does not
control can serve without an inbound port, a public address, or a certificate,
and the gateway keeps one hardened door rather than one per operator.

The agent declares one number, how many requests it can run at once. Every
other limit belongs to the model: the artifact's own maximum input, the class
ceilings, and the latency objectives are the network's, applied to every
operator serving that model, and a value sent here can only lower them.

Nothing in this module retains request content. Tokens pass through to the
reply and are not written down.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

log = logging.getLogger("microtensor.serving.agent")

HELLO: Final[str] = "hello"
REQUEST: Final[str] = "request"
RESPONSE: Final[str] = "response"
HEARTBEAT: Final[str] = "heartbeat"
CANCEL: Final[str] = "cancel"

PROTOCOL_VERSION: Final[int] = 1
MIN_CONCURRENCY: Final[int] = 1
HEARTBEAT_SECONDS: Final[float] = 15.0
RECONNECT_CEILING_SECONDS: Final[float] = 60.0


class AgentError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Settings:
    gateway: str
    hotkey: str
    model: str
    artifact_digest: str
    engine_url: str
    concurrency: int = 8
    worker: str = ""

    def __post_init__(self) -> None:
        if not self.gateway:
            raise AgentError("no gateway to dial; there is no default")
        if not self.hotkey:
            raise AgentError("an operator serves under its own hotkey")
        if not self.model or not self.artifact_digest:
            raise AgentError("an operator serves a named model pinned by its manifest digest")
        if not self.engine_url:
            raise AgentError("no local engine to drive")
        if self.concurrency < MIN_CONCURRENCY:
            raise AgentError(f"concurrency must be at least {MIN_CONCURRENCY}")


@dataclass(slots=True)
class Inflight:
    """What the agent is running, so a heartbeat reports capacity honestly."""

    limit: int
    running: int = 0
    served: int = 0
    failed: int = 0
    started: float = field(default_factory=time.monotonic)

    @property
    def free(self) -> int:
        return max(0, self.limit - self.running)

    @property
    def idle_seconds(self) -> float:
        return 0.0 if self.running else time.monotonic() - self.started

    def take(self) -> None:
        if not self.free:
            raise AgentError("the gateway routed past the declared concurrency")
        self.running += 1

    def release(self, *, ok: bool) -> None:
        self.running = max(0, self.running - 1)
        if ok:
            self.served += 1
        else:
            self.failed += 1
        if not self.running:
            self.started = time.monotonic()


def hello(settings: Settings) -> dict[str, Any]:
    """The opening frame. Connecting and being admitted is the registration."""
    return {
        "type": HELLO,
        "version": PROTOCOL_VERSION,
        "hotkey": settings.hotkey,
        "model": settings.model,
        "artifact_digest": settings.artifact_digest,
        "concurrency": settings.concurrency,
        "worker": settings.worker,
    }


def heartbeat(state: Inflight) -> dict[str, Any]:
    return {
        "type": HEARTBEAT,
        "running": state.running,
        "free": state.free,
        "served": state.served,
        "failed": state.failed,
        "idle_seconds": round(state.idle_seconds, 3),
    }


def response(
    correlation: str,
    *,
    text: str = "",
    prompt_tokens: list[int] | None = None,
    completion_tokens: list[int] | None = None,
    finish_reason: str = "stop",
    error: str = "",
) -> dict[str, Any]:
    """What the agent returns, and the whole of what verification needs.

    The token ids are the evidence. A validator holding the certified artifact
    re-derives what it would have produced for this prompt and compares, so an
    operator that served something cheaper is visible without having committed
    to anything beyond the tokens it already generated.
    """
    if error:
        return {"type": RESPONSE, "correlation": correlation, "error": error}
    return {
        "type": RESPONSE,
        "correlation": correlation,
        "text": text,
        "finish_reason": finish_reason,
        "tokens": {
            "prompt": list(prompt_tokens or []),
            "completion": list(completion_tokens or []),
        },
        "usage": {
            "prompt_tokens": len(prompt_tokens or []),
            "completion_tokens": len(completion_tokens or []),
        },
    }


def backoff(attempt: int, ceiling: float = RECONNECT_CEILING_SECONDS) -> float:
    """Reconnect delay, doubling and capped.

    A gateway restart drops every operator at once, so a fixed retry would
    bring them all back in the same instant and do it again. Doubling spreads
    the return; the ceiling stops an operator that has been down a while from
    waiting an hour to notice the gateway is back.
    """
    if attempt <= 0:
        return 0.0
    return min(float(ceiling), 2.0 ** min(attempt - 1, 16))


class Engine:
    """The local inference server, driven over HTTP.

    The operator starts this themselves and the agent never configures it.
    What the agent needs back is the generated text and the token ids on both
    sides, which every engine can report, so no engine is patched and the
    artifact served is the one the certificate names.
    """

    def __init__(self, url: str, timeout: float = 1800.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    async def generate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("bind an engine before serving")


def decode_frame(raw: str | bytes) -> dict[str, Any]:
    try:
        found = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise AgentError(f"the gateway sent a frame that is not json: {exc}") from exc
    if not isinstance(found, dict):
        raise AgentError("a frame must be an object")
    if not found.get("type"):
        raise AgentError("a frame must name its type")
    return found


async def serve_one(engine: Engine, frame: Mapping[str, Any], state: Inflight) -> dict[str, Any]:
    """One routed request, start to finish, never raising into the socket.

    A failure returns an error frame rather than dropping the connection: the
    gateway needs to route the next request somewhere, and a dropped leg looks
    like an operator that vanished rather than one request that went wrong.
    """
    correlation = str(frame.get("correlation", ""))
    state.take()
    try:
        found = await engine.generate(frame.get("request", {}))
        answer = response(
            correlation,
            text=str(found.get("text", "")),
            prompt_tokens=list(found.get("prompt_tokens", [])),
            completion_tokens=list(found.get("completion_tokens", [])),
            finish_reason=str(found.get("finish_reason", "stop")),
        )
        state.release(ok=True)
        return answer
    except asyncio.CancelledError:
        state.release(ok=False)
        raise
    except Exception as exc:
        log.warning("request %s failed: %s", correlation or "?", exc)
        state.release(ok=False)
        return response(correlation, error=f"{type(exc).__name__}: {exc}")
