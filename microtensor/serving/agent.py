from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Final

log = logging.getLogger("microtensor.serving.agent")

HELLO: Final[str] = "hello"
REQUEST: Final[str] = "request"
RESPONSE: Final[str] = "response"
HEARTBEAT: Final[str] = "heartbeat"
CANCEL: Final[str] = "cancel"
CHUNK: Final[str] = "chunk"

PROTOCOL_VERSION: Final[int] = 1
MIN_CONCURRENCY: Final[int] = 1
HEARTBEAT_SECONDS: Final[float] = 15.0
RECONNECT_CEILING_SECONDS: Final[float] = 60.0
DEFAULT_MAX_TOKENS: Final[int] = 512
HEALTH_TIMEOUT: Final[float] = 5.0


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


def chunk(correlation: str, delta: str) -> dict[str, Any]:
    return {"type": CHUNK, "correlation": correlation, "delta": delta}


def backoff(attempt: int, ceiling: float = RECONNECT_CEILING_SECONDS) -> float:
    if attempt <= 0:
        return 0.0
    return min(float(ceiling), 2.0 ** min(attempt - 1, 16))


class Engine:
    def __init__(self, url: str, timeout: float = 1800.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    async def generate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("bind an engine before serving")

    async def stream(
        self, request: Mapping[str, Any], on_delta: Callable[[str], Awaitable[None]]
    ) -> dict[str, Any]:
        return await self.generate(request)


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
    state.take()
    return await serve_taken(engine, frame, state)


async def serve_taken(
    engine: Engine,
    frame: Mapping[str, Any],
    state: Inflight,
    on_delta: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    correlation = str(frame.get("correlation", ""))
    try:
        request = frame.get("request", {})
        if on_delta is None:
            found = await engine.generate(request)
        else:
            found = await engine.stream(request, on_delta)
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


def _httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise AgentError("httpx is required to drive a local engine") from exc
    return httpx


def _websockets() -> Any:
    try:
        import websockets
    except ImportError as exc:
        raise AgentError("websockets is required to dial the gateway") from exc
    return websockets


class HttpEngine(Engine):
    @staticmethod
    def _body(request: Mapping[str, Any]) -> dict[str, Any]:
        prompt = str(request.get("prompt", ""))
        if not prompt:
            raise AgentError("the gateway sent a request with no prompt")
        body: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": int(request.get("max_tokens", DEFAULT_MAX_TOKENS)),
            "temperature": float(request.get("temperature", 0.0)),
            "return_tokens": True,
        }
        for name in ("top_p", "top_k", "seed", "repeat_penalty"):
            if request.get(name) is not None:
                body[name] = request[name]
        stop = request.get("stop")
        if stop:
            body["stop"] = list(stop)
        return body

    @staticmethod
    def _answer(text: str, prompt_tokens: list[int], raw: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": [int(t) for t in raw.get("tokens", [])],
            "finish_reason": "length" if raw.get("stopped_limit") else "stop",
        }

    async def _prompt_tokens(self, client: Any, prompt: str) -> list[int]:
        counted = await client.post("/tokenize", json={"content": prompt})
        counted.raise_for_status()
        return [int(t) for t in counted.json().get("tokens", [])]

    async def generate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        httpx = _httpx()
        body = dict(self._body(request), stream=False)

        async with httpx.AsyncClient(base_url=self.url, timeout=self.timeout) as client:
            prompt_tokens = await self._prompt_tokens(client, body["prompt"])
            found = await client.post("/completion", json=body)
            found.raise_for_status()
            answer = found.json()

        return self._answer(str(answer.get("content", "")), prompt_tokens, answer)

    async def stream(
        self, request: Mapping[str, Any], on_delta: Callable[[str], Awaitable[None]]
    ) -> dict[str, Any]:
        httpx = _httpx()
        body = dict(self._body(request), stream=True)
        text: list[str] = []
        answer: dict[str, Any] = {}

        async with httpx.AsyncClient(base_url=self.url, timeout=self.timeout) as client:
            prompt_tokens = await self._prompt_tokens(client, body["prompt"])
            async with client.stream("POST", "/completion", json=body) as found:
                found.raise_for_status()
                async for line in found.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        piece = json.loads(line[5:].strip())
                    except ValueError:
                        continue
                    delta = str(piece.get("content", ""))
                    if delta:
                        text.append(delta)
                        await on_delta(delta)
                    if piece.get("stop"):
                        answer = piece

        return self._answer("".join(text), prompt_tokens, answer)

    async def ready(self) -> bool:
        httpx = _httpx()
        try:
            async with httpx.AsyncClient(base_url=self.url, timeout=HEALTH_TIMEOUT) as client:
                found = await client.get("/health")
            return bool(found.status_code == 200)
        except Exception:
            return False


def _drop(running: dict[str, asyncio.Task[None]], key: str, _: asyncio.Task[None]) -> None:
    running.pop(key, None)


async def _send(socket: Any, frame: Mapping[str, Any]) -> None:
    await socket.send(json.dumps(frame, separators=(",", ":")))


async def _beat(socket: Any, state: Inflight, every: float = HEARTBEAT_SECONDS) -> None:
    while True:
        await asyncio.sleep(every)
        await _send(socket, heartbeat(state))


async def _answer(socket: Any, engine: Engine, frame: Mapping[str, Any], state: Inflight) -> None:
    correlation = str(frame.get("correlation", ""))

    async def on_delta(delta: str) -> None:
        await _send(socket, chunk(correlation, delta))

    found = await serve_taken(engine, frame, state, on_delta)
    await _send(socket, found)


async def session(settings: Settings, engine: Engine) -> None:
    websockets = _websockets()
    state = Inflight(limit=settings.concurrency)
    running: dict[str, asyncio.Task[None]] = {}

    async with websockets.connect(settings.gateway, max_size=None) as socket:
        await _send(socket, hello(settings))
        log.info("dialled %s as %s for %s", settings.gateway, settings.hotkey[:12], settings.model)
        pulse = asyncio.create_task(_beat(socket, state))
        try:
            async for raw in socket:
                frame = decode_frame(raw)
                kind = str(frame.get("type", ""))
                correlation = str(frame.get("correlation", ""))

                if kind == REQUEST:
                    if not state.free:
                        await _send(socket, response(correlation, error="at declared concurrency"))
                        continue
                    state.take()
                    task = asyncio.create_task(_answer(socket, engine, frame, state))
                    running[correlation] = task
                    task.add_done_callback(partial(_drop, running, correlation))
                elif kind == CANCEL:
                    found = running.pop(correlation, None)
                    if found is not None:
                        found.cancel()
                elif kind == HEARTBEAT:
                    await _send(socket, heartbeat(state))
        finally:
            pulse.cancel()
            for task in list(running.values()):
                task.cancel()
            if running:
                await asyncio.gather(*running.values(), return_exceptions=True)


async def run(settings: Settings, engine: Engine, *, stop: asyncio.Event | None = None) -> None:
    attempt = 0
    while stop is None or not stop.is_set():
        try:
            await session(settings, engine)
            attempt = 0
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            attempt += 1
            delay = backoff(attempt)
            log.warning("dial failed (%s); retrying in %.0fs", exc, delay)
            if stop is None:
                await asyncio.sleep(delay)
                continue
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
                return
            except asyncio.TimeoutError:
                continue
