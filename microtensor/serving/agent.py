from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
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

PROTOCOL_VERSION: Final[int] = 2
MIN_CONCURRENCY: Final[int] = 1
HEARTBEAT_SECONDS: Final[float] = 15.0
RECONNECT_CEILING_SECONDS: Final[float] = 60.0
DEFAULT_MAX_TOKENS: Final[int] = 512
HEALTH_TIMEOUT: Final[float] = 5.0


class AgentError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Served:
    model: str
    artifact_digest: str
    engines: tuple[str, ...]
    concurrency: int = 4
    specialist: tuple[str, ...] = ()
    router: Any = None

    def __post_init__(self) -> None:
        if not self.model or not self.artifact_digest:
            raise AgentError("a served model is a name pinned by its manifest digest")
        if not self.engines:
            raise AgentError(f"no local engine to drive for {self.model}")
        if self.concurrency < MIN_CONCURRENCY:
            raise AgentError(f"concurrency for {self.model} must be at least {MIN_CONCURRENCY}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "artifact_digest": self.artifact_digest,
            "concurrency": self.concurrency,
        }


def _urls(engines: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(engines, str):
        return tuple(u.strip() for u in engines.split(",") if u.strip())
    return tuple(str(u).strip() for u in engines if str(u).strip())


def served(
    model: str,
    artifact_digest: str,
    engines: str | Iterable[str],
    concurrency: int = 4,
    specialist: str | Iterable[str] = (),
    router: Any = None,
) -> Served:
    return Served(
        model=model,
        artifact_digest=artifact_digest,
        engines=_urls(engines),
        concurrency=concurrency,
        specialist=_urls(specialist),
        router=router,
    )


@dataclass(frozen=True, slots=True)
class Settings:
    gateway: str
    hotkey: str
    serves: tuple[Served, ...]
    worker: str = ""

    def __post_init__(self) -> None:
        if not self.gateway:
            raise AgentError("no gateway to dial; there is no default")
        if not self.hotkey:
            raise AgentError("an operator serves under its own hotkey")
        if not self.serves:
            raise AgentError("an operator declares at least one model to serve")
        names = [s.model for s in self.serves]
        if len(set(names)) != len(names):
            raise AgentError("a model is declared twice; one entry names one model")

    @property
    def concurrency(self) -> int:
        return sum(s.concurrency for s in self.serves)

    def of(self, model: str) -> Served | None:
        return next((s for s in self.serves if s.model == model), None)


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


@dataclass(slots=True)
class Capacity:
    limits: dict[str, Inflight]

    @classmethod
    def of(cls, serves: Sequence[Served]) -> Capacity:
        return cls({s.model: Inflight(limit=s.concurrency) for s in serves})

    def free(self, model: str) -> int:
        held = self.limits.get(model)
        return held.free if held is not None else 0

    def take(self, model: str) -> None:
        held = self.limits.get(model)
        if held is None:
            raise AgentError(f"the gateway routed {model}, which this operator does not serve")
        held.take()

    def release(self, model: str, *, ok: bool) -> None:
        held = self.limits.get(model)
        if held is not None:
            held.release(ok=ok)

    @property
    def running(self) -> int:
        return sum(held.running for held in self.limits.values())

    @property
    def idle_seconds(self) -> float:
        if self.running:
            return 0.0
        return min((held.idle_seconds for held in self.limits.values()), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            model: {
                "running": held.running,
                "free": held.free,
                "served": held.served,
                "failed": held.failed,
            }
            for model, held in sorted(self.limits.items())
        }


def hello(settings: Settings) -> dict[str, Any]:
    return {
        "type": HELLO,
        "version": PROTOCOL_VERSION,
        "hotkey": settings.hotkey,
        "worker": settings.worker,
        "concurrency": settings.concurrency,
        "models": [s.to_dict() for s in settings.serves],
    }


def heartbeat(state: Capacity) -> dict[str, Any]:
    return {
        "type": HEARTBEAT,
        "running": state.running,
        "idle_seconds": round(state.idle_seconds, 3),
        "models": state.to_dict(),
    }


def response(
    correlation: str,
    *,
    model: str = "",
    text: str = "",
    prompt_tokens: list[int] | None = None,
    completion_tokens: list[int] | None = None,
    finish_reason: str = "stop",
    error: str = "",
) -> dict[str, Any]:
    if error:
        return {"type": RESPONSE, "correlation": correlation, "model": model, "error": error}
    return {
        "type": RESPONSE,
        "correlation": correlation,
        "model": model,
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

    async def ready(self) -> bool:
        return True


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


class Pool:
    def __init__(self, serves: Sequence[Served], build: Callable[[str], Engine] = HttpEngine):
        self._engines: dict[str, list[Engine]] = {
            entry.model: [build(url) for url in entry.engines] for entry in serves
        }
        self._specialists: dict[str, list[Engine]] = {
            entry.model: [build(url) for url in entry.specialist]
            for entry in serves
            if entry.specialist
        }
        self._routers: dict[str, Any] = {
            entry.model: entry.router for entry in serves if entry.router is not None
        }
        self._running: dict[int, int] = {}

    def router(self, model: str) -> Any:
        return self._routers.get(model)

    def has_specialist(self, model: str) -> bool:
        return bool(self._specialists.get(model))

    def take_specialist(self, model: str) -> Engine:
        held = self._specialists.get(model)
        if not held:
            raise AgentError(f"no specialist loaded for {model}")
        chosen = min(held, key=lambda e: self._running.get(id(e), 0))
        self._running[id(chosen)] = self._running.get(id(chosen), 0) + 1
        return chosen

    def models(self) -> list[str]:
        return sorted(self._engines)

    def engines(self, model: str) -> list[Engine]:
        return list(self._engines.get(model, []))

    def take(self, model: str) -> Engine:
        held = self._engines.get(model)
        if not held:
            raise AgentError(f"no engine loaded for {model}")
        chosen = min(held, key=lambda e: self._running.get(id(e), 0))
        self._running[id(chosen)] = self._running.get(id(chosen), 0) + 1
        return chosen

    def release(self, engine: Engine) -> None:
        key = id(engine)
        self._running[key] = max(0, self._running.get(key, 0) - 1)

    async def unready(self) -> list[str]:
        missing: list[str] = []
        for model, held in sorted(self._engines.items()):
            for engine in held:
                if not await engine.ready():
                    missing.append(f"{model} at {engine.url}")
        return missing


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


async def serve_one(pool: Pool, frame: Mapping[str, Any], state: Capacity) -> dict[str, Any]:
    model = str(frame.get("model", ""))
    state.take(model)
    return await serve_taken(pool, frame, state)


async def run_system(
    pool: Pool,
    model: str,
    request: Mapping[str, Any],
    on_delta: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[dict[str, Any], Engine | None, Engine | None]:
    from microtensor.serving import cascade

    front = pool.take(model)
    specialist: Engine | None = None
    asked = dict(request)

    if pool.router(model) is not None and pool.has_specialist(model):
        asked.setdefault("logprobs", True)
        first = await front.generate(asked)
        answered = cascade.Answer(
            legs=[
                cascade.Leg(
                    role=cascade.FRONT,
                    text=str(first.get("text", "")),
                    prompt_tokens=list(first.get("prompt_tokens", [])),
                    completion_tokens=list(first.get("completion_tokens", [])),
                    finish_reason=str(first.get("finish_reason", "stop")),
                )
            ]
        )
        answered.router_features = cascade.features_of(first, len(first.get("prompt_tokens") or []))
        if pool.router(model).choose(answered.router_features) == cascade.ESCALATE:
            specialist = pool.take_specialist(model)
            if on_delta is None:
                second = await specialist.generate(request)
            else:
                second = await specialist.stream(request, on_delta)
            answered.escalated = True
            answered.legs.append(
                cascade.Leg(
                    role=cascade.SPECIALIST,
                    text=str(second.get("text", "")),
                    prompt_tokens=list(second.get("prompt_tokens", [])),
                    completion_tokens=list(second.get("completion_tokens", [])),
                    finish_reason=str(second.get("finish_reason", "stop")),
                )
            )
        elif on_delta is not None and answered.served.text:
            await on_delta(answered.served.text)
        return answered.to_dict(), front, specialist

    if on_delta is None:
        found = await front.generate(request)
    else:
        found = await front.stream(request, on_delta)
    return dict(found) | {"escalated": False, "answered_by": "front"}, front, None


async def serve_taken(
    pool: Pool,
    frame: Mapping[str, Any],
    state: Capacity,
    on_delta: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    correlation = str(frame.get("correlation", ""))
    model = str(frame.get("model", ""))
    engine: Engine | None = None
    specialist: Engine | None = None
    try:
        found, engine, specialist = await run_system(
            pool, model, frame.get("request", {}), on_delta
        )
        answer = response(
            correlation,
            model=model,
            text=str(found.get("text", "")),
            prompt_tokens=list(found.get("prompt_tokens", [])),
            completion_tokens=list(found.get("completion_tokens", [])),
            finish_reason=str(found.get("finish_reason", "stop")),
        )
        answer["escalated"] = bool(found.get("escalated"))
        answer["answered_by"] = str(found.get("answered_by", "front"))
        if found.get("router_features"):
            answer["router_features"] = dict(found["router_features"])
        state.release(model, ok=True)
        return answer
    except asyncio.CancelledError:
        state.release(model, ok=False)
        raise
    except Exception as exc:
        log.warning("request %s on %s failed: %s", correlation or "?", model or "?", exc)
        state.release(model, ok=False)
        return response(correlation, model=model, error=f"{type(exc).__name__}: {exc}")
    finally:
        if engine is not None:
            pool.release(engine)
        if specialist is not None:
            pool.release(specialist)


def _drop(running: dict[str, asyncio.Task[None]], key: str, _: asyncio.Task[None]) -> None:
    running.pop(key, None)


async def _send(socket: Any, frame: Mapping[str, Any]) -> None:
    await socket.send(json.dumps(frame, separators=(",", ":")))


async def _beat(socket: Any, state: Capacity, every: float = HEARTBEAT_SECONDS) -> None:
    while True:
        await asyncio.sleep(every)
        await _send(socket, heartbeat(state))


async def _answer(socket: Any, pool: Pool, frame: Mapping[str, Any], state: Capacity) -> None:
    correlation = str(frame.get("correlation", ""))

    async def on_delta(delta: str) -> None:
        await _send(socket, chunk(correlation, delta))

    found = await serve_taken(pool, frame, state, on_delta)
    await _send(socket, found)


async def session(settings: Settings, pool: Pool) -> None:
    websockets = _websockets()
    state = Capacity.of(settings.serves)
    running: dict[str, asyncio.Task[None]] = {}

    async with websockets.connect(settings.gateway, max_size=None) as socket:
        await _send(socket, hello(settings))
        log.info(
            "dialled %s as %s for %s",
            settings.gateway,
            settings.hotkey[:12],
            ", ".join(s.model for s in settings.serves),
        )
        pulse = asyncio.create_task(_beat(socket, state))
        try:
            async for raw in socket:
                frame = decode_frame(raw)
                kind = str(frame.get("type", ""))
                correlation = str(frame.get("correlation", ""))
                model = str(frame.get("model", ""))

                if kind == REQUEST:
                    if settings.of(model) is None:
                        await _send(
                            socket,
                            response(
                                correlation, model=model, error="this operator does not serve it"
                            ),
                        )
                        continue
                    if not state.free(model):
                        await _send(
                            socket,
                            response(correlation, model=model, error="at declared concurrency"),
                        )
                        continue
                    state.take(model)
                    task = asyncio.create_task(_answer(socket, pool, frame, state))
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


async def run(settings: Settings, pool: Pool, *, stop: asyncio.Event | None = None) -> None:
    attempt = 0
    while stop is None or not stop.is_set():
        try:
            await session(settings, pool)
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
