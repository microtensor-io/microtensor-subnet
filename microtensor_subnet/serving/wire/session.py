from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Final

from microtensor_subnet.serving.wire import protocol

log = logging.getLogger("microtensor_subnet.serving.wire")

MIN_CONCURRENCY: Final[int] = 1
HEARTBEAT_SECONDS: Final[float] = 15.0
RECONNECT_CEILING_SECONDS: Final[float] = 60.0

Handler = Callable[[str, Mapping[str, Any]], dict[str, Any]]


class SessionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Served:
    model: str
    artifact_digest: str
    concurrency: int = 1

    def __post_init__(self) -> None:
        if not self.model or not self.artifact_digest:
            raise SessionError("a served model is a name pinned by its artifact digest")
        if self.concurrency < MIN_CONCURRENCY:
            raise SessionError(f"concurrency for {self.model} must be at least {MIN_CONCURRENCY}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "artifact_digest": self.artifact_digest,
            "concurrency": self.concurrency,
        }


@dataclass(frozen=True, slots=True)
class Settings:
    gateway: str
    hotkey: str
    serves: tuple[Served, ...]
    worker: str = ""

    def __post_init__(self) -> None:
        if not self.gateway:
            raise SessionError("no gateway to dial; there is no default")
        if not self.hotkey:
            raise SessionError("an operator serves under its own hotkey")
        if not self.serves:
            raise SessionError("an operator declares at least one model to serve")
        names = [s.model for s in self.serves]
        if len(set(names)) != len(names):
            raise SessionError("a model is declared twice; one entry names one model")

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
            raise SessionError("the gateway routed past the declared concurrency")
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
    def of(cls, serves: tuple[Served, ...]) -> Capacity:
        return cls({s.model: Inflight(limit=s.concurrency) for s in serves})

    def free(self, model: str) -> int:
        held = self.limits.get(model)
        return held.free if held is not None else 0

    def take(self, model: str) -> None:
        held = self.limits.get(model)
        if held is None:
            raise SessionError(f"the gateway routed {model}, which this operator does not serve")
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


def backoff(attempt: int, ceiling: float = RECONNECT_CEILING_SECONDS) -> float:
    if attempt <= 0:
        return 0.0
    return min(float(ceiling), 2.0 ** min(attempt - 1, 16))


def _websockets() -> Any:
    try:
        import websockets
    except ImportError as exc:
        raise SessionError("websockets is required to dial the gateway") from exc
    return websockets


async def _send(socket: Any, frame: Mapping[str, Any]) -> None:
    await socket.send(protocol.encode(frame))


async def _beat(socket: Any, state: Capacity, every: float = HEARTBEAT_SECONDS) -> None:
    while True:
        await asyncio.sleep(every)
        await _send(socket, protocol.heartbeat(state.running, state.idle_seconds, state.to_dict()))


async def serve_one(handler: Handler, frame: Mapping[str, Any], state: Capacity) -> dict[str, Any]:
    correlation = str(frame.get("correlation", ""))
    model = str(frame.get("model", ""))
    raw = frame.get("request")
    request: dict[str, Any] = dict(raw) if isinstance(raw, Mapping) else {}
    started = time.perf_counter()
    try:
        found = await asyncio.to_thread(handler, model, request)
    except asyncio.CancelledError:
        state.release(model, ok=False)
        raise
    except Exception as exc:
        log.warning("request %s on %s failed: %s", correlation or "?", model or "?", exc)
        state.release(model, ok=False)
        return protocol.response(correlation, model=model, error=f"{type(exc).__name__}: {exc}")
    state.release(model, ok=not found.get("error"))
    if found.get("error"):
        return protocol.response(correlation, model=model, error=str(found["error"]))
    return protocol.response(
        correlation,
        model=model,
        answer=found.get("answer"),
        p=float(found.get("p", 0.0)),
        floor_met=bool(found.get("floor_met")),
        verified=bool(found.get("verified")),
        trace=str(found.get("trace", "")),
        usage=found.get("usage"),
        latency_ms=float(found.get("latency_ms") or (time.perf_counter() - started) * 1000),
        finish_reason=str(found.get("finish_reason", "stop")),
    )


def _drop(running: dict[str, asyncio.Task[None]], key: str, _: asyncio.Task[None]) -> None:
    running.pop(key, None)


async def _answer(socket: Any, handler: Handler, frame: Mapping[str, Any], state: Capacity) -> None:
    await _send(socket, await serve_one(handler, frame, state))


async def session(settings: Settings, handler: Handler) -> None:
    websockets = _websockets()
    state = Capacity.of(settings.serves)
    running: dict[str, asyncio.Task[None]] = {}

    async with websockets.connect(settings.gateway, max_size=None) as socket:
        await _send(
            socket,
            protocol.hello(
                settings.hotkey,
                settings.worker,
                [s.to_dict() for s in settings.serves],
                settings.concurrency,
            ),
        )
        log.info(
            "dialled %s as %s for %s",
            settings.gateway,
            settings.hotkey[:12],
            ", ".join(s.model for s in settings.serves),
        )
        pulse = asyncio.create_task(_beat(socket, state))
        try:
            async for raw in socket:
                frame = protocol.decode(raw)
                kind = str(frame.get("type", ""))
                correlation = str(frame.get("correlation", ""))
                model = str(frame.get("model", ""))
                if kind == protocol.REQUEST:
                    if settings.of(model) is None:
                        await _send(
                            socket,
                            protocol.response(
                                correlation, model=model, error="this operator does not serve it"
                            ),
                        )
                        continue
                    if not state.free(model):
                        await _send(
                            socket,
                            protocol.response(
                                correlation, model=model, error="at declared concurrency"
                            ),
                        )
                        continue
                    state.take(model)
                    task = asyncio.create_task(_answer(socket, handler, frame, state))
                    running[correlation] = task
                    task.add_done_callback(partial(_drop, running, correlation))
                elif kind == protocol.CANCEL:
                    found = running.pop(correlation, None)
                    if found is not None:
                        found.cancel()
                elif kind == protocol.HEARTBEAT:
                    await _send(
                        socket,
                        protocol.heartbeat(state.running, state.idle_seconds, state.to_dict()),
                    )
        finally:
            pulse.cancel()
            for task in list(running.values()):
                task.cancel()
            if running:
                await asyncio.gather(*running.values(), return_exceptions=True)


async def run(settings: Settings, handler: Handler, *, stop: asyncio.Event | None = None) -> None:
    attempt = 0
    while stop is None or not stop.is_set():
        try:
            await session(settings, handler)
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
