from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Final

from microtensor_subnet.serving.wire import protocol
from microtensor_subnet.serving.wire.protocol import Signer

log = logging.getLogger("microtensor_subnet.serving.wire")

MIN_CONCURRENCY: Final[int] = 1
HEARTBEAT_SECONDS: Final[float] = 15.0
RECONNECT_CEILING_SECONDS: Final[float] = 60.0

Handler = Callable[[str, str, Mapping[str, Any]], dict[str, Any]]
OnChunk = Callable[[Mapping[str, Any]], Awaitable[None]]
ChatHandler = Callable[[str, str, Mapping[str, Any], OnChunk], Awaitable[dict[str, Any]]]
Send = Callable[[Mapping[str, Any]], Awaitable[None]]
Stats = Callable[[], Sequence[Mapping[str, Any]]]


class SessionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Served:
    model: str
    artifact_digest: str
    gpu: str
    concurrency: int = 1
    kind: str = protocol.SYSTEM
    context: int = 0
    tool_calling: bool = False
    vision: bool = False
    engine: str = ""

    def __post_init__(self) -> None:
        if not self.model or not self.artifact_digest:
            raise SessionError("a served model is a name pinned by its artifact digest")
        if not self.gpu:
            raise SessionError(f"{self.model} must name the gpu that holds it")
        if self.concurrency < MIN_CONCURRENCY:
            raise SessionError(f"concurrency for {self.model} must be at least {MIN_CONCURRENCY}")
        if self.kind not in protocol.KINDS:
            raise SessionError(f"{self.model} is offered as {self.kind!r}; kinds are system, model")

    def to_dict(self) -> dict[str, Any]:
        offer: dict[str, Any] = {
            "model": self.model,
            "kind": self.kind,
            "artifact_digest": self.artifact_digest,
            "concurrency": self.concurrency,
            "gpu": self.gpu,
        }
        if self.engine:
            offer["engine"] = self.engine
        if self.kind == protocol.MODEL:
            offer["context"] = self.context
            offer["tool_calling"] = self.tool_calling
            if self.vision:
                offer["vision"] = True
        return offer


@dataclass(frozen=True, slots=True)
class Settings:
    gateway: str
    hotkey: str
    serves: tuple[Served, ...]
    gpus: tuple[Mapping[str, Any], ...] = ()
    worker: str = ""
    signer: Signer | None = None
    agent: Mapping[str, Any] = field(default_factory=dict)
    stats: Stats | None = None

    def __post_init__(self) -> None:
        if not self.gateway:
            raise SessionError("no gateway to dial; there is no default")
        if not self.hotkey:
            raise SessionError("an operator serves under its own hotkey")
        if not self.serves:
            raise SessionError("an operator declares at least one model to serve")
        gpus = [s.gpu for s in self.serves]
        if len(set(gpus)) != len(gpus):
            raise SessionError("a gpu is declared twice; one gpu holds one model")
        announced = {str(g.get("id", "")) for g in self.gpus}
        missing = sorted(set(gpus) - announced)
        if announced and missing:
            raise SessionError(
                f"models name gpus the hello does not announce: {', '.join(missing)}"
            )

    @property
    def concurrency(self) -> int:
        return sum(s.concurrency for s in self.serves)

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(sorted({s.model for s in self.serves}))

    def of(self, gpu: str) -> Served | None:
        return next((s for s in self.serves if s.gpu == gpu), None)

    def holding(self, model: str) -> tuple[Served, ...]:
        return tuple(s for s in self.serves if s.model == model)

    def announced(self) -> tuple[dict[str, Any], ...]:
        if self.gpus:
            return tuple(dict(g) for g in self.gpus)
        return tuple({"id": s.gpu} for s in self.serves)

    def hello(self) -> dict[str, Any]:
        frame = protocol.hello(
            self.hotkey,
            self.worker,
            [s.to_dict() for s in self.serves],
            self.concurrency,
            gpus=self.announced(),
            agent=self.agent,
        )
        if self.signer is None:
            raise SessionError("a protocol 5 hello is signed by the hotkey; no signer was given")
        return protocol.signed(frame, self.signer)


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
    models: dict[str, str]

    @classmethod
    def of(cls, serves: tuple[Served, ...]) -> Capacity:
        return cls(
            {s.gpu: Inflight(limit=s.concurrency) for s in serves},
            {s.gpu: s.model for s in serves},
        )

    def free(self, gpu: str) -> int:
        held = self.limits.get(gpu)
        return held.free if held is not None else 0

    def take(self, gpu: str) -> None:
        held = self.limits.get(gpu)
        if held is None:
            raise SessionError(f"the gateway routed to gpu {gpu}, which this operator lacks")
        held.take()

    def release(self, gpu: str, *, ok: bool) -> None:
        held = self.limits.get(gpu)
        if held is not None:
            held.release(ok=ok)

    def freest(self, model: str) -> str:
        candidates = [gpu for gpu, held in self.models.items() if held == model]
        if not candidates:
            return ""
        return max(candidates, key=lambda gpu: (self.free(gpu), gpu))

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
            gpu: {
                "model": self.models.get(gpu, ""),
                "running": held.running,
                "free": held.free,
                "served": held.served,
                "failed": held.failed,
            }
            for gpu, held in sorted(self.limits.items())
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


async def _report(settings: Settings, state: Capacity) -> dict[str, Any]:
    reported: list[Mapping[str, Any]] = []
    if settings.stats is not None:
        try:
            reported = list(await asyncio.to_thread(settings.stats))
        except Exception as exc:
            log.debug("gpu stats unavailable: %s", exc)
    seen = {str(r.get("id", "")) for r in reported}
    for gpu in settings.announced():
        if str(gpu.get("id", "")) not in seen:
            reported.append({"id": gpu.get("id", "")})
    return protocol.heartbeat(state.running, state.idle_seconds, state.to_dict(), gpus=reported)


async def _beat(
    socket: Any, settings: Settings, state: Capacity, every: float = HEARTBEAT_SECONDS
) -> None:
    while True:
        await asyncio.sleep(every)
        await _send(socket, await _report(settings, state))


def gpu_of(frame: Mapping[str, Any], labels: Mapping[str, str], state: Capacity) -> str:
    named = str(frame.get("gpu", ""))
    if named:
        return labels.get(named, named)
    return state.freest(str(frame.get("model", "")))


async def serve_one(
    handler: Handler, frame: Mapping[str, Any], state: Capacity, gpu: str
) -> dict[str, Any]:
    correlation = str(frame.get("correlation", ""))
    model = str(frame.get("model", ""))
    raw = frame.get("request")
    request: dict[str, Any] = dict(raw) if isinstance(raw, Mapping) else {}
    started = time.perf_counter()
    try:
        found = await asyncio.to_thread(handler, gpu, model, request)
    except asyncio.CancelledError:
        state.release(gpu, ok=False)
        raise
    except Exception as exc:
        log.warning("request %s on %s failed: %s", correlation or "?", model or "?", exc)
        state.release(gpu, ok=False)
        return protocol.response(correlation, model=model, error=f"{type(exc).__name__}: {exc}")
    state.release(gpu, ok=not found.get("error"))
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


async def serve_chat(
    handler: ChatHandler, frame: Mapping[str, Any], state: Capacity, send: Send, gpu: str
) -> dict[str, Any]:
    correlation = str(frame.get("correlation", ""))
    model = str(frame.get("model", ""))
    raw = frame.get("request")
    request: dict[str, Any] = dict(raw) if isinstance(raw, Mapping) else {}
    started = time.perf_counter()
    seq = 0

    async def on_chunk(delta: Mapping[str, Any]) -> None:
        nonlocal seq
        out = protocol.chunk(correlation, model=model, seq=seq, delta=delta)
        seq += 1
        await send(out)

    try:
        found = await handler(gpu, model, request, on_chunk)
    except asyncio.CancelledError:
        state.release(gpu, ok=False)
        raise
    except Exception as exc:
        log.warning("chat %s on %s failed: %s", correlation or "?", model or "?", exc)
        state.release(gpu, ok=False)
        return protocol.response(correlation, model=model, error=f"{type(exc).__name__}: {exc}")
    state.release(gpu, ok=not found.get("error"))
    if found.get("error"):
        return protocol.response(correlation, model=model, error=str(found["error"]))
    return protocol.chat_response(
        correlation,
        model=model,
        content=str(found.get("content") or ""),
        reasoning=str(found.get("reasoning") or ""),
        tool_calls=list(found.get("tool_calls") or []),
        finish_reason=str(found.get("finish_reason") or "stop"),
        usage=found.get("usage"),
        tokens=found.get("tokens"),
        artifact_digest=str(found.get("artifact_digest") or ""),
        latency_ms=float(found.get("latency_ms") or (time.perf_counter() - started) * 1000),
        ttft_ms=float(found.get("ttft_ms") or 0.0),
    )


def _drop(running: dict[str, asyncio.Task[None]], key: str, _: asyncio.Task[None]) -> None:
    running.pop(key, None)


async def _answer(
    socket: Any, handler: Handler, frame: Mapping[str, Any], state: Capacity, gpu: str
) -> None:
    await _send(socket, await serve_one(handler, frame, state, gpu))


async def _answer_chat(
    socket: Any, handler: ChatHandler, frame: Mapping[str, Any], state: Capacity, gpu: str
) -> None:
    async def send(found: Mapping[str, Any]) -> None:
        await _send(socket, found)

    await _send(socket, await serve_chat(handler, frame, state, send, gpu))


def refusal(
    settings: Settings,
    refused: set[str],
    frame: Mapping[str, Any],
    chat: ChatHandler | None,
    gpu: str,
) -> str:
    model = str(frame.get("model", ""))
    served = settings.of(gpu)
    if served is None or served.model != model or model in refused:
        return "this operator does not serve it"
    raw = frame.get("request")
    asked_chat = isinstance(raw, Mapping) and raw.get("kind") == protocol.CHAT
    if asked_chat and served.kind != protocol.MODEL:
        return f"this operator serves {model} as a system, not for chat"
    if asked_chat and chat is None:
        return "this operator runs no chat handler"
    if not asked_chat and served.kind == protocol.MODEL:
        return f"this operator serves {model} for chat only"
    return ""


def welcomed(
    settings: Settings, frame: Mapping[str, Any], refused: set[str], labels: dict[str, str]
) -> None:
    raw = frame.get("gpus")
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, Mapping) and entry.get("label") and entry.get("id"):
                labels[str(entry["label"])] = str(entry["id"])
    for reason in frame.get("refused") or []:
        log.warning("the gateway kept one gpu out: %s", reason)
    version = int(frame.get("version") or 0)
    if not version or version > protocol.SYSTEMS_ONLY_VERSION:
        return
    dropped = sorted({s.model for s in settings.serves if s.kind != protocol.SYSTEM})
    if dropped:
        log.warning(
            "the gateway speaks protocol %d, which carries no chat; not serving %s",
            version,
            ", ".join(dropped),
        )
    refused.update(dropped)


async def session(settings: Settings, handler: Handler, *, chat: ChatHandler | None = None) -> None:
    websockets = _websockets()
    state = Capacity.of(settings.serves)
    running: dict[str, asyncio.Task[None]] = {}
    refused: set[str] = set()
    labels: dict[str, str] = {}

    async with websockets.connect(settings.gateway, max_size=None) as socket:
        await _send(socket, settings.hello())
        log.info(
            "dialled %s as %s with %d gpus for %s",
            settings.gateway,
            settings.hotkey[:12],
            len(settings.serves),
            ", ".join(settings.models),
        )
        pulse = asyncio.create_task(_beat(socket, settings, state))
        try:
            async for raw in socket:
                frame = protocol.decode(raw)
                kind = str(frame.get("type", ""))
                correlation = str(frame.get("correlation", ""))
                model = str(frame.get("model", ""))
                if kind == protocol.WELCOME:
                    welcomed(settings, frame, refused, labels)
                elif kind == protocol.REQUEST:
                    gpu = gpu_of(frame, labels, state)
                    refused_now = refusal(settings, refused, frame, chat, gpu)
                    if refused_now:
                        await _send(
                            socket, protocol.response(correlation, model=model, error=refused_now)
                        )
                        continue
                    if not state.free(gpu):
                        await _send(
                            socket,
                            protocol.response(
                                correlation, model=model, error="at declared concurrency"
                            ),
                        )
                        continue
                    state.take(gpu)
                    served = settings.of(gpu)
                    if chat is not None and served is not None and served.kind == protocol.MODEL:
                        task = asyncio.create_task(_answer_chat(socket, chat, frame, state, gpu))
                    else:
                        task = asyncio.create_task(_answer(socket, handler, frame, state, gpu))
                    running[correlation] = task
                    task.add_done_callback(partial(_drop, running, correlation))
                elif kind == protocol.CANCEL:
                    found = running.pop(correlation, None)
                    if found is not None:
                        found.cancel()
                elif kind == protocol.HEARTBEAT:
                    await _send(socket, await _report(settings, state))
        finally:
            pulse.cancel()
            for task in list(running.values()):
                task.cancel()
            if running:
                await asyncio.gather(*running.values(), return_exceptions=True)


async def run(
    settings: Settings,
    handler: Handler,
    *,
    chat: ChatHandler | None = None,
    stop: asyncio.Event | None = None,
) -> None:
    attempt = 0
    while stop is None or not stop.is_set():
        try:
            await session(settings, handler, chat=chat)
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
