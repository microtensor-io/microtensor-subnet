from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Final

from microtensor_subnet.serving.wire import protocol
from microtensor_subnet.serving.wire.protocol import Signer

log = logging.getLogger("microtensor_subnet.serving.wire")

MIN_CONCURRENCY: Final[int] = 1
HEARTBEAT_SECONDS: Final[float] = 10.0
READY_RETRY_SECONDS: Final[float] = 60.0
_RETRIES: set[asyncio.Task[None]] = set()
RECONNECT_CEILING_SECONDS: Final[float] = 60.0
HEALTHY_SESSION_SECONDS: Final[float] = 60.0
STATS_SECONDS: Final[float] = 5.0
HANDLER_THREADS: Final[int] = 128
HANDLER_POOL: Final[ThreadPoolExecutor] = ThreadPoolExecutor(
    max_workers=HANDLER_THREADS, thread_name_prefix="mt-request"
)
STATS_POOL: Final[ThreadPoolExecutor] = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="mt-gpu-stats"
)
CHALLENGE_SECONDS: Final[float] = 30.0
DRAIN_SECONDS: Final[float] = 600.0
DRAINING: Final[str] = "draining"

Handler = Callable[..., dict[str, Any]]
OnChunk = Callable[[Mapping[str, Any]], Awaitable[None]]
ChatHandler = Callable[[str, str, Mapping[str, Any], OnChunk], Awaitable[dict[str, Any]]]
Send = Callable[[Mapping[str, Any]], Awaitable[None]]
Stats = Callable[[], Sequence[Mapping[str, Any]]]
Phases = Callable[[], Mapping[str, str]]
OnAssign = Callable[[Sequence[Mapping[str, Any]]], None]


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
    image_digest: str = ""

    def __post_init__(self) -> None:
        if not self.model or not self.artifact_digest:
            raise SessionError("a served model is a name pinned by its artifact digest")
        if not self.gpu:
            raise SessionError(f"{self.model} must name the gpu that holds it")
        if self.concurrency < MIN_CONCURRENCY:
            raise SessionError(f"concurrency for {self.model} must be at least {MIN_CONCURRENCY}")
        if self.kind not in protocol.KINDS:
            raise SessionError(f"{self.model} is offered as {self.kind!r}; kinds are system, model")

    def ready(self, instance: str) -> dict[str, Any]:
        return protocol.ready(
            instance,
            model=self.model,
            kind=self.kind,
            artifact_digest=self.artifact_digest,
            image_digest=self.image_digest,
            concurrency=self.concurrency,
            engine=self.engine,
            context=self.context,
            tool_calling=self.tool_calling,
            vision=self.vision,
        )


@dataclass(frozen=True, slots=True)
class Settings:
    gateway: str
    hotkey: str
    serves: tuple[Served, ...] = ()
    gpus: tuple[Mapping[str, Any], ...] = ()
    worker: str = ""
    signer: Signer | None = None
    agent: Mapping[str, Any] = field(default_factory=dict)
    stats: Stats | None = None
    phases: Phases | None = None
    on_assign: OnAssign | None = None

    def __post_init__(self) -> None:
        if not self.gateway:
            raise SessionError("no gateway to dial; there is no default")
        if not self.hotkey:
            raise SessionError("an operator serves under its own hotkey")
        if not self.serves and not self.gpus:
            raise SessionError("an operator announces at least one gpu")
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
    def models(self) -> tuple[str, ...]:
        return tuple(sorted({s.model for s in self.serves}))

    def of(self, gpu: str) -> Served | None:
        return next((s for s in self.serves if s.gpu == gpu), None)

    def announced(self) -> tuple[dict[str, Any], ...]:
        if self.gpus:
            return tuple(dict(g) for g in self.gpus)
        return tuple({"id": s.gpu} for s in self.serves)

    def hello(self, nonce: str) -> dict[str, Any]:
        frame = protocol.hello(
            self.hotkey, self.worker, nonce=nonce, gpus=self.announced(), agent=self.agent
        )
        if self.signer is None:
            raise SessionError("a hello is signed by the hotkey; no signer was given")
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
    draining: set[str] = field(default_factory=set)

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
        candidates = [
            gpu for gpu, held in self.models.items() if held == model and gpu not in self.draining
        ]
        if not candidates:
            return ""
        return max(candidates, key=lambda gpu: (self.free(gpu), gpu))

    def in_flight(self, gpu: str) -> int:
        held = self.limits.get(gpu)
        return held.running if held is not None else 0

    @property
    def running(self) -> int:
        return sum(held.running for held in self.limits.values())

    def drained(self, gpu: str = "") -> bool:
        if gpu:
            return self.in_flight(gpu) == 0
        return all(self.in_flight(g) == 0 for g in self.draining)

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


@dataclass(slots=True)
class Link:
    labels: dict[str, str] = field(default_factory=dict)
    ids: dict[str, str] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)
    welcomed: bool = False

    def learn(self, entries: Any) -> None:
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, Mapping) or not entry.get("label") or not entry.get("id"):
                continue
            label, gpu = str(entry["label"]), str(entry["id"])
            self.labels[label] = gpu
            self.ids[gpu] = label
            if entry.get("status"):
                self.statuses[label] = str(entry["status"])

    def gpu(self, instance: str) -> str:
        return self.labels.get(instance, instance)

    def label(self, gpu: str) -> str:
        return self.ids.get(gpu, "")

    def assigned(self, entries: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if not isinstance(entries, list):
            return found
        for entry in entries:
            if not isinstance(entry, Mapping) or not entry.get("model"):
                continue
            label = str(entry.get("gpu", ""))
            found.append(
                {
                    "gpu": self.gpu(label),
                    "label": label,
                    "model": str(entry["model"]),
                    "artifact_digest": str(entry.get("artifact_digest", "")),
                }
            )
        return found


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


async def _retry_ready(
    socket: Any, settings: Settings, link: Link, label: str, delay: float = READY_RETRY_SECONDS
) -> None:
    await asyncio.sleep(delay)
    for served in settings.serves:
        if link.label(served.gpu) != label:
            continue
        try:
            await _send(socket, served.ready(label))
        except Exception as exc:
            log.debug("ready for %s could not be resent: %s", label, exc)
        return


def _stats_by_id(
    settings: Settings, reported: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    found = {str(r.get("id", "")): dict(r) for r in reported}
    for gpu in settings.announced():
        found.setdefault(str(gpu.get("id", "")), {}).setdefault(
            "memory_mb", int(gpu.get("memory_mb") or 0)
        )
    return found


async def _report(settings: Settings, state: Capacity, link: Link) -> dict[str, Any]:
    reported: list[Mapping[str, Any]] = []
    if settings.stats is not None:
        try:
            loop = asyncio.get_running_loop()
            found = loop.run_in_executor(STATS_POOL, settings.stats)
            reported = list(await asyncio.wait_for(found, timeout=STATS_SECONDS))
        except Exception as exc:
            log.debug("gpu stats unavailable: %s", exc)
    phases: Mapping[str, str] = {}
    if settings.phases is not None:
        try:
            phases = dict(settings.phases())
        except Exception as exc:
            log.debug("phases unavailable: %s", exc)
    stats = _stats_by_id(settings, reported)
    instances: list[dict[str, Any]] = []
    for gpu in settings.announced():
        gpu_id = str(gpu.get("id", ""))
        label = link.label(gpu_id)
        if not label:
            continue
        if gpu_id in state.limits:
            phase = protocol.DRAINING if gpu_id in state.draining else protocol.READY_PHASE
        elif phases.get(gpu_id) in (protocol.PULLING, protocol.LOADING):
            phase = str(phases[gpu_id])
        else:
            continue
        card = stats.get(gpu_id, {})
        total = int(card.get("memory_mb") or 0)
        used = int(card.get("memory_used_mb") or 0)
        instances.append(
            protocol.instance_report(
                label,
                phase=phase,
                in_flight=state.in_flight(gpu_id),
                gpu_utilisation=float(card.get("utilisation") or 0.0),
                free_vram_mb=max(0, total - used) if total else 0,
                memory_used_mb=used,
            )
        )
    return protocol.heartbeat(instances)


async def _beat(
    socket: Any, settings: Settings, state: Capacity, link: Link, every: float = HEARTBEAT_SECONDS
) -> None:
    while True:
        await asyncio.sleep(every)
        await _send(socket, await _report(settings, state, link))


def _request_of(frame: Mapping[str, Any]) -> dict[str, Any]:
    raw = frame.get("request")
    return dict(raw) if isinstance(raw, Mapping) else {}


def _takes_first(handler: Handler) -> bool:
    try:
        parameters = list(inspect.signature(handler).parameters.values())
    except (TypeError, ValueError):
        return False
    return len(parameters) >= 4 or any(
        p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters
    )


async def serve_one(
    handler: Handler,
    frame: Mapping[str, Any],
    state: Capacity,
    gpu: str,
    instance: str = "",
    send: Send | None = None,
) -> dict[str, Any]:
    request_id = str(frame.get("id", ""))
    model = str(frame.get("model", ""))
    request = _request_of(frame)
    started = time.perf_counter()
    loop = asyncio.get_running_loop()
    first: list[float] = []

    def on_first() -> None:
        if first:
            return
        first.append((time.perf_counter() - started) * 1000)
        if send is not None:
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(send(protocol.chunk(request_id, seq=0, delta={})))
            )

    try:
        if _takes_first(handler):
            found = await asyncio.get_running_loop().run_in_executor(
                HANDLER_POOL, handler, gpu, model, request, on_first
            )
        else:
            found = await asyncio.get_running_loop().run_in_executor(
                HANDLER_POOL, handler, gpu, model, request
            )
    except asyncio.CancelledError:
        state.release(gpu, ok=False)
        raise
    except Exception as exc:
        log.warning("request %s on %s failed: %s", request_id or "?", model or "?", exc)
        state.release(gpu, ok=False)
        return protocol.error(request_id, f"{type(exc).__name__}: {exc}")
    state.release(gpu, ok=not found.get("error"))
    if found.get("error"):
        return protocol.error(request_id, str(found["error"]))
    total = (time.perf_counter() - started) * 1000
    return protocol.done(
        request_id,
        instance=instance,
        model=model,
        usage=found.get("usage"),
        ttft_ms=first[0] if first else float(found.get("ttft_ms") or total),
        total_ms=total,
        answer=found.get("answer"),
        p=round(float(found.get("p", 0.0)), 6),
        floor_met=bool(found.get("floor_met")),
        verified=bool(found.get("verified")),
        trace=str(found.get("trace", ""))[:4000],
        finish_reason=str(found.get("finish_reason", "stop")),
    )


async def serve_chat(
    handler: ChatHandler,
    frame: Mapping[str, Any],
    state: Capacity,
    send: Send,
    gpu: str,
    instance: str = "",
) -> dict[str, Any]:
    request_id = str(frame.get("id", ""))
    model = str(frame.get("model", ""))
    request = _request_of(frame)
    started = time.perf_counter()
    first = 0.0
    seq = 0

    async def on_chunk(delta: Mapping[str, Any]) -> None:
        nonlocal seq, first
        if not first:
            first = (time.perf_counter() - started) * 1000
        out = protocol.chunk(request_id, seq=seq, delta=delta)
        seq += 1
        await send(out)

    try:
        found = await handler(gpu, model, request, on_chunk)
    except asyncio.CancelledError:
        state.release(gpu, ok=False)
        raise
    except Exception as exc:
        log.warning("chat %s on %s failed: %s", request_id or "?", model or "?", exc)
        state.release(gpu, ok=False)
        return protocol.error(request_id, f"{type(exc).__name__}: {exc}")
    state.release(gpu, ok=not found.get("error"))
    if found.get("error"):
        return protocol.error(request_id, str(found["error"]))
    total = (time.perf_counter() - started) * 1000
    return protocol.done(
        request_id,
        instance=instance,
        model=model,
        usage=found.get("usage"),
        ttft_ms=first or float(found.get("ttft_ms") or total),
        total_ms=total,
        kind=protocol.CHAT,
        content=str(found.get("content") or ""),
        reasoning=str(found.get("reasoning") or ""),
        tool_calls=[dict(call) for call in found.get("tool_calls") or []],
        finish_reason=str(found.get("finish_reason") or "stop"),
        tokens={
            "prompt": [int(t) for t in (found.get("tokens") or {}).get("prompt") or ()],
            "completion": [int(t) for t in (found.get("tokens") or {}).get("completion") or ()],
        },
        artifact_digest=str(found.get("artifact_digest") or ""),
    )


def _drop(running: dict[str, asyncio.Task[None]], key: str, _: asyncio.Task[None]) -> None:
    running.pop(key, None)


def refusal(
    settings: Settings,
    state: Capacity,
    frame: Mapping[str, Any],
    chat: ChatHandler | None,
    gpu: str,
    now: float,
) -> str:
    model = str(frame.get("model", ""))
    served = settings.of(gpu)
    if served is None or served.model != model:
        return "this operator does not serve it"
    deadline = float(frame.get("deadline") or 0.0)
    if deadline and deadline < now:
        return "the deadline passed before the request arrived"
    if gpu in state.draining:
        return DRAINING
    request = _request_of(frame)
    asked_chat = request.get("kind") == protocol.CHAT
    if asked_chat and served.kind != protocol.MODEL:
        return f"this operator serves {model} as a system, not for chat"
    if asked_chat and chat is None:
        return "this operator runs no chat handler"
    if not asked_chat and served.kind == protocol.MODEL:
        return f"this operator serves {model} for chat only"
    if not state.free(gpu):
        return "at declared concurrency"
    return ""


async def _challenge(frames: Any) -> str:
    try:
        raw = await asyncio.wait_for(frames.__anext__(), CHALLENGE_SECONDS)
    except asyncio.TimeoutError as exc:
        raise SessionError("the gateway sent no challenge") from exc
    except StopAsyncIteration as exc:
        raise SessionError("the gateway closed before the challenge") from exc
    frame = protocol.decode(raw)
    if frame.get("type") != protocol.CHALLENGE or not frame.get("nonce"):
        raise SessionError(f"expected a challenge, the gateway sent {frame.get('type')!r}")
    version = int(frame.get("version") or 0)
    if version != protocol.VERSION:
        raise SessionError(
            f"the gateway speaks protocol {version}; this agent speaks {protocol.VERSION}"
        )
    return str(frame["nonce"])


async def _finish_drains(
    socket: Any, state: Capacity, pending: set[str], link: Link, running: int
) -> None:
    for gpu in sorted(pending):
        if gpu and state.in_flight(gpu) == 0:
            await _send(socket, protocol.drained(link.label(gpu)))
            pending.discard(gpu)
    if "" in pending and running == 0:
        await _send(socket, protocol.drained(""))
        pending.discard("")


async def session(
    settings: Settings,
    handler: Handler,
    *,
    chat: ChatHandler | None = None,
    drain: asyncio.Event | None = None,
) -> None:
    websockets = _websockets()
    state = Capacity.of(settings.serves)
    running: dict[str, asyncio.Task[None]] = {}
    link = Link()
    pending: set[str] = set()

    async with websockets.connect(settings.gateway, max_size=None) as socket:
        frames = socket.__aiter__()
        nonce = await _challenge(frames)
        await _send(socket, settings.hello(nonce))
        log.info(
            "dialled %s as %s with %d gpus for %s",
            settings.gateway,
            settings.hotkey[:12],
            len(settings.announced()),
            ", ".join(settings.models) or "nothing yet",
        )
        pulse: asyncio.Task[None] | None = None
        draining = asyncio.create_task(drain.wait()) if drain is not None else None

        async def handle(raw: str | bytes) -> None:
            nonlocal pulse
            frame = protocol.decode(raw)
            kind = str(frame.get("type", ""))
            request_id = str(frame.get("id", ""))
            if kind == protocol.WELCOME:
                link.learn(frame.get("gpus"))
                link.welcomed = True
                if settings.on_assign is not None:
                    settings.on_assign(link.assigned(frame.get("assign")))
                for served in settings.serves:
                    label = link.label(served.gpu)
                    if label:
                        await _send(socket, served.ready(label))
                    else:
                        log.warning("the gateway gave no label to gpu %s", served.gpu)
                if pulse is None:
                    pulse = asyncio.create_task(_beat(socket, settings, state, link))
                    await _send(socket, await _report(settings, state, link))
            elif kind == protocol.ACCEPTED:
                label = str(frame.get("instance", ""))
                link.statuses[label] = str(frame.get("status", ""))
                log.info("instance %s accepted as %s", label, link.statuses[label])
            elif kind == protocol.REFUSED:
                label = str(frame.get("instance", ""))
                log.warning("instance %s refused: %s", label, frame.get("reason"))
                retry = asyncio.create_task(_retry_ready(socket, settings, link, label))
                _RETRIES.add(retry)
                retry.add_done_callback(_RETRIES.discard)
            elif kind == protocol.ASSIGN:
                if settings.on_assign is not None:
                    settings.on_assign(link.assigned(frame.get("assign")))
            elif kind == protocol.REQUEST:
                instance = str(frame.get("instance", ""))
                gpu = link.gpu(instance) if instance else state.freest(str(frame.get("model")))
                refused = refusal(settings, state, frame, chat, gpu, time.time())
                if refused:
                    await _send(socket, protocol.error(request_id, refused))
                    return
                state.take(gpu)
                holder = settings.of(gpu)
                label = instance or link.label(gpu)
                if chat is not None and holder is not None and holder.kind == protocol.MODEL:

                    async def send(found: Mapping[str, Any]) -> None:
                        await _send(socket, found)

                    task = asyncio.create_task(
                        _finished(socket, serve_chat(chat, frame, state, send, gpu, label))
                    )
                else:

                    async def send_frame(found: Mapping[str, Any]) -> None:
                        await _send(socket, found)

                    task = asyncio.create_task(
                        _finished(socket, serve_one(handler, frame, state, gpu, label, send_frame))
                    )
                running[request_id] = task
                task.add_done_callback(partial(_drop, running, request_id))
                task.add_done_callback(
                    lambda _: asyncio.ensure_future(
                        _finish_drains(socket, state, pending, link, len(running))
                    )
                )
            elif kind == protocol.CANCEL:
                found = running.pop(request_id, None)
                if found is not None:
                    found.cancel()
            elif kind == protocol.HEARTBEAT:
                await _send(socket, await _report(settings, state, link))
            elif kind == protocol.DRAIN:
                instance = str(frame.get("instance", ""))
                if instance:
                    gpu = link.gpu(instance)
                    state.draining.add(gpu)
                    pending.add(gpu)
                else:
                    state.draining.update(state.limits)
                    pending.add("")
                await _finish_drains(socket, state, pending, link, len(running))

        try:
            while True:
                receive = asyncio.ensure_future(frames.__anext__())
                waiting = {receive} | ({draining} if draining is not None else set())
                finished, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                if draining is not None and draining in finished:
                    receive.cancel()
                    await _drain_out(socket, state, running)
                    return
                try:
                    raw = receive.result()
                except StopAsyncIteration:
                    return
                await handle(raw)
        finally:
            if pulse is not None:
                pulse.cancel()
            if draining is not None:
                draining.cancel()
            for task in list(running.values()):
                task.cancel()
            if running:
                await asyncio.gather(*running.values(), return_exceptions=True)


async def _finished(socket: Any, answer: Awaitable[Mapping[str, Any]]) -> None:
    await _send(socket, await answer)


async def _drain_out(socket: Any, state: Capacity, running: dict[str, asyncio.Task[None]]) -> None:
    state.draining.update(state.limits)
    await _send(socket, protocol.drain(""))
    deadline = time.monotonic() + DRAIN_SECONDS
    while running and time.monotonic() < deadline:
        await asyncio.wait(list(running.values()), timeout=1.0)
    await _send(socket, protocol.drained(""))
    log.info("drained; %d requests were still open", len(running))


async def run(
    settings: Settings,
    handler: Handler,
    *,
    chat: ChatHandler | None = None,
    stop: asyncio.Event | None = None,
    drain: asyncio.Event | None = None,
) -> None:
    attempt = 0
    while stop is None or not stop.is_set():
        if drain is not None and drain.is_set():
            return
        started = time.monotonic()
        try:
            await session(settings, handler, chat=chat, drain=drain)
            attempt = 0
            if drain is not None and drain.is_set():
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if time.monotonic() - started >= HEALTHY_SESSION_SECONDS:
                attempt = 0
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
