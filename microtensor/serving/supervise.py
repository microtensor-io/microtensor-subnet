from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from microtensor.serving import cascade, engines, plan
from microtensor.serving.agent import Pool, Served, Settings, served, session
from microtensor.serving.client import ServerError

log = logging.getLogger("microtensor.serving.supervise")

CATALOGUE_PATH = "/v1/public/inference/catalogue"
CATALOGUE_TIMEOUT = 30
REVIEW_SECONDS = 300.0
IDLE_SECONDS = 60.0
FIRST_PORT = 18080
READY_ATTEMPTS = 120
READY_PAUSE = 2.0
MANIFEST_NAME = "manifest.json"


class SuperviseError(RuntimeError):
    pass


def catalogue(server: str, timeout: int = CATALOGUE_TIMEOUT) -> list[dict[str, Any]]:
    url = urllib.parse.urljoin(server.rstrip("/") + "/", CATALOGUE_PATH.lstrip("/"))
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            found = json.loads(answer.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ServerError(f"{exc.code}: could not read the catalogue") from exc
    except urllib.error.URLError as exc:
        raise ServerError(f"could not reach {url}: {exc.reason}") from exc
    except (OSError, ValueError) as exc:
        raise ServerError(f"could not read {url}: {exc}") from exc
    return list(found.get("models", []))


@dataclass(frozen=True, slots=True)
class Engine:
    model: str
    artifact: Path
    port: int
    process: Any = None
    role: str = cascade.FRONT

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def free_disk(where: Path) -> int:
    where.mkdir(parents=True, exist_ok=True)
    return int(shutil.disk_usage(where).free)


def usable_memory() -> int:
    try:
        import psutil
    except ImportError:
        return 0
    return int(psutil.virtual_memory().available)


@dataclass(frozen=True, slots=True)
class Accelerator:
    vendor: str
    name: str
    memory_bytes: int


def _ask(argv: list[str], timeout: float = 20.0) -> str:
    try:
        found = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return found.stdout if found.returncode == 0 else ""


def _nvidia() -> list[Accelerator]:
    text = _ask(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    found: list[Accelerator] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        found.append(
            Accelerator(vendor="nvidia", name=parts[0], memory_bytes=int(parts[1]) * 1024 * 1024)
        )
    return found


def _amd() -> list[Accelerator]:
    text = _ask(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    found: list[Accelerator] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].lower().startswith("card"):
            continue
        total = next((int(p) for p in parts[1:] if p.isdigit()), 0)
        if total:
            found.append(Accelerator(vendor="amd", name=parts[0], memory_bytes=total))
    return found


def _apple() -> list[Accelerator]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    shared = usable_memory()
    if not shared:
        return []
    return [
        Accelerator(
            vendor="apple", name=platform.processor() or "apple silicon", memory_bytes=shared
        )
    ]


def accelerators() -> list[Accelerator]:
    for probe in (_nvidia, _amd, _apple):
        found = probe()
        if found:
            return found
    return []


def accelerator_memory(found: Sequence[Accelerator] | None = None) -> int:
    held = list(found if found is not None else accelerators())
    if not held:
        return 0
    if held[0].vendor == "apple":
        return held[0].memory_bytes
    return max(a.memory_bytes for a in held)


def require_accelerator() -> list[Accelerator]:
    found = accelerators()
    if found:
        return found
    raise SuperviseError(
        "an inference miner needs a GPU. Looked for nvidia-smi, rocm-smi and "
        "apple silicon and found none. Certified systems are ranked on a single "
        "CPU thread for determinism, but serving is judged on what a client "
        "waits for, and a CPU cannot hold that pace"
    )


def format_of(artifact: Path) -> str:
    manifest = cascade.manifest_of(artifact)
    return str((manifest.get("load") or {}).get("format", "")).lower()


def launcher(
    binary: str = "",
    threads: int = 0,
    context: int = engines.DEFAULT_CONTEXT,
    gpu_layers: int = -1,
    share: float = 0.85,
    concurrency: int = 8,
) -> Callable[[Engine], Any]:
    def start(engine: Engine) -> Any:
        declared = format_of(engine.artifact)
        try:
            plan_for = engines.launch(
                declared,
                cascade.entrypoint_of(engine.artifact)
                if engines.backend_for(declared) == engines.LLAMA
                else engine.artifact,
                engine.port,
                binary=binary,
                context=context,
                concurrency=concurrency,
                threads=threads,
                gpu_layers=gpu_layers,
                share=share,
            )
        except (engines.EngineError, cascade.CascadeError) as exc:
            raise SuperviseError(str(exc)) from exc
        log.info(
            "starting %s (%s) on port %d with %s",
            engine.model,
            engine.role,
            engine.port,
            plan_for.backend,
        )
        return subprocess.Popen(
            list(plan_for.argv), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    return start


def client_for(artifact: Path) -> Callable[[str], Any]:
    backend = engines.backend_for(format_of(artifact))
    if backend == engines.SGLANG:
        return engines.SGLangEngine
    from microtensor.serving.agent import HttpEngine

    return HttpEngine


async def measure(engine: Engine, tokens: int = 32) -> float:
    from microtensor.serving.agent import HttpEngine

    driver = HttpEngine(engine.url)
    started = time.perf_counter()
    try:
        found = await driver.generate(
            {"prompt": "Count from one to twenty.", "max_tokens": tokens, "temperature": 0.0}
        )
    except Exception as exc:
        raise SuperviseError(f"{engine.model} could not be measured: {exc}") from exc
    produced = len(found.get("completion_tokens") or [])
    if produced <= 0:
        raise SuperviseError(f"{engine.model} produced nothing when measured")
    return ((time.perf_counter() - started) * 1000.0) / produced


@dataclass(slots=True)
class Bench:
    server: str
    hotkey: str
    artifacts: Path
    capacity: plan.Capacity
    policy: plan.Policy = field(default_factory=plan.Policy)
    worker: str = ""
    gateway: str = ""
    fetch: Callable[[plan.Offer, Path], Path] | None = None
    start: Callable[[Engine], Any] | None = None
    declare: Callable[[str, str], None] | None = None
    read: Callable[[str], list[dict[str, Any]]] = catalogue
    ready: Callable[[Engine], Awaitable[bool]] | None = None
    disk: Callable[[Path], int] = free_disk
    memory: Callable[[], int] = accelerator_memory
    engines: dict[str, Engine] = field(default_factory=dict)
    specialists: dict[str, Engine] = field(default_factory=dict)
    routers: dict[str, Any] = field(default_factory=dict)
    clients: dict[str, Any] = field(default_factory=dict)
    timed: dict[str, float] = field(default_factory=dict)
    bench: Callable[[Engine], Awaitable[float]] | None = None
    last: plan.Plan | None = None

    def held(self) -> dict[str, str]:
        return {model: self._digest_of(model) for model in self.engines}

    def _digest_of(self, model: str) -> str:
        if self.last is None:
            return ""
        for entry in self.last.hold:
            if entry.model == model:
                return entry.artifact_digest
        return ""

    def _port(self) -> int:
        taken = {engine.port for engine in self.engines.values()}
        taken |= {engine.port for engine in self.specialists.values()}
        port = FIRST_PORT
        while port in taken:
            port += 1
        return port

    def build(self, url: str) -> Any:
        from microtensor.serving.agent import HttpEngine

        return self.clients.get(url, HttpEngine)(url)

    def serves(self) -> tuple[Served, ...]:
        if self.last is None:
            return ()
        found: list[Served] = []
        for entry in self.last.hold:
            engine = self.engines.get(entry.model)
            if engine is None:
                continue
            second = self.specialists.get(entry.model)
            found.append(
                served(
                    entry.model,
                    entry.artifact_digest,
                    engine.url,
                    entry.concurrency,
                    specialist=second.url if second else (),
                    router=self.routers.get(entry.model),
                )
            )
            self.clients[engine.url] = client_for(engine.artifact)
            if second is not None:
                self.clients[second.url] = client_for(second.artifact)
        return tuple(found)

    def settings(self) -> Settings:
        return Settings(
            gateway=self.gateway,
            hotkey=self.hotkey,
            serves=self.serves(),
            worker=self.worker,
        )

    def measure(self) -> plan.Capacity:
        disk = self.disk(self.artifacts) or self.capacity.disk_bytes
        memory = self.memory() or self.capacity.memory_bytes
        return plan.Capacity(
            disk_bytes=min(disk, self.capacity.disk_bytes),
            memory_bytes=min(memory, self.capacity.memory_bytes),
            slots=self.capacity.slots,
            max_models=self.capacity.max_models,
        )

    def decide(self) -> plan.Plan:
        offers = self.read(self.server)
        return plan.build(
            offers,
            self.measure(),
            held=self.held(),
            policy=self.policy,
            measured=self.timed,
        )

    async def apply(self, found: plan.Plan) -> plan.Plan:
        for model in found.drop:
            self.stop(model)

        kept: list[plan.Holding] = []
        for entry in found.hold:
            offer = next((o for o in found.fetch if o.model == entry.model), None)
            if offer is not None or entry.model not in self.engines:
                try:
                    await self.raise_one(entry, offer)
                except SuperviseError as exc:
                    log.warning("could not bring up %s: %s", entry.model, exc)
                    continue
            kept.append(entry)

        self.last = plan.Plan(
            hold=tuple(kept),
            fetch=found.fetch,
            drop=found.drop,
            skipped=found.skipped,
            disk_bytes=found.disk_bytes,
            memory_bytes=found.memory_bytes,
            reasons=found.reasons,
        )
        return self.last

    def _share(self) -> float:
        held = max(1, len(self.engines) + len(self.specialists) + 1)
        return engines.share_for(held)

    async def raise_one(self, entry: plan.Holding, offer: plan.Offer | None) -> None:
        ceiling = offer.tpot_ms if offer is not None else 0.0
        if self.fetch is None or self.start is None:
            raise SuperviseError("this bench has no way to fetch or start an engine")
        if offer is not None:
            self.stop(entry.model)
            artifact = self.fetch(offer, self.artifacts)
        else:
            existing = self.engines.get(entry.model)
            artifact = existing.artifact if existing else self.artifacts / entry.model
        if entry.model in self.engines:
            return

        engine = self._raise_engine(entry.model, artifact, cascade.FRONT)
        self.engines[entry.model] = engine
        if self.ready is not None and not await self.ready(engine):
            self.stop(entry.model)
            raise SuperviseError(f"{entry.model} never answered on {engine.url}")

        found = system_of(artifact)
        if not found.single:
            await self._raise_rest(entry.model, artifact, found)

        if self.bench is not None and entry.model not in self.timed:
            try:
                self.timed[entry.model] = await self.bench(engine)
                log.info(
                    "%s runs at %.0f ms a token on this machine",
                    entry.model,
                    self.timed[entry.model],
                )
            except SuperviseError:
                self.stop(entry.model)
                raise
            if ceiling and self.timed[entry.model] > ceiling:
                self.stop(entry.model)
                raise SuperviseError(
                    f"{entry.model} runs at {self.timed[entry.model]:.0f} ms a token here "
                    f"and its envelope allows {ceiling:.0f}"
                )

        if self.declare is not None:
            try:
                self.declare(entry.model, entry.artifact_digest)
            except ServerError as exc:
                log.warning("could not declare %s: %s", entry.model, exc)

    def _raise_engine(self, model: str, artifact: Path, role: str) -> Engine:
        if self.start is None:
            raise SuperviseError("this bench has no way to start an engine")
        engine = Engine(model=model, artifact=artifact, port=self._port(), role=role)
        return Engine(
            model=engine.model,
            artifact=engine.artifact,
            port=engine.port,
            process=self.start(engine),
            role=role,
        )

    async def _raise_rest(self, model: str, artifact: Path, found: cascade.System) -> None:
        root = artifact if artifact.is_dir() else artifact.parent

        if found.specialist is not None:
            where = root / (found.specialist.path or cascade.SPECIALIST)
            if not where.exists():
                raise SuperviseError(
                    f"{model} declares a specialist at {where}, which the archive does not carry"
                )
            engine = self._raise_engine(model, where, cascade.SPECIALIST)
            self.specialists[model] = engine
            if self.ready is not None and not await self.ready(engine):
                self.stop(model)
                raise SuperviseError(f"the specialist for {model} never answered on {engine.url}")

        if found.router is not None:
            where = root / (found.router.path or f"{cascade.ROUTER}.onnx")
            if not where.exists():
                raise SuperviseError(
                    f"{model} declares a router at {where}, which the archive does not carry"
                )
            self.routers[model] = cascade.load_router(where, found.router_features)
            log.info(
                "%s is a system: front, router on %s, specialist",
                model,
                ", ".join(found.router_features) or "no features",
            )

    def stop(self, model: str) -> None:
        for engine in (self.specialists.pop(model, None),):
            if engine is not None and engine.process is not None:
                with contextlib.suppress(Exception):
                    engine.process.terminate()
        self.routers.pop(model, None)
        engine = self.engines.pop(model, None)
        if engine is None:
            return
        process = engine.process
        if process is None:
            return
        with contextlib.suppress(Exception):
            process.terminate()
        log.info("stopped %s", model)

    def stop_all(self) -> None:
        for model in list(self.engines):
            self.stop(model)


async def wait_ready(engine: Engine, probe: Callable[[str], Awaitable[bool]]) -> bool:
    for _ in range(READY_ATTEMPTS):
        if await probe(engine.url):
            return True
        await asyncio.sleep(READY_PAUSE)
    return False


async def supervise(
    bench: Bench,
    *,
    stop: asyncio.Event | None = None,
    review_seconds: float = REVIEW_SECONDS,
    idle_seconds: float = IDLE_SECONDS,
    serve: Callable[[Settings, Pool], Coroutine[Any, Any, None]] = session,
) -> None:
    while stop is None or not stop.is_set():
        try:
            wanted = bench.decide()
        except ServerError as exc:
            log.warning("could not read the catalogue: %s", exc)
            if await _pause(stop, idle_seconds):
                return
            continue

        held = await bench.apply(wanted)
        if not held.hold:
            log.info("nothing to serve yet; %d model(s) skipped", len(held.skipped))
            if await _pause(stop, idle_seconds):
                return
            continue

        log.info("serving %s", ", ".join(f"{h.model} at {h.concurrency}" for h in held.hold))
        serving: asyncio.Task[None] = asyncio.create_task(
            serve(bench.settings(), Pool(bench.serves(), build=bench.build))
        )
        try:
            await _until_stale(bench, serving, stop, review_seconds)
        finally:
            serving.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await serving


async def _until_stale(
    bench: Bench,
    serving: asyncio.Task[None],
    stop: asyncio.Event | None,
    review_seconds: float,
) -> None:
    while not serving.done():
        if await _pause(stop, review_seconds):
            return
        try:
            wanted = bench.decide()
        except ServerError as exc:
            log.warning("could not review the catalogue: %s", exc)
            continue
        if plan.changed(wanted, bench.held()):
            log.info("the catalogue moved; redialling with %s", ", ".join(wanted.models))
            return


async def _pause(stop: asyncio.Event | None, seconds: float) -> bool:
    if stop is None:
        await asyncio.sleep(seconds)
        return False
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
        return True
    except asyncio.TimeoutError:
        return False


def summary(found: plan.Plan) -> str:
    if not found.hold:
        return "nothing servable"
    parts = [f"{h.model}({h.concurrency})" for h in found.hold]
    return ", ".join(parts)


def offers_of(rows: Sequence[Mapping[str, Any]]) -> list[plan.Offer]:
    return [plan.Offer.from_wire(row) for row in rows]


def _matches(found: str, wanted: str) -> bool:
    left = found.split(":")[-1]
    right = wanted.split(":")[-1]
    shortest = min(len(left), len(right))
    return shortest >= 16 and left[:shortest] == right[:shortest]


def artifact_of(offer: plan.Offer, where: Path, timeout: int = 900) -> Path:
    from microtensor.core.hashing import digest_file
    from microtensor.registry.fetch import Unfetchable, fetcher_for, parse_source

    if not offer.archive_repo:
        raise SuperviseError(f"{offer.model} publishes no archive to fetch from")

    digest = offer.artifact_digest.split(":")[-1]
    landed = where / digest[:16]
    if (landed / MANIFEST_NAME).exists():
        return landed

    source = offer.archive_repo
    if "://" not in source and not source.startswith("hf:"):
        source = f"hf:{source}"
    scheme, locator = parse_source(source)
    fetcher = fetcher_for(scheme)

    staging = where / f"staging-{digest[:16]}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    try:
        fetcher(locator, MANIFEST_NAME, staging / MANIFEST_NAME, timeout)
        manifest = json.loads((staging / MANIFEST_NAME).read_text(encoding="utf-8"))
        wanted = [str(name) for name in (manifest.get("files") or {})]
        if not wanted:
            wanted = [str((manifest.get("load") or {}).get("entrypoint", ""))]
        for name in wanted:
            if not name or name == MANIFEST_NAME:
                continue
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            fetcher(locator, name, target, timeout)
    except (Unfetchable, ValueError, OSError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise SuperviseError(f"{offer.model} could not be fetched: {exc}") from exc

    entry = cascade.entrypoint_of(staging)
    found = digest_file(entry)
    declared = str(manifest.get("artifact_digest", "")) or offer.artifact_digest
    if not _matches(found, declared):
        shutil.rmtree(staging, ignore_errors=True)
        raise SuperviseError(
            f"{offer.model} fetched to {found.split(':')[-1][:16]}, which is not the "
            f"certified {declared.split(':')[-1][:16]}"
        )

    shutil.rmtree(landed, ignore_errors=True)
    staging.replace(landed)
    log.info("fetched %s to %s", offer.model, landed)
    return landed


def system_of(artifact: Path) -> cascade.System:
    return cascade.System.from_manifest(cascade.manifest_of(artifact))


async def engine_ready(engine: Engine) -> bool:
    from microtensor.serving.agent import HttpEngine

    async def probe(url: str) -> bool:
        return await HttpEngine(url).ready()

    return await wait_ready(engine, probe)
