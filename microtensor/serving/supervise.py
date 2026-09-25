from __future__ import annotations

import asyncio
import contextlib
import json
import logging
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

from microtensor.serving import plan
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


SERVABLE_FORMATS: frozenset[str] = frozenset({"gguf"})


def format_of(artifact: Path) -> str:
    found = artifact if artifact.is_dir() else artifact.parent
    manifest = found / MANIFEST_NAME
    if manifest.exists():
        try:
            return str(json.loads(manifest.read_text(encoding="utf-8")).get("format", ""))
        except ValueError:
            return ""
    return artifact.suffix.lstrip(".").lower()


def weights_in(artifact: Path) -> Path:
    if artifact.is_file():
        return artifact
    for child in sorted(artifact.glob("*.gguf")):
        return child
    raise SuperviseError(f"no servable weights under {artifact}")


def launcher(
    binary: str, threads: int = 0, context: int = 4096, gpu_layers: int = -1
) -> Callable[[Engine], Any]:
    def start(engine: Engine) -> Any:
        declared = format_of(engine.artifact)
        if declared and declared not in SERVABLE_FORMATS:
            raise SuperviseError(
                f"{engine.model} is {declared}, and only "
                f"{', '.join(sorted(SERVABLE_FORMATS))} can be served and verified today"
            )
        argv = [
            binary,
            "--model",
            str(weights_in(engine.artifact)),
            "--host",
            "127.0.0.1",
            "--port",
            str(engine.port),
            "--ctx-size",
            str(context),
        ]
        if threads:
            argv += ["--threads", str(threads)]
        if gpu_layers:
            argv += ["--n-gpu-layers", str(gpu_layers)]
        log.info("starting %s on port %d", engine.model, engine.port)
        return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return start


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
    memory: Callable[[], int] = usable_memory
    engines: dict[str, Engine] = field(default_factory=dict)
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
        port = FIRST_PORT
        while port in taken:
            port += 1
        return port

    def serves(self) -> tuple[Served, ...]:
        if self.last is None:
            return ()
        found: list[Served] = []
        for entry in self.last.hold:
            engine = self.engines.get(entry.model)
            if engine is None:
                continue
            found.append(served(entry.model, entry.artifact_digest, engine.url, entry.concurrency))
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

        engine = Engine(model=entry.model, artifact=artifact, port=self._port())
        engine = Engine(
            model=engine.model,
            artifact=engine.artifact,
            port=engine.port,
            process=self.start(engine),
        )
        self.engines[entry.model] = engine
        if self.ready is not None and not await self.ready(engine):
            self.stop(entry.model)
            raise SuperviseError(f"{entry.model} never answered on {engine.url}")

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

    def stop(self, model: str) -> None:
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
            serve(bench.settings(), Pool(bench.serves()))
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

    where.mkdir(parents=True, exist_ok=True)
    digest = offer.artifact_digest.split(":")[-1]
    landed = where / f"{digest[:16]}.gguf"
    if landed.exists() and _matches(digest_file(landed), digest):
        return landed

    source = offer.archive_repo
    if "://" not in source and not source.startswith("hf:"):
        source = f"hf:{source}"
    scheme, locator = parse_source(source)
    fetcher = fetcher_for(scheme)

    staging = where / f"staging-{digest[:16]}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        fetcher(locator, f"{digest[:16]}.gguf", staging, timeout)
    except Unfetchable as exc:
        staging.unlink(missing_ok=True)
        raise SuperviseError(str(exc)) from exc

    found = digest_file(staging)
    if not _matches(found, digest):
        staging.unlink(missing_ok=True)
        raise SuperviseError(
            f"{offer.model} fetched to {found[:16]}, which is not the certified {digest[:16]}"
        )
    staging.replace(landed)
    log.info("fetched %s to %s", offer.model, landed)
    return landed


async def engine_ready(engine: Engine) -> bool:
    from microtensor.serving.agent import HttpEngine

    async def probe(url: str) -> bool:
        return await HttpEngine(url).ready()

    return await wait_ready(engine, probe)
