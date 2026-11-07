from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import httpx

from microtensor_subnet.core.constants import RELEASE_VERSION
from microtensor_subnet.core.hashing import DIGEST_PREFIX
from microtensor_subnet.serving import gpus, plan, update, wire
from microtensor_subnet.serving.engines import BACKENDS, EngineError
from microtensor_subnet.serving.host.answer import Held, answer
from microtensor_subnet.serving.host.config import HostConfig
from microtensor_subnet.serving.host.fetch import (
    Fetched,
    FetchedModel,
    FetchError,
    fetch,
    fetch_model,
)
from microtensor_subnet.serving.host.image import (
    ImageError,
    ImageHeld,
    answer_image,
    certified_image,
    close_image,
    open_image,
)
from microtensor_subnet.serving.host.plain import (
    PlainError,
    PlainHeld,
    attach_model,
    close_model,
    open_model,
)
from microtensor_subnet.serving.host.plain import chat as chat_model
from microtensor_subnet.serving.tokens import TokenError
from microtensor_subnet.serving.wire import protocol
from microtensor_subnet.validator.sandbox import SandboxError, SandboxHost, host_ip
from microtensor_subnet.validator.sandbox.docker import Docker, DockerError
from microtensor_subnet.validator.sandbox.services import Services

log = logging.getLogger("microtensor_subnet.serving.host")

CATALOGUE_PATH = "/v1/public/inference/catalogue"
GENERAL: Final[str] = "general"
Declare = Callable[[str, str], None]
Catalogue = Callable[[str], list[dict[str, Any]]]
ModelFetcher = Callable[[Mapping[str, Any], Path], FetchedModel]


class HostError(RuntimeError):
    pass


SERVE_FAILURES: Final[tuple[type[Exception], ...]] = (
    FetchError,
    SandboxError,
    HostError,
    PlainError,
    EngineError,
    TokenError,
    DockerError,
    ImageError,
    OSError,
)


def read_catalogue(server: str, timeout: float = 30.0) -> list[dict[str, Any]]:
    with httpx.Client(timeout=timeout) as http:
        response = http.get(server.rstrip("/") + CATALOGUE_PATH)
    response.raise_for_status()
    found = response.json()
    return [dict(row) for row in found.get("models") or []]


class ServingHost:
    def __init__(
        self,
        config: HostConfig,
        *,
        hotkey: str,
        server: str,
        gateway: str,
        policy: plan.Policy | None = None,
        pinned: Mapping[str, str] | None = None,
        worker: str = "",
        docker: Docker | None = None,
        declare: Declare | None = None,
        catalogue: Catalogue = read_catalogue,
        fetcher: Callable[[Mapping[str, Any], Any], Fetched] = fetch,
        model_fetcher: ModelFetcher | None = None,
        accept_licences: bool = False,
        engine_url: str = "",
        signer: wire.Signer | None = None,
        stats: wire.Stats | None = None,
        updater: update.Updater | None = None,
        shim_urls: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.hotkey = hotkey
        self.server = server
        self.gateway = gateway
        self.policy = policy or plan.Policy()
        self.pinned = dict(pinned) if pinned is not None else None
        self.worker = worker
        self.docker = docker or Docker()
        self.declare = declare
        self.catalogue = catalogue
        self.fetcher = fetcher
        self.model_fetcher: ModelFetcher = model_fetcher or self._fetch_model
        self.accept_licences = accept_licences
        self.engine_url = engine_url
        self.shim_urls = {str(k): str(v).rstrip("/") for k, v in dict(shim_urls or {}).items()}
        self.signer = signer
        self.stats = stats
        self.updater = updater
        self.cards: tuple[gpus.Card, ...] = config.gpu_cards
        self.services = Services(host_ip(), config.host_services_port)
        self.sandboxes = SandboxHost(config, self.docker, self.services, host_ip())
        self.pinned_hosts: dict[str, SandboxHost] = {}
        self.opened_by: dict[str, SandboxHost] = {}
        self.held: dict[str, Held] = {}
        self.plain: dict[str, PlainHeld] = {}
        self.images_held: dict[str, ImageHeld] = {}
        self.entries: dict[str, dict[str, Any]] = {}
        self.phases_by: dict[str, str] = {}
        self.preferred: dict[str, str] = {}
        self.drain = asyncio.Event()
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if not self._started:
            self.services.start()
            self._started = True

    def stop(self) -> None:
        for gpu in [*self.held, *self.plain, *self.images_held]:
            self.release(gpu)
        if self._started:
            self.services.stop()
            self._started = False
        try:
            self.docker.sweep()
        except Exception as exc:
            log.warning("docker sweep skipped: %s", exc)

    def _fetch_model(self, entry: Mapping[str, Any], cache_dir: Path) -> FetchedModel:
        return fetch_model(entry, cache_dir, token=self.config.hf_token)

    @property
    def devices(self) -> tuple[str, ...]:
        return tuple(card.device for card in self.cards)

    def card(self, device: str) -> gpus.Card:
        found = next((card for card in self.cards if card.device == device), None)
        if found is None:
            raise HostError(f"gpu {device} is not one this host was given")
        return found

    def device_of(self, gpu_id: str) -> str:
        found = next((card for card in self.cards if card.id == gpu_id), None)
        return found.device if found is not None else gpu_id

    def holding(self) -> dict[str, tuple[str, str]]:
        found = {gpu: (held.model, held.artifact_digest) for gpu, held in self.held.items()}
        found.update({gpu: (held.model, held.artifact_digest) for gpu, held in self.plain.items()})
        found.update(
            {gpu: (held.model, held.artifact_digest) for gpu, held in self.images_held.items()}
        )
        return found

    def images(self) -> frozenset[str]:
        if self.engine_url:
            return frozenset(BACKENDS)
        return frozenset(self.config.images)

    def phases(self) -> dict[str, str]:
        return {self.card(device).id: phase for device, phase in self.phases_by.items()}

    def assign(self, entries: Sequence[Mapping[str, Any]]) -> None:
        wished = {
            self.device_of(str(entry.get("gpu", ""))): str(entry.get("model", ""))
            for entry in entries
            if entry.get("model")
        }
        if wished != self.preferred:
            log.info("the gateway suggests %s", wished or "nothing")
        self.preferred = wished

    def plan_now(self, offers: Sequence[Mapping[str, Any]]) -> plan.Plan:
        self.entries = {str(row.get("model") or row.get("id") or ""): dict(row) for row in offers}
        gpu_gb = gpus.smallest_gb(self.cards) or self.config.gpu_gb
        capacity = plan.Capacity(gpus=self.devices, gpu_gb=gpu_gb, images=self.images())
        rules = dataclasses.replace(self.policy, accept_licences=self.accept_licences)
        return plan.build(
            offers,
            capacity,
            held=self.holding(),
            policy=rules,
            pinned=self.pinned,
            preferred=self.preferred,
        )

    def release(self, gpu: str) -> None:
        self.phases_by.pop(gpu, None)
        image = self.images_held.pop(gpu, None)
        if image is not None:
            if not image.container:
                return
            try:
                close_image(self.docker, image)
            except Exception as exc:
                log.warning("closing %s on gpu %s failed: %s", image.model, gpu, exc)
            return
        plain = self.plain.pop(gpu, None)
        if plain is not None:
            try:
                close_model(self.docker, plain)
            except Exception as exc:
                log.warning("closing %s on gpu %s failed: %s", plain.model, gpu, exc)
            return
        held = self.held.pop(gpu, None)
        if held is None:
            return
        try:
            self.opened_by.pop(gpu, self.sandboxes).close(held.sandbox)
        except Exception as exc:
            log.warning("closing %s on gpu %s failed: %s", held.model, gpu, exc)

    def engine_for(self, entry: Mapping[str, Any]) -> SandboxHost:
        images = entry.get("images")
        engine = images.get("engine") if isinstance(images, Mapping) else None
        image = str(engine.get("image") or "") if isinstance(engine, Mapping) else ""
        digest = str(engine.get("digest") or "") if isinstance(engine, Mapping) else ""
        if not image or not digest:
            return self.sandboxes
        reference = self.docker.ensure_digest(image, digest)
        found = self.pinned_hosts.get(reference)
        if found is None:
            config = dataclasses.replace(self.config, vllm_image=reference)
            found = SandboxHost(config, self.docker, self.services, host_ip())
            self.pinned_hosts[reference] = found
        return found

    def take_up(self, holding: plan.Holding) -> None:
        entry = self.entries.get(holding.model)
        if entry is None:
            raise HostError(f"{holding.model} is not in the catalogue")
        if holding.gpu not in self.devices:
            raise HostError(f"gpu {holding.gpu} is not one this host was given")
        self.phases_by[holding.gpu] = protocol.PULLING
        try:
            if holding.gpu in self.shim_urls:
                self._take_shim(holding, entry)
            elif entry.get("kind") == GENERAL:
                self._take_model(holding, entry)
            elif certified_image(entry)[0]:
                self._take_image(holding, entry)
            else:
                self._take_system(holding, entry)
        finally:
            self.phases_by.pop(holding.gpu, None)

    def _take_shim(self, holding: plan.Holding, entry: Mapping[str, Any]) -> None:
        url = self.shim_urls[holding.gpu]
        self.phases_by[holding.gpu] = protocol.LOADING
        deadline = time.monotonic() + self.config.engine_ready_seconds
        served = ""
        while True:
            try:
                with httpx.Client(timeout=5.0) as http:
                    response = http.get(url + "/health")
                if response.status_code == 200:
                    served = str(dict(response.json()).get("model") or "")
                    break
            except (httpx.HTTPError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise HostError(f"the shim at {url} did not become ready")
            time.sleep(5.0)
        if served and served != holding.model:
            raise HostError(f"the shim at {url} serves {served}, not {holding.model}")
        self.images_held[holding.gpu] = ImageHeld(
            model=holding.model,
            artifact_digest=holding.artifact_digest,
            image_digest=certified_image(entry)[1],
            entry=dict(entry),
            gpu=holding.gpu,
            slot=self.devices.index(holding.gpu),
            container="",
            network="",
            url=url,
        )
        if self.declare is not None:
            self.declare(holding.model, holding.artifact_digest)
        log.info("%s ready on gpu %s from the shim at %s", holding.model, holding.gpu, url)

    def _take_system(self, holding: plan.Holding, entry: Mapping[str, Any]) -> None:
        fetched = self.fetcher(entry, self.config.cache_dir)
        slot = self.devices.index(holding.gpu)
        sandboxes = self.engine_for(entry)
        self.phases_by[holding.gpu] = protocol.LOADING
        sandbox = sandboxes.open(
            artifact_dir=fetched.artifact_dir,
            slm_dir=fetched.slm_dir,
            router_dir=fetched.router_dir,
            gpu=holding.gpu,
            config=fetched.config,
            sample_database=b"",
            fault_plan=None,
            desk=None,
            slot=slot,
            databases={},
            item_databases={},
        )
        self.held[holding.gpu] = Held(
            model=holding.model,
            artifact_digest=holding.artifact_digest,
            entry=dict(entry),
            sandbox=sandbox,
            gpu=holding.gpu,
            slot=slot,
        )
        self.opened_by[holding.gpu] = sandboxes
        if self.declare is not None:
            self.declare(holding.model, holding.artifact_digest)
        log.info("%s ready on gpu %s", holding.model, holding.gpu)

    def _take_image(self, holding: plan.Holding, entry: Mapping[str, Any]) -> None:
        slot = self.devices.index(holding.gpu)
        held = open_image(
            self.docker,
            entry,
            gpu=holding.gpu,
            slot=slot,
            ready_seconds=self.config.engine_ready_seconds,
        )
        if held.artifact_digest.lower() != holding.artifact_digest.lower():
            close_image(self.docker, held)
            raise HostError(f"{holding.model} is certified under another artifact than the plan")
        self.images_held[holding.gpu] = held
        if self.declare is not None:
            self.declare(holding.model, holding.artifact_digest)
        log.info("%s ready on gpu %s from its certified image", holding.model, holding.gpu)

    def _take_model(self, holding: plan.Holding, entry: Mapping[str, Any]) -> None:
        model = holding.model
        if not self.accept_licences:
            licence = str(entry.get("licence") or "an unnamed licence")
            link = str(entry.get("licence_url") or "no published link")
            raise HostError(
                f"{model} is published under {licence} ({link}); "
                "read it and pass --accept-licences to hold it"
            )
        fetched = self.model_fetcher(entry, self.config.cache_dir)
        if fetched.artifact_digest != holding.artifact_digest.lower().removeprefix(DIGEST_PREFIX):
            raise HostError(f"{model} fetched under a different digest than the plan holds")
        self.phases_by[holding.gpu] = protocol.LOADING
        if self.engine_url:
            held = attach_model(fetched, entry, self.engine_url, gpu=holding.gpu)
        else:
            held = open_model(
                self.docker,
                fetched,
                entry,
                gpu=holding.gpu,
                images=self.config.images,
                mem_fraction=self.config.mem_fraction,
                ready_seconds=self.config.engine_ready_seconds,
            )
        self.plain[holding.gpu] = held
        if self.declare is not None:
            self.declare(model, holding.artifact_digest)
        log.info("%s ready on gpu %s", model, holding.gpu)

    def apply(self, found: plan.Plan) -> None:
        wanted = plan.wanted_of(found)
        for gpu in found.drop:
            self.release(gpu)
        for holding in found.hold:
            pair = (holding.model, holding.artifact_digest)
            if self.holding().get(holding.gpu) == pair:
                continue
            if holding.gpu in self.holding():
                self.release(holding.gpu)
            try:
                self.take_up(holding)
            except SERVE_FAILURES as exc:
                log.warning("%s could not be served on gpu %s: %s", holding.model, holding.gpu, exc)
        for gpu in list(self.holding()):
            if gpu not in wanted:
                self.release(gpu)

    def handler(
        self,
        gpu: str,
        model: str,
        request: Mapping[str, Any],
        on_first: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        device = self.device_of(gpu)
        image = self.images_held.get(device)
        if image is not None and image.model == model:
            return answer_image(image, request, timeout=self.config.request_seconds)
        held = self.held.get(device)
        if held is None or held.model != model:
            return {"error": "this operator does not hold that system on that gpu"}
        return answer(held, request, timeout=self.config.request_seconds, on_first=on_first)

    async def chat(
        self, gpu: str, model: str, request: Mapping[str, Any], on_chunk: wire.OnChunk
    ) -> dict[str, Any]:
        held = self.plain.get(self.device_of(gpu))
        if held is None or held.model != model:
            return {"error": "this operator does not hold that model on that gpu"}
        return await chat_model(held, request, on_chunk, timeout=self.config.request_seconds)

    def agent(self) -> dict[str, Any]:
        return {"version": RELEASE_VERSION, "engines": dict(self.config.images)}

    def pending_update(self) -> update.Release | None:
        if self.updater is None or not self.updater.due(self.config.review_seconds):
            return None
        try:
            return self.updater.pending()
        except Exception as exc:
            log.warning("update check failed: %s", exc)
            return None

    def apply_update(self) -> bool:
        if self.updater is None:
            return False
        outcome = self.updater.check()
        if outcome.get("applied"):
            log.warning("a newer agent image was pulled; this process will be replaced")
            return True
        if outcome.get("reason") and self.updater.last_error:
            log.warning("update check: %s", outcome["reason"])
        return False

    def settings(self) -> wire.Settings | None:
        offered = [
            wire.Served(
                model=held.model,
                artifact_digest=held.artifact_digest,
                gpu=self.card(gpu).id,
                concurrency=1,
            )
            for gpu, held in self.held.items()
        ]
        offered += [
            wire.Served(
                model=held.model,
                artifact_digest=held.artifact_digest,
                gpu=self.card(gpu).id,
                concurrency=1,
                image_digest=held.image_digest,
            )
            for gpu, held in self.images_held.items()
        ]
        offered += [
            wire.Served(
                model=held.model,
                artifact_digest=held.artifact_digest,
                gpu=self.card(gpu).id,
                concurrency=(
                    min(held.concurrency, self.config.max_concurrency)
                    if self.config.max_concurrency
                    else held.concurrency
                ),
                kind="model",
                context=held.context,
                tool_calling=held.tool_calling,
                vision=held.vision,
                engine=self.config.images.get(held.spec.backend, held.spec.backend),
            )
            for gpu, held in self.plain.items()
        ]
        serves = tuple(sorted(offered, key=lambda served: (served.gpu, served.model)))
        if not serves and not self.cards:
            return None
        return wire.Settings(
            gateway=self.gateway,
            hotkey=self.hotkey,
            serves=serves,
            gpus=tuple(card.announced() for card in self.cards),
            worker=self.worker,
            signer=self.signer,
            agent=self.agent(),
            stats=self.stats,
            phases=self.phases,
            on_assign=self.assign,
        )

    def _dial(self, settings: wire.Settings, review: asyncio.Event) -> asyncio.Task[None]:
        return asyncio.create_task(
            wire.run(settings, self.handler, chat=self.chat, stop=review, drain=self.drain)
        )

    def _watch_signals(self, halt: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()

        def stopping() -> None:
            log.warning("stop requested; draining open requests")
            self.drain.set()
            halt.set()

        for name in ("SIGTERM", "SIGINT"):
            number = getattr(signal, name, None)
            if number is None:
                continue
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.add_signal_handler(number, stopping)

    async def run(self, *, stop: asyncio.Event | None = None) -> None:
        self.start()
        halt = stop or asyncio.Event()
        self._watch_signals(halt)
        dial: asyncio.Task[None] | None = None
        review = asyncio.Event()
        try:
            while not halt.is_set():
                pending = await asyncio.to_thread(self.pending_update)
                if pending is not None:
                    log.warning("agent image %s is published; draining first", pending.digest[:19])
                    self.drain.set()
                    if dial is not None:
                        await asyncio.gather(dial, return_exceptions=True)
                        dial = None
                    if await asyncio.to_thread(self.apply_update):
                        halt.set()
                        break
                    self.drain = asyncio.Event()
                try:
                    offers = await asyncio.to_thread(self.catalogue, self.server)
                except Exception as exc:
                    log.warning("catalogue unreadable (%s); keeping what is held", exc)
                    offers = list(self.entries.values())
                found = self.plan_now(offers)
                for model, reason in sorted(found.reasons.items()):
                    log.info("skip %s: %s", model, reason)
                before = self.settings()
                if (dial is None or dial.done()) and before is not None:
                    review = asyncio.Event()
                    dial = self._dial(before, review)
                if plan.changed(found, self.holding()):
                    await asyncio.to_thread(self.apply, found)
                settings = self.settings()
                if dial is None or dial.done() or settings != before:
                    if dial is not None:
                        review.set()
                        dial.cancel()
                        await asyncio.gather(dial, return_exceptions=True)
                        dial = None
                    if settings is None:
                        log.warning(
                            "nothing to serve; reviewing in %.0fs", self.config.review_seconds
                        )
                    else:
                        review = asyncio.Event()
                        dial = self._dial(settings, review)
                if self.drain.is_set():
                    break
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(halt.wait(), timeout=self.config.review_seconds)
        finally:
            if dial is not None:
                if self.drain.is_set():
                    await asyncio.gather(dial, return_exceptions=True)
                review.set()
                dial.cancel()
                await asyncio.gather(dial, return_exceptions=True)
            self.stop()


__all__ = [
    "CATALOGUE_PATH",
    "GENERAL",
    "FetchError",
    "Fetched",
    "FetchedModel",
    "Held",
    "HostConfig",
    "HostError",
    "ImageHeld",
    "PlainHeld",
    "ServingHost",
    "answer",
    "fetch",
    "fetch_model",
    "read_catalogue",
]
