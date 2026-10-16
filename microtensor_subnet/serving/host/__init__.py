from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import httpx

from microtensor_subnet.core.hashing import DIGEST_PREFIX
from microtensor_subnet.serving import plan, wire
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
from microtensor_subnet.serving.host.plain import (
    PlainError,
    PlainHeld,
    attach_model,
    close_model,
    open_model,
)
from microtensor_subnet.serving.host.plain import chat as chat_model
from microtensor_subnet.serving.tokens import TokenError
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
        self.services = Services(host_ip(), config.host_services_port)
        self.sandboxes = SandboxHost(config, self.docker, self.services, host_ip())
        self.held: dict[str, Held] = {}
        self.plain: dict[str, PlainHeld] = {}
        self.entries: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if not self._started:
            self.services.start()
            self._started = True

    def stop(self) -> None:
        for model in [*self.held, *self.plain]:
            self.release(model)
        if self._started:
            self.services.stop()
            self._started = False
        self.docker.sweep()

    def _fetch_model(self, entry: Mapping[str, Any], cache_dir: Path) -> FetchedModel:
        return fetch_model(entry, cache_dir, token=self.config.hf_token)

    def holding(self) -> dict[str, str]:
        found = {model: held.artifact_digest for model, held in self.held.items()}
        found.update({model: held.artifact_digest for model, held in self.plain.items()})
        return found

    def images(self) -> frozenset[str]:
        if self.engine_url:
            return frozenset(BACKENDS)
        return frozenset(self.config.images)

    def plan_now(self, offers: Sequence[Mapping[str, Any]]) -> plan.Plan:
        self.entries = {str(row.get("model") or row.get("id") or ""): dict(row) for row in offers}
        capacity = plan.Capacity(
            gpus=self.config.gpus, gpu_gb=self.config.gpu_gb, images=self.images()
        )
        rules = dataclasses.replace(self.policy, accept_licences=self.accept_licences)
        return plan.build(offers, capacity, held=self.holding(), policy=rules, pinned=self.pinned)

    def release(self, model: str) -> None:
        plain = self.plain.pop(model, None)
        if plain is not None:
            try:
                close_model(self.docker, plain)
            except Exception as exc:
                log.warning("closing %s failed: %s", model, exc)
            return
        held = self.held.pop(model, None)
        if held is None:
            return
        try:
            self.sandboxes.close(held.sandbox)
        except Exception as exc:
            log.warning("closing %s failed: %s", model, exc)

    def take_up(self, holding: plan.Holding) -> None:
        entry = self.entries.get(holding.model)
        if entry is None:
            raise HostError(f"{holding.model} is not in the catalogue")
        if entry.get("kind") == GENERAL:
            self._take_model(holding, entry)
            return
        fetched = self.fetcher(entry, self.config.cache_dir)
        slot = self.config.gpus.index(holding.gpu) if holding.gpu in self.config.gpus else 0
        sandbox = self.sandboxes.open(
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
        self.held[holding.model] = Held(
            model=holding.model,
            artifact_digest=holding.artifact_digest,
            entry=entry,
            sandbox=sandbox,
            gpu=holding.gpu,
            slot=slot,
        )
        if self.declare is not None:
            self.declare(holding.model, holding.artifact_digest)
        log.info("%s ready on gpu %s", holding.model, holding.gpu)

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
        self.plain[model] = held
        if self.declare is not None:
            self.declare(model, holding.artifact_digest)
        log.info("%s ready on gpu %s", model, holding.gpu)

    def apply(self, found: plan.Plan) -> None:
        for model in found.drop:
            self.release(model)
        wanted = {offer.model for offer in found.fetch}
        for holding in found.hold:
            if holding.model not in wanted:
                continue
            if holding.model in self.held or holding.model in self.plain:
                self.release(holding.model)
            try:
                self.take_up(holding)
            except (
                FetchError,
                SandboxError,
                HostError,
                PlainError,
                EngineError,
                TokenError,
                DockerError,
                OSError,
            ) as exc:
                log.warning("%s could not be served: %s", holding.model, exc)

    def handler(self, model: str, request: Mapping[str, Any]) -> dict[str, Any]:
        held = self.held.get(model)
        if held is None:
            return {"error": "this operator does not hold that system"}
        return answer(held, request, timeout=self.config.request_seconds)

    async def chat(
        self, model: str, request: Mapping[str, Any], on_chunk: wire.OnChunk
    ) -> dict[str, Any]:
        held = self.plain.get(model)
        if held is None:
            return {"error": "this operator does not hold that model"}
        return await chat_model(held, request, on_chunk, timeout=self.config.request_seconds)

    def settings(self) -> wire.Settings | None:
        offered = [
            wire.Served(model=model, artifact_digest=held.artifact_digest, concurrency=1)
            for model, held in self.held.items()
        ]
        offered += [
            wire.Served(
                model=model,
                artifact_digest=held.artifact_digest,
                concurrency=held.concurrency,
                kind="model",
                context=held.context,
                tool_calling=held.tool_calling,
                vision=held.vision,
            )
            for model, held in self.plain.items()
        ]
        serves = tuple(sorted(offered, key=lambda served: served.model))
        if not serves:
            return None
        return wire.Settings(
            gateway=self.gateway, hotkey=self.hotkey, serves=serves, worker=self.worker
        )

    async def run(self, *, stop: asyncio.Event | None = None) -> None:
        self.start()
        halt = stop or asyncio.Event()
        try:
            while not halt.is_set():
                try:
                    offers = await asyncio.to_thread(self.catalogue, self.server)
                except Exception as exc:
                    log.warning("catalogue unreadable (%s); keeping what is held", exc)
                    offers = list(self.entries.values())
                found = self.plan_now(offers)
                for model, reason in sorted(found.reasons.items()):
                    log.info("skip %s: %s", model, reason)
                await asyncio.to_thread(self.apply, found)
                settings = self.settings()
                if settings is None:
                    log.warning("nothing to serve; reviewing in %.0fs", self.config.review_seconds)
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(halt.wait(), timeout=self.config.review_seconds)
                    continue
                review = asyncio.Event()
                dial = asyncio.create_task(
                    wire.run(settings, self.handler, chat=self.chat, stop=review)
                )
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(halt.wait(), timeout=self.config.review_seconds)
                fresh = await asyncio.to_thread(self._refresh)
                if halt.is_set() or (fresh is not None and plan.changed(fresh, self.holding())):
                    review.set()
                    dial.cancel()
                    await asyncio.gather(dial, return_exceptions=True)
                    continue
                review.set()
                dial.cancel()
                await asyncio.gather(dial, return_exceptions=True)
        finally:
            self.stop()

    def _refresh(self) -> plan.Plan | None:
        try:
            return self.plan_now(self.catalogue(self.server))
        except Exception as exc:
            log.warning("catalogue unreadable at review (%s)", exc)
            return None


__all__ = [
    "CATALOGUE_PATH",
    "GENERAL",
    "FetchError",
    "Fetched",
    "FetchedModel",
    "Held",
    "HostConfig",
    "HostError",
    "PlainHeld",
    "ServingHost",
    "answer",
    "fetch",
    "fetch_model",
    "read_catalogue",
]
