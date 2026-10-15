from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from microtensor_subnet.serving import plan, wire
from microtensor_subnet.serving.host.answer import Held, answer
from microtensor_subnet.serving.host.config import HostConfig
from microtensor_subnet.serving.host.fetch import Fetched, FetchError, fetch
from microtensor_subnet.validator.sandbox import SandboxError, SandboxHost, host_ip
from microtensor_subnet.validator.sandbox.docker import Docker
from microtensor_subnet.validator.sandbox.services import Services

log = logging.getLogger("microtensor_subnet.serving.host")

CATALOGUE_PATH = "/v1/public/inference/catalogue"
Declare = Callable[[str, str], None]
Catalogue = Callable[[str], list[dict[str, Any]]]


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
        self.services = Services(host_ip(), config.host_services_port)
        self.sandboxes = SandboxHost(config, self.docker, self.services, host_ip())
        self.held: dict[str, Held] = {}
        self.entries: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if not self._started:
            self.services.start()
            self._started = True

    def stop(self) -> None:
        for model in list(self.held):
            self.release(model)
        if self._started:
            self.services.stop()
            self._started = False
        self.docker.sweep()

    def holding(self) -> dict[str, str]:
        return {model: held.artifact_digest for model, held in self.held.items()}

    def plan_now(self, offers: Sequence[Mapping[str, Any]]) -> plan.Plan:
        self.entries = {str(row.get("model") or row.get("id") or ""): dict(row) for row in offers}
        capacity = plan.Capacity(gpus=self.config.gpus, gpu_gb=self.config.gpu_gb)
        return plan.build(
            offers, capacity, held=self.holding(), policy=self.policy, pinned=self.pinned
        )

    def release(self, model: str) -> None:
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

    def apply(self, found: plan.Plan) -> None:
        for model in found.drop:
            self.release(model)
        wanted = {offer.model for offer in found.fetch}
        for holding in found.hold:
            if holding.model not in wanted or holding.model in self.held:
                continue
            try:
                self.take_up(holding)
            except (FetchError, SandboxError, HostError, OSError) as exc:
                log.warning("%s could not be served: %s", holding.model, exc)

    def handler(self, model: str, request: Mapping[str, Any]) -> dict[str, Any]:
        held = self.held.get(model)
        if held is None:
            return {"error": "this operator does not hold that system"}
        return answer(held, request, timeout=self.config.request_seconds)

    def settings(self) -> wire.Settings | None:
        serves = tuple(
            wire.Served(model=model, artifact_digest=held.artifact_digest, concurrency=1)
            for model, held in sorted(self.held.items())
        )
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
                dial = asyncio.create_task(wire.run(settings, self.handler, stop=review))
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
    "FetchError",
    "Fetched",
    "Held",
    "HostConfig",
    "HostError",
    "ServingHost",
    "answer",
    "fetch",
    "read_catalogue",
]
