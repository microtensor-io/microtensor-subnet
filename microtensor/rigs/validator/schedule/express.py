from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from microtensor.rigs.validator.config import Settings
from microtensor.rigs.validator.pool import PoolClient, ServerError
from microtensor.rigs.validator.scoring.reliability import UptimeLedger
from microtensor.rigs.validator.session.client import AgentClient, AgentError

log = logging.getLogger("validator.express")

UNREACHABLE_TICKS = 3
BUSY_PERCENT = 50
SIGNAL_HISTORY = 500
LEDGER_SAVE_SECONDS = 60.0


@dataclass(frozen=True)
class Signal:
    rig_id: str
    hotkey: str
    kind: str
    detail: str
    at: float

    def payload(self) -> dict[str, Any]:
        return {
            "rig_id": self.rig_id,
            "hotkey": self.hotkey,
            "kind": self.kind,
            "detail": self.detail,
            "at": self.at,
        }


class Limiter:
    def __init__(self, max_inflight: int, per_miner: int) -> None:
        self._global = asyncio.Semaphore(max(1, max_inflight))
        self._per_miner = max(1, per_miner)
        self._miners: dict[str, asyncio.Semaphore] = {}

    def _miner(self, owner: str) -> asyncio.Semaphore:
        semaphore = self._miners.get(owner)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self._per_miner)
            self._miners[owner] = semaphore
        return semaphore

    @contextlib.asynccontextmanager
    async def slot(self, owner: str) -> AsyncIterator[None]:
        async with self._miner(owner or "-"), self._global:
            yield


def owner_of(rig: dict[str, Any]) -> str:
    return str(rig.get("hotkey", "") or rig.get("id", "") or "")


def utilisation_signal(rig: dict[str, Any], utilisation: dict[str, Any]) -> tuple[str, str] | None:
    placed = int(rig.get("jobs_running", 0) or 0)
    gpus = [card for card in utilisation.get("gpus") or [] if isinstance(card, dict)]
    if not gpus:
        return None
    busy = []
    for card in gpus:
        try:
            percent = int(card.get("gpu_percent", card.get("utilization", 0)) or 0)
        except (TypeError, ValueError):
            percent = 0
        processes = card.get("processes") or []
        if percent >= BUSY_PERCENT or processes:
            busy.append(str(card.get("uuid", "") or "?"))
    if placed > 0 and not busy:
        return "idle_while_placed", f"{placed} jobs placed, every card idle"
    if placed == 0 and busy:
        return "busy_without_assignment", f"cards busy with nothing assigned: {', '.join(busy[:4])}"
    return None


class ExpressLane:
    def __init__(
        self,
        settings: Settings,
        pool: PoolClient,
        agents: AgentClient,
        ledger: UptimeLedger,
        limiter: Limiter | None = None,
    ) -> None:
        self.settings = settings
        self.pool = pool
        self.agents = agents
        self.ledger = ledger
        self.limiter = limiter or Limiter(settings.max_inflight, settings.per_miner)
        self.unreachable: dict[str, int] = {}
        self.early_due: set[str] = set()
        self.signals: collections.deque[Signal] = collections.deque(maxlen=SIGNAL_HISTORY)
        self.last_probe: dict[str, dict[str, Any]] = {}
        self._last_save = 0.0

    def take_early_due(self) -> set[str]:
        taken = set(self.early_due)
        self.early_due.clear()
        return taken

    def record(self, rig: dict[str, Any], kind: str, detail: str) -> None:
        signal = Signal(str(rig.get("id", "")), owner_of(rig), kind, detail, time.time())
        self.signals.append(signal)
        log.info(
            "signal rig=%s hotkey=%s kind=%s detail=%s", signal.rig_id, signal.hotkey, kind, detail
        )

    async def probe(self, rig: dict[str, Any]) -> dict[str, Any]:
        rig_id = str(rig.get("id", ""))
        report: dict[str, Any] = {"rig_id": rig_id, "at": time.time(), "online": False}
        async with self.limiter.slot(owner_of(rig)):
            try:
                version = await self.agents.version(rig)
                ping = await self.agents.ping(rig)
                utilisation = await self.agents.utilisation(rig)
            except AgentError as exc:
                report["error"] = exc.detail
                self.ledger.observe(rig_id, False)
                count = self.unreachable.get(rig_id, 0) + 1
                self.unreachable[rig_id] = count
                if count == UNREACHABLE_TICKS:
                    self.early_due.add(rig_id)
                    self.record(
                        rig,
                        "unreachable",
                        f"{count} consecutive express ticks failed: {exc.detail[:120]}",
                    )
                self.last_probe[rig_id] = report
                return report
        self.unreachable.pop(rig_id, None)
        self.ledger.observe(rig_id, True)
        report.update(
            {
                "online": True,
                "agent_version": str(version.get("version", "") or ""),
                "draining": bool(version.get("draining", False)),
                "pong": bool(ping.get("ok", True)),
                "utilisation": utilisation,
            }
        )
        signal = utilisation_signal(rig, utilisation)
        if signal is not None:
            self.record(rig, *signal)
        self.last_probe[rig_id] = report
        return report

    async def tick(self) -> int:
        try:
            roster = await self.pool.rigs()
        except ServerError as exc:
            log.warning("express tick: roster unavailable: %s", exc.detail)
            return 0
        rigs = [
            rig
            for rig in roster.get("rigs") or []
            if rig.get("online") and str(rig.get("state", "")) != "removed"
        ]
        if rigs:
            await asyncio.gather(*(self.probe(rig) for rig in rigs), return_exceptions=True)
        if time.time() - self._last_save >= LEDGER_SAVE_SECONDS:
            self.ledger.save()
            self._last_save = time.time()
        return len(rigs)

    async def loop(self, gate: Any = None) -> None:
        while True:
            if gate is not None:
                await gate()
            started = time.monotonic()
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("express tick failed: %s", exc)
            delay = self.settings.express_seconds - (time.monotonic() - started)
            await asyncio.sleep(max(1.0, delay))
