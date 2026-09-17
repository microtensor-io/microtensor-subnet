from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import hmac
import logging
import secrets
import shlex
import time
from dataclasses import dataclass, field
from typing import Any

from microtensor.rigs.protocol.failures import Failure, FailureClass, Verdict, worst
from microtensor.rigs.validator.config import Settings
from microtensor.rigs.validator.pool import PoolClient, ServerError
from microtensor.rigs.validator.recovery import gpu_wedge
from microtensor.rigs.validator.schedule.express import ExpressLane, Limiter, owner_of
from microtensor.rigs.validator.scoring.reliability import UptimeLedger
from microtensor.rigs.validator.session.client import AgentClient, AgentError, Session
from microtensor.rigs.validator.session.ssh import SshError, classify
from microtensor.rigs.validator.verify import (
    challenge,
    duplicates,
    exclusivity,
    host_reality,
    specs,
    version,
)
from microtensor.rigs.validator.verify import scrape as scrape_check
from microtensor.rigs.validator.verify.payloads import source

log = logging.getLogger("validator.deep")

POLL_SECONDS = 60.0
DEFAULT_INTERVAL = 3600
VERIFIABLE_STATES = ("validating", "probation", "active", "failed")
INSPECTOR = "inspector.py"
CLEANUP_TIMEOUT = 30.0


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def summary_of(verdict: Verdict) -> str:
    if verdict.failure is not None:
        return verdict.failure.detail[:200]
    return str(verdict.evidence.get("summary", "ok"))[:200]


@dataclass
class Outcome:
    rig_id: str
    hotkey: str
    seed: str
    started_at: dt.datetime
    verdicts: list[Verdict] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    finished_at: dt.datetime | None = None
    failure: Failure | None = None

    @property
    def passed(self) -> bool:
        return self.failure is None

    @property
    def elapsed_ms(self) -> float:
        end = self.finished_at or utcnow()
        return (end - self.started_at).total_seconds() * 1000.0

    @property
    def failing_check(self) -> str:
        for verdict in self.verdicts:
            if verdict.failure is not None and verdict.failure is self.failure:
                return verdict.check
        for verdict in self.verdicts:
            if verdict.failure is not None:
                return verdict.check
        return ""

    def finish(self) -> Outcome:
        self.finished_at = utcnow()
        self.failure = worst(self.verdicts)
        self.detail["reason"] = self.failure.detail if self.failure else ""
        self.detail["failing_check"] = self.failing_check
        self.detail["elapsed_ms"] = round(self.elapsed_ms, 1)
        return self

    def payload(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failure": self.failure.kind.value if self.failure else "",
            "seed": self.seed[:80],
            "detail": self.detail,
            "checks": [
                {"check": verdict.check, "passed": verdict.passed, "value": summary_of(verdict)}
                for verdict in self.verdicts
            ],
            "started_at": self.started_at.isoformat(),
            "finished_at": (self.finished_at or utcnow()).isoformat(),
        }


def trimmed_host(host: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "init_root_differs",
        "init_os_release_readable",
        "init_hostname",
        "init_comm",
        "init_cgroup",
        "detect_virt",
        "dmi",
        "kernel",
        "boot_id",
        "machine_id",
        "init_environ_container",
        "dockerenv",
        "host_dockerenv",
    )
    return {key: host.get(key) for key in keep if key in host}


def trimmed_scrape(data: dict[str, Any]) -> dict[str, Any]:
    nvml = data.get("nvml") or {}
    host = data.get("host") or {}
    probes = data.get("probes") or {}
    return {
        "driver": nvml.get("driver"),
        "cuda": nvml.get("cuda"),
        "nvml_version": nvml.get("nvml_version"),
        "nvml_errors": nvml.get("errors"),
        "hashes": data.get("hashes"),
        "docker": data.get("docker"),
        "probes": {
            "sysbox": (probes.get("sysbox") or {}).get("ok"),
            "sysbox_detail": {
                k: v for k, v in (probes.get("sysbox") or {}).items() if k != "stdout"
            },
            "storage_quota": (probes.get("storage_quota") or {}).get("ok"),
            "idmapped": probes.get("idmapped"),
            "image": probes.get("image"),
        },
        "host": {
            "cpu": host.get("cpu"),
            "memory": host.get("memory"),
            "disk": host.get("disk"),
            "os": host.get("os"),
            "boot_id": host.get("boot_id"),
            "uptime_seconds": host.get("uptime_seconds"),
            "nvidia_devices": host.get("nvidia_devices"),
        },
        "gpus": [
            {
                key: value
                for key, value in card.items()
                if key
                in (
                    "index",
                    "name",
                    "uuid",
                    "memory_total_mb",
                    "memory_used_mb",
                    "compute_capability",
                    "power_limit_w",
                    "power_default_w",
                    "utilization",
                    "memory_utilization",
                    "mig_mode",
                    "virtualization",
                    "pci_bus_id",
                    "minor",
                    "errors",
                )
            }
            for card in data.get("gpus") or []
        ],
        "elapsed_ms": data.get("elapsed_ms"),
    }


class DeepPass:
    def __init__(
        self,
        settings: Settings,
        pool: PoolClient,
        agents: AgentClient,
        ledger: UptimeLedger,
        express: ExpressLane | None = None,
    ) -> None:
        self.settings = settings
        self.pool = pool
        self.agents = agents
        self.ledger = ledger
        self.express = express
        self.limiter = Limiter(settings.max_inflight, settings.per_miner)
        self.busy: set[str] = set()
        self.server_interval = DEFAULT_INTERVAL
        self.last_outcomes: dict[str, Outcome] = {}

    @property
    def interval(self) -> int:
        return self.settings.deep_seconds or self.server_interval or DEFAULT_INTERVAL

    def locally_due(self, rig: dict[str, Any], now: dt.datetime) -> bool:
        if str(rig.get("state", "")) not in VERIFIABLE_STATES or not rig.get("online", False):
            return False
        last = str((rig.get("last_pass") or {}).get("at", "") or "")
        if not last:
            return True
        try:
            moment = dt.datetime.fromisoformat(last)
        except ValueError:
            return True
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=dt.timezone.utc)
        return (now - moment).total_seconds() >= self.interval

    async def due_rigs(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        roster = await self.pool.rigs()
        roster_rigs = [rig for rig in roster.get("rigs") or [] if isinstance(rig, dict)]
        self.server_interval = int(
            roster.get("deep_pass_seconds", DEFAULT_INTERVAL) or DEFAULT_INTERVAL
        )
        min_version = str(roster.get("agent_min_version", "") or "")
        if self.settings.deep_seconds:
            now = utcnow()
            due = [rig for rig in roster_rigs if self.locally_due(rig, now)]
        else:
            answer = await self.pool.rigs(due=True)
            due = [rig for rig in answer.get("rigs") or [] if isinstance(rig, dict)]
            min_version = str(answer.get("agent_min_version", min_version) or min_version)
        chosen = {str(rig.get("id", "")) for rig in due}
        early = self.express.take_early_due() if self.express is not None else set()
        for rig in roster_rigs:
            rig_id = str(rig.get("id", ""))
            if (
                rig_id in early
                and rig_id not in chosen
                and str(rig.get("state", "")) in VERIFIABLE_STATES
            ):
                due.append(rig)
                chosen.add(rig_id)
        return due, roster_rigs, min_version

    async def run_once(self) -> int:
        due, roster_rigs, min_version = await self.due_rigs()
        pending = [rig for rig in due if str(rig.get("id", "")) not in self.busy]
        if not pending:
            return 0
        log.info(
            "deep pass on %s rigs (%s due, interval %ss)", len(pending), len(due), self.interval
        )
        await asyncio.gather(
            *(self.guarded(rig, roster_rigs, min_version) for rig in pending),
            return_exceptions=True,
        )
        return len(pending)

    async def guarded(
        self, rig: dict[str, Any], roster_rigs: list[dict[str, Any]], min_version: str
    ) -> Outcome | None:
        rig_id = str(rig.get("id", ""))
        self.busy.add(rig_id)
        try:
            async with self.limiter.slot(owner_of(rig)):
                return await self.verify(rig, roster_rigs, min_version)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("deep pass on rig %s aborted: %s", rig_id, exc)
            return None
        finally:
            self.busy.discard(rig_id)

    async def verify(
        self, rig: dict[str, Any], roster_rigs: list[dict[str, Any]], min_version: str
    ) -> Outcome | None:
        rig_id = str(rig.get("id", ""))
        request = challenge.new_request()
        outcome = Outcome(
            rig_id=rig_id, hotkey=owner_of(rig), seed=str(request.seed), started_at=utcnow()
        )
        try:
            detail = await self.pool.rig(rig_id)
        except ServerError as exc:
            log.warning(
                "rig %s: detail unavailable, verifying from the roster view: %s", rig_id, exc.detail
            )
            detail = dict(rig)
        if "connection" not in detail:
            detail["connection"] = rig.get("connection") or {}
        try:
            await self.pipeline(detail, roster_rigs, min_version, request, outcome)
        except challenge.ChallengeUnavailable as exc:
            log.error("rig %s: deep pass refused to run: %s", rig_id, exc)
            return None
        outcome.finish()
        await self.report(outcome)
        return outcome

    async def pipeline(
        self,
        rig: dict[str, Any],
        roster_rigs: list[dict[str, Any]],
        min_version: str,
        request: Any,
        outcome: Outcome,
    ) -> None:
        seed = outcome.seed
        verdicts = outcome.verdicts
        detail = outcome.detail
        version_verdict, answer = await version.fetch(self.agents, rig, min_version, seed)
        verdicts.append(version_verdict)
        detail["agent_version"] = str(answer.get("version", "") or "")
        if version_verdict.failure is not None:
            return
        try:
            async with self.agents.open(rig) as session:
                await self.in_session(session, rig, roster_rigs, request, outcome)
        except AgentError as exc:
            verdicts.append(
                Verdict(
                    "session", failure=exc.failure(seed), evidence={"summary": exc.detail[:120]}
                )
            )

    async def in_session(
        self,
        session: Session,
        rig: dict[str, Any],
        roster_rigs: list[dict[str, Any]],
        request: Any,
        outcome: Outcome,
    ) -> None:
        seed = outcome.seed
        verdicts = outcome.verdicts
        detail = outcome.detail
        settings = self.settings
        verdicts.append(session.attestation_verdict(seed))
        detail["attestation"] = session.attestation_summary()
        session_dir = f"{session.grant.root_dir.rstrip('/')}/mt-{request.seed:016x}"
        session.dir = session_dir
        made = await session.run(
            f"mkdir -p {shlex.quote(session_dir)} && chmod 700 {shlex.quote(session_dir)}",
            CLEANUP_TIMEOUT,
        )
        crash = classify(made, seed)
        if crash is not None:
            verdicts.append(
                Verdict(
                    "session",
                    failure=crash,
                    evidence={"summary": "could not create the session directory"},
                )
            )
            return
        try:
            host_verdict, host = await host_reality.run(
                session, session_dir, settings.check_timeout, seed
            )
            verdicts.append(host_verdict)
            detail["host"] = trimmed_host(host)
            if host_verdict.failure is not None and host_verdict.failure.kind in (
                FailureClass.SSH_TRANSPORT,
            ):
                return

            scraped = await scrape_check.run(session, settings.check_timeout, seed)
            verdicts.append(scraped.verdict)
            gpu_uuids = [
                str(card.get("uuid", "")) for card in rig.get("gpus") or [] if card.get("uuid")
            ]
            if scraped.data is not None and scraped.spec is not None:
                detail["spec"] = scraped.spec
                detail["scrape"] = trimmed_scrape(scraped.data)
                spec_verdict = specs.check(scraped.data, rig, seed)
                verdicts.append(spec_verdict)
                detail["specs"] = spec_verdict.evidence
                dup_verdict = duplicates.check(
                    scraped.data.get("gpus") or [], str(rig.get("id", "")), roster_rigs, seed
                )
                verdicts.append(dup_verdict)
                detail["duplicates"] = dup_verdict.evidence
                seen = [
                    str(card.get("uuid", ""))
                    for card in scraped.data.get("gpus") or []
                    if card.get("uuid")
                ]
                gpu_uuids = [uuid for uuid in gpu_uuids if uuid in seen] or gpu_uuids
            elif (
                scraped.verdict.failure is not None
                and scraped.verdict.failure.kind is FailureClass.SSH_TRANSPORT
            ):
                return

            challenge_verdict = await challenge.run(
                session, settings, session_dir, request, challenge.tier_of(rig), gpu_uuids
            )
            verdicts.append(challenge_verdict)
            detail["challenge"] = {
                k: v for k, v in challenge_verdict.evidence.items() if k != "digest"
            }
            if (
                challenge_verdict.failure is not None
                and challenge_verdict.failure.kind is FailureClass.SSH_TRANSPORT
            ):
                return

            inspector_verdict, inspected = await self.inspect(session, session_dir, seed)
            verdicts.append(inspector_verdict)
            detail["inspector"] = inspector_verdict.evidence
            if (
                inspector_verdict.failure is not None
                and inspector_verdict.failure.kind is FailureClass.SSH_TRANSPORT
            ):
                return

            ours, listing_error = await exclusivity.own_containers(session)
            alive = (
                {
                    int(p.get("pid", 0) or 0)
                    for p in inspected.get("processes") or []
                    if isinstance(p, dict)
                }
                if inspected
                else None
            )
            gpus = (scraped.data or {}).get("gpus") or []
            exclusivity_verdict = exclusivity.check(
                gpus, exclusivity.jobs_of(rig), ours, seed, alive_pids=alive
            )
            if listing_error:
                exclusivity_verdict.evidence["listing_error"] = listing_error[:200]
            verdicts.append(exclusivity_verdict)
            detail["exclusivity"] = exclusivity_verdict.evidence

            if worst(verdicts) is None:
                sweep = await gpu_wedge.sweep(session)
                detail["recovery"] = {"gpu_wedge": sweep.payload()}
        finally:
            await session.run(f"rm -rf {shlex.quote(session_dir)}", CLEANUP_TIMEOUT)

    async def inspect(
        self, session: Session, session_dir: str, seed: str
    ) -> tuple[Verdict, dict[str, Any]]:
        secret = secrets.token_hex(32)
        remote = f"{session_dir}/{INSPECTOR}"
        try:
            await session.upload(source(INSPECTOR), remote, 0o600)
            result, parsed = await session.run_json(
                f"{shlex.quote(session.python)} -I {shlex.quote(remote)} {secret} {session.nonce}",
                self.settings.check_timeout,
            )
        except SshError as exc:
            return Verdict("inspector", failure=exc.failure(seed)), {}
        if parsed is None:
            crash = classify(result, seed) or Failure(
                FailureClass.AGENT_CRASH, "the inspector printed no result", seed=seed
            )
            return Verdict("inspector", failure=crash, elapsed_ms=result.elapsed_ms), {}
        expected = hmac.new(
            bytes.fromhex(secret), session.nonce.encode("ascii"), hashlib.sha256
        ).hexdigest()
        reply = str(parsed.get("handshake", "") or "")
        evidence: dict[str, Any] = {
            "handshake_ok": hmac.compare_digest(expected, reply),
            "processes": len(parsed.get("processes") or []),
            "containers": len(parsed.get("containers") or []),
            "containers_error": str(parsed.get("containers_error", "") or "")[:200],
            "summary": f"{len(parsed.get('processes') or [])} processes, {len(parsed.get('containers') or [])} containers",
        }
        if not evidence["handshake_ok"]:
            return (
                Verdict(
                    "inspector",
                    failure=Failure(
                        FailureClass.AGENT_CRASH,
                        "the inspector handshake does not match this session",
                        seed=seed,
                    ),
                    evidence=evidence,
                    elapsed_ms=result.elapsed_ms,
                ),
                {},
            )
        return Verdict("inspector", evidence=evidence, elapsed_ms=result.elapsed_ms), parsed

    async def report(self, outcome: Outcome) -> None:
        self.last_outcomes[outcome.rig_id] = outcome
        posted = "posted"
        try:
            await self.pool.verification(outcome.rig_id, outcome.payload())
        except ServerError as exc:
            posted = f"refused ({exc.status}: {exc.detail[:120]})"
        log.info(
            "verification rig=%s hotkey=%s class=%s seed=%s elapsed_ms=%.0f failing=%s reason=%s %s",
            outcome.rig_id,
            outcome.hotkey,
            outcome.failure.kind.value if outcome.failure else "PASS",
            outcome.seed,
            outcome.elapsed_ms,
            outcome.failing_check or "-",
            (outcome.failure.detail if outcome.failure else "")[:160].replace("\n", " "),
            posted,
        )

    async def loop(self, gate: Any = None) -> None:
        while True:
            if gate is not None:
                await gate()
            started = time.monotonic()
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("deep pass loop failed: %s", exc)
            await asyncio.sleep(max(5.0, POLL_SECONDS - (time.monotonic() - started)))
