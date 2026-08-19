from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from microtensor.chain.wallet import verify_bytes
from microtensor.coordinator.collect import (
    ReportRejected,
    intake,
    quorum_reached,
)
from microtensor.coordinator.config import config_hash, served_config
from microtensor.coordinator.report import Report, canonical
from microtensor.coordinator.reputation import update as update_standing
from microtensor.coordinator.settle import Entry
from microtensor.coordinator.settle import build as build_settlement
from microtensor.coordinator.store import CoordinatorStore
from microtensor.coordinator.tokens import TOKEN_HEADER, KeyRing, TokenInvalid
from microtensor.coordinator.tokens import verify as verify_token
from microtensor.core.constants import (
    COORDINATOR_QUORUM,
    REPORT_MAX_BYTES,
)

log = logging.getLogger("microtensor.coordinator")

SIGNATURE_HEADER = "x-mt-signature"
HOTKEY_HEADER = "x-mt-hotkey"
TIMESTAMP_HEADER = "x-mt-timestamp"
CLOCK_SKEW_SECONDS = 120

UNKNOWN_HOTKEY = "hotkey is not registered on this subnet"
NO_PERMIT = "hotkey holds no validator permit"
BAD_SIGNATURE = "signature does not verify against the declaring hotkey"
STALE_REQUEST = "request timestamp is outside the accepted window"
TOO_LARGE = "report exceeds the size limit"


class Unauthorised(RuntimeError):
    pass


@dataclass(slots=True)
class Registry:
    """Who may talk to the coordinator, read from the metagraph.

    Kept behind an interface so the API can be exercised without a chain. The
    production implementation refreshes from the metagraph; the rule it
    enforces is the same either way.
    """

    permitted: dict[str, int]

    def uid(self, hotkey: str) -> int | None:
        return self.permitted.get(hotkey)

    def authorise(self, hotkey: str) -> int:
        uid = self.uid(hotkey)
        if uid is None:
            raise Unauthorised(UNKNOWN_HOTKEY)
        return uid


def signing_bytes(method: str, path: str, timestamp: str, body: bytes) -> bytes:
    """What a worker signs. Method and path are covered so a signed report
    cannot be replayed against a different endpoint."""
    return b"\n".join([method.upper().encode(), path.encode(), timestamp.encode(), body])


def verify_request(
    *,
    method: str,
    path: str,
    hotkey: str,
    timestamp: str,
    signature: str,
    body: bytes,
    registry: Registry,
    now: float | None = None,
) -> int:
    """Authenticate one request and return the caller's uid."""
    uid = registry.authorise(hotkey)

    try:
        stamped = float(timestamp)
    except ValueError as exc:
        raise Unauthorised(STALE_REQUEST) from exc

    if abs((now if now is not None else time.time()) - stamped) > CLOCK_SKEW_SECONDS:
        raise Unauthorised(STALE_REQUEST)

    if not verify_bytes(hotkey, signing_bytes(method, path, timestamp, body), signature):
        raise Unauthorised(BAD_SIGNATURE)

    return uid


@dataclass(slots=True)
class Coordinator:
    """The service logic, independent of the HTTP layer.

    Every endpoint below is a thin call onto this, so the whole surface is
    testable without starting a server.
    """

    store: CoordinatorStore
    registry: Registry
    keyring: KeyRing | None = None
    corpus_version: str = ""
    engine_version: str = ""
    corpora: dict[str, Any] = None  # type: ignore[assignment]
    catalogue: dict[str, Entry] = None  # type: ignore[assignment]
    coldkeys: dict[str, str] = None  # type: ignore[assignment]
    uid_by_hotkey: dict[str, int] = None  # type: ignore[assignment]
    reserve: Callable[[], dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if not self.catalogue:
            self.catalogue = {}
        if not self.coldkeys:
            self.coldkeys = {}
        if not self.uid_by_hotkey:
            self.uid_by_hotkey = {}
        if not self.corpora:
            self.corpora = {}

    def current_round(self) -> dict[str, Any]:
        row = self.store.latest_round()
        if row is None:
            return {}
        config = served_config(self.corpus_version or "")
        return {
            "corpus_digest": self.corpus_digest(),
            "round": row["round_index"],
            "seed_block": row["seed_block"],
            "close_block": row["close_block"],
            "block_hash": row["block_hash"],
            "config_hash": row["config_hash"] or config_hash(config),
            "anchored": bool(row["anchored_at"]),
            "config": config,
        }

    def assignment(self, hotkey: str) -> dict[str, Any]:
        """This worker's share of the round, or no document at all.

        The `systems` key is present only when the round has actually been
        assigned. Returning an empty list before `mt coordinator assign` has run
        would tell every worker it was legitimately idle, so nobody would
        measure, quorum would never be reached, and the whole fleet would abstain
        while logging that everything was fine. Absence of the key is how a
        worker tells "you have nothing to do" from "there is nothing to do yet".
        """
        row = self.store.latest_round()
        if row is None:
            return {"round": None}

        index = int(row["round_index"])
        if self.store.expected_reports(index) == 0:
            return {"round": index}

        return {
            "round": index,
            "systems": self.store.assignment_for(index, hotkey),
        }

    def settlement(self, round_index: int) -> dict[str, Any] | None:
        return self.store.settlement(round_index)

    def reports(self, round_index: int) -> list[dict[str, Any]]:
        """Published so the settlement is recomputable by anyone."""
        return self.store.reports_payload(round_index)

    def health(self) -> dict[str, Any]:
        row = self.store.latest_round()
        if row is None:
            return {"round": None, "workers": 0, "quorum": False}
        index = int(row["round_index"])
        expected = self.store.expected_reports(index)
        received = self.store.report_count(index)
        return {
            "round": index,
            "expected_reports": expected,
            "received_reports": received,
            "quorum": quorum_reached(expected, received, COORDINATOR_QUORUM),
            "settled": self.store.settlement(index) is not None,
            "divergence_rate": round(self.store.divergence_rate(index), 4),
            "advisory_workers": list(self.store.advisory()),
            "anchored": bool(row["anchored_at"]),
        }

    def reputation(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.store.standings()]

    def submit(self, body: dict[str, Any], raw: bytes) -> dict[str, Any]:
        if len(raw) > REPORT_MAX_BYTES:
            raise ReportRejected(TOO_LARGE)

        report = Report.from_dict(body)
        assigned = self.store.assigned_digests(report.round_index, report.worker_hotkey)
        already = self.store.reporters(report.round_index, report.system_digest)

        from microtensor.coordinator.collect import accept

        accept(
            report,
            assigned=assigned,
            engine_version=self.engine_version,
            corpus_version=self.corpus_version,
            already=already,
            corpus_digest=self.corpus_digest(),
        )

        self.store.record_report(report)
        return {"accepted": True, "digest": report.digest()}

    def settle(self, round_index: int) -> dict[str, Any] | None:
        """Reconcile, settle, and publish once quorum is reached.

        Returns None while the round is still collecting, so a caller polling
        this cannot accidentally publish a settlement built from a third of
        the measurements.

        A round with nothing to measure is the exception. Quorum over zero
        assignments is undefined rather than reached, so a subnet with no
        commitments yet would publish nothing and a declared hold would never
        pay. Such a round settles on the hold alone and says so: no reports, no
        frontier, and a weight vector that names only the reserved uid.
        """
        expected = self.store.expected_reports(round_index)
        received = self.store.report_count(round_index)
        if not quorum_reached(expected, received, COORDINATOR_QUORUM):
            if expected > 0 or received > 0:
                return None
            return self._settle_on_hold(round_index)

        existing = self.store.settlement(round_index)
        if existing is not None:
            return existing

        by_system = self.store.reports_by_system(round_index)
        advisory = self.store.advisory()
        result = intake(by_system, advisory=advisory)

        self.store.record_divergences(round_index, result.divergences)
        self._update_reputation(round_index, result)

        row = self.store.round(round_index) or {}
        assignment = self.store.full_assignment(round_index)
        from microtensor.coordinator.assign import under_replicated

        settlement = build_settlement(
            round_index,
            config_hash=str(row.get("config_hash", "")),
            corpus_version=self.corpus_version,
            reconciled=result.reconciled,
            catalogue=self.catalogue,
            report_digests=self.store.report_digests(round_index),
            unscored=result.unscored,
            under_replicated=under_replicated(assignment),
            advisory=advisory,
            coldkeys=self.coldkeys,
            previous=self._previous_weights(round_index),
            reserved=self._reserved(),
        )
        self.store.publish(settlement)
        log.info(
            "round %d settled: %d systems, %d unscored, %d divergences",
            round_index,
            len(settlement.frontier),
            len(settlement.unscored),
            len(result.divergences),
        )
        return self.store.settlement(round_index)

    def corpus_digest(self) -> str:
        from microtensor.tasks.corpus import corpus_digest

        return corpus_digest(self.corpora) if self.corpora else ""

    def corpus(self, track: str) -> dict[str, Any]:
        """The task set for one track, as the coordinator holds it.

        Served so every worker measures the same tasks. A worker that brought
        its own corpus would produce numbers nobody else can reproduce, and the
        reconciliation majority would be counting configurations rather than
        measurements.
        """
        held = self.corpora.get(track)
        if held is None:
            return {}
        return {
            "track": held.track,
            "version": held.version,
            "digest": held.digest(),
            "tasks": [
                {
                    "ref": task.ref,
                    "prompt": task.prompt,
                    "gold": task.gold,
                    "partition": task.partition,
                    "inputs": task.inputs,
                    "max_output_tokens": task.max_output_tokens,
                }
                for task in held.tasks
            ],
        }

    def _settle_on_hold(self, round_index: int) -> dict[str, Any] | None:
        existing = self.store.settlement(round_index)
        if existing is not None:
            return existing

        held = self._reserved()
        if not held:
            return None

        row = self.store.round(round_index) or {}
        settlement = build_settlement(
            round_index,
            config_hash=str(row.get("config_hash", "")),
            corpus_version=self.corpus_version,
            reconciled=(),
            catalogue={},
            report_digests=[],
            coldkeys=self.coldkeys,
            reserved=held,
        )
        self.store.publish(settlement)
        log.info(
            "round %d had nothing to measure; settled on the hold for %s alone",
            round_index,
            held["hotkey"],
        )
        return self.store.settlement(round_index)

    def _reserved(self) -> dict[str, Any]:
        """Resolve the control plane's hold against this metagraph.

        A hotkey the metagraph does not carry is dropped rather than settled
        around. The hold names a uid in the published settlement and every
        worker checks that uid against its own metagraph, so publishing one
        that cannot be resolved here would be rejected everywhere it landed.
        """
        if self.reserve is None:
            return {}

        try:
            held = self.reserve() or {}
        except Exception as exc:
            log.warning("the reserved emission could not be read (%s); holding nothing", exc)
            return {}

        hotkey = str(held.get("hotkey", ""))
        share = float(held.get("share", 0.0))
        if not hotkey or share <= 0.0:
            return {}

        uid = self.uid_by_hotkey.get(hotkey)
        if uid is None:
            log.warning(
                "%s is reserved %.2f%% of emission but is not on this metagraph; holding nothing",
                hotkey,
                share * 100,
            )
            return {}

        log.info("holding %.2f%% of emission for %s at uid %d", share * 100, hotkey, uid)
        return {"hotkey": hotkey, "uid": uid, "share": share}

    def _previous_weights(self, round_index: int) -> dict[str, float]:
        """The last published blend, so the EMA has something to smooth against.

        Taken from the coordinator's own previous settlement rather than from
        any worker's local history, because every worker adopts this vector and
        they must all smooth against the same prior.
        """
        for earlier in range(round_index - 1, max(round_index - 8, -1), -1):
            published = self.store.settlement(earlier)
            if published:
                return {str(k): float(v) for k, v in published.get("blended", {}).items()}
        return {}

    def _update_reputation(self, round_index: int, result: Any) -> None:
        diverged = {d.worker_hotkey for d in result.divergences}
        agreed = {h for r in result.reconciled for h in r.agreed}

        for hotkey in sorted(agreed | diverged):
            standing = self.store.standing(hotkey)
            self.store.save_standing(
                update_standing(standing, agreed=hotkey not in diverged, round_index=round_index)
            )


def build_app(coordinator: Coordinator) -> Any:
    """The HTTP surface. FastAPI is imported here so the module stays usable,
    and testable, on a host that has no web stack installed."""
    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as exc:  # pragma: no cover - exercised by the extra
        raise RuntimeError(
            'the coordinator API needs the web stack: pip install ".[coordinator]"'
        ) from exc

    app = FastAPI(title="Microtensor coordinator", version="1")

    def authenticate(request: Request, body: bytes) -> str:
        hotkey = str(request.headers.get(HOTKEY_HEADER, ""))
        try:
            verify_request(
                method=request.method,
                path=request.url.path,
                hotkey=hotkey,
                timestamp=str(request.headers.get(TIMESTAMP_HEADER, "")),
                signature=str(request.headers.get(SIGNATURE_HEADER, "")),
                body=body,
                registry=coordinator.registry,
            )
        except Unauthorised as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return hotkey

    def authorised(request: Request, hotkey: str) -> None:
        """Require a control plane token when one can be checked.

        Enforced only once a key is known. A coordinator running without a
        control plane has no key, no way to check a token, and no business
        refusing a worker over one; the moment it has read the key, a token
        becomes mandatory and a revoked worker stops being served on its next
        request.
        """
        keyring = coordinator.keyring
        if keyring is None or not keyring.load():
            return

        try:
            verify_token(
                str(request.headers.get(TOKEN_HEADER, "")),
                keyring.public_key,
                hotkey=hotkey,
            )
        except TokenInvalid as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.get("/v1/round/current")
    def round_current() -> dict[str, Any]:
        return coordinator.current_round()

    @app.get("/v1/corpus")
    async def corpus_index(request: Request) -> dict[str, Any]:
        hotkey = authenticate(request, b"")
        authorised(request, hotkey)
        return {
            "digest": coordinator.corpus_digest(),
            "version": coordinator.corpus_version,
            "tracks": sorted(coordinator.corpora),
        }

    @app.get("/v1/corpus/{track}")
    async def corpus_track(track: str, request: Request) -> dict[str, Any]:
        hotkey = authenticate(request, b"")
        authorised(request, hotkey)
        found = coordinator.corpus(track)
        if not found:
            raise HTTPException(status_code=404, detail=f"no corpus is served for {track!r}")
        return found

    @app.get("/v1/assignment/{hotkey}")
    async def assignment(hotkey: str, request: Request) -> dict[str, Any]:
        caller = authenticate(request, b"")
        if caller != hotkey:
            raise HTTPException(status_code=403, detail="a worker may only read its own work")
        authorised(request, caller)
        return coordinator.assignment(hotkey)

    @app.post("/v1/report")
    async def report(request: Request) -> dict[str, Any]:
        raw = await request.body()
        hotkey = authenticate(request, raw)
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="body is not valid JSON") from exc

        if body.get("worker_hotkey") != hotkey:
            raise HTTPException(status_code=403, detail="report is signed by another hotkey")

        authorised(request, hotkey)

        try:
            return coordinator.submit(body, raw)
        except ReportRejected as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/settlement/{round_index}")
    def settlement(round_index: int) -> dict[str, Any]:
        found = coordinator.settlement(round_index) or coordinator.settle(round_index)
        if found is None:
            raise HTTPException(status_code=404, detail="no settlement published yet")
        return found

    @app.get("/v1/reports/{round_index}")
    def reports(round_index: int) -> dict[str, Any]:
        return {"round": round_index, "reports": coordinator.reports(round_index)}

    @app.get("/v1/frontier/{track}/{hardware_class}")
    def frontier(track: str, hardware_class: str) -> dict[str, Any]:
        row = coordinator.store.latest_round()
        if row is None:
            return {"frontier": []}
        published = coordinator.store.settlement(int(row["round_index"])) or {}
        entries = [
            e
            for e in published.get("frontier", [])
            if e.get("track") == track and e.get("class") == hardware_class
        ]
        return {"round": published.get("round"), "frontier": entries}

    @app.get("/v1/reputation")
    def reputation() -> dict[str, Any]:
        return {"workers": coordinator.reputation()}

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return coordinator.health()

    return app


__all__ = [
    "Coordinator",
    "Registry",
    "Unauthorised",
    "build_app",
    "canonical",
    "signing_bytes",
    "verify_request",
]
