from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from microtensor.chain.rounds import Round
from microtensor.coordinator.assign import System, Worker
from microtensor.coordinator.chain import ChainSource
from microtensor.coordinator.settle import Entry, Settlement

log = logging.getLogger("microtensor.coordinator")

CREDENTIAL_HEADER = "x-mt-credential"
BAD_SCHEME = "the server URL must be http or https"
FELL_BACK = (
    "the control plane is unreachable, so the round was derived from chain instead; "
    "the schedule is the chain's until it returns"
)


class ServerUnreachable(RuntimeError):
    """The control plane did not answer.

    Never fatal. The subnet has to keep running with the server down, so every
    caller of this falls back to chain rather than stopping.
    """


class ServerRefused(RuntimeError):
    """The control plane answered and said no.

    Distinct from unreachable because the response differs: a refusal means the
    ingest credential is wrong or revoked, and quietly carrying on would hide
    that behind a round that looks normal.
    """


@dataclass(slots=True)
class ServerClient:
    base_url: str
    credential: str = ""
    timeout: int = 30
    retries: int = 3
    backoff: float = 2.0

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        url = self.base_url.rstrip("/") + path
        if urllib.parse.urlparse(url).scheme not in ("http", "https"):
            raise ValueError(f"{BAD_SCHEME}: {self.base_url!r}")

        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode() if body else b""
        headers = {"content-type": "application/json", "accept": "application/json"}
        if self.credential:
            headers[CREDENTIAL_HEADER] = self.credential

        request = urllib.request.Request(  # noqa: S310 - scheme checked above
            url, data=raw or None, method=method, headers=headers
        )

        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(  # noqa: S310 - scheme checked above
                    request, timeout=self.timeout
                ) as response:
                    return json.loads(response.read() or b"null")
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                if exc.code in (401, 403):
                    raise ServerRefused(
                        f"{url} refused this coordinator ({exc.code}); check the ingest credential"
                    ) from exc
                if exc.code == 429 or exc.code >= 500:
                    last = exc
                    if attempt + 1 < self.retries:
                        time.sleep(self.backoff * (2**attempt))
                        continue
                    raise ServerUnreachable(f"{url} returned {exc.code}") from exc
                raise ServerRefused(f"{url} returned {exc.code}: {exc}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff * (2**attempt))

        raise ServerUnreachable(f"{url} after {self.retries} attempts: {last}")

    def current_round(self) -> dict[str, Any] | None:
        found: dict[str, Any] | None = self._call("GET", "/v1/control/round/next")
        return found

    def config(self) -> dict[str, Any] | None:
        found: dict[str, Any] | None = self._call("GET", "/v1/control/config")
        return found

    def authorised_workers(self) -> list[dict[str, Any]] | None:
        found = self._call("GET", "/v1/control/workers")
        return list(found) if found is not None else None

    def token_key(self) -> str:
        found = self._call("GET", "/v1/public/keys") or {}
        return str(found.get("token_public_key", ""))

    def reserved(self) -> dict[str, Any]:
        """The emission the control plane is holding for a named hotkey.

        An unreachable server yields no hold rather than the last one seen. A
        stale hold is a payment nobody authorised this round, and holding
        nothing is the state the mechanism already knows how to settle.
        """
        found = self._call("GET", "/v1/control/emission") or {}
        if found.get("paused"):
            return {"paused": True}
        return {
            "hotkey": str(found.get("reserved_hotkey", "")),
            "share": float(found.get("reserved_share", 0.0)),
        }

    def push_settlement(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._call("POST", "/v1/ingest/settlement", payload) or {}

    def push_reports(self, round_index: int, reports: list[dict[str, Any]]) -> dict[str, Any]:
        return (
            self._call(
                "POST", "/v1/ingest/reports", {"round_index": round_index, "reports": reports}
            )
            or {}
        )

    def push_assignments(
        self, round_index: int, assignment: Mapping[str, Sequence[str]]
    ) -> dict[str, Any]:
        return (
            self._call(
                "POST",
                "/v1/ingest/assignments",
                {
                    "round_index": round_index,
                    "assignment": {k: list(v) for k, v in assignment.items()},
                },
            )
            or {}
        )


@dataclass(slots=True)
class ServerSource:
    """The round comes from the server; the participants still come from chain.

    Only the schedule and the config move. Commitments stay a chain read
    because that is the thing nobody can forge and the thing the whole
    mechanism is anchored to: a server that could name the participants could
    quietly exclude one.

    An unreachable server falls back to the chain-derived round, so a control
    plane outage costs the published schedule and nothing else.

    Satisfies RoundSource structurally rather than by inheritance. Subclassing
    a Protocol hands the class Protocol's own __init__, which silently discards
    the dataclass one and leaves every field unset.
    """

    chain: ChainSource
    client: ServerClient | None = None
    config_hash: str = ""
    degraded: bool = False

    def open_round(self) -> Round:
        if self.client is None:
            return self.chain.open_round()

        try:
            current = self.client.current_round()
        except ServerUnreachable as exc:
            if not self.degraded:
                log.warning("%s (%s)", FELL_BACK, exc)
                self.degraded = True
            return self.chain.open_round()

        if not current:
            log.info("the control plane has opened no round; deriving it from chain")
            return self.chain.open_round()

        if self.degraded:
            log.info("the control plane is back; the schedule is its own again")
            self.degraded = False

        self.config_hash = str(current.get("config_hash", ""))
        derived = self.chain.open_round()
        published = int(current["round"])

        if published != derived.index:
            raise ServerRefused(
                f"the control plane is on round {published} but this chain gives "
                f"{derived.index}; refusing to measure across that gap"
            )

        return derived

    def seed(self, round_: Round) -> str:
        return self.chain.seed(round_)

    def systems(self, round_: Round) -> tuple[Sequence[System], dict[str, Entry]]:
        return self.chain.systems(round_)

    def coldkeys(self) -> dict[str, str]:
        return self.chain.coldkeys()

    def uids(self) -> dict[str, int]:
        return self.chain.uids()

    def workers(self) -> Sequence[Worker]:
        """Permitted on chain, and authorised by the server when it answers.

        The chain decides who may hold a validator permit; the server decides
        who may take a coordinated assignment. A worker the server has not
        authorised keeps its permit and can still run standalone, which is the
        property that keeps this a network you coordinate rather than one you
        switch off.
        """
        permitted = list(self.chain.workers())
        if self.client is None:
            return permitted

        try:
            authorised = self.client.authorised_workers()
        except (ServerUnreachable, ServerRefused) as exc:
            log.warning(
                "could not read the authorised worker list (%s); falling back to the "
                "chain permit alone for this round",
                exc,
            )
            return permitted

        if authorised is None:
            return permitted

        allowed = {str(row.get("hotkey", "")) for row in authorised}
        kept = [w for w in permitted if w.hotkey in allowed]
        if len(kept) != len(permitted):
            log.info(
                "%d of %d permitted validators are authorised for coordinated rounds",
                len(kept),
                len(permitted),
            )
        return kept


def publish_round(
    client: ServerClient,
    settlement: Settlement,
    *,
    reports: list[dict[str, Any]],
    assignment: Mapping[str, Sequence[str]],
) -> dict[str, int]:
    """Hand the archive everything the settlement was computed from.

    Reports go first. A settlement whose reports never arrived is a result
    nobody can recompute, which is the one thing the archive exists to prevent,
    so the order makes a partial push leave evidence without a conclusion rather
    than a conclusion without evidence.
    """
    stored = {"reports": 0, "assignments": 0, "settlement": 0}

    if reports:
        answer = client.push_reports(settlement.round_index, reports)
        stored["reports"] = int(answer.get("stored", 0))

    if assignment:
        answer = client.push_assignments(settlement.round_index, assignment)
        stored["assignments"] = int(answer.get("stored", 0))

    body = settlement.body()
    body["signature"] = settlement.signature
    answer = client.push_settlement(body)
    stored["settlement"] = int(answer.get("stored", 0))

    log.info(
        "round %d archived on the control plane: %d reports, %d assignments",
        settlement.round_index,
        stored["reports"],
        stored["assignments"],
    )
    return stored
