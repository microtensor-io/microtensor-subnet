from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from microtensor.coordinator.api import (
    HOTKEY_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    signing_bytes,
)
from microtensor.coordinator.collect import intake
from microtensor.coordinator.config import MISMATCH, config_hash, matches
from microtensor.coordinator.report import Report
from microtensor.coordinator.settle import (
    RESERVE_MAX,
    Entry,
    catalogue_from,
    merkle_root,
    normalise_reserved,
)
from microtensor.coordinator.settle import build as build_settlement
from microtensor.coordinator.tokens import TOKEN_HEADER
from microtensor.core.constants import (
    COORDINATOR_BACKOFF_SECONDS,
    COORDINATOR_RETRIES,
    COORDINATOR_TIMEOUT_SECONDS,
)

log = logging.getLogger("microtensor.validator")

REPORTS_ROOT_MISMATCH = "the published reports do not hash to the settlement's reports_root"
BAD_SCHEME = "the coordinator URL must be http or https"
WEIGHTS_MISMATCH = "the settlement's weights do not recompute from the published reports"


def _detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read() or b"null")
    except (ValueError, OSError):
        return ""
    return str(body.get("detail", "")) if isinstance(body, dict) else ""


class CoordinatorUnreachable(RuntimeError):
    """The coordinator could not be reached inside the round window.

    Not fatal. A worker that cannot reach the coordinator holds its last
    settled vector rather than halting: an outage pauses the network, and
    weights keep flowing throughout.
    """


class CoordinatorRefused(RuntimeError):
    """The coordinator answered and said no.

    Kept apart from CoordinatorUnreachable because the right response differs.
    A transport failure holds the last vector so the subnet keeps setting
    weights. A refusal means this worker is misconfigured or unauthorised, and
    quietly holding would hide that behind a working-looking pause.
    """


class SettlementRejected(RuntimeError):
    """The published settlement does not recompute from its own inputs.

    A worker that submits a vector it has not checked is a relay, and a network
    of relays has no consensus. This is the error that keeps that from
    happening quietly.
    """


@dataclass(slots=True)
class CoordinatorClient:
    base_url: str
    hotkey: str
    wallet: Any = None
    token: str = ""
    timeout: int = COORDINATOR_TIMEOUT_SECONDS
    retries: int = COORDINATOR_RETRIES
    backoff: float = COORDINATOR_BACKOFF_SECONDS

    def _sign(self, method: str, path: str, body: bytes) -> dict[str, str]:
        stamp = f"{time.time():.0f}"
        headers = {
            HOTKEY_HEADER: self.hotkey,
            TIMESTAMP_HEADER: stamp,
            "content-type": "application/json",
        }
        if self.token:
            headers[TOKEN_HEADER] = self.token
        if self.wallet is not None:
            from microtensor.chain.wallet import sign_bytes

            headers[SIGNATURE_HEADER] = sign_bytes(
                self.wallet, signing_bytes(method, path, stamp, body)
            )
        return headers

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode() if body else b""
        url = self.base_url.rstrip("/") + path
        if urllib.parse.urlparse(url).scheme not in ("http", "https"):
            raise ValueError(f"{BAD_SCHEME}: {self.base_url!r}")
        request = urllib.request.Request(  # noqa: S310 - scheme checked above
            url, data=raw or None, method=method, headers=self._sign(method, path, raw)
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
                if exc.code == 401:
                    raise CoordinatorRefused(
                        f"{url} does not recognise this hotkey (401); check it holds a "
                        f"validator permit on this subnet and that the clock is correct"
                    ) from exc
                if exc.code == 403:
                    detail = _detail(exc)
                    raise CoordinatorRefused(
                        f"{url} recognises this hotkey and refused it (403"
                        f"{': ' + detail if detail else ''}); the hotkey is known, so this "
                        f"is an authorisation rule rather than registration or the clock"
                    ) from exc
                if exc.code == 429 or exc.code >= 500:
                    last = exc
                    if attempt + 1 < self.retries:
                        time.sleep(self.backoff * (2**attempt))
                        continue
                    raise CoordinatorUnreachable(
                        f"{url} returned {exc.code} after {self.retries} attempts"
                    ) from exc
                detail = _detail(exc)
                raise CoordinatorRefused(
                    f"{url} returned {exc.code}: {detail or exc}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff * (2**attempt))

        raise CoordinatorUnreachable(f"{url} after {self.retries} attempts: {last}")

    def current_round(self) -> dict[str, Any]:
        return self._call("GET", "/v1/round/current") or {}

    def assignment(self) -> tuple[int | None, tuple[str, ...] | None]:
        """The round and the systems this worker measures.

        The systems are None when there is no assignment document at all. None
        and () mean different things and must not be collapsed: () is the
        coordinator telling us we have nothing to do this round, None is the
        endpoint having nothing for us, which is a fault rather than an idle
        round. The round travels with it so a coordinator that has moved on can
        be detected instead of measured against.
        """
        found = self._call("GET", f"/v1/assignment/{self.hotkey}")
        if found is None or "systems" not in found:
            return None, None
        round_index = found.get("round")
        return (
            None if round_index is None else int(round_index),
            tuple(s["system_digest"] for s in found["systems"]),
        )

    def submit(self, report: Report) -> dict[str, Any]:
        body = report.body()
        body["signature"] = report.signature
        return self._call("POST", "/v1/report", body) or {}

    def reported(self, round_index: int, worker_hotkey: str) -> set[str]:
        found = self._call("GET", f"/v1/reports/{round_index}") or {}
        return {
            str(row.get("system_digest", ""))
            for row in found.get("reports", ())
            if row.get("worker_hotkey") == worker_hotkey and row.get("system_digest")
        }

    def corpus_index(self) -> dict[str, Any]:
        found: dict[str, Any] | None = self._call("GET", "/v1/corpus")
        return found or {}

    def corpus(self, track: str) -> dict[str, Any] | None:
        found: dict[str, Any] | None = self._call("GET", f"/v1/corpus/{track}")
        return found

    def weights(self) -> dict[int, float]:
        found = self._call("GET", "/v1/weights") or {}
        if found.get("paused"):
            return {}
        return {int(uid): float(value) for uid, value in (found.get("weights") or {}).items()}

    def settlement(self, round_index: int) -> dict[str, Any] | None:
        found: dict[str, Any] | None = self._call("GET", f"/v1/settlement/{round_index}")
        return found

    def reports(self, round_index: int) -> list[dict[str, Any]]:
        found = self._call("GET", f"/v1/reports/{round_index}") or {}
        return list(found.get("reports", []))


def verify_config(served: dict[str, Any], anchored: str) -> None:
    """Refuse a config the chain was never told about.

    This is what separates a coordinator serving specs over HTTP from one whose
    rules are provable. A mismatch aborts the round on this worker rather than
    measuring against ceilings nobody committed to.
    """
    if not matches(served, anchored):
        raise SettlementRejected(
            f"{MISMATCH}: served {config_hash(served)} against anchored {anchored or 'nothing'}"
        )


UID_MISMATCH = "the settlement names a uid the metagraph does not give that hotkey"
UNKNOWN_HOTKEY = "the settlement names a hotkey absent from the metagraph"
RESERVE_TOO_LARGE = "the settlement holds back more emission than this worker will sign"


def cross_check(published: dict[str, Any], uid_by_hotkey: Mapping[str, int]) -> None:
    """Check the settlement's own inputs against the chain.

    Recomputation alone proves the weights follow from the published inputs. It
    does not prove the inputs are honest, and the coordinator supplies them. The
    part a worker can check independently is identity: every entry names a
    hotkey the metagraph knows, at the uid the metagraph gives it. A coordinator
    cannot redirect emission to a uid of its choosing without failing this.
    """
    for row in published.get("catalogue", []):
        hotkey = str(row.get("miner", ""))
        if hotkey not in uid_by_hotkey:
            raise SettlementRejected(f"{UNKNOWN_HOTKEY}: {hotkey}")
        if int(row.get("uid", -1)) != uid_by_hotkey[hotkey]:
            raise SettlementRejected(
                f"{UID_MISMATCH}: {hotkey} is uid {uid_by_hotkey[hotkey]}, "
                f"settlement says {row.get('uid')}"
            )

    held = published.get("reserved") or {}
    if held:
        hotkey = str(held.get("hotkey", ""))
        if hotkey not in uid_by_hotkey:
            raise SettlementRejected(f"{UNKNOWN_HOTKEY}: {hotkey}")
        if int(held.get("uid", -1)) != uid_by_hotkey[hotkey]:
            raise SettlementRejected(
                f"{UID_MISMATCH}: {hotkey} is uid {uid_by_hotkey[hotkey]}, "
                f"settlement says {held.get('uid')}"
            )


def verify_settlement(
    published: dict[str, Any],
    reports: list[dict[str, Any]],
    catalogue: dict[str, Entry] | None = None,
    reserve_ceiling: float = RESERVE_MAX,
) -> None:
    """Recompute the settlement from the published reports and compare.

    Step 7 of the worker round, and it is not optional. Without it the worker
    is relaying a number it never checked, and a compromised coordinator
    publishing a false settlement would be caught by nobody.

    The catalogue defaults to the one the settlement publishes, because a worker
    measured only its assigned subset and its local `rounds_observed` differs
    from its peers' for honest reasons. Pair this with `cross_check` to test the
    published inputs against the metagraph.

    A declared reserve is folded in the same way the coordinator folded it, so
    a hold reproduces rather than reading as a divergence. It reproduces because
    it is stated: an undeclared hold changes the weights without changing the
    inputs and fails here like any other tampering.

    The dropped set is taken from the settlement for the same reason as the
    advisory set: it is liveness state only the coordinator observes, it changes
    who is measured, and recomputing without it would disagree with an honest
    settlement the moment any miner went quiet.

    Reconciliation uses the advisory set the settlement declares. That set is
    reputation state only the coordinator holds, and excluding a worker changes
    which value wins a majority, so recomputing without it would disagree with
    an honest settlement the moment any worker was advisory.
    """
    if catalogue is None:
        catalogue = catalogue_from(published.get("catalogue", []))

    parsed = [Report.from_dict(r) for r in reports]

    root = merkle_root([r.digest() for r in parsed])
    if root != published.get("reports_root"):
        raise SettlementRejected(
            f"{REPORTS_ROOT_MISMATCH}: computed {root} against {published.get('reports_root')}"
        )

    by_system: dict[str, list[Report]] = {}
    for report in parsed:
        by_system.setdefault(report.system_digest, []).append(report)

    result = intake(by_system, advisory=published.get("advisory", ()))
    recomputed = build_settlement(
        int(published["round"]),
        config_hash=str(published.get("config_hash", "")),
        corpus_version=str(published.get("corpus_version", "")),
        reconciled=result.reconciled,
        catalogue=catalogue,
        report_digests=[r.digest() for r in parsed],
        unscored=result.unscored,
        under_replicated=published.get("under_replicated", ()),
        reserved=published.get("reserved"),
        dropped=published.get("dropped"),
    )

    held = normalise_reserved(published.get("reserved"))
    if held and float(held["share"]) > reserve_ceiling:
        raise SettlementRejected(
            f"{RESERVE_TOO_LARGE}: {held['share']:.4f} held for {held['hotkey']}, "
            f"ceiling is {reserve_ceiling:.4f}"
        )

    published_weights = {
        str(k): round(float(v), 9) for k, v in published.get("weights", {}).items()
    }
    mine = {str(k): round(float(v), 9) for k, v in recomputed.weights.items()}

    if published_weights != mine:
        raise SettlementRejected(f"{WEIGHTS_MISMATCH}: computed {mine} against {published_weights}")
