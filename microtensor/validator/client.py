from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
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
from microtensor.coordinator.settle import Entry, merkle_root
from microtensor.coordinator.settle import build as build_settlement
from microtensor.core.constants import (
    COORDINATOR_BACKOFF_SECONDS,
    COORDINATOR_RETRIES,
    COORDINATOR_TIMEOUT_SECONDS,
)

log = logging.getLogger("microtensor.validator")

REPORTS_ROOT_MISMATCH = "the published reports do not hash to the settlement's reports_root"
BAD_SCHEME = "the coordinator URL must be http or https"
WEIGHTS_MISMATCH = "the settlement's weights do not recompute from the published reports"


class CoordinatorUnreachable(RuntimeError):
    """The coordinator could not be reached inside the round window.

    Not fatal. A worker that cannot reach the coordinator settles standalone
    rather than halting, because a coordinator outage must not stop the subnet
    from setting weights.
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
    timeout: int = COORDINATOR_TIMEOUT_SECONDS
    retries: int = COORDINATOR_RETRIES
    backoff: float = COORDINATOR_BACKOFF_SECONDS

    # ------------------------------------------------------------- transport

    def _sign(self, method: str, path: str, body: bytes) -> dict[str, str]:
        stamp = f"{time.time():.0f}"
        headers = {
            HOTKEY_HEADER: self.hotkey,
            TIMESTAMP_HEADER: stamp,
            "content-type": "application/json",
        }
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
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff * (2**attempt))

        raise CoordinatorUnreachable(f"{url} after {self.retries} attempts: {last}")

    # ------------------------------------------------------------- endpoints

    def current_round(self) -> dict[str, Any]:
        return self._call("GET", "/v1/round/current") or {}

    def assignment(self) -> tuple[str, ...]:
        found = self._call("GET", f"/v1/assignment/{self.hotkey}") or {}
        return tuple(s["system_digest"] for s in found.get("systems", []))

    def submit(self, report: Report) -> dict[str, Any]:
        body = report.body()
        body["signature"] = report.signature
        return self._call("POST", "/v1/report", body) or {}

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


def verify_settlement(
    published: dict[str, Any],
    reports: list[dict[str, Any]],
    catalogue: dict[str, Entry],
) -> None:
    """Recompute the settlement from the published reports and compare.

    Step 7 of the worker round, and it is not optional. Without it the worker
    is relaying a number it never checked, and a compromised coordinator
    publishing a false settlement would be caught by nobody.
    """
    parsed = [Report.from_dict(r) for r in reports]

    root = merkle_root([r.digest() for r in parsed])
    if root != published.get("reports_root"):
        raise SettlementRejected(
            f"{REPORTS_ROOT_MISMATCH}: computed {root} against {published.get('reports_root')}"
        )

    by_system: dict[str, list[Report]] = {}
    for report in parsed:
        by_system.setdefault(report.system_digest, []).append(report)

    result = intake(by_system)
    recomputed = build_settlement(
        int(published["round"]),
        config_hash=str(published.get("config_hash", "")),
        corpus_version=str(published.get("corpus_version", "")),
        reconciled=result.reconciled,
        catalogue=catalogue,
        report_digests=[r.digest() for r in parsed],
        unscored=result.unscored,
        under_replicated=published.get("under_replicated", ()),
    )

    published_weights = {
        str(k): round(float(v), 9) for k, v in published.get("weights", {}).items()
    }
    mine = {str(k): round(float(v), 9) for k, v in recomputed.weights.items()}

    if published_weights != mine:
        raise SettlementRejected(f"{WEIGHTS_MISMATCH}: computed {mine} against {published_weights}")
