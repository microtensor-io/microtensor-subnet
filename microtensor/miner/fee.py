from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from microtensor.chain.wallet import hotkey_address, sign_bytes

log = logging.getLogger("microtensor.miner.fee")

TIMEOUT_SECONDS = 20
RAO_PER_TAO = 1_000_000_000
HOTKEY_HEADER = "x-mt-hotkey"
TIMESTAMP_HEADER = "x-mt-timestamp"
SIGNATURE_HEADER = "x-mt-signature"
REPORT_PATH = "/v1/arena/fee/report"


class FeeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FeePolicy:
    """What one submission costs, read from the public surface.

    Disabled is a real answer: a server that charges nothing says so, and the
    miner CLI treats that differently from a server it could not reach.
    """

    enabled: bool
    fee_tao: float
    pay_to: str
    contract_version: int = 1

    @classmethod
    def from_wire(cls, raw: Any) -> FeePolicy:
        found = dict(raw or {})
        fee = float(found.get("fee_tao") or 0.0)
        pay_to = str(found.get("pay_to") or "")
        return cls(
            enabled=bool(found.get("enabled")) and fee > 0 and bool(pay_to),
            fee_tao=fee,
            pay_to=pay_to,
            contract_version=int(found.get("contract_version") or 1),
        )

    @property
    def rao(self) -> int:
        return round(self.fee_tao * RAO_PER_TAO)


@dataclass(frozen=True, slots=True)
class FeeStatus:
    state: str
    paid: bool
    required: bool
    reason: str = ""
    payment_reference: str = ""
    payment_block: int = 0
    fee_tao: float = 0.0
    pay_to: str = ""

    @classmethod
    def from_wire(cls, raw: Any) -> FeeStatus:
        found = dict(raw or {})
        return cls(
            state=str(found.get("state") or "unpaid"),
            paid=bool(found.get("paid")),
            required=bool(found.get("required")),
            reason=str(found.get("reason") or ""),
            payment_reference=str(found.get("payment_reference") or ""),
            payment_block=int(found.get("payment_block") or 0),
            fee_tao=float(found.get("fee_tao") or 0.0),
            pay_to=str(found.get("pay_to") or ""),
        )


def signing_bytes(method: str, path: str, timestamp: str, body: bytes) -> bytes:
    return b"\n".join([method.upper().encode(), path.encode(), timestamp.encode(), body])


def _url(base: str, path: str) -> str:
    url = base.rstrip("/") + path
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise FeeError(f"the server URL must be http or https: {base!r}")
    return url


def _call(
    url: str, *, method: str = "GET", body: bytes = b"", headers: dict[str, str] | None = None
) -> Any:
    request = urllib.request.Request(  # noqa: S310 - scheme checked by _url
        url,
        data=body or None,
        method=method,
        headers={"accept": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            return json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = str(json.loads(exc.read() or b"{}").get("detail", ""))
        except Exception:
            detail = ""
        raise FeeError(f"{url} returned {exc.code}{': ' + detail if detail else ''}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise FeeError(f"{url} unreachable: {exc}") from exc


def fetch_policy(base: str) -> FeePolicy:
    return FeePolicy.from_wire(_call(_url(base, "/v1/arena/fee")))


def fetch_status(base: str, hotkey: str, manifest_digest: str) -> FeeStatus:
    digest = short(manifest_digest)
    return FeeStatus.from_wire(_call(_url(base, f"/v1/arena/fee/{hotkey}/{digest}")))


def short(manifest_digest: str) -> str:
    body = manifest_digest.split(":", 1)[1] if ":" in manifest_digest else manifest_digest
    return body.strip().lower()[:32]


def report(
    base: str,
    wallet: Any,
    *,
    manifest_digest: str,
    round_index: int,
    extrinsic_hash: str,
    block: int,
) -> FeeStatus:
    """Tell the server which transfer paid for which artifact, signed by the hotkey.

    The signature covers the method, the path, a timestamp and the exact body,
    so a report cannot be replayed against another endpoint or altered in
    flight, and the server binds the payment to the signing hotkey alone.
    """
    hotkey = hotkey_address(wallet)
    payload: dict[str, Any] = {
        "manifest_digest": short(manifest_digest),
        "round_index": int(round_index),
        "extrinsic_hash": extrinsic_hash,
        "block": int(block),
        "hotkey": hotkey,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    timestamp = str(time.time())
    signature = sign_bytes(wallet, signing_bytes("POST", REPORT_PATH, timestamp, body))
    headers: dict[str, str] = {
        "content-type": "application/json",
        HOTKEY_HEADER: hotkey,
        TIMESTAMP_HEADER: timestamp,
        SIGNATURE_HEADER: signature,
    }
    answer = _call(_url(base, REPORT_PATH), method="POST", body=body, headers=headers)
    return FeeStatus.from_wire(answer)


def gate(base: str, hotkey: str, manifest_digest: str) -> str:
    """Why this artifact would not be catalogued, or an empty string.

    An unreachable server is reported as a warning rather than a refusal: the
    commit still lands, and the coordinator decides at discovery. Only a server
    that answers "unpaid" stops the publish, because that one is certain.
    """
    try:
        policy = fetch_policy(base)
    except FeeError as exc:
        log.warning("could not read the submission fee policy (%s); publishing anyway", exc)
        return ""
    if not policy.enabled:
        return ""
    try:
        status = fetch_status(base, hotkey, manifest_digest)
    except FeeError as exc:
        log.warning("could not read the submission fee status (%s); publishing anyway", exc)
        return ""
    if status.paid:
        return ""
    return (
        f"the submission fee of {policy.fee_tao:g} TAO for {short(manifest_digest)} is "
        f"{status.state}{': ' + status.reason if status.reason else ''}; "
        "run `mt miner fee pay` first, or the coordinator will not catalogue this artifact"
    )
