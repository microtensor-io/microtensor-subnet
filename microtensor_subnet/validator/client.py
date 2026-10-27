from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Final

import httpx

from microtensor_subnet.chain.wallet import sign_bytes

HOTKEY_HEADER: Final[str] = "x-mt-hotkey"
TIMESTAMP_HEADER: Final[str] = "x-mt-timestamp"
SIGNATURE_HEADER: Final[str] = "x-mt-signature"
TIMEOUT_SECONDS: Final[float] = 60.0
BUNDLE_TIMEOUT_SECONDS: Final[float] = 600.0


ARENA_NETWORKS: dict[str, str] = {"test": "testnet", "finney": "mainnet"}


class ServerError(RuntimeError):
    pass


def signing_bytes(method: str, path: str, timestamp: str, body: bytes) -> bytes:
    return b"\n".join([method.upper().encode(), path.encode(), timestamp.encode(), body])


@dataclass(slots=True)
class ServerClient:
    base_url: str
    hotkey: str
    wallet: Any = None
    timeout: float = TIMEOUT_SECONDS

    def _headers(self, method: str, path: str, body: bytes) -> dict[str, str]:
        stamp = f"{time.time():.0f}"
        headers = {"accept": "application/json", "content-type": "application/json"}
        if self.wallet is not None:
            headers[HOTKEY_HEADER] = self.hotkey
            headers[TIMESTAMP_HEADER] = stamp
            headers[SIGNATURE_HEADER] = sign_bytes(
                self.wallet, signing_bytes(method, path, stamp, body)
            )
        return headers

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None, *, signed: bool = False
    ) -> Any:
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode() if body else b""
        headers = (
            self._headers(method, path, raw)
            if signed
            else {"accept": "application/json", "content-type": "application/json"}
        )
        try:
            with httpx.Client(base_url=self.base_url.rstrip("/"), timeout=self.timeout) as http:
                response = http.request(method, path, content=raw or None, headers=headers)
        except httpx.HTTPError as exc:
            raise ServerError(f"{method} {path}: {exc}") from exc
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise ServerError(
                f"{method} {path} answered {response.status_code}: {response.text[:300]}"
            )
        return response.json() if response.content else None

    def live_arenas(self, network: str) -> list[str]:
        found = self._request("GET", "/v2/arenas") or {}
        wanted = ARENA_NETWORKS.get(network, network)
        return sorted(
            str(a.get("id"))
            for a in found.get("arenas") or []
            if a.get("state") == "live" and a.get("network") == wanted and a.get("id")
        )

    def arena(self, arena_id: str, block: int | None = None) -> dict[str, Any] | None:
        query = f"?block={block}" if block is not None else ""
        found = self._request("GET", f"/v2/arenas/{arena_id}{query}")
        return dict(found) if found else None

    def epoch(self, arena_id: str, index: int) -> dict[str, Any] | None:
        found = self._request("GET", f"/v2/arenas/{arena_id}/epochs/{index}")
        return dict(found) if found else None

    def document(self, arena_id: str, index: int, kind: str) -> dict[str, Any] | None:
        found = self._request("GET", f"/v2/arenas/{arena_id}/epochs/{index}/documents/{kind}")
        return dict(found) if found else None

    def bundle(self, arena_id: str, index: int) -> bytes:
        try:
            with httpx.Client(
                base_url=self.base_url.rstrip("/"), timeout=BUNDLE_TIMEOUT_SECONDS
            ) as http:
                response = http.get(f"/v2/arenas/{arena_id}/epochs/{index}/bundle")
        except httpx.HTTPError as exc:
            raise ServerError(f"bundle download failed: {exc}") from exc
        if response.status_code != 200:
            raise ServerError(f"bundle download answered {response.status_code}")
        return bytes(response.content)

    def public_keys(self) -> dict[str, Any]:
        found = self._request("GET", "/v1/public/keys")
        return dict(found or {})

    def register_key(self, public_key: str, expires_block: int, signature: str) -> dict[str, Any]:
        found = self._request(
            "POST",
            "/v2/validators/keys",
            {"public_key": public_key, "expires_block": expires_block, "signature": signature},
            signed=True,
        )
        return dict(found or {})

    def relay_token(self) -> dict[str, Any]:
        found = self._request("POST", "/v2/validators/relay-token", {}, signed=True)
        return dict(found or {})

    def ledger_snapshot(self, arena_id: str, index: int) -> dict[str, Any] | None:
        return self.document(arena_id, index, "snapshot")

    def roster(self, arena_id: str, index: int) -> dict[str, Any] | None:
        return self.document(arena_id, index, "roster")

    def submit_report(self, report: dict[str, Any]) -> dict[str, Any]:
        found = self._request("POST", "/v2/reports", report, signed=True)
        return dict(found or {})

    def submit_artifact_result(self, body: dict[str, Any]) -> dict[str, Any]:
        found = self._request("POST", "/v2/reports/artifact", body, signed=True)
        if not isinstance(found, dict) or not found.get("accepted"):
            raise ServerError(f"POST /v2/reports/artifact was not accepted: {found!r}"[:300])
        return dict(found)
