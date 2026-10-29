from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx

from microtensor_subnet.chain.wallet import sign_bytes

HOTKEY_HEADER: Final[str] = "x-mt-hotkey"
TIMESTAMP_HEADER: Final[str] = "x-mt-timestamp"
SIGNATURE_HEADER: Final[str] = "x-mt-signature"
TIMEOUT_SECONDS: Final[float] = 60.0
BUNDLE_TIMEOUT_SECONDS: Final[float] = 600.0
BLOB_TIMEOUT_SECONDS: Final[float] = 3600.0
UPLOAD_TIMEOUT_SECONDS: Final[float] = 300.0
PROGRESS_TIMEOUT_SECONDS: Final[float] = 5.0
GZIP: Final[str] = "application/gzip"
CHUNK_BYTES: Final[int] = 8 * 1024 * 1024


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

    def _headers(
        self, method: str, path: str, body: bytes, content_type: str = "application/json"
    ) -> dict[str, str]:
        stamp = f"{time.time():.0f}"
        headers = {"accept": "application/json", "content-type": content_type}
        if self.wallet is not None:
            headers[HOTKEY_HEADER] = self.hotkey
            headers[TIMESTAMP_HEADER] = stamp
            headers[SIGNATURE_HEADER] = sign_bytes(
                self.wallet, signing_bytes(method, path, stamp, body)
            )
        return headers

    def _send(
        self,
        method: str,
        path: str,
        raw: bytes,
        *,
        content_type: str,
        signed: bool,
        timeout: float | None,
    ) -> Any:
        headers = (
            self._headers(method, path, raw, content_type)
            if signed
            else {"accept": "application/json", "content-type": content_type}
        )
        try:
            with httpx.Client(
                base_url=self.base_url.rstrip("/"), timeout=timeout or self.timeout
            ) as http:
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

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        signed: bool = False,
        timeout: float | None = None,
    ) -> Any:
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode() if body else b""
        return self._send(
            method, path, raw, content_type="application/json", signed=signed, timeout=timeout
        )

    def _download(self, path: str, *, timeout: float) -> httpx.Response:
        try:
            with httpx.Client(
                base_url=self.base_url.rstrip("/"), timeout=timeout, follow_redirects=True
            ) as http:
                return http.get(path)
        except httpx.HTTPError as exc:
            raise ServerError(f"GET {path} failed: {exc}") from exc

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
        response = self._download(
            f"/v2/arenas/{arena_id}/epochs/{index}/bundle", timeout=BUNDLE_TIMEOUT_SECONDS
        )
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

    def submit_artifact_result(
        self, body: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        found = self._request("POST", "/v2/reports/artifact", body, signed=True, timeout=timeout)
        if not isinstance(found, dict) or not found.get("accepted"):
            raise ServerError(f"POST /v2/reports/artifact was not accepted: {found!r}"[:300])
        return dict(found)

    def report_progress(self, body: dict[str, Any]) -> dict[str, Any]:
        found = self._request(
            "POST", "/v2/reports/progress", body, signed=True, timeout=PROGRESS_TIMEOUT_SECONDS
        )
        return dict(found or {})

    def profiles(self) -> dict[str, Any]:
        found = self._request("GET", "/v2/profiles")
        return dict(found or {})

    def mirror(self, arena_id: str, index: int) -> dict[str, Any] | None:
        found = self._request("GET", f"/v2/arenas/{arena_id}/epochs/{index}/mirror")
        return dict(found) if found else None

    def inputs(self, arena_id: str, index: int) -> dict[str, Any] | None:
        found = self._request("GET", f"/v2/arenas/{arena_id}/epochs/{index}/inputs")
        return dict(found) if found else None

    def receipts(self, arena_id: str, index: int) -> dict[str, Any] | None:
        found = self._request("GET", f"/v2/arenas/{arena_id}/epochs/{index}/receipts")
        return dict(found) if found else None

    def epoch_validators(self, arena_id: str, index: int) -> list[dict[str, Any]]:
        found = self._request("GET", f"/v2/arenas/{arena_id}/epochs/{index}/validators")
        rows = found.get("validators") if isinstance(found, dict) else found
        return [dict(r) for r in rows or [] if isinstance(r, dict)]

    def evidence(self, arena_id: str, index: int) -> list[dict[str, Any]]:
        found = self._request("GET", f"/v2/arenas/{arena_id}/epochs/{index}/evidence")
        rows = found.get("evidence") if isinstance(found, dict) else found
        return [dict(r) for r in rows or [] if isinstance(r, dict)]

    def bans(self, arena_id: str) -> list[dict[str, Any]]:
        found = self._request("GET", f"/v2/arenas/{arena_id}/bans")
        rows = found.get("bans") if isinstance(found, dict) else found
        return [dict(r) for r in rows or [] if isinstance(r, dict)]

    def inputs_bundle(self, arena_id: str, index: int, expected_sha256: str = "") -> bytes:
        response = self._download(
            f"/v2/arenas/{arena_id}/epochs/{index}/inputs/bundle", timeout=BUNDLE_TIMEOUT_SECONDS
        )
        if response.status_code != 200:
            raise ServerError(f"inputs bundle download answered {response.status_code}")
        blob = bytes(response.content)
        if expected_sha256 and hashlib.sha256(blob).hexdigest() != expected_sha256.lower():
            raise ServerError("the inputs bundle does not match its published sha256")
        return blob

    def _stream_mirror(self, sha256: str, sink: Callable[[bytes], Any]) -> None:
        digest = hashlib.sha256()
        try:
            with (
                httpx.Client(
                    base_url=self.base_url.rstrip("/"),
                    timeout=BLOB_TIMEOUT_SECONDS,
                    follow_redirects=True,
                ) as http,
                http.stream("GET", f"/v2/mirror/{sha256}") as response,
            ):
                if response.status_code == 404:
                    raise ServerError(f"the mirror holds no blob {sha256}")
                if response.status_code != 200:
                    raise ServerError(f"mirror blob {sha256} answered {response.status_code}")
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    digest.update(chunk)
                    sink(chunk)
        except httpx.HTTPError as exc:
            raise ServerError(f"mirror blob {sha256} failed: {exc}") from exc
        if digest.hexdigest() != sha256:
            raise ServerError(f"mirror blob {sha256} does not match its sha256")

    def mirror_blob(self, sha256: str) -> bytes:
        wanted = sha256.strip().lower()
        chunks: list[bytes] = []
        self._stream_mirror(wanted, chunks.append)
        return b"".join(chunks)

    def mirror_file(self, sha256: str, into: Path) -> Path:
        wanted = sha256.strip().lower()
        into = Path(into)
        if into.is_file():
            if _file_sha256(into) == wanted:
                return into
            into.unlink()
        into.parent.mkdir(parents=True, exist_ok=True)
        partial = into.with_name(into.name + ".part")
        try:
            with partial.open("wb") as handle:
                self._stream_mirror(wanted, handle.write)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        os.replace(partial, into)
        return into

    def upload_transcript(self, arena_id: str, index: int, compressed: bytes) -> dict[str, Any]:
        found = self._send(
            "POST",
            f"/v2/arenas/{arena_id}/epochs/{index}/transcripts",
            compressed,
            content_type=GZIP,
            signed=True,
            timeout=UPLOAD_TIMEOUT_SECONDS,
        )
        if not isinstance(found, dict) or not found.get("transcript_sha256"):
            raise ServerError(f"the transcript upload was not accepted: {found!r}"[:300])
        return dict(found)

    def miner_relay_token(self) -> dict[str, Any]:
        found = self._request("POST", "/v2/miners/relay-token", {}, signed=True)
        return dict(found or {})

    def register_profile(self, profiles: list[str]) -> dict[str, Any]:
        found = self._request(
            "POST", "/v2/validators/profile", {"profiles": list(profiles)}, signed=True
        )
        return dict(found or {})

    def post_evidence(self, body: dict[str, Any]) -> dict[str, Any]:
        found = self._request("POST", "/v2/reports/evidence", body, signed=True)
        return dict(found or {})

    def confirm_evidence(self, evidence_id: str, body: dict[str, Any]) -> dict[str, Any]:
        found = self._request(
            "POST", f"/v2/reports/evidence/{evidence_id}/confirm", body, signed=True
        )
        return dict(found or {})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
