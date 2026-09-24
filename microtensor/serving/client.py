from __future__ import annotations

import contextlib
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from microtensor.chain.wallet import hotkey_address, sign_bytes

log = logging.getLogger("microtensor.serving.client")

TIMEOUT_SECONDS = 30
HOTKEY_HEADER = "x-mt-hotkey"
TIMESTAMP_HEADER = "x-mt-timestamp"
SIGNATURE_HEADER = "x-mt-signature"

REGISTER_PATH = "/v1/inference/operators/register"
COLLATERAL_PATH = "/v1/inference/operators/collateral"
MODELS_PATH = "/v1/inference/operators/models"
ME_PATH = "/v1/inference/operators/me"


class ServerError(RuntimeError):
    pass


def signing_bytes(method: str, path: str, timestamp: str, body: bytes) -> bytes:
    return b"\n".join([method.upper().encode(), path.encode(), timestamp.encode(), body])


def _url(base: str, path: str) -> str:
    return urllib.parse.urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def _call(url: str, *, method: str, body: bytes | None, headers: dict[str, str]) -> Any:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as answer:
            raw = answer.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        with contextlib.suppress(ValueError):
            detail = str(json.loads(detail).get("detail", detail))
        raise ServerError(f"{exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ServerError(f"could not reach {url}: {exc.reason}") from exc
    return json.loads(raw) if raw else {}


def _signed(
    base: str, wallet: Any, *, method: str, path: str, payload: dict[str, Any] | None = None
) -> Any:
    hotkey = hotkey_address(wallet)
    body = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        if payload is not None
        else b""
    )
    timestamp = str(time.time())
    headers = {
        HOTKEY_HEADER: hotkey,
        TIMESTAMP_HEADER: timestamp,
        SIGNATURE_HEADER: sign_bytes(wallet, signing_bytes(method, path, timestamp, body)),
    }
    if payload is not None:
        headers["content-type"] = "application/json"
    return _call(_url(base, path), method=method, body=body or None, headers=headers)


def register(base: str, wallet: Any, *, label: str = "", coldkey: str = "") -> dict[str, Any]:
    payload = {"label": label, "coldkey": coldkey}
    return dict(_signed(base, wallet, method="POST", path=REGISTER_PATH, payload=payload))


def post_collateral(base: str, wallet: Any, *, reference: str, block: int) -> dict[str, Any]:
    payload = {"reference": reference.lower(), "block": int(block)}
    return dict(_signed(base, wallet, method="POST", path=COLLATERAL_PATH, payload=payload))


def declare(base: str, wallet: Any, *, model: str, artifact_digest: str) -> dict[str, Any]:
    payload = {"model": model, "artifact_digest": artifact_digest.lower()}
    return dict(_signed(base, wallet, method="POST", path=MODELS_PATH, payload=payload))


def me(base: str, wallet: Any) -> dict[str, Any]:
    return dict(_signed(base, wallet, method="GET", path=ME_PATH))
