from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

PROBE_PATH = "/v1/operators/probe"
CREDENTIAL_HEADER = "x-mt-credential"
TIMEOUT_SECONDS = 180.0


@dataclass(frozen=True, slots=True)
class Answered:
    ok: bool
    answer: Any = None
    p: float = 0.0
    floor_met: bool = False
    verified: bool = False
    artifact_digest: str = ""
    latency_ms: float = 0.0
    error: str = ""


def ask(
    gateway: str,
    credential: str,
    *,
    hotkey: str,
    model: str,
    task: dict[str, Any],
    floor: float,
    timeout: float = TIMEOUT_SECONDS,
) -> Answered:
    body = {"hotkey": hotkey, "model": model, "task": task, "floor": floor}
    try:
        with httpx.Client(base_url=gateway.rstrip("/"), timeout=timeout) as http:
            response = http.post(PROBE_PATH, json=body, headers={CREDENTIAL_HEADER: credential})
    except httpx.HTTPError as exc:
        return Answered(False, error=f"gateway unreachable: {exc.__class__.__name__}")
    if response.status_code != 200:
        detail = response.text[:200]
        return Answered(False, error=f"gateway answered {response.status_code}: {detail}")
    found = response.json()
    return Answered(
        True,
        answer=found.get("answer"),
        p=float(found.get("p") or 0.0),
        floor_met=bool(found.get("floor_met")),
        verified=bool(found.get("verified")),
        artifact_digest=str(found.get("artifact_digest") or ""),
        latency_ms=float(found.get("latency_ms") or 0.0),
    )
