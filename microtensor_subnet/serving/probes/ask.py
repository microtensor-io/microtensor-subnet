from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

PROBE_PATH = "/v1/operators/probe"
CREDENTIAL_HEADER = "x-mt-credential"
TIMEOUT_SECONDS = 180.0
CHAT_MAX_TOKENS = 512


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


@dataclass(frozen=True, slots=True)
class Chatted:
    ok: bool
    content: str = ""
    reasoning: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    finish_reason: str = ""
    prompt_ids: tuple[int, ...] = ()
    completion_ids: tuple[int, ...] = ()
    usage: dict[str, Any] = field(default_factory=dict)
    artifact_digest: str = ""
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    error: str = ""


def _ids(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(int(token) for token in raw)


def chatted(found: Mapping[str, Any]) -> Chatted:
    if found.get("error"):
        return Chatted(False, error=str(found["error"])[:300])
    tokens = found.get("tokens")
    held: Mapping[str, Any] = tokens if isinstance(tokens, Mapping) else {}
    calls = found.get("tool_calls")
    listed = calls if isinstance(calls, list) else []
    usage = found.get("usage")
    try:
        prompt, completion = _ids(held.get("prompt")), _ids(held.get("completion"))
    except (TypeError, ValueError):
        return Chatted(False, error="the answer carried token ids that are not integers")
    return Chatted(
        True,
        content=str(found.get("content") or ""),
        reasoning=str(found.get("reasoning") or ""),
        tool_calls=tuple(dict(c) for c in listed if isinstance(c, Mapping)),
        finish_reason=str(found.get("finish_reason") or ""),
        prompt_ids=prompt,
        completion_ids=completion,
        usage=dict(usage) if isinstance(usage, Mapping) else {},
        artifact_digest=str(found.get("artifact_digest") or ""),
        latency_ms=float(found.get("latency_ms") or 0.0),
        ttft_ms=float(found.get("ttft_ms") or 0.0),
    )


def ask_chat(
    gateway: str,
    credential: str,
    *,
    hotkey: str,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
    max_tokens: int = CHAT_MAX_TOKENS,
    timeout: float = TIMEOUT_SECONDS,
) -> Chatted:
    body = {
        "hotkey": hotkey,
        "model": model,
        "kind": "chat",
        "messages": [dict(m) for m in messages],
        "tools": [dict(t) for t in tools or []],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
    }
    try:
        with httpx.Client(base_url=gateway.rstrip("/"), timeout=timeout) as http:
            response = http.post(PROBE_PATH, json=body, headers={CREDENTIAL_HEADER: credential})
    except httpx.HTTPError as exc:
        return Chatted(False, error=f"gateway unreachable: {exc.__class__.__name__}")
    if response.status_code != 200:
        detail = response.text[:200]
        return Chatted(False, error=f"gateway answered {response.status_code}: {detail}")
    try:
        found = response.json()
    except ValueError:
        return Chatted(False, error="the gateway answered with something not json")
    if not isinstance(found, Mapping):
        return Chatted(False, error="the gateway answered with something not an object")
    return chatted(found)
