from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from microtensor.chain.wallet import hotkey_address, sign_bytes
from microtensor.serving import canonical, verify
from microtensor.serving.client import (
    HOTKEY_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    ServerError,
    signing_bytes,
)

log = logging.getLogger("microtensor.serving.probe")

TIMEOUT_SECONDS = 300
CREDENTIAL_HEADER = "x-mt-credential"

OPERATORS_PATH = "/v1/inference/validators/operators"
PROBES_PATH = "/v1/inference/validators/probes"
WITHDRAW_PATH = "/v1/inference/validators/withdraw"
GATEWAY_PROBE_PATH = "/v1/operators/probe"

DEFAULT_MAX_TOKENS = 48
PROMPTS: tuple[str, ...] = (
    "List three primary colours.",
    "What is the capital of Japan?",
    "Explain what a hash function does, in one sentence.",
    "Name two things you would find in a kitchen.",
    "Count from one to five.",
    "Give a one line definition of gravity.",
    "What comes after Tuesday?",
    "Summarise why water boils, briefly.",
)


class ProbeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Answered:
    hotkey: str
    model: str
    artifact_digest: str
    text: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
    latency_ms: float
    escalated: bool = False
    answered_by: str = "front"
    router_features: dict[str, float] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.completion_tokens


def _url(base: str, path: str) -> str:
    return urllib.parse.urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def _call(url: str, *, method: str, body: bytes | None, headers: dict[str, str]) -> Any:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as answer:
            raw = answer.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise ServerError(f"{exc.code}: {detail[:300]}") from exc
    except urllib.error.URLError as exc:
        raise ServerError(f"could not reach {url}: {exc.reason}") from exc
    except (OSError, ValueError) as exc:
        raise ServerError(f"could not read {url}: {exc}") from exc
    return json.loads(raw) if raw else {}


def _signed(
    base: str, wallet: Any, *, method: str, path: str, payload: dict[str, Any] | None = None
) -> Any:
    body = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        if payload is not None
        else b""
    )
    timestamp = str(time.time())
    headers = {
        HOTKEY_HEADER: hotkey_address(wallet),
        TIMESTAMP_HEADER: timestamp,
        SIGNATURE_HEADER: sign_bytes(wallet, signing_bytes(method, path, timestamp, body)),
    }
    if payload is not None:
        headers["content-type"] = "application/json"
    return _call(_url(base, path), method=method, body=body or None, headers=headers)


def to_probe(server: str, wallet: Any, *, model: str = "") -> list[dict[str, Any]]:
    path = OPERATORS_PATH + (f"?model={urllib.parse.quote(model)}" if model else "")
    found = _signed(server, wallet, method="GET", path=path)
    return list(found.get("operators", []))


def ask(
    gateway: str,
    credential: str,
    *,
    hotkey: str,
    model: str,
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> Answered:
    payload = {
        "hotkey": hotkey,
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    headers = {"content-type": "application/json", CREDENTIAL_HEADER: credential}
    started = time.perf_counter()
    found = _call(_url(gateway, GATEWAY_PROBE_PATH), method="POST", body=body, headers=headers)
    tokens = dict(found.get("tokens") or {})
    return Answered(
        hotkey=hotkey,
        model=model,
        artifact_digest=str(found.get("artifact_digest", "")),
        text=str(found.get("text", "")),
        prompt_tokens=[int(t) for t in tokens.get("prompt", [])],
        completion_tokens=[int(t) for t in tokens.get("completion", [])],
        latency_ms=(time.perf_counter() - started) * 1000.0,
        escalated=bool(found.get("escalated")),
        answered_by=str(found.get("answered_by", "front")),
        router_features={k: float(v) for k, v in dict(found.get("router_features") or {}).items()},
    )


def judge(model: Any, answered: Answered, calibration: verify.Calibration) -> verify.Judgement:
    if answered.empty:
        return verify.Judgement(
            verdict=verify.UNPROVEN,
            score=0.0,
            threshold=calibration.threshold,
            reason="the operator returned no tokens",
        )
    logits = canonical.canonical_logits(model, answered.prompt_tokens, answered.completion_tokens)
    found = verify.statistic(logits, answered.completion_tokens, mode=verify.GREEDY)
    return verify.judge(found, calibration)


def artifact_model(artifact: Path) -> Any:
    return canonical.open_artifact(artifact)


def report(
    server: str, wallet: Any, *, hotkey: str, model: str, failed: bool, verdict: str = ""
) -> dict[str, Any]:
    payload = {
        "operator_hotkey": hotkey,
        "model": model,
        "failed": bool(failed),
        "verdict": verdict,
    }
    return dict(_signed(server, wallet, method="POST", path=PROBES_PATH, payload=payload))


def withdraw(server: str, wallet: Any, *, hotkey: str, reason: str) -> dict[str, Any]:
    payload = {"operator_hotkey": hotkey, "reason": reason}
    return dict(_signed(server, wallet, method="POST", path=WITHDRAW_PATH, payload=payload))


def prompt_for(rng: random.Random) -> str:
    return rng.choice(PROMPTS)


def routing_holds(answered: Answered, router: Any, features: Mapping[str, float]) -> bool:
    from microtensor.serving import cascade

    if router is None:
        return not answered.escalated
    chosen = str(router.choose(features))
    return (chosen == cascade.ESCALATE) is answered.escalated


def judge_system(
    engines: Mapping[str, Any],
    answered: Answered,
    calibration: verify.Calibration,
    router: Any = None,
    features: Mapping[str, float] | None = None,
) -> verify.Judgement:
    from microtensor.serving import cascade

    if router is not None:
        found = dict(features or answered.router_features)
        if not found:
            return verify.Judgement(
                verdict=verify.UNPROVEN,
                score=0.0,
                threshold=calibration.threshold,
                reason="no router features to check the escalation against",
            )
        if not routing_holds(answered, router, found):
            return verify.Judgement(
                verdict=verify.CHEAT,
                score=0.0,
                threshold=calibration.threshold,
                reason=f"the router says {'escalate' if answered.escalated else 'resolve'} "
                f"and the operator did the opposite",
            )

    role = cascade.SPECIALIST if answered.escalated else cascade.FRONT
    engine = engines.get(role)
    if engine is None:
        return verify.Judgement(
            verdict=verify.UNPROVEN,
            score=0.0,
            threshold=calibration.threshold,
            reason=f"this validator holds no {role} artifact for {answered.model}",
        )
    return judge(engine, answered, calibration)
