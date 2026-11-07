from __future__ import annotations

import json
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from microtensor_subnet.artifact import contract
from microtensor_subnet.validator import grade
from microtensor_subnet.validator.context import render_prompt
from microtensor_subnet.validator.pools import router as router_pool
from microtensor_subnet.validator.sandbox import Sandbox, call_main, tools
from microtensor_subnet.validator.sandbox.slm import SlmClient, SlmError

DEFAULT_FLOOR = 0.9
ROUTE_MODEL: Final[str] = "model"
MODEL_SEED: Final[int] = 0


@dataclass(slots=True)
class Held:
    model: str
    artifact_digest: str
    entry: dict[str, Any]
    sandbox: Sandbox
    gpu: str
    slot: int
    lock: threading.Lock = field(default_factory=threading.Lock)


def _text(answer: Any) -> str:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, Mapping):
        for key in ("sql", "answer", "text", "output"):
            if isinstance(answer.get(key), str):
                return str(answer[key])
    return json.dumps(answer, ensure_ascii=False) if answer is not None else ""


def _probability(held: Held, task: Mapping[str, Any], found: Any) -> float:
    if not held.sandbox.router_url:
        return 1.0
    query = render_prompt(str(held.entry.get("slm_prompt") or ""), task)
    sql = str(held.entry.get("grader", "")) == grade.SQL_EXECUTION
    draft = router_pool.normalise(_text(found), sql=sql)
    probability, _ = router_pool.decide(held.sandbox.router_url, query, draft)
    return probability


def answer_from_model(
    held: Held,
    request: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    timeout: float,
    on_first: Callable[[], None] | None = None,
) -> dict[str, Any]:
    entry = held.entry
    limit = int(entry.get("max_output") or 0)
    asked = int(request.get("max_tokens") or 0) or limit
    prompt = render_prompt(str(entry.get("slm_prompt") or ""), task)
    if on_first is not None:
        on_first()
    started = time.perf_counter()
    try:
        made = SlmClient(held.sandbox.slm_url, timeout=timeout).generate(
            prompt,
            max_tokens=min(asked, limit) if limit else asked,
            stop=tuple(str(token) for token in entry.get("stop") or ()),
            seed=MODEL_SEED,
        )
    except SlmError as exc:
        return {"error": f"model: {exc}"}
    probability = _probability(held, task, made.text)
    return {
        "answer": made.text,
        "p": probability,
        "floor_met": probability >= float(request.get("floor") or DEFAULT_FLOOR),
        "verified": True,
        "trace": "",
        "usage": {"prompt_tokens": made.prompt_tokens, "completion_tokens": made.completion_tokens},
        "latency_ms": (time.perf_counter() - started) * 1000,
        "ttft_ms": made.ttft_ms,
        "finish_reason": made.finish_reason or "stop",
    }


def answer(
    held: Held,
    request: Mapping[str, Any],
    *,
    timeout: float,
    on_first: Callable[[], None] | None = None,
) -> dict[str, Any]:
    raw = request.get("task")
    task: dict[str, Any] = dict(raw) if isinstance(raw, Mapping) else {}
    if str(held.entry.get("route") or "") == ROUTE_MODEL:
        return answer_from_model(held, request, task, timeout=timeout, on_first=on_first)
    floor = float(request.get("floor") or DEFAULT_FLOOR)
    correlation = str(request.get("correlation") or secrets.token_hex(6))
    session = held.sandbox.session
    started = time.perf_counter()
    with held.lock:
        schema = str(task.get("schema") or "")
        if schema:
            session.register_item(correlation, tools.database_from_schema(schema))
        session.reset_counters()
        with session.lock:
            session.current_item = correlation
            session.first_call = on_first
        try:
            call = call_main(
                held.sandbox.main_url,
                contract.HARNESS_RUN,
                {"item_id": correlation, "task": task},
                timeout=timeout,
            )
            if not call.ok:
                return {"error": f"harness: {call.error}"}
            found = call.body.get("answer")
            verified = bool((call.body.get("self_check") or {}).get("pass"))
            trace = str(call.body.get("trace") or "")
            probability = _probability(held, task, found)
            with session.lock:
                usage = {
                    "prompt_tokens": session.tokens_in,
                    "completion_tokens": session.tokens_out,
                }
                session.current_item = ""
        finally:
            with session.lock:
                session.first_call = None
            session.forget_item(correlation)
    return {
        "answer": found,
        "p": probability,
        "floor_met": verified and probability >= floor,
        "verified": verified,
        "trace": trace,
        "usage": usage,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "finish_reason": "stop",
    }
