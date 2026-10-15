from __future__ import annotations

import json
import secrets
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from microtensor_subnet.artifact import contract
from microtensor_subnet.validator import grade
from microtensor_subnet.validator.context import render_prompt
from microtensor_subnet.validator.pools import router as router_pool
from microtensor_subnet.validator.sandbox import Sandbox, call_main, tools

DEFAULT_FLOOR = 0.9


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


def answer(held: Held, request: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
    raw = request.get("task")
    task: dict[str, Any] = dict(raw) if isinstance(raw, Mapping) else {}
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
            probability = 1.0
            if held.sandbox.router_url:
                query = render_prompt(str(held.entry.get("slm_prompt") or ""), task)
                sql = str(held.entry.get("grader", "")) == grade.SQL_EXECUTION
                draft = router_pool.normalise(_text(found), sql=sql)
                probability, _ = router_pool.decide(held.sandbox.router_url, query, draft)
            with session.lock:
                usage = {
                    "prompt_tokens": session.tokens_in,
                    "completion_tokens": session.tokens_out,
                }
                session.current_item = ""
        finally:
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
