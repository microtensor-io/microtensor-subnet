from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from microtensor.core.items import Item
from microtensor.scoring.stats import percentile
from microtensor.validator import grade
from microtensor.validator.context import ESCALATION, SYSTEM, EpochContext, render_prompt
from microtensor.validator.relay import REFERENCE, RelayError, TicketDesk
from microtensor.validator.results import ItemRun, ReferenceRun

log = logging.getLogger("microtensor.validator.reference")
REFERENCE_MINER = "reference"
REDO_MAX_TOKENS = 512


def baseline_model(context: EpochContext) -> dict[str, Any]:
    models = list(context.definition.get("baseline_models") or [])
    if not models:
        raise ValueError("the arena names no baseline model")
    return dict(models[0])


def redo_prompt(context: EpochContext, item: Item) -> list[dict[str, str]]:
    template = str(
        context.definition.get("reference_prompt")
        or "You answer with one SQLite query and nothing else.\n\n"
        "Schema:\n{schema}\n\nQuestion: {question}"
    )
    return [{"role": "user", "content": render_prompt(template, item.task)}]


def redo_cap_micros(context: EpochContext, model: dict[str, Any], prompt_chars: int) -> int:
    usd_in = float(model.get("usd_in_per_mtok") or 0.0)
    usd_out = float(model.get("usd_out_per_mtok") or 0.0)
    estimate = prompt_chars // 3 + 64
    return round(estimate * usd_in + REDO_MAX_TOKENS * usd_out) * 2 + 1


def _answer(
    context: EpochContext, desk: TicketDesk, model: dict[str, Any], item: Item, purpose: str
) -> ItemRun:
    messages = redo_prompt(context, item)
    chars = sum(len(m["content"]) for m in messages)
    try:
        opaque = desk.issue(
            miner=REFERENCE_MINER,
            item_id=item.item_id,
            purpose=purpose,
            cap_micros=redo_cap_micros(context, model, chars),
        )
        answer = desk.complete(
            opaque, model=str(model["model"]), messages=messages, max_tokens=REDO_MAX_TOKENS
        )
    except RelayError as exc:
        return ItemRun(
            item.item_id, item.family, "", False, 0.0, f"relay: {exc.code}", answered=False
        )
    cost = desk.cost_for(opaque)
    desk.forget(opaque)
    try:
        verdict = grade.grade(context.grader, answer.text, item.label, context.items.databases)
    except grade.GradeError as exc:
        verdict = grade.Grade(0.0, False, str(exc))
    return ItemRun(
        item.item_id,
        item.family,
        answer.text,
        verdict.success,
        verdict.metric,
        verdict.reason,
        prompt_tokens=answer.receipt.tokens_in,
        completion_tokens=answer.receipt.tokens_out,
        latency_ms=float(answer.receipt.latency_ms),
        cost_micros=cost,
    )


def run(
    context: EpochContext,
    desk: TicketDesk,
    *,
    escalation_items: Sequence[Item] | None = None,
    system_items: Sequence[Item] | None = None,
) -> ReferenceRun:
    model = baseline_model(context)
    system_rows = list(system_items) if system_items is not None else context.items.items(SYSTEM)
    redo_rows = (
        list(escalation_items) if escalation_items is not None else context.items.items(ESCALATION)
    )
    system_answers = {
        item.item_id: _answer(context, desk, model, item, REFERENCE) for item in system_rows
    }
    redo_answers = {
        item.item_id: _answer(context, desk, model, item, REFERENCE) for item in redo_rows
    }
    answered = [r for r in system_answers.values() if r.answered]
    return ReferenceRun(
        system_answers=system_answers,
        redo_answers=redo_answers,
        cost_micros=sum(r.cost_micros for r in system_answers.values()),
        latency_p95_ms=percentile([r.latency_ms for r in answered], 0.95),
    )
