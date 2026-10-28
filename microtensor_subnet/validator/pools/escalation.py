from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from microtensor_subnet.artifact import contract
from microtensor_subnet.core.items import Item
from microtensor_subnet.validator.context import ESCALATION, EpochContext
from microtensor_subnet.validator.progress import Progress, Tally
from microtensor_subnet.validator.relay import REPAIR, RelayError, TicketDesk
from microtensor_subnet.validator.results import EscalationRun, ItemRun, ReferenceRun
from microtensor_subnet.validator.sandbox import Sandbox, call_main

log = logging.getLogger("microtensor_subnet.validator.pools.escalation")
TICKET_MULTIPLE = 1.5


def wrong_drafts(
    context: EpochContext, reference: ReferenceRun, reference_drafts: Mapping[str, tuple[str, bool]]
) -> list[tuple[Item, str]]:
    found: list[tuple[Item, str]] = []
    for item in context.items.items(ESCALATION):
        draft = reference_drafts.get(item.item_id)
        if draft is None or draft[1]:
            continue
        if item.item_id not in reference.redo_answers:
            continue
        found.append((item, draft[0]))
    return found


def run(
    context: EpochContext,
    sandbox: Sandbox,
    desk: TicketDesk,
    *,
    miner: str,
    drafts: Sequence[tuple[Item, str]],
    reference: ReferenceRun,
    timeout: float,
    token_price: float,
    progress: Progress | None = None,
) -> EscalationRun:
    repairs: list[ItemRun] = []
    session = sandbox.session
    tally = Tally(progress, context.pool_size(ESCALATION), len(drafts))
    for item, draft in tally.over(drafts):
        redo = reference.redo_answers.get(item.item_id)
        redo_cost = redo.cost_micros if redo is not None else 0
        cap = max(1, int(redo_cost * TICKET_MULTIPLE))
        try:
            opaque = desk.issue(miner=miner, item_id=item.item_id, purpose=REPAIR, cap_micros=cap)
        except RelayError as exc:
            repairs.append(
                ItemRun(
                    item.item_id, item.family, "", False, 0.0, f"ticket: {exc.code}", answered=False
                )
            )
            continue
        session.reset_counters()
        with session.lock:
            session.current_item = item.item_id
        call = call_main(
            sandbox.main_url,
            contract.ESCALATION_REPAIR,
            {
                "item_id": item.item_id,
                "task": dict(item.task),
                "draft": draft,
                "trace": "",
                "errors": "the draft returned the wrong result",
                "ticket": opaque,
            },
            timeout=timeout,
        )
        with session.lock:
            slm_tokens = session.tokens_in + session.tokens_out
            session.current_item = ""
        cost = desk.cost_for(opaque) + round(slm_tokens * token_price)
        desk.forget(opaque)
        if not call.ok:
            repairs.append(
                ItemRun(
                    item.item_id,
                    item.family,
                    "",
                    False,
                    0.0,
                    call.error,
                    latency_ms=call.latency_ms,
                    cost_micros=cost,
                    answered=False,
                )
            )
            continue
        output: Any = call.body.get("answer")
        verdict = context.grade(item, output)
        repairs.append(
            ItemRun(
                item.item_id,
                item.family,
                output,
                verdict.success,
                verdict.metric,
                verdict.reason,
                latency_ms=call.latency_ms,
                cost_micros=cost,
            )
        )
    redo_rows = [
        reference.redo_answers[item.item_id]
        for item, _ in drafts
        if item.item_id in reference.redo_answers
    ]
    redo_pass = sum(1 for r in redo_rows if r.success) / len(redo_rows) if redo_rows else 0.0
    return EscalationRun(
        repairs=repairs,
        redo_pass_rate=redo_pass,
        redo_cost_micros=sum(r.cost_micros for r in redo_rows),
    )
