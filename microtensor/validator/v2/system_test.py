from __future__ import annotations

import logging
from collections.abc import Sequence

from microtensor.artifact import contract
from microtensor.core.items import Item
from microtensor.validator.v2 import grade
from microtensor.validator.v2.context import SYSTEM, EpochContext
from microtensor.validator.v2.relay import SYSTEM as SYSTEM_PURPOSE
from microtensor.validator.v2.relay import RelayError, TicketDesk
from microtensor.validator.v2.results import FloorRun, ItemRun, ReferenceRun, SystemRun
from microtensor.validator.v2.sandbox import Sandbox, call_main

log = logging.getLogger("microtensor.validator.v2.system_test")
TICKET_MULTIPLE = 1.5


def run(
    context: EpochContext,
    sandbox: Sandbox,
    desk: TicketDesk,
    *,
    miner: str,
    reference: ReferenceRun,
    timeout: float,
    token_price: float,
    items: Sequence[Item] | None = None,
) -> SystemRun:
    chosen = list(items) if items is not None else context.items.items(SYSTEM)
    session = sandbox.session
    floors: list[FloorRun] = []
    for floor in context.floors:
        runs: list[ItemRun] = []
        for item in chosen:
            redo = reference.system_answers.get(item.item_id)
            cap = max(1, int((redo.cost_micros if redo else 0) * TICKET_MULTIPLE))
            try:
                opaque = desk.issue(
                    miner=miner,
                    item_id=f"{item.item_id}@{floor}",
                    purpose=SYSTEM_PURPOSE,
                    cap_micros=cap,
                )
            except RelayError as exc:
                runs.append(
                    ItemRun(
                        item.item_id,
                        item.family,
                        "",
                        False,
                        0.0,
                        f"ticket: {exc.code}",
                        answered=False,
                    )
                )
                continue
            session.reset_counters()
            with session.lock:
                session.current_item = item.item_id
            call = call_main(
                sandbox.main_url,
                contract.SYSTEM_RUN,
                {
                    "item_id": item.item_id,
                    "task": dict(item.task),
                    "floor": floor,
                    "ticket": opaque,
                },
                timeout=timeout,
            )
            with session.lock:
                slm_tokens = session.tokens_in + session.tokens_out
                session.current_item = ""
            escalated_cost = desk.cost_for(opaque)
            desk.forget(opaque)
            cost = escalated_cost + round(slm_tokens * token_price)
            if not call.ok:
                runs.append(
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
            output = call.body.get("answer")
            route = str(call.body.get("route", ""))
            probability = call.body.get("p")
            try:
                verdict = grade.grade(context.grader, output, item.label, context.items.databases)
            except grade.GradeError as exc:
                verdict = grade.Grade(0.0, False, str(exc))
            runs.append(
                ItemRun(
                    item.item_id,
                    item.family,
                    output,
                    verdict.success,
                    verdict.metric,
                    verdict.reason,
                    latency_ms=call.latency_ms,
                    cost_micros=cost,
                    escalated=route == "escalated" or escalated_cost > 0,
                    probability=float(probability)
                    if isinstance(probability, (int, float))
                    else None,
                )
            )
        floors.append(FloorRun(floor=floor, runs=runs))
    return SystemRun(
        floors=floors,
        reference_cost_micros=reference.cost_micros,
        reference_latency_p95_ms=reference.latency_p95_ms,
    )
