from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from microtensor.artifact import contract
from microtensor.core.items import Item
from microtensor.validator.v2 import grade
from microtensor.validator.v2.context import HARNESS, EpochContext, render_prompt
from microtensor.validator.v2.results import Condition, HarnessRun, ItemRun
from microtensor.validator.v2.sandbox import Sandbox, SandboxHost, call_main
from microtensor.validator.v2.slm import SlmClient, SlmError

log = logging.getLogger("microtensor.validator.v2.harness_test")
BARE = "B0"


def _graded(
    context: EpochContext, item: Item, output: Any, latency: float, ok: bool, reason: str
) -> ItemRun:
    if not ok:
        return ItemRun(
            item.item_id,
            item.family,
            output,
            False,
            0.0,
            reason,
            latency_ms=latency,
            answered=False,
        )
    try:
        verdict = grade.grade(context.grader, output, item.label, context.items.databases)
    except grade.GradeError as exc:
        verdict = grade.Grade(0.0, False, str(exc))
    return ItemRun(
        item.item_id,
        item.family,
        output,
        verdict.success,
        verdict.metric,
        verdict.reason,
        latency_ms=latency,
    )


def run_condition(
    context: EpochContext,
    sandbox: Sandbox,
    name: str,
    items: Sequence[Item],
    *,
    timeout: float,
) -> Condition:
    session = sandbox.session
    session.reset_counters()
    runs: list[ItemRun] = []
    for item in items:
        with session.lock:
            session.current_item = item.item_id
        call = call_main(
            sandbox.main_url,
            contract.HARNESS_RUN,
            {"item_id": item.item_id, "task": dict(item.task)},
            timeout=timeout,
        )
        output = call.body.get("answer") if call.ok else None
        runs.append(_graded(context, item, output, call.latency_ms, call.ok, call.error))
    with session.lock:
        tokens = session.tokens_in + session.tokens_out
        session.current_item = ""
    return Condition(name=name, runs=runs, tokens=tokens)


def run_bare(
    context: EpochContext, client: SlmClient, items: Sequence[Item], *, max_tokens: int = 256
) -> Condition:
    runs: list[ItemRun] = []
    tokens = 0
    for item in items:
        prompt = render_prompt(context.slm_prompt, item.task)
        try:
            generation = client.generate(prompt, max_tokens=max_tokens, stop=("\n\n", ";"))
        except SlmError as exc:
            runs.append(_graded(context, item, "", 0.0, False, str(exc)))
            continue
        tokens += generation.prompt_tokens + generation.completion_tokens
        runs.append(_graded(context, item, generation.text, generation.total_ms, True, ""))
    return Condition(name=BARE, runs=runs, tokens=tokens)


def run(
    context: EpochContext,
    host: SandboxHost,
    sandbox: Sandbox,
    *,
    items: Sequence[Item] | None = None,
    timeout: float,
) -> HarnessRun:
    chosen = list(items) if items is not None else context.items.items(HARNESS)
    miner = run_condition(context, sandbox, "H", chosen, timeout=timeout)
    baselines = [run_bare(context, SlmClient(sandbox.slm_url), chosen)]
    variants: list[Path] = [context.reference.harness_dir, *context.reference.variant_dirs]
    for index, variant in enumerate(variants, start=1):
        try:
            host.restart_main(sandbox, variant)
        except Exception as exc:
            log.warning("reference harness %d could not start: %s", index, exc)
            continue
        baselines.append(run_condition(context, sandbox, f"B{index}", chosen, timeout=timeout))
    return HarnessRun(miner=miner, baselines=baselines)
