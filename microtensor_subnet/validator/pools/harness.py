from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from microtensor_subnet.artifact import contract
from microtensor_subnet.core.items import Item
from microtensor_subnet.scoring.stats import percentile
from microtensor_subnet.validator.context import HARNESS, EpochContext, render_prompt
from microtensor_subnet.validator.progress import BASELINE_PHASE, MINER_PHASE, Progress, Tally
from microtensor_subnet.validator.results import Condition, HarnessRun, ItemRun
from microtensor_subnet.validator.sandbox import Sandbox, SandboxHost, call_main
from microtensor_subnet.validator.sandbox.slm import SlmClient, SlmError

log = logging.getLogger("microtensor_subnet.validator.pools.harness")
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
    verdict = context.grade(item, output)
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
    items: Iterable[Item],
    *,
    timeout: float,
) -> Condition:
    session = sandbox.session
    session.reset_counters()
    runs: list[ItemRun] = []
    added: list[float] = []
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
        added.append(max(0.0, call.latency_ms - session.model_ms(item.item_id)))
    with session.lock:
        tokens = session.tokens_in + session.tokens_out
        session.current_item = ""
    return Condition(name=name, runs=runs, tokens=tokens, added_ms=added)


def run_bare(
    context: EpochContext,
    client: SlmClient,
    items: Iterable[Item],
    *,
    max_tokens: int | None = None,
) -> Condition:
    budget = max_tokens or context.max_tokens
    runs: list[ItemRun] = []
    tokens = 0
    for item in items:
        prompt = render_prompt(context.slm_prompt, item.task)
        try:
            generation = client.generate(prompt, max_tokens=budget, stop=context.stop)
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
    progress: Progress | None = None,
) -> HarnessRun:
    chosen = list(items) if items is not None else context.items.items(HARNESS)
    variants: list[Path] = [context.reference.harness_dir, *context.reference.variant_dirs]
    tally = Tally(
        progress,
        context.pool_size(HARNESS),
        len(chosen) * (len(variants) + 2),
        detail={"phase": MINER_PHASE},
    )
    miner = run_condition(context, sandbox, "H", tally.over(chosen), timeout=timeout)
    tally.mark(phase=BASELINE_PHASE)
    baselines = [run_bare(context, SlmClient(sandbox.slm_url), tally.over(chosen))]
    limit = float(context.harness_limits.get("added_wall_p95_ms") or 0)
    added = percentile(miner.added_ms, 0.95)
    limits_ok = not limit or added <= limit
    reason = "" if limits_ok else f"added wall time p95 {added:.0f} ms is over {limit:.0f} ms"
    for index, variant in enumerate(variants, start=1):
        try:
            host.restart_main(sandbox, variant)
        except Exception as exc:
            log.warning("reference harness %d could not start: %s", index, exc)
            tally.advance(len(chosen))
            continue
        baselines.append(
            run_condition(context, sandbox, f"B{index}", tally.over(chosen), timeout=timeout)
        )
    return HarnessRun(miner=miner, baselines=baselines, limits_ok=limits_ok, limits_reason=reason)
