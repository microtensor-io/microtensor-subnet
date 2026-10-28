from __future__ import annotations

import logging
from collections.abc import Sequence

from microtensor_subnet.core.items import Item
from microtensor_subnet.scoring.stats import percentile
from microtensor_subnet.validator.context import SLM, EpochContext, render_prompt
from microtensor_subnet.validator.progress import Progress, Tally
from microtensor_subnet.validator.results import ItemRun, SlmRun
from microtensor_subnet.validator.sandbox.slm import SlmClient, SlmError

log = logging.getLogger("microtensor_subnet.validator.pools.slm")


def _run_item(context: EpochContext, client: SlmClient, item: Item, max_tokens: int) -> ItemRun:
    prompt = render_prompt(context.slm_prompt, item.task)
    try:
        generation = client.generate(prompt, max_tokens=max_tokens, stop=context.stop)
    except SlmError as exc:
        return ItemRun(item.item_id, item.family, "", False, 0.0, str(exc), answered=False)
    verdict = context.grade(item, generation.text)
    return ItemRun(
        item.item_id,
        item.family,
        generation.text,
        verdict.success,
        verdict.metric,
        verdict.reason,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        latency_ms=generation.total_ms,
        ttft_ms=generation.ttft_ms,
        tpot_ms=generation.tpot_ms,
    )


def run(
    context: EpochContext,
    client: SlmClient,
    *,
    memory_bytes: int,
    parameters: int,
    max_tokens: int | None = None,
    items: Sequence[Item] | None = None,
    progress: Progress | None = None,
) -> SlmRun:
    budget = max_tokens or context.max_tokens
    chosen = list(items) if items is not None else context.items.originals(SLM)
    twins = context.items.twins_of(SLM)
    runs: list[ItemRun] = []
    twin_runs: list[ItemRun] = []
    tally = Tally(progress, len(chosen))
    for item in tally.over(chosen):
        runs.append(_run_item(context, client, item, budget))
        twin = twins.get(item.item_id)
        if twin is not None:
            twin_runs.append(_run_item(context, client, twin, budget))
    answered = [r for r in runs if r.answered]
    return SlmRun(
        runs=runs,
        twins=twin_runs,
        ttft_p95_ms=percentile([r.ttft_ms for r in answered], 0.95),
        tpot_p95_ms=percentile([r.tpot_ms for r in answered if r.completion_tokens > 1], 0.95),
        vram_peak_bytes=memory_bytes,
        parameters=parameters,
    )
