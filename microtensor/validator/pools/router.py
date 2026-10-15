from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from microtensor.artifact import contract
from microtensor.core.items import Item
from microtensor.scoring.stats import percentile
from microtensor.validator import grade
from microtensor.validator.context import ROUTER, EpochContext, render_prompt
from microtensor.validator.results import RouterRun, SourceRun

log = logging.getLogger("microtensor.validator.pools.router")

OWN = "own"
REFERENCE = "reference"
PEER = "peer"
ADVERSARIAL = "adversarial"
SOURCES = (OWN, REFERENCE, PEER, ADVERSARIAL)
PEER_CAP_PER_COLDKEY = 50
DRAFTS_PER_SOURCE = 250
LATENCY_GATE_MS = 100.0
_WS = re.compile(r"\s+")
_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)


def normalise(draft: str, *, sql: bool) -> str:
    text = unicodedata.normalize("NFC", str(draft or ""))
    if sql:
        try:
            import sqlglot

            return str(sqlglot.transpile(text, read="sqlite", write="sqlite", pretty=False)[0])
        except Exception:
            text = _COMMENT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def decide(
    router_url: str, query: str, draft: str, *, timeout: float = 30.0
) -> tuple[float, float]:
    with httpx.Client(timeout=timeout) as http:
        response = http.post(
            router_url.rstrip("/") + contract.ROUTER_DECIDE,
            json={"input": {"query": query, "draft": draft}},
        )
    response.raise_for_status()
    body = response.json()
    answers = body.get("answers") or []
    probability = float(answers[0].get("probability")) if answers else 0.0
    return min(1.0, max(0.0, probability)), float(body.get("latency_ms") or 0.0)


def _label(context: EpochContext, item: Item, draft: str) -> int:
    try:
        return (
            1
            if grade.grade(context.grader, draft, item.label, context.items.databases).success
            else 0
        )
    except grade.GradeError:
        return 0


def build_sources(
    context: EpochContext,
    *,
    own: Mapping[str, tuple[str, bool]],
    reference: Mapping[str, tuple[str, bool]],
    peers: Mapping[str, Mapping[str, tuple[str, bool]]],
    peer_coldkeys: Mapping[str, str],
) -> dict[str, list[tuple[Item, str, int]]]:
    items = {item.item_id: item for item in context.items.items(ROUTER)}
    sql = context.grader == grade.SQL_EXECUTION
    found: dict[str, list[tuple[Item, str, int]]] = {s: [] for s in SOURCES}
    for item_id, item in items.items():
        if item_id in own:
            draft, ok = own[item_id]
            found[OWN].append((item, normalise(draft, sql=sql), 1 if ok else 0))
        if item_id in reference:
            draft, ok = reference[item_id]
            found[REFERENCE].append((item, normalise(draft, sql=sql), 1 if ok else 0))
    per_coldkey: dict[str, int] = {}
    for hotkey, drafts in sorted(peers.items()):
        coldkey = peer_coldkeys.get(hotkey, hotkey)
        for item_id, item in items.items():
            if item_id not in drafts or per_coldkey.get(coldkey, 0) >= PEER_CAP_PER_COLDKEY:
                continue
            draft, ok = drafts[item_id]
            found[PEER].append((item, normalise(draft, sql=sql), 1 if ok else 0))
            per_coldkey[coldkey] = per_coldkey.get(coldkey, 0) + 1
    for row in context.items.adversarial:
        found_item: Item | None = items.get(str(row.get("item_id", "")))
        draft = str(row.get("draft", ""))
        if found_item is None and isinstance(row.get("task"), dict):
            found_item = Item(
                str(row.get("item_id")),
                str(row.get("family", "")),
                dict(row["task"]),
                dict(row.get("label") or {}),
            )
        if found_item is None or not draft:
            continue
        label = int(row["y"]) if row.get("y") is not None else _label(context, found_item, draft)
        found[ADVERSARIAL].append((found_item, normalise(draft, sql=sql), label))
    for source, rows in found.items():
        found[source] = rows[:DRAFTS_PER_SOURCE]
    return found


def run(
    context: EpochContext,
    router_url: str,
    sources: Mapping[str, Sequence[tuple[Item, str, int]]],
) -> RouterRun:
    runs: list[SourceRun] = []
    latencies: list[float] = []
    for source, rows in sources.items():
        probabilities: list[float] = []
        labels: list[int] = []
        for item, draft, label in rows:
            query = render_prompt(context.slm_prompt, item.task)
            try:
                probability, latency = decide(router_url, query, draft)
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                log.warning("router failed on %s: %s", item.item_id, exc)
                return RouterRun(sources=[], loaded=False, reason=f"router failed: {exc}")
            probabilities.append(probability)
            labels.append(label)
            latencies.append(latency)
        runs.append(SourceRun(source=source, probabilities=probabilities, labels=labels))
    return RouterRun(sources=runs, latency_p95_ms=percentile(latencies, 0.95))


def as_detail(run_: RouterRun) -> dict[str, Any]:
    return {
        "sources": {s.source: len(s.labels) for s in run_.sources},
        "latency_p95_ms": run_.latency_p95_ms,
        "loaded": run_.loaded,
        "reason": run_.reason,
    }
