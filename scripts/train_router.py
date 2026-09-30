from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microtensor.core.constants import ALLOWED_ROUTER_FEATURES
from microtensor.core.protocol import ArtifactFormat, LoadManifest
from microtensor.core.system import EscalationRef
from microtensor.core.tracks import get_track
from microtensor.harness.engines.gguf import GgufEngine
from microtensor.harness.engines.router import (
    Clause,
    Decision,
    ThresholdRouter,
    grams,
    typicality,
)
from microtensor.harness.sdk import EscalationCall, Runtime, engine_small, openai_escalation
from microtensor.scoring.metrics import score_task
from microtensor.tasks.corpus import load_corpus

FOLDS = 5
QUANTILES = tuple(q / 20 for q in range(1, 20))


@dataclass(frozen=True, slots=True)
class Row:
    features: dict[str, float]
    small: float
    escalated: float


def _never(_: str) -> EscalationCall:
    raise RuntimeError("recording runs never escalate")


def record(args: argparse.Namespace) -> tuple[list[Row], list[str]]:
    track = get_track(args.track)
    engine = GgufEngine()
    engine.load(
        args.model,
        LoadManifest(
            format=ArtifactFormat.GGUF,
            quantization="",
            entrypoint=args.model.name,
            max_input={"tokens": args.max_input_tokens},
        ),
    )
    ask = (
        openai_escalation(args.escalation_url, args.escalation_model)
        if args.escalation_url
        else None
    )
    tasks = load_corpus(args.corpus, args.track).tasks[: args.limit or None]
    prompts: list[str] = []
    raw: list[tuple[dict[str, float], float, float]] = []
    for task in tasks:
        runtime = Runtime(
            args.harness,
            ThresholdRouter((), Decision.RESOLVE),
            sorted(ALLOWED_ROUTER_FEATURES),
            engine_small(engine, chat=track.chat, max_output_tokens=task.max_output_tokens),
            _never,
            escalation=EscalationRef(model="local/none", revision="0" * 40),
            system_digest="sha256:local",
            hotkey="local",
        )
        trace = runtime.run(0, task.ref, task.prompt, task.inputs)
        small = score_task(track.metric, trace.final, task.gold)
        escalated = (
            score_task(track.metric, ask(trace.small.prompt).output, task.gold)
            if ask is not None
            else args.escalation_accuracy
        )
        prompts.append(trace.small.prompt)
        raw.append((dict(trace.router.features), small, escalated))
    rows: list[Row] = []
    for index, (features, small, escalated) in enumerate(raw):
        fold = index % FOLDS
        reference = frozenset().union(
            *(grams(p) for i, p in enumerate(prompts) if i % FOLDS != fold)
        )
        features["input_typicality"] = typicality(prompts[index], reference)
        rows.append(Row(features=features, small=small, escalated=escalated))
    return rows, prompts


def outcome(rows: Sequence[Row], router: ThresholdRouter, cost_weight: float) -> dict[str, float]:
    quality = escalations = waste = misses = 0.0
    for row in rows:
        up = router.decide(row.features) is Decision.ESCALATE
        quality += row.escalated if up else row.small
        escalations += up
        waste += up and row.small >= 0.5
        misses += (not up) and row.small < 0.5
    n = max(1, len(rows))
    return {
        "objective": quality / n - cost_weight * escalations / n,
        "quality": quality / n,
        "escalation_rate": escalations / n,
        "waste": waste / n,
        "misses": misses / n,
    }


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.floor(q * (len(ordered) - 1))))]


def fit(rows: Sequence[Row], cost_weight: float, max_clauses: int) -> ThresholdRouter:
    chosen: list[Clause] = []
    best = outcome(rows, ThresholdRouter((), Decision.RESOLVE), cost_weight)["objective"]
    for _ in range(max_clauses):
        found: tuple[float, Clause] | None = None
        for feature in sorted(ALLOWED_ROUTER_FEATURES):
            values = [row.features.get(feature, 0.0) for row in rows]
            if len(set(values)) < 2:
                continue
            for q in QUANTILES:
                value = _quantile(values, q)
                for op in ("lt", "gt"):
                    clause = Clause(feature=feature, op=op, value=value, decision=Decision.ESCALATE)
                    router = ThresholdRouter((*chosen, clause), Decision.RESOLVE)
                    score = outcome(rows, router, cost_weight)["objective"]
                    if found is None or score > found[0]:
                        found = (score, clause)
        if found is None or found[0] <= best + 1e-9:
            break
        best = found[0]
        chosen.append(found[1])
    return ThresholdRouter(tuple(chosen), Decision.RESOLVE)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit a router on the small model's own train split results."
    )
    parser.add_argument("--model", type=Path, required=True, help="the small model GGUF")
    parser.add_argument("--harness", type=Path, required=True, help="the harness package")
    parser.add_argument("--corpus", type=Path, required=True, help="the arena's train split")
    parser.add_argument("--track", required=True)
    parser.add_argument("--out", type=Path, required=True, help="router.json to write")
    parser.add_argument("--escalation-url", default="", help="OpenAI compatible endpoint")
    parser.add_argument("--escalation-model", default="")
    parser.add_argument(
        "--escalation-accuracy", type=float, default=0.9, help="assumed without an endpoint"
    )
    parser.add_argument(
        "--cost-weight", type=float, default=0.1, help="quality given up to avoid one escalation"
    )
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    rows, prompts = record(args)
    if not rows:
        raise SystemExit("the train split gave nothing to fit on")
    router = fit(rows, args.cost_weight, args.max_clauses)
    typical = sorted(frozenset().union(*(grams(p) for p in prompts)))
    document: dict[str, Any] = {
        "form": "threshold",
        "clauses": [
            {"feature": c.feature, "op": c.op, "value": c.value, "decision": c.decision.value}
            for c in router.clauses
        ],
        "default": Decision.RESOLVE.value,
        "typical": typical[:200_000],
    }
    args.out.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    never = outcome(rows, ThresholdRouter((), Decision.RESOLVE), args.cost_weight)
    fitted = outcome(rows, router, args.cost_weight)
    print(f"tasks {len(rows)}")
    for name, found in (("no router", never), ("fitted", fitted)):
        print(
            f"  {name:<10} quality {found['quality']:.3f}  escalated {found['escalation_rate']:.2f}"
            f"  waste {found['waste']:.2f}  misses {found['misses']:.2f}"
        )
    print(f"features {sorted({c.feature for c in router.clauses})}")
    print(f"wrote {args.out}; declare these features in system.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
