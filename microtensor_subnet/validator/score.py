from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from microtensor_subnet.core.reports import PoolScore
from microtensor_subnet.scoring import pools
from microtensor_subnet.scoring.stats import (
    Z_ONE_SIDED_95,
    bootstrap_seed,
    ladder,
    paired_lb,
    wilson_lb,
)
from microtensor_subnet.validator.results import (
    ArtifactOutcome,
    EscalationRun,
    HarnessRun,
    RouterRun,
    SlmRun,
    SourceRun,
    SystemRun,
)

TWIN_TOLERANCE: Final[float] = 0.03
EFFICIENCY_RANGE: Final[float] = 0.25
EFFICIENCY_WEIGHTS: Final[dict[str, float]] = {"tpot": 0.1, "ttft": 0.4, "vram": 0.3, "n": 0.2}
TOKEN_ALLOWANCE: Final[float] = 1.5
BRIER_WEIGHT: Final[float] = 0.2
ROUTER_MEAN_WEIGHT: Final[float] = 0.7
ROUTER_MIN_WEIGHT: Final[float] = 0.3
ROUTER_LATENCY_GATE_MS: Final[float] = 100.0
ESCALATION_MARGIN: Final[float] = 0.02
COST_WEIGHT: Final[float] = 0.7
LATENCY_WEIGHT: Final[float] = 0.3
GAIN_CAP: Final[float] = math.log(10)
MIN_ITEMS: Final[int] = 400
DIGITS: Final[int] = 6
WEIGHT_KEYS: Final[dict[str, str]] = {
    "ttft": "ttft",
    "memory": "vram",
    "parameters": "n",
    "tpot": "tpot",
}
FLOORS_BRIER: Final[str] = "floors_brier"
BRIER_AUROC: Final[str] = "brier_auroc"


@dataclass(frozen=True, slots=True)
class Rules:
    wilson_z: float = Z_ONE_SIDED_95
    twin_tolerance: float = TWIN_TOLERANCE
    twin_slope: float = 0.0
    efficiency_form: str = "log"
    efficiency_range: float = EFFICIENCY_RANGE
    efficiency_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(EFFICIENCY_WEIGHTS)
    )
    edge_per_group: float = 0.0
    edge_max: float = 0.0
    per_item_envelope: bool = False
    violation_share: float = 0.01
    cap: float | None = 1.0
    burn_empty_pools: bool = False
    router_kind: str = FLOORS_BRIER
    router_latency_ms: float = ROUTER_LATENCY_GATE_MS

    @classmethod
    def from_definition(cls, definition: Mapping[str, Any]) -> Rules:
        block = _mapping(definition.get("scoring"))
        twin = _mapping(block.get("twin"))
        efficiency = _mapping(block.get("efficiency"))
        edge = _mapping(block.get("edge_bonus"))
        envelope = _mapping(definition.get("envelope"))
        router = _mapping(_mapping(definition.get("limits")).get("router"))
        weights = dict(EFFICIENCY_WEIGHTS)
        for key, value in _mapping(efficiency.get("weights")).items():
            if key in WEIGHT_KEYS:
                weights[WEIGHT_KEYS[key]] = float(value)
        applies = edge.get("applies", True) is not False
        return cls(
            wilson_z=float(block.get("wilson_z") or Z_ONE_SIDED_95),
            twin_tolerance=float(twin.get("tolerance", TWIN_TOLERANCE)),
            twin_slope=float(twin.get("slope") or 0.0),
            efficiency_form=str(efficiency.get("form") or "log"),
            efficiency_range=float(efficiency.get("range", EFFICIENCY_RANGE)),
            efficiency_weights=weights,
            edge_per_group=float(edge.get("per_group") or 0.0) if applies else 0.0,
            edge_max=float(edge.get("max") or 0.0) if applies else 0.0,
            per_item_envelope=block.get("envelope_rule") == "per_item",
            violation_share=float(envelope.get("violation_share", 0.01)),
            cap=(None if block.get("cap") is None else float(block["cap"]))
            if "cap" in block
            else 1.0,
            burn_empty_pools=block.get("empty_pools") == "burn",
            router_kind=str(block.get("router") or FLOORS_BRIER),
            router_latency_ms=float(router.get("latency_gate_ms") or ROUTER_LATENCY_GATE_MS),
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _log_ratio(reference: float, measured: float) -> float:
    if reference <= 0 or measured <= 0:
        return 0.0
    return math.log(reference / measured)


def _gain(reference: float, measured: float) -> float:
    if reference <= 0 or measured <= 0:
        return 0.0
    return min(1.0, max(0.0, (reference - measured) / reference))


def efficiency(run: SlmRun, reference: SlmRun, rules: Rules | None = None) -> float:
    chosen = rules or Rules()
    weights = chosen.efficiency_weights
    pairs = {
        "tpot": (reference.tpot_p95_ms, run.tpot_p95_ms),
        "ttft": (reference.ttft_p95_ms, run.ttft_p95_ms),
        "vram": (float(reference.vram_peak_bytes), float(run.vram_peak_bytes)),
        "n": (float(reference.parameters), float(run.parameters)),
    }
    if chosen.efficiency_form == "linear":
        return sum(weights[key] * _gain(*pair) for key, pair in pairs.items())
    value = sum(weights[key] * _log_ratio(*pair) for key, pair in pairs.items())
    return max(-1.0, min(1.0, value))


def twin_factor(run: SlmRun, rules: Rules) -> tuple[float, float]:
    if not run.twins:
        return 0.0, 1.0
    excess = max(0.0, (run.pass_rate - run.twin_rate) - rules.twin_tolerance)
    if rules.twin_slope > 0:
        return excess, 1 - min(1.0, excess / rules.twin_slope)
    return excess, 1 - excess


def envelope_share(run: SlmRun, budgets: Mapping[str, Any]) -> float:
    answered = [r for r in run.runs if r.answered]
    if not answered:
        return 1.0
    ttft = float(budgets.get("ttft_ms") or 0)
    tpot = float(budgets.get("tpot_ms") or 0)
    rate = float(budgets.get("min_tokens_per_s") or 0)
    slowest = 1000.0 / rate if rate > 0 else 0.0
    over = 0
    for r in answered:
        late = bool(ttft) and r.ttft_ms > ttft
        slow = r.completion_tokens > 1 and (
            (bool(tpot) and r.tpot_ms > tpot) or (bool(slowest) and r.tpot_ms > slowest)
        )
        over += 1 if late or slow else 0
    return over / len(answered)


def _outcomes(runs: Sequence[Any]) -> tuple[list[float], list[str]]:
    return [1.0 if r.success else 0.0 for r in runs], [r.family for r in runs]


def slm_score(
    run: SlmRun,
    reference: SlmRun,
    *,
    gate_ok: bool,
    provenance_ok: bool,
    base: SlmRun | None,
    min_items: int = MIN_ITEMS,
    rules: Rules | None = None,
    envelope_ok: bool = True,
    edge_groups: int = 0,
) -> PoolScore:
    chosen = rules or Rules()
    n = len(run.runs)
    successes = sum(1 for r in run.runs if r.success)
    lower = wilson_lb(successes, n, chosen.wilson_z)
    twin_gap, factor = twin_factor(run, chosen)
    eff = efficiency(run, reference, chosen)
    bonus = min(chosen.edge_max, chosen.edge_per_group * edge_groups)
    passed = gate_ok and provenance_ok and envelope_ok
    score = lower * factor * (1 + chosen.efficiency_range * eff + bonus) if passed else 0.0
    score = max(0.0, score if chosen.cap is None else min(chosen.cap, score))
    _, families = _outcomes(run.runs)
    by_item = {r.item_id: r for r in reference.runs}
    paired = [1.0 if by_item[r.item_id].success else 0.0 for r in run.runs if r.item_id in by_item]
    aligned = [1.0 if r.success else 0.0 for r in run.runs if r.item_id in by_item]
    aligned_families = [r.family for r in run.runs if r.item_id in by_item]
    versus_reference = paired_lb(aligned, paired, aligned_families).lower_bound if paired else 0.0
    beats_base = None
    if base is not None:
        base_by_item = {r.item_id: r for r in base.runs}
        b = [
            1.0 if base_by_item[r.item_id].success else 0.0
            for r in run.runs
            if r.item_id in base_by_item
        ]
        a = [1.0 if r.success else 0.0 for r in run.runs if r.item_id in base_by_item]
        f = [r.family for r in run.runs if r.item_id in base_by_item]
        beats_base = paired_lb(a, b, f).lower_bound if b else 0.0
    eligible = passed and n >= min_items and versus_reference > 0
    return PoolScore(
        pool=pools.SLM,
        score=round(score, DIGITS),
        lower_bound=lower,
        point=round(run.pass_rate, DIGITS),
        n=n,
        eligible=eligible,
        detail={
            "twin_rate": round(run.twin_rate, DIGITS),
            "twin_gap": round(twin_gap, DIGITS),
            "twin_factor": round(factor, DIGITS),
            "efficiency": round(eff, DIGITS),
            "edge_bonus": round(bonus, DIGITS),
            "ttft_p95_ms": round(run.ttft_p95_ms, 1),
            "tpot_p95_ms": round(run.tpot_p95_ms, 2),
            "vram_peak_bytes": run.vram_peak_bytes,
            "parameters": run.parameters,
            "versus_reference_lb": versus_reference,
            "beats_base_lb": beats_base,
            "gate": gate_ok,
            "provenance": provenance_ok,
            "envelope": envelope_ok,
            "families": len(set(families)),
        },
    )


def harness_score(run: HarnessRun, *, min_items: int = MIN_ITEMS) -> PoolScore:
    if not run.limits_ok:
        return PoolScore(
            pools.HARNESS,
            0.0,
            0.0,
            run.miner.pass_rate,
            len(run.miner.runs),
            False,
            {"reason": run.limits_reason},
        )
    if not run.baselines:
        return PoolScore(
            pools.HARNESS,
            0.0,
            0.0,
            run.miner.pass_rate,
            len(run.miner.runs),
            False,
            {"reason": "no baseline"},
        )
    best = run.best_baseline()
    by_item = {r.item_id: r for r in best.runs}
    a = [1.0 if r.success else 0.0 for r in run.miner.runs if r.item_id in by_item]
    b = [1.0 if by_item[r.item_id].success else 0.0 for r in run.miner.runs if r.item_id in by_item]
    families = [r.family for r in run.miner.runs if r.item_id in by_item]
    paired = paired_lb(a, b, families)
    allowance = (
        min(1.0, TOKEN_ALLOWANCE * best.tokens / run.miner.tokens) if run.miner.tokens > 0 else 1.0
    )
    score = max(0.0, paired.lower_bound) * allowance
    return PoolScore(
        pool=pools.HARNESS,
        score=round(min(1.0, score), DIGITS),
        lower_bound=paired.lower_bound,
        point=round(paired.delta, DIGITS),
        n=paired.n,
        eligible=score > 0 and paired.n >= min_items,
        detail={
            "baseline": best.name,
            "baseline_pass_rate": round(best.pass_rate, DIGITS),
            "miner_pass_rate": round(run.miner.pass_rate, DIGITS),
            "baselines": {c.name: round(c.pass_rate, DIGITS) for c in run.baselines},
            "tokens_miner": run.miner.tokens,
            "tokens_baseline": best.tokens,
            "allowance": round(allowance, DIGITS),
            "standard_error": paired.standard_error,
        },
    )


def _loss(p: float, y: int, floors: Sequence[float]) -> float:
    total = 0.0
    weight = 1.0 / len(floors)
    for floor in floors:
        kept_wrong = 1.0 if p >= floor and y == 0 else 0.0
        sent_right = 1.0 if p < floor and y == 1 else 0.0
        total += weight * (floor * kept_wrong + (1 - floor) * sent_right)
    return total + BRIER_WEIGHT * (p - y) ** 2


def auroc(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    positives = sum(1 for y in labels if y == 1)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return 0.5
    order = sorted(range(len(probabilities)), key=lambda i: probabilities[i])
    ranks = [0.0] * len(order)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and probabilities[order[end + 1]] == probabilities[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1
        start = end + 1
    rank_sum = sum(rank for rank, y in zip(ranks, labels, strict=True) if y == 1)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _brier_auroc(source: SourceRun) -> float | None:
    count = len(source.labels)
    mean_label = sum(source.labels) / count
    base = sum((mean_label - y) ** 2 for y in source.labels) / count
    if base <= 0:
        return None
    brier = (
        sum((p - y) ** 2 for p, y in zip(source.probabilities, source.labels, strict=True)) / count
    )
    separation = auroc(source.probabilities, source.labels)
    return 0.5 * max(0.0, 1 - brier / base) + 0.5 * max(0.0, 2 * separation - 1)


def source_score(
    source: SourceRun, floors: Sequence[float], kind: str = FLOORS_BRIER
) -> float | None:
    if not source.labels:
        return None
    if kind == BRIER_AUROC:
        return _brier_auroc(source)
    mean_label = sum(source.labels) / len(source.labels)
    base = sum(_loss(mean_label, y, floors) for y in source.labels) / len(source.labels)
    if base <= 0:
        return None
    loss = sum(
        _loss(p, y, floors) for p, y in zip(source.probabilities, source.labels, strict=True)
    ) / len(source.labels)
    return max(0.0, 1 - loss / base)


def router_value(run: RouterRun, floors: Sequence[float], kind: str = FLOORS_BRIER) -> float:
    found = (source_score(src, floors, kind) for src in run.sources)
    scores = [s for s in found if s is not None]
    if not scores:
        return 0.0
    return ROUTER_MEAN_WEIGHT * sum(scores) / len(scores) + ROUTER_MIN_WEIGHT * min(scores)


def _resample(
    run: RouterRun,
    families: Mapping[str, str],
    items_by_source: Mapping[str, Sequence[str]],
    chosen: Sequence[str],
) -> RouterRun:
    wanted: dict[str, int] = {}
    for family in chosen:
        wanted[family] = wanted.get(family, 0) + 1
    sources: list[SourceRun] = []
    for source in run.sources:
        ids = items_by_source.get(source.source, [])
        probabilities: list[float] = []
        labels: list[int] = []
        for index, item_id in enumerate(ids[: len(source.labels)]):
            repeat = wanted.get(families.get(item_id, item_id), 0)
            probabilities += [source.probabilities[index]] * repeat
            labels += [source.labels[index]] * repeat
        sources.append(SourceRun(source.source, probabilities, labels))
    return RouterRun(sources=sources, latency_p95_ms=run.latency_p95_ms)


def router_score(
    run: RouterRun,
    reference: RouterRun | None,
    *,
    floors: Sequence[float],
    families: Mapping[str, str],
    items_by_source: Mapping[str, Sequence[str]],
    seed: str,
    min_drafts: int = MIN_ITEMS,
    rules: Rules | None = None,
) -> PoolScore:
    chosen = rules or Rules()
    kind = chosen.router_kind
    gate = run.loaded and run.latency_p95_ms < chosen.router_latency_ms
    value = router_value(run, floors, kind) if gate else 0.0
    total = sum(len(s.labels) for s in run.sources)
    margin = None
    if reference is not None and gate:
        family_names = sorted({families.get(i, i) for ids in items_by_source.values() for i in ids})
        import random

        rng = random.Random(bootstrap_seed(seed, "router"))  # noqa: S311
        draws: list[float] = []
        for _ in range(200):
            picked = [family_names[rng.randrange(len(family_names))] for _ in family_names]
            mine = router_value(_resample(run, families, items_by_source, picked), floors, kind)
            theirs = router_value(
                _resample(reference, families, items_by_source, picked), floors, kind
            )
            draws.append(mine - theirs)
        draws.sort()
        margin = round(draws[max(0, int(0.05 * len(draws)) - 1)], DIGITS) if draws else 0.0
    return PoolScore(
        pool=pools.ROUTER,
        score=round(max(0.0, min(1.0, value)), DIGITS),
        lower_bound=margin if margin is not None else 0.0,
        point=round(value, DIGITS),
        n=total,
        eligible=gate and total >= min_drafts and (margin is not None and margin > 0),
        detail={
            "sources": {
                s.source: round(source_score(s, floors, kind) or 0.0, DIGITS) for s in run.sources
            },
            "latency_p95_ms": round(run.latency_p95_ms, 2),
            "gate": gate,
            "reason": run.reason,
            "margin_over_reference_lb": margin,
        },
    )


def escalation_value(run: EscalationRun) -> tuple[float, float]:
    n = len(run.repairs)
    lower = wilson_lb(sum(1 for r in run.repairs if r.success), n)
    gate = lower >= run.redo_pass_rate - ESCALATION_MARGIN
    saving = (
        max(0.0, 1 - run.repair_cost_micros / run.redo_cost_micros)
        if run.redo_cost_micros > 0
        else 0.0
    )
    return (saving if gate else 0.0), lower


def escalation_score(
    run: EscalationRun, reference: EscalationRun | None, *, seed: str, min_items: int = MIN_ITEMS
) -> PoolScore:
    value, lower = escalation_value(run)
    margin = None
    if reference is not None:
        by_family: dict[str, list[tuple[Any, Any]]] = {}
        theirs = {r.item_id: r for r in reference.repairs}
        for mine in run.repairs:
            other = theirs.get(mine.item_id)
            if other is not None:
                by_family.setdefault(mine.family, []).append((mine, other))

        def metric(rows: Sequence[Any]) -> float:
            pairs = list(rows)
            mine_run = EscalationRun(
                [m for m, _ in pairs], run.redo_pass_rate, run.redo_cost_micros
            )
            their_run = EscalationRun(
                [o for _, o in pairs], reference.redo_pass_rate, reference.redo_cost_micros
            )
            return escalation_value(mine_run)[0] - escalation_value(their_run)[0]

        margin = _bootstrap_pairs(by_family, metric, bootstrap_seed(seed, "escalation"))
    return PoolScore(
        pool=pools.ESCALATION,
        score=round(value, DIGITS),
        lower_bound=lower,
        point=round(sum(1 for r in run.repairs if r.success) / len(run.repairs), DIGITS)
        if run.repairs
        else 0.0,
        n=len(run.repairs),
        eligible=value > 0
        and len(run.repairs) >= min_items
        and (margin is not None and margin > 0),
        detail={
            "redo_pass_rate": round(run.redo_pass_rate, DIGITS),
            "repair_cost_micros": run.repair_cost_micros,
            "redo_cost_micros": run.redo_cost_micros,
            "margin_over_reference_lb": margin,
        },
    )


def _bootstrap_pairs(
    families: Mapping[str, Sequence[Any]], metric: Any, seed: int, resamples: int = 200
) -> float:
    import random

    names = sorted(families)
    if not names:
        return 0.0
    rng = random.Random(seed)  # noqa: S311
    draws: list[float] = []
    for _ in range(resamples):
        chosen = [names[rng.randrange(len(names))] for _ in names]
        rows = [row for name in chosen for row in families[name]]
        draws.append(float(metric(rows)) if rows else 0.0)
    draws.sort()
    return round(draws[max(0, int(0.05 * len(draws)) - 1)], DIGITS)


def system_score(run: SystemRun, *, min_items: int = MIN_ITEMS) -> PoolScore:
    values: list[float] = []
    detail: dict[str, Any] = {"floors": {}}
    from microtensor_subnet.scoring.stats import percentile

    for floor in run.floors:
        n = len(floor.runs)
        lower = wilson_lb(sum(1 for r in floor.runs if r.success), n)
        cost = floor.cost_micros
        latency = percentile(floor.latencies, 0.95)
        gain = COST_WEIGHT * _log_ratio(
            float(run.reference_cost_micros), float(cost)
        ) + LATENCY_WEIGHT * _log_ratio(run.reference_latency_p95_ms, latency)
        value = (max(0.0, min(GAIN_CAP, gain)) / GAIN_CAP) if lower >= floor.floor else 0.0
        values.append(value)
        detail["floors"][str(floor.floor)] = {
            "pass_rate": round(floor.pass_rate, DIGITS),
            "lower_bound": lower,
            "cost_micros": cost,
            "latency_p95_ms": round(latency, 1),
            "escalated": round(sum(1 for r in floor.runs if r.escalated) / n, DIGITS) if n else 0.0,
            "value": round(value, DIGITS),
        }
    score = sum(values) / len(values) if values else 0.0
    n = min((len(f.runs) for f in run.floors), default=0)
    detail["reference_cost_micros"] = run.reference_cost_micros
    detail["reference_latency_p95_ms"] = round(run.reference_latency_p95_ms, 1)
    return PoolScore(
        pool=pools.SYSTEM,
        score=round(score, DIGITS),
        lower_bound=round(
            min((f["lower_bound"] for f in detail["floors"].values()), default=0.0), DIGITS
        ),
        point=round(score, DIGITS),
        n=n,
        eligible=score > 0 and n >= min_items,
        detail=detail,
    )


def published(previous: float | None, score: PoolScore) -> float:
    return ladder(previous, score.score, pools.LADDER_STEP)


def standings_for(
    outcomes: Mapping[str, ArtifactOutcome],
    scores: Mapping[str, Sequence[PoolScore]],
    previous: Mapping[tuple[str, str, str], float],
    artifact_shas: Mapping[str, str],
    commit_blocks: Mapping[str, int],
) -> dict[str, list[pools.Standing]]:
    by_pool: dict[str, list[pools.Standing]] = {pool: [] for pool in pools.POOLS}
    for hotkey, pool_scores in scores.items():
        for score in pool_scores:
            value = published(
                previous.get((hotkey, artifact_shas.get(hotkey, ""), score.pool)), score
            )
            by_pool[score.pool].append(
                pools.Standing(
                    hotkey=hotkey,
                    score=value,
                    eligible=score.eligible,
                    commit_block=commit_blocks.get(hotkey, 0),
                )
            )
    return by_pool
