from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from microtensor_subnet.core.reports import PoolScore
from microtensor_subnet.scoring import pools
from microtensor_subnet.scoring.stats import (
    Z_ONE_SIDED_95,
    ladder,
    paired_lb,
    percentile,
    wilson_lb,
)
from microtensor_subnet.validator import compress
from microtensor_subnet.validator.results import (
    ArtifactOutcome,
    SlmRun,
    SystemRun,
)

TWIN_TOLERANCE: Final[float] = 0.03
EFFICIENCY_RANGE: Final[float] = 0.25
EFFICIENCY_WEIGHTS: Final[dict[str, float]] = {"tpot": 0.1, "ttft": 0.4, "vram": 0.3, "n": 0.2}
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
    min_items: int = MIN_ITEMS

    @classmethod
    def from_definition(cls, definition: Mapping[str, Any]) -> Rules:
        block = _mapping(definition.get("scoring"))
        twin = _mapping(block.get("twin"))
        efficiency = _mapping(block.get("efficiency"))
        edge = _mapping(block.get("edge_bonus"))
        envelope = _mapping(definition.get("envelope"))
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
            min_items=int(block.get("min_items") or MIN_ITEMS),
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
    held = budgets.get("long_input")
    long: Mapping[str, Any] = held if isinstance(held, Mapping) else {}
    above = int(long.get("above_tokens") or 0)
    long_ttft = float(long.get("ttft_p95_ms") or 0)
    long_e2e = float(long.get("e2e_p95_ms") or 0)
    slowest = 1000.0 / rate if rate > 0 else 0.0
    over = 0
    for r in answered:
        is_long = bool(above) and r.prompt_tokens > above
        limit = long_ttft if is_long and long_ttft else ttft
        late = bool(limit) and r.ttft_ms > limit
        late = late or (is_long and bool(long_e2e) and r.latency_ms > long_e2e)
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
    max_tokens: int = 0,
) -> PoolScore:
    chosen = rules or Rules()
    n = len(run.runs)
    capped = sum(1 for r in run.runs if max_tokens and r.completion_tokens >= max_tokens)
    loops = sum(1 for r in run.runs if compress.looped(r.output))
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
            "passed": successes,
            "cap_rate": round(capped / n, DIGITS) if n else 0.0,
            "loop_rate": round(loops / n, DIGITS) if n else 0.0,
        },
    )


def system_score(run: SystemRun, *, min_items: int = MIN_ITEMS) -> PoolScore:
    values: list[float] = []
    detail: dict[str, Any] = {"floors": {}}
    for floor in run.floors:
        n = len(floor.runs)
        lower = wilson_lb(sum(1 for r in floor.runs if r.success), n)
        cost = floor.cost_micros
        latency = percentile(
            [r.latency_ms + r.model_ms(run.model_ttft_ms, run.model_tpot_ms) for r in floor.runs],
            0.95,
        )
        reference_per_item = (
            run.reference_cost_micros / run.reference_items if run.reference_items else 0.0
        )
        gain = COST_WEIGHT * _log_ratio(
            reference_per_item, cost / n if n else 0.0
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
            "n": n,
            "met": lower >= floor.floor,
        }
    score = sum(values) / len(values) if values else 0.0
    n = min((len(f.runs) for f in run.floors), default=0)
    detail["reference_cost_micros"] = run.reference_cost_micros
    detail["reference_latency_p95_ms"] = round(run.reference_latency_p95_ms, 1)
    detail["reference_items"] = run.reference_items
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
