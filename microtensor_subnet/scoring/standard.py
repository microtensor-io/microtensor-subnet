from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from microtensor_subnet.scoring.metrics import Evidence, Metric, MetricError, Param, register
from microtensor_subnet.scoring.stats import Z_ONE_SIDED_95, percentile, wilson_lb

PART: Final[Param] = Param("text", default="slm")
TIMINGS: Final[tuple[str, ...]] = ("latency_ms", "ttft_ms", "tpot_ms")


def _pass_rate(evidence: Evidence, params: Mapping[str, Any]) -> float:
    runs = evidence.part(params["part"])
    return sum(1 for run in runs if run.success) / len(runs)


def _mean_item_score(evidence: Evidence, params: Mapping[str, Any]) -> float:
    runs = evidence.part(params["part"])
    return sum(float(run.metric) for run in runs) / len(runs)


def _wilson_lower_bound(evidence: Evidence, params: Mapping[str, Any]) -> float:
    runs = evidence.part(params["part"])
    if float(params["z"]) <= 0:
        raise MetricError("z must be above 0")
    return wilson_lb(sum(1 for run in runs if run.success), len(runs), float(params["z"]))


def _p95_latency_ms(evidence: Evidence, params: Mapping[str, Any]) -> float:
    answered = [run for run in evidence.part(params["part"]) if run.answered]
    if not answered:
        raise MetricError(f"no answered items in {params['part']}")
    return percentile([float(getattr(run, params["field"])) for run in answered], 0.95)


PASS_RATE = register(Metric("pass_rate", _pass_rate, {"part": PART}))
MEAN_ITEM_SCORE = register(Metric("mean_item_score", _mean_item_score, {"part": PART}))
WILSON_LOWER_BOUND = register(
    Metric(
        "wilson_lower_bound",
        _wilson_lower_bound,
        {"part": PART, "z": Param("number", default=Z_ONE_SIDED_95)},
    )
)
P95_LATENCY_MS = register(
    Metric(
        "p95_latency_ms",
        _p95_latency_ms,
        {"part": PART, "field": Param("text", default="latency_ms", choices=TIMINGS)},
    )
)
