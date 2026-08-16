from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final

from microtensor.core.constants import ACCURACY_DECIMALS, FIXED_FRACTION, ROTATING_FRACTION
from microtensor.core.protocol import TaskOutcome

Metric = Callable[[Any, Any], float]

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_WHITESPACE = re.compile(r"\s+")


def quantise(score: float) -> float:
    return round(max(0.0, min(1.0, score)), ACCURACY_DECIMALS)


def _normalise_text(value: Any) -> str:
    return _WHITESPACE.sub(" ", str(value).strip().lower())


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {_normalise_text(value)} if value.strip() else set()
    if isinstance(value, dict):
        return {f"{_normalise_text(k)}={_normalise_text(v)}" for k, v in value.items()}
    if isinstance(value, Iterable):
        return {_normalise_text(v) for v in value if str(v).strip()}
    return {_normalise_text(value)}


def f1(predicted: set[str], gold: set[str]) -> float:
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0
    matched = len(predicted & gold)
    if matched == 0:
        return 0.0
    precision = matched / len(predicted)
    recall = matched / len(gold)
    return 2 * precision * recall / (precision + recall)


def fbeta(predicted: set[str], gold: set[str], beta: float = 2.0) -> float:
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0
    matched = len(predicted & gold)
    if matched == 0:
        return 0.0
    precision = matched / len(predicted)
    recall = matched / len(gold)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def execution_pass_rate(output: Any, gold: Any) -> float:
    if isinstance(gold, dict) and "tests" in gold and "entry_point" in gold:
        from microtensor.scoring import execution

        return execution.execute_pass_rate(
            str(output), str(gold["entry_point"]), execution.parse_tests(gold["tests"])
        )
    if isinstance(output, dict):
        cases = output.get("cases")
        if isinstance(cases, Sequence) and cases:
            passed = sum(1 for c in cases if bool(c))
            return passed / len(cases)
        if "passed" in output:
            return 1.0 if bool(output["passed"]) else 0.0
    if isinstance(gold, dict) and "expected" in gold:
        return 1.0 if _normalise_text(output) == _normalise_text(gold["expected"]) else 0.0
    return 1.0 if bool(output) else 0.0


def schema_conformance(output: Any, gold: Any) -> float:
    required = _as_set((gold or {}).get("required_keys") if isinstance(gold, dict) else gold)
    if not required:
        return 1.0
    if not isinstance(output, dict):
        return 0.0
    present = {_normalise_text(k) for k in output}
    return len(required & present) / len(required)


def extraction_f1(output: Any, gold: Any) -> float:
    return f1(_as_set(output), _as_set(gold))


def span_accuracy(output: Any, gold: Any) -> float:
    return fbeta(_as_set(output), _as_set(gold), beta=2.0)


def exact_match_numeric(output: Any, gold: Any, tolerance: float = 1e-6) -> float:
    expected = _extract_number(gold)
    actual = _extract_number(output)
    if expected is None or actual is None:
        return 1.0 if _normalise_text(output) == _normalise_text(gold) else 0.0
    if math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        return 1.0
    return 0.0


def _extract_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        for key in ("value", "answer", "result"):
            if key in value:
                return _extract_number(value[key])
        return None
    match = _NUMBER.search(str(value))
    return float(match.group()) if match else None


def rubric_f1_tool_calls(output: Any, gold: Any) -> float:
    gold_calls = _as_set((gold or {}).get("tool_calls") if isinstance(gold, dict) else None)
    pred_calls = _as_set((output or {}).get("tool_calls") if isinstance(output, dict) else None)

    gold_points = _as_set((gold or {}).get("rubric") if isinstance(gold, dict) else gold)
    pred_points = _as_set((output or {}).get("covers") if isinstance(output, dict) else output)

    rubric = f1(pred_points, gold_points)
    if not gold_calls:
        return rubric
    calls = f1(pred_calls, gold_calls)
    return 0.5 * rubric + 0.5 * calls


METRICS: Final[dict[str, Metric]] = {
    "execution_pass_rate": execution_pass_rate,
    "schema_conformance": schema_conformance,
    "extraction_f1": extraction_f1,
    "span_accuracy": span_accuracy,
    "exact_match_numeric": exact_match_numeric,
    "rubric_f1_tool_calls": rubric_f1_tool_calls,
}


def get_metric(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError:
        raise KeyError(f"unknown metric {name!r}; known: {sorted(METRICS)}") from None


def score_task(metric: str, output: Any, gold: Any) -> float:
    try:
        return quantise(get_metric(metric)(output, gold))
    except (TypeError, ValueError, AttributeError, ZeroDivisionError):
        return 0.0


def aggregate(outcomes: Sequence[TaskOutcome]) -> float:
    if not outcomes:
        return 0.0
    return quantise(sum(o.score for o in outcomes) / len(outcomes))


def partition_scores(outcomes: Sequence[TaskOutcome]) -> tuple[float, float, int, int]:
    rotating = [o for o in outcomes if o.partition == "rotating"]
    fixed = [o for o in outcomes if o.partition == "fixed"]
    return aggregate(rotating), aggregate(fixed), len(rotating), len(fixed)


def combine_partitions(score_rotating: float, score_fixed: float) -> float:
    return quantise(ROTATING_FRACTION * score_rotating + FIXED_FRACTION * score_fixed)
