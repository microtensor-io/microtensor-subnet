from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final

from microtensor.core.constants import (
    ACCURACY_DECIMALS,
    FIXED_FRACTION,
    NOVEL_FRACTION,
    ROTATING_FRACTION,
)
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
    if isinstance(gold, dict) and "expected" in gold:
        return 1.0 if _normalise_text(output) == _normalise_text(gold["expected"]) else 0.0
    return 0.0


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


_SPAN_KEYS = ("unsupported", "unsupported_spans", "spans")
_EMPTY_ANSWERS = frozenset({"", "none", "[]", "{}", "null"})


def _gold_spans(gold: Any) -> set[str]:
    """The unsupported spans a task hides, wherever the corpus put them.

    Attached tests arrive as {"tests": [{"unsupported": [...]}]}; a task that
    carries its own gold gives {"unsupported": [...]} or a bare list. Every
    shape reduces to one set, so a corpus edit cannot silently zero a track.
    """
    if isinstance(gold, str):
        try:
            gold = json.loads(gold)
        except ValueError:
            return _as_set(gold)
    if isinstance(gold, dict):
        for key in _SPAN_KEYS:
            if key in gold:
                return _as_set(gold[key])
        cases = gold.get("tests")
        if isinstance(cases, list | tuple):
            found: set[str] = set()
            for case in cases:
                found |= _gold_spans(case)
            return found
        return set()
    return _as_set(gold)


def _output_spans(output: Any) -> set[str]:
    """What the model flagged: JSON in the declared shape, a bare list, or one
    span per line for a model that ignored the shape. Prose that says nothing
    is unsupported counts as an empty set."""
    parsed = _parse_calls(output)
    if isinstance(parsed, dict):
        for key in _SPAN_KEYS:
            if key in parsed:
                return _as_set(parsed[key])
        return set()
    if isinstance(parsed, list | tuple):
        return _as_set(parsed)
    text = _CALL_NOISE.sub(" ", str(output if output is not None else "")).strip()
    if _normalise_text(text) in _EMPTY_ANSWERS:
        return set()
    return {_normalise_text(line) for line in text.splitlines() if line.strip()}


def span_accuracy(output: Any, gold: Any) -> float:
    return fbeta(_output_spans(output), _gold_spans(gold), beta=2.0)


def exact_match_numeric(output: Any, gold: Any, tolerance: float = 1e-6) -> float:
    expected = _extract_number(gold)
    actual = _extract_number(output)
    if expected is None or actual is None:
        return 1.0 if _normalise_text(output) == _normalise_text(gold) else 0.0
    if math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        return 1.0
    return 0.0


def _extract_number(value: Any) -> float | None:
    if isinstance(value, (int | float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        for key in ("value", "answer", "result"):
            if key in value:
                return _extract_number(value[key])
        return None
    match = _NUMBER.search(str(value))
    return float(match.group()) if match else None


_CALL_NOISE = re.compile(r"```(?:json)?|</?tool_call>|</?function_call>", re.IGNORECASE)


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        return _normalise_text(value)
    if isinstance(value, dict):
        return {_normalise_text(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    return _normalise_text(value)


def _call_keys(calls: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(calls, dict):
        calls = [calls]
    if not isinstance(calls, list | tuple):
        return keys
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if isinstance(function, dict):
            call = function
        name = call.get("name", "")
        arguments = call.get("arguments", call.get("parameters", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"_raw": arguments}
        if not str(name).strip():
            continue
        packed = json.dumps(_canonical(arguments), sort_keys=True, separators=(",", ":"))
        keys.add(f"{_normalise_text(name)}({packed})")
    return keys


def _parse_calls(output: Any) -> Any:
    if isinstance(output, dict | list):
        return output
    text = _CALL_NOISE.sub(" ", str(output if output is not None else "")).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    starts = sorted(i for i in (text.find("{"), text.find("[")) if i >= 0)
    for start in starts:
        try:
            value, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        return value
    return None


def _gold_cases(gold: Any) -> list[dict[str, Any]]:
    if isinstance(gold, str):
        try:
            gold = json.loads(gold)
        except ValueError:
            return []
    if not isinstance(gold, dict):
        return []
    if "tool_calls" in gold or "rubric" in gold:
        return [gold]
    cases = gold.get("tests")
    if isinstance(cases, list | tuple):
        return [c for c in cases if isinstance(c, dict)]
    return []


def rubric_f1_tool_calls(output: Any, gold: Any) -> float:
    cases = _gold_cases(gold)
    gold_calls: set[str] = set()
    gold_points: set[str] = set()
    for case in cases:
        gold_calls |= _call_keys(case.get("tool_calls"))
        gold_points |= _as_set(case.get("rubric"))
    if not gold_calls and not gold_points:
        return 0.0

    parsed = _parse_calls(output)
    if isinstance(parsed, dict):
        pred_calls = _call_keys(parsed.get("tool_calls", parsed if "name" in parsed else []))
        pred_points = _as_set(parsed.get("covers"))
    else:
        pred_calls = _call_keys(parsed)
        pred_points = set()

    if not gold_calls:
        return f1(pred_points, gold_points)
    calls = f1(pred_calls, gold_calls)
    if not gold_points:
        return calls
    return 0.5 * f1(pred_points, gold_points) + 0.5 * calls


def entity_micro_f1(output: Any, gold: Any) -> float:
    """Per-document F1 proxy for entity extraction.

    The ranked quality is corpus-level micro-F1, aggregated across every
    document per partition, which lives in the validator. This per-task value
    exists so the pipeline has a number and so a malformed output reads as zero
    here too.
    """
    from microtensor.scoring.extraction import gold_entities, micro_f1, parse_entities

    return micro_f1([parse_entities(output)], [gold_entities(gold)])


_LABEL_PREFIX = re.compile(r"^\s*(?:label|intent|answer)\s*[:=]\s*", re.IGNORECASE)
_LABEL_TRIM = "\"'`*.,;:!?()[]{}<>"


def _label(value: Any) -> str:
    text = ""
    for line in str(value if value is not None else "").splitlines():
        candidate = _LABEL_PREFIX.sub("", line.strip()).strip().strip(_LABEL_TRIM).strip()
        if candidate:
            text = candidate
            break
    return re.sub(r"[\s\-]+", "_", _normalise_text(text))


def _expected_label(gold: Any) -> str:
    if isinstance(gold, dict):
        if "expected" in gold:
            return _label(gold["expected"])
        cases = gold.get("tests")
        if isinstance(cases, list | tuple):
            for case in cases:
                if isinstance(case, dict) and "expected" in case:
                    return _label(case["expected"])
        return ""
    return _label(gold)


def label_accuracy(output: Any, gold: Any) -> float:
    expected = _expected_label(gold)
    if not expected:
        return 0.0
    return 1.0 if _label(output) == expected else 0.0


def map_at_iou(output: Any, gold: Any) -> float:
    """Per-image proxy: did the detector emit any well-formed box for this image.

    The ranked quality for detection is COCO mAP, computed at dataset level per
    partition, not an average of per-image scores; that lives in the validator.
    This exists only so the per-task pipeline has a value and so a malformed
    output reads as zero here too.
    """
    from microtensor.scoring.detection import parse_detections

    return 1.0 if parse_detections(output) else 0.0


METRICS: Final[dict[str, Metric]] = {
    "execution_pass_rate": execution_pass_rate,
    "label_accuracy": label_accuracy,
    "map_at_iou": map_at_iou,
    "entity_micro_f1": entity_micro_f1,
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


def partition_scores(
    outcomes: Sequence[TaskOutcome],
) -> tuple[float, float, float, int, int, int]:
    rotating = [o for o in outcomes if o.partition == "rotating"]
    fixed = [o for o in outcomes if o.partition == "fixed"]
    novel = [o for o in outcomes if o.partition == "novel"]
    return (
        aggregate(rotating),
        aggregate(fixed),
        aggregate(novel),
        len(rotating),
        len(fixed),
        len(novel),
    )


def combine_partitions(
    score_rotating: float,
    score_fixed: float,
    score_novel: float = 0.0,
    n_rotating: int = 1,
    n_fixed: int = 1,
    n_novel: int = 0,
) -> float:
    parts = [
        (ROTATING_FRACTION, score_rotating, n_rotating),
        (FIXED_FRACTION, score_fixed, n_fixed),
        (NOVEL_FRACTION, score_novel, n_novel),
    ]
    served = [(weight, score) for weight, score, count in parts if count > 0]
    total = sum(weight for weight, _ in served)
    if total <= 0.0:
        return 0.0
    return quantise(sum(weight * score for weight, score in served) / total)
