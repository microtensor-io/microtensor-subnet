from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from microtensor.core.constants import (
    ALLOWED_ROUTER_FEATURES,
    ROUTER_ALLOWED_OPS,
    ROUTER_MAX_BYTES,
)
from microtensor.harness.contract import EngineLoadError, Response


class Decision(str, Enum):
    RESOLVE = "resolve"
    ESCALATE = "escalate"


class RouterError(EngineLoadError):
    pass


OPS: dict[str, Callable[[float, float], bool]] = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}


@dataclass(frozen=True, slots=True)
class Clause:
    feature: str
    op: str
    value: float
    decision: Decision

    def holds(self, features: Mapping[str, float]) -> bool:
        return bool(OPS[self.op](features[self.feature], self.value))


class Router:
    """Interpreted routing policy. Participants supply data, never code."""

    features: tuple[str, ...]

    def decide(self, features: Mapping[str, float]) -> Decision:
        raise NotImplementedError


class ThresholdRouter(Router):
    def __init__(self, clauses: Sequence[Clause], default: Decision) -> None:
        self.clauses = tuple(clauses)
        self.default = default
        self.features = tuple(dict.fromkeys(c.feature for c in self.clauses))

    def decide(self, features: Mapping[str, float]) -> Decision:
        for clause in self.clauses:
            if clause.holds(features):
                return clause.decision
        return self.default


class LinearRouter(Router):
    def __init__(self, session: Any, features: Sequence[str], threshold: float) -> None:
        self.session = session
        self.features = tuple(features)
        self.threshold = threshold

    def decide(self, features: Mapping[str, float]) -> Decision:
        import numpy as np

        vector = np.array([[features[name] for name in self.features]], dtype=np.float32)
        name = self.session.get_inputs()[0].name
        raw = self.session.run(None, {name: vector})[0]
        return Decision.ESCALATE if float(raw.flat[0]) > self.threshold else Decision.RESOLVE


def _clause(raw: Any, index: int) -> Clause:
    if not isinstance(raw, dict):
        raise RouterError(f"clause {index} is not an object")
    try:
        feature = str(raw["feature"])
        op = str(raw["op"])
        decision = Decision(str(raw["decision"]))
        value = float(raw["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RouterError(f"clause {index} is malformed: {exc}") from exc

    if feature not in ALLOWED_ROUTER_FEATURES:
        raise RouterError(f"clause {index} reads {feature!r}, which is not a permitted feature")
    if op not in OPS:
        raise RouterError(f"clause {index} uses operator {op!r}; permitted: {sorted(OPS)}")
    if not math.isfinite(value):
        raise RouterError(f"clause {index} compares against a non-finite value")
    return Clause(feature=feature, op=op, value=value, decision=decision)


def load_threshold(payload: dict[str, Any]) -> ThresholdRouter:
    raw_clauses = payload.get("clauses")
    if not isinstance(raw_clauses, list) or not raw_clauses:
        raise RouterError("a threshold router must carry a non-empty clause list")
    try:
        default = Decision(str(payload.get("default", Decision.RESOLVE.value)))
    except ValueError as exc:
        raise RouterError(f"unknown default decision: {exc}") from exc
    return ThresholdRouter([_clause(c, i) for i, c in enumerate(raw_clauses)], default)


def _check_graph(model: Any) -> None:
    used = {node.op_type for node in model.graph.node}
    disallowed = sorted(used - set(ROUTER_ALLOWED_OPS))
    if disallowed:
        raise RouterError(
            f"router graph contains disallowed operators {disallowed}; "
            f"permitted: {sorted(ROUTER_ALLOWED_OPS)}"
        )
    if len(model.graph.output) != 1:
        raise RouterError("a router graph must declare exactly one output")
    if len(model.graph.input) != 1:
        raise RouterError("a router graph must declare exactly one input")


def load_linear(path: Path, features: Sequence[str], threshold: float) -> LinearRouter:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RouterError(f"the onnx runtime is unavailable: {exc}") from exc

    _check_graph(onnx.load(str(path)))

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return LinearRouter(session, features, threshold)


def load_router(path: Path, declared_features: Sequence[str]) -> Router:
    if not path.is_file():
        raise RouterError(f"{path} is not a router artifact")

    size = path.stat().st_size
    if size > ROUTER_MAX_BYTES:
        raise RouterError(
            f"router artifact is {size} bytes, over the {ROUTER_MAX_BYTES} byte limit"
        )

    unpermitted = [f for f in declared_features if f not in ALLOWED_ROUTER_FEATURES]
    if unpermitted:
        raise RouterError(f"router declares features that are not permitted: {unpermitted}")

    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RouterError(f"router is not valid json: {exc}") from exc
        if not isinstance(payload, dict):
            raise RouterError("a router document must be a json object")

        form = str(payload.get("form", ""))
        if form == "threshold":
            router = load_threshold(payload)
        elif form == "linear":
            raise RouterError("a linear router must be supplied as an onnx graph, not json")
        else:
            raise RouterError(f"unknown router form {form!r}; permitted: threshold, linear")

        undeclared = sorted(set(router.features) - set(declared_features))
        if undeclared:
            raise RouterError(f"router reads features it did not declare: {undeclared}")
        return router

    if path.suffix == ".onnx":
        threshold_path = path.with_suffix(".threshold")
        if not threshold_path.is_file():
            raise RouterError("a linear router must ship a .threshold file beside its graph")
        try:
            threshold = float(threshold_path.read_text(encoding="utf-8").strip())
        except (ValueError, UnicodeDecodeError) as exc:
            raise RouterError(f"router threshold is not a number: {exc}") from exc
        if not declared_features:
            raise RouterError("a linear router must declare the features it reads")
        return load_linear(path, declared_features, threshold)

    raise RouterError(f"unrecognised router artifact {path.name!r}; expected .json or .onnx")


def features_from(
    response: Response,
    *,
    prompt_tokens: int,
    schema_valid: bool = True,
) -> dict[str, float]:
    """Every feature is computed here, by the validator, from what the front emitted.

    Nothing an artifact reports about itself reaches the router, which is what
    makes the purity condition checkable rather than declarative.
    """
    logprobs = tuple(response.logprobs)
    entropies = tuple(response.entropies)
    total = float(sum(logprobs))
    return {
        "seq_logprob": total,
        "seq_logprob_norm": total / len(logprobs) if logprobs else 0.0,
        "output_tokens": float(response.output_tokens),
        "mean_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
        "max_entropy": max(entropies) if entropies else 0.0,
        "schema_valid": 1.0 if schema_valid else 0.0,
        "input_tokens": float(prompt_tokens),
    }


def decide(router: Router | None, features: Mapping[str, float]) -> Decision:
    if router is None:
        return Decision.RESOLVE
    missing = [name for name in router.features if name not in features]
    if missing:
        raise RouterError(f"the router reads features the validator did not compute: {missing}")
    return router.decide(features)
