from __future__ import annotations

import importlib
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

NAME: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{0,47}(\.[a-z][a-z0-9_]{0,47})?\Z")
KINDS: Final[dict[str, tuple[type, ...]]] = {
    "number": (int, float),
    "integer": (int,),
    "text": (str,),
    "flag": (bool,),
}
PROVIDERS: Final[tuple[str, ...]] = ("microtensor_subnet.scoring.standard",)


class MetricError(ValueError):
    pass


class UnknownMetric(MetricError):
    def __init__(self, names: Iterable[str]) -> None:
        self.names = tuple(sorted(set(names)))
        super().__init__(f"unknown metric {', '.join(self.names)}")


@dataclass(frozen=True, slots=True)
class Param:
    kind: str
    required: bool = False
    default: Any = None
    choices: tuple[Any, ...] = ()

    def checked(self, metric: str, key: str, value: Any) -> Any:
        flag = self.kind == "flag"
        if isinstance(value, bool) != flag or not isinstance(value, KINDS[self.kind]):
            raise MetricError(f"metric {metric}: parameter {key} must be {self.kind}")
        if isinstance(value, float) and not math.isfinite(value):
            raise MetricError(f"metric {metric}: parameter {key} must be finite")
        if self.choices and value not in self.choices:
            allowed = ", ".join(str(c) for c in self.choices)
            raise MetricError(f"metric {metric}: parameter {key} must be one of {allowed}")
        return value


@dataclass(frozen=True, slots=True)
class Results:
    runs: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    figures: Mapping[str, float] = field(default_factory=dict)
    name: str = ""
    weights_sha256: str = ""
    commit_block: int = 0

    def part(self, name: str) -> tuple[Any, ...]:
        return tuple(self.runs.get(name, ()))

    def ids(self, name: str) -> set[str]:
        return {str(run.item_id) for run in self.part(name)}

    def only(self, keep: Mapping[str, set[str]]) -> Results:
        return Results(
            runs={
                name: tuple(run for run in runs if str(run.item_id) in keep.get(name, set()))
                for name, runs in self.runs.items()
            },
            figures=dict(self.figures),
            name=self.name,
            weights_sha256=self.weights_sha256,
            commit_block=self.commit_block,
        )


@dataclass(frozen=True, slots=True)
class Evidence:
    subject: Results
    baseline: Results | None = None
    earlier: tuple[Results, ...] = ()

    def part(self, name: str) -> tuple[Any, ...]:
        found = self.subject.part(name)
        if not found:
            raise MetricError(f"no scored items in {name}")
        return found

    def pairs(self, name: str) -> list[tuple[Any, Any]]:
        if self.baseline is None:
            raise MetricError("this metric needs the baseline's results")
        theirs = {str(run.item_id): run for run in self.baseline.part(name)}
        found = [
            (run, theirs[str(run.item_id)])
            for run in self.subject.part(name)
            if str(run.item_id) in theirs
        ]
        if not found:
            raise MetricError(f"no items in {name} in common with the baseline")
        return found


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    compute: Callable[[Evidence, Mapping[str, Any]], float]
    params: Mapping[str, Param] = field(default_factory=dict)
    compares: bool = False
    grader: str = ""

    def checked(self, given: Mapping[str, Any] | None) -> dict[str, Any]:
        raw = dict(given or {})
        extra = sorted(set(raw) - set(self.params))
        if extra:
            raise MetricError(f"metric {self.name} takes no parameter {', '.join(extra)}")
        found: dict[str, Any] = {}
        for key, param in self.params.items():
            if key in raw:
                found[key] = param.checked(self.name, key, raw[key])
            elif param.required:
                raise MetricError(f"metric {self.name} needs the parameter {key}")
            else:
                found[key] = param.default
        return found

    def value(self, evidence: Evidence, params: Mapping[str, Any]) -> float:
        found = float(self.compute(evidence, params))
        if not math.isfinite(found):
            raise MetricError(f"metric {self.name} gave {found}")
        return found


_registry: dict[str, Metric] = {}
_loaded = False


def register(metric: Metric) -> Metric:
    if not NAME.match(metric.name):
        raise MetricError(
            f"metric name {metric.name!r} must be lowercase letters, digits and underscores, "
            "optionally after a grader name and a dot"
        )
    for key, param in metric.params.items():
        if param.kind not in KINDS:
            raise MetricError(f"metric {metric.name}: parameter {key} has no kind {param.kind}")
        if param.required and param.default is not None:
            raise MetricError(f"metric {metric.name}: required parameter {key} has a default")
    held = _registry.get(metric.name)
    if held is not None and held != metric:
        raise MetricError(f"metric {metric.name} is already registered")
    _registry[metric.name] = metric
    return metric


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in PROVIDERS:
        importlib.import_module(module)


def lookup(name: str) -> Metric:
    _load()
    found = _registry.get(name)
    if found is None:
        raise UnknownMetric([name])
    return found


def missing(names: Iterable[str]) -> list[str]:
    _load()
    return sorted({name for name in names if name not in _registry})


def known() -> list[str]:
    _load()
    return sorted(_registry)
