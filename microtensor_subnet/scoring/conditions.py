from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from microtensor_subnet.core.reports import PoolScore
from microtensor_subnet.scoring import metrics

BASELINE: Final[str] = "baseline"
KEYS: Final[tuple[str, ...]] = ("baseline", "rules")
RULE_KEYS: Final[tuple[str, ...]] = ("name", "metric", "params", "against", "compare", "threshold")
OPTIONAL_RULE_KEYS: Final[frozenset[str]] = frozenset({"params"})
NAME: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{0,47}\Z")
MODEL: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z"
)
REVISION: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}\Z")
MAX_RULES: Final[int] = 16
DIGITS: Final[int] = 6
COMPARES: Final[dict[str, Callable[[float, float], float]]] = {
    "above": lambda reference, threshold: reference + threshold,
    "at_least": lambda reference, threshold: reference + threshold,
    "below": lambda reference, threshold: reference - threshold,
    "at_most": lambda reference, threshold: reference - threshold,
    "ratio_at_least": lambda reference, threshold: reference * threshold,
    "ratio_at_most": lambda reference, threshold: reference * threshold,
}
HOLDS: Final[dict[str, Callable[[float, float], bool]]] = {
    "above": lambda measured, target: measured > target,
    "at_least": lambda measured, target: measured >= target,
    "below": lambda measured, target: measured < target,
    "at_most": lambda measured, target: measured <= target,
    "ratio_at_least": lambda measured, target: measured >= target,
    "ratio_at_most": lambda measured, target: measured <= target,
}


class ConditionError(ValueError):
    pass


class NeedsUpdate(ConditionError):
    def __init__(self, names: Sequence[str]) -> None:
        self.names = tuple(sorted(set(names)))
        super().__init__(
            f"the arena's conditions use metrics this validator does not know: "
            f"{', '.join(self.names)}; update the validator"
        )


def _number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConditionError(f"{where} must be a mapping")
    return value


def _exact(
    block: Mapping[str, Any], keys: Sequence[str], where: str, optional: frozenset[str]
) -> None:
    for key in block:
        if key not in keys:
            raise ConditionError(f"unknown key {key!r} in {where}")
    for key in keys:
        if key not in block and key not in optional:
            raise ConditionError(f"{where}.{key} is required")


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    metric: str
    params: Mapping[str, Any]
    against: str | float
    compare: str
    threshold: float

    @property
    def baseline(self) -> bool:
        return self.against == BASELINE

    def target(self, reference: float) -> float:
        return round(COMPARES[self.compare](reference, self.threshold), DIGITS)

    def holds(self, measured: float, reference: float) -> bool:
        return HOLDS[self.compare](round(measured, DIGITS), self.target(reference))

    def wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "params": dict(self.params),
            "against": self.against,
            "compare": self.compare,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class Conditions:
    model: str
    revision: str
    rules: tuple[Rule, ...]
    digest: str

    @property
    def baseline(self) -> dict[str, str]:
        return {"model": self.model, "revision": self.revision}

    @property
    def needs_baseline(self) -> bool:
        return any(rule.baseline for rule in self.rules)

    def wire(self) -> dict[str, Any]:
        return {"baseline": self.baseline, "rules": [rule.wire() for rule in self.rules]}


@dataclass(frozen=True, slots=True)
class Check:
    rule: Rule
    passed: bool
    measured: float | None = None
    reference: float | None = None
    target: float | None = None
    items: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.rule.wire(),
            "passed": self.passed,
            "measured": self.measured,
            "reference": self.reference,
            "target": self.target,
            "items": self.items,
            "reason": self.reason,
        }


def digest(block: Mapping[str, Any]) -> str:
    encoded = json.dumps(block, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rule(raw: Any, index: int) -> Rule:
    where = f"conditions.rules[{index}]"
    block = _mapping(raw, where)
    _exact(block, RULE_KEYS, where, OPTIONAL_RULE_KEYS)
    name = block["name"]
    if not isinstance(name, str) or not NAME.match(name):
        raise ConditionError(f"{where}.name must be lowercase letters, digits and underscores")
    metric = block["metric"]
    if not isinstance(metric, str) or not metrics.NAME.match(metric):
        raise ConditionError(f"{where}.metric must name a metric from the registry")
    params = dict(_mapping(block.get("params") or {}, f"{where}.params"))
    against = block["against"]
    if against != BASELINE and not _number(against):
        raise ConditionError(f"{where}.against must be {BASELINE} or a number")
    compare = block["compare"]
    if compare not in COMPARES:
        raise ConditionError(f"{where}.compare must be one of {', '.join(COMPARES)}")
    threshold = block["threshold"]
    if not _number(threshold):
        raise ConditionError(f"{where}.threshold must be a finite number")
    if compare.startswith("ratio_") and float(threshold) <= 0:
        raise ConditionError(f"{where}.threshold must be above 0 for a ratio")
    return Rule(
        name=name,
        metric=metric,
        params=params,
        against=against if against == BASELINE else float(against),
        compare=compare,
        threshold=float(threshold),
    )


def parse(definition: Mapping[str, Any]) -> Conditions | None:
    if "conditions" not in definition:
        return None
    block = _mapping(definition["conditions"], "conditions")
    _exact(block, KEYS, "conditions", frozenset())
    baseline = _mapping(block["baseline"], "conditions.baseline")
    _exact(baseline, ("model", "revision"), "conditions.baseline", frozenset())
    model, revision = baseline["model"], baseline["revision"]
    if not isinstance(model, str) or not MODEL.match(model):
        raise ConditionError("conditions.baseline.model must be a Hugging Face repository id")
    if not isinstance(revision, str) or not REVISION.match(revision):
        raise ConditionError("conditions.baseline.revision must be a 40 hex commit sha")
    raw = block["rules"]
    if not isinstance(raw, list) or not raw:
        raise ConditionError("conditions.rules must list at least one condition")
    if len(raw) > MAX_RULES:
        raise ConditionError(f"conditions.rules holds at most {MAX_RULES} conditions")
    rules = tuple(_rule(entry, index) for index, entry in enumerate(raw))
    names = [rule.name for rule in rules]
    if len(set(names)) != len(names):
        raise ConditionError("conditions.rules repeats a name")
    references = definition.get("references")
    if (
        isinstance(references, Mapping)
        and references.get("slm")
        and (references.get("slm"), references.get("slm_revision")) != (model, revision)
    ):
        raise ConditionError(
            "conditions.baseline must be the arena's reference model at its revision"
        )
    return Conditions(model=model, revision=revision, rules=rules, digest=digest(block))


def load(definition: Mapping[str, Any]) -> Conditions | None:
    found = parse(definition)
    if found is None:
        return None
    unknown = metrics.missing(rule.metric for rule in found.rules)
    if unknown:
        raise NeedsUpdate(unknown)
    rules = []
    for rule in found.rules:
        try:
            params = metrics.lookup(rule.metric).checked(rule.params)
        except metrics.MetricError as exc:
            raise ConditionError(f"condition {rule.name}: {exc}") from exc
        rules.append(
            Rule(rule.name, rule.metric, params, rule.against, rule.compare, rule.threshold)
        )
    return Conditions(found.model, found.revision, tuple(rules), found.digest)


def _evaluate(
    rule: Rule, submission: metrics.Results, baseline: metrics.Results | None
) -> Check:
    metric = metrics.lookup(rule.metric)
    if not rule.baseline:
        measured = round(metric.value(submission, rule.params), DIGITS)
        reference = round(float(rule.against), DIGITS)
        held = submission
    else:
        if baseline is None:
            return Check(rule, False, reason="the baseline's results are missing")
        keep = {part: submission.ids(part) & baseline.ids(part) for part in submission.runs}
        held = submission.only(keep)
        if not any(keep.values()):
            return Check(rule, False, reason="no items in common with the baseline")
        measured = round(metric.value(held, rule.params), DIGITS)
        reference = round(metric.value(baseline.only(keep), rule.params), DIGITS)
    items = len({str(run.item_id) for runs in held.runs.values() for run in runs})
    passed = rule.holds(measured, reference)
    return Check(rule, passed, measured, reference, rule.target(reference), items)


def evaluate(
    conditions: Conditions, submission: metrics.Results, baseline: metrics.Results | None
) -> list[Check]:
    found: list[Check] = []
    for rule in conditions.rules:
        try:
            found.append(_evaluate(rule, submission, baseline))
        except metrics.UnknownMetric:
            raise
        except Exception as exc:
            found.append(Check(rule, False, reason=f"the metric failed: {exc}"[:200]))
    return found


def failed(checks: Sequence[Check]) -> list[Check]:
    return [check for check in checks if not check.passed]


def gate(scores: Sequence[PoolScore], checks: Sequence[Check]) -> list[PoolScore]:
    missed = failed(checks)
    if not missed:
        return list(scores)
    reason = f"fails condition {missed[0].rule.name}"
    return [
        PoolScore(
            score.pool,
            0.0,
            score.lower_bound,
            score.point,
            score.n,
            False,
            {**score.detail, "reason": score.detail.get("reason") or reason},
        )
        for score in scores
    ]


@dataclass(slots=True)
class Applied:
    conditions: Conditions | None = None
    checks: dict[str, list[Check]] = field(default_factory=dict)

    def wire(self, hotkey: str) -> list[dict[str, Any]]:
        return [check.to_dict() for check in self.checks.get(hotkey, [])]
