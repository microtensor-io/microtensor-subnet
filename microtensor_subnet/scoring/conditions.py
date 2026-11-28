from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from microtensor_subnet.core.reports import PoolScore
from microtensor_subnet.scoring import metrics
from microtensor_subnet.scoring.pools import POOLS

BASELINE: Final[str] = "baseline"
KEYS: Final[tuple[str, ...]] = ("baseline", "rules")
RULE_KEYS: Final[tuple[str, ...]] = (
    "name",
    "metric",
    "params",
    "against",
    "compare",
    "margin",
    "ratio",
    "pools",
)
REQUIRED_RULE_KEYS: Final[tuple[str, ...]] = ("name", "metric", "against", "compare")
NAME: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{0,47}\Z")
MODEL: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z"
)
REVISION: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}\Z")
MAX_RULES: Final[int] = 16
DIGITS: Final[int] = 6
HOLDS: Final[dict[str, Callable[[float, float], bool]]] = {
    "above": lambda measured, target: measured > target,
    "at_least": lambda measured, target: measured >= target,
    "below": lambda measured, target: measured < target,
    "at_most": lambda measured, target: measured <= target,
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
    block: Mapping[str, Any], keys: Sequence[str], where: str, required: Sequence[str]
) -> None:
    for key in block:
        if key not in keys:
            raise ConditionError(f"unknown key {key!r} in {where}")
    for key in required:
        if key not in block:
            raise ConditionError(f"{where}.{key} is required")


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    metric: str
    params: Mapping[str, Any]
    against: str | float
    compare: str
    margin: float = 0.0
    ratio: float = 1.0
    pools: tuple[str, ...] | None = None

    @property
    def baseline(self) -> bool:
        return self.against == BASELINE

    def gates(self, pool: str) -> bool:
        return self.pools is None or pool in self.pools

    def target(self, reference: float) -> float:
        return round(reference * self.ratio + self.margin, DIGITS)

    def holds(self, measured: float, reference: float) -> bool:
        return HOLDS[self.compare](round(measured, DIGITS), self.target(reference))

    def wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "params": dict(self.params),
            "against": self.against,
            "compare": self.compare,
            "margin": self.margin,
            "ratio": self.ratio,
            "pools": list(self.pools) if self.pools is not None else None,
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
        return any(rule.baseline or metrics.lookup(rule.metric).compares for rule in self.rules)

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


def _pools(value: Any, where: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ConditionError(f"{where}.pools must list at least one pool")
    for pool in value:
        if pool not in POOLS:
            raise ConditionError(f"{where}.pools names {pool!r}; pools are {', '.join(POOLS)}")
    if len(set(value)) != len(value):
        raise ConditionError(f"{where}.pools repeats a pool")
    return tuple(str(pool) for pool in value)


def _rule(raw: Any, index: int) -> Rule:
    where = f"conditions.rules[{index}]"
    block = _mapping(raw, where)
    _exact(block, RULE_KEYS, where, REQUIRED_RULE_KEYS)
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
    if compare not in HOLDS:
        raise ConditionError(f"{where}.compare must be one of {', '.join(HOLDS)}")
    margin = block.get("margin", 0)
    if not _number(margin):
        raise ConditionError(f"{where}.margin must be a finite number")
    ratio = block.get("ratio", 1)
    if not _number(ratio) or float(ratio) <= 0:
        raise ConditionError(f"{where}.ratio must be a number above 0")
    return Rule(
        name=name,
        metric=metric,
        params=params,
        against=against if against == BASELINE else float(against),
        compare=compare,
        margin=float(margin),
        ratio=float(ratio),
        pools=_pools(block.get("pools"), where),
    )


def parse(definition: Mapping[str, Any]) -> Conditions | None:
    if "conditions" not in definition:
        return None
    block = _mapping(definition["conditions"], "conditions")
    _exact(block, KEYS, "conditions", KEYS)
    baseline = _mapping(block["baseline"], "conditions.baseline")
    _exact(baseline, ("model", "revision"), "conditions.baseline", ("model", "revision"))
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
        metric = metrics.lookup(rule.metric)
        try:
            params = metric.checked(rule.params)
        except metrics.MetricError as exc:
            raise ConditionError(f"condition {rule.name}: {exc}") from exc
        if metric.compares and rule.baseline:
            raise ConditionError(
                f"condition {rule.name}: {rule.metric} already compares the submission with the "
                "baseline, so against must be a number"
            )
        rules.append(
            Rule(
                rule.name,
                rule.metric,
                params,
                rule.against,
                rule.compare,
                rule.margin,
                rule.ratio,
                rule.pools,
            )
        )
    return Conditions(found.model, found.revision, tuple(rules), found.digest)


def _items(results: metrics.Results) -> int:
    return len({str(run.item_id) for runs in results.runs.values() for run in runs})


def _evaluate(rule: Rule, evidence: metrics.Evidence) -> Check:
    metric = metrics.lookup(rule.metric)
    if not rule.baseline:
        measured = round(metric.value(evidence, rule.params), DIGITS)
        reference = round(float(rule.against), DIGITS)
        items = _items(evidence.subject)
    else:
        baseline = evidence.baseline
        if baseline is None:
            return Check(rule, False, reason="the baseline's results are missing")
        submission = evidence.subject
        keep = {part: submission.ids(part) & baseline.ids(part) for part in submission.runs}
        if not any(keep.values()):
            return Check(rule, False, reason="no items in common with the baseline")
        held = metrics.Evidence(submission.only(keep), None, evidence.earlier)
        theirs = metrics.Evidence(baseline.only(keep))
        measured = round(metric.value(held, rule.params), DIGITS)
        reference = round(metric.value(theirs, rule.params), DIGITS)
        items = _items(held.subject)
    passed = rule.holds(measured, reference)
    return Check(rule, passed, measured, reference, rule.target(reference), items)


def evaluate(conditions: Conditions, evidence: metrics.Evidence) -> list[Check]:
    found: list[Check] = []
    for rule in conditions.rules:
        try:
            found.append(_evaluate(rule, evidence))
        except metrics.UnknownMetric:
            raise
        except Exception as exc:
            found.append(Check(rule, False, reason=f"the metric failed: {exc}"[:200]))
    return found


def failed(checks: Sequence[Check]) -> list[Check]:
    return [check for check in checks if not check.passed]


def gate(scores: Sequence[PoolScore], checks: Sequence[Check]) -> list[PoolScore]:
    missed = failed(checks)
    found: list[PoolScore] = []
    for score in scores:
        first = next((check for check in missed if check.rule.gates(score.pool)), None)
        if first is None:
            found.append(score)
            continue
        reason = score.detail.get("reason") or f"fails condition {first.rule.name}"
        found.append(
            PoolScore(
                score.pool,
                0.0,
                score.lower_bound,
                score.point,
                score.n,
                False,
                {**score.detail, "reason": reason},
            )
        )
    return found
