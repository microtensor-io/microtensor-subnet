from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from typing import Any, Final

from microtensor.scoring.metrics import score_task

CANDIDATE_LIMIT: Final[int] = 8


def _key(gold: Any) -> str:
    try:
        return json.dumps(gold, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(gold)


def candidates(golds: Sequence[Any], limit: int = CANDIDATE_LIMIT) -> list[Any]:
    counts: Counter[str] = Counter()
    seen: dict[str, Any] = {}
    for gold in golds:
        key = _key(gold)
        counts[key] += 1
        seen.setdefault(key, gold)
    return [seen[key] for key, _ in counts.most_common(limit)]


def constant_baseline(metric: str, golds: Sequence[Any], limit: int = CANDIDATE_LIMIT) -> float:
    if not golds:
        return 0.0
    best = 0.0
    for answer in candidates(golds, limit):
        total = sum(score_task(metric, answer, gold) for gold in golds)
        best = max(best, total / len(golds))
    return best


def skill(quality: float, baseline: float) -> float:
    if baseline <= 0.0:
        return max(0.0, quality)
    if baseline >= 1.0:
        return 0.0
    return max(0.0, (quality - baseline) / (1.0 - baseline))
