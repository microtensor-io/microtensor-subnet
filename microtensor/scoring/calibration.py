from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from microtensor.core.constants import ACCURACY_DECIMALS
from microtensor.scoring.metrics import (
    decision_brier,
    decision_judgements,
    decision_level_errors,
)

BINS: Final[int] = 10


def _bin(confidence: float, bins: int) -> int:
    return min(bins - 1, max(0, int(confidence * bins)))


def reliability(rows: Sequence[tuple[float, bool]], bins: int = BINS) -> list[dict[str, Any]]:
    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, correct in rows:
        grouped[_bin(confidence, bins)].append((confidence, correct))
    table: list[dict[str, Any]] = []
    for index, members in enumerate(grouped):
        count = len(members)
        table.append(
            {
                "lower": round(index / bins, 2),
                "upper": round((index + 1) / bins, 2),
                "count": count,
                "confidence": round(math.fsum(c for c, _ in members) / count, ACCURACY_DECIMALS)
                if count
                else 0.0,
                "accuracy": round(sum(1 for _, ok in members if ok) / count, ACCURACY_DECIMALS)
                if count
                else 0.0,
            }
        )
    return table


def expected_calibration_error(rows: Sequence[tuple[float, bool]], bins: int = BINS) -> float:
    if not rows:
        return 0.0
    total = len(rows)
    gap = math.fsum(
        row["count"] / total * abs(row["accuracy"] - row["confidence"])
        for row in reliability(rows, bins)
        if row["count"]
    )
    return round(gap, ACCURACY_DECIMALS)


def summarise(pairs: Iterable[tuple[Any, Any]]) -> dict[str, Any]:
    rows: list[tuple[float, bool]] = []
    qualities: list[float] = []
    levels: list[float] = []
    for output, gold in pairs:
        rows.extend(decision_judgements(output, gold))
        qualities.append(decision_brier(output, gold))
        levels.extend(decision_level_errors(output, gold))
    return {
        "questions": len(rows),
        "accuracy": round(sum(1 for _, ok in rows if ok) / len(rows), ACCURACY_DECIMALS)
        if rows
        else 0.0,
        "brier_quality": round(math.fsum(qualities) / len(qualities), ACCURACY_DECIMALS)
        if qualities
        else 0.0,
        "ece": expected_calibration_error(rows),
        "level_mae": round(math.fsum(levels) / len(levels), ACCURACY_DECIMALS) if levels else None,
        "reliability": reliability(rows),
    }


def partition_report(
    partitions: Mapping[str, Iterable[tuple[Any, Any]]], novel: str
) -> dict[str, Any]:
    everything: list[tuple[Any, Any]] = []
    held_out: list[tuple[Any, Any]] = []
    for name, pairs in partitions.items():
        materialised = list(pairs)
        everything.extend(materialised)
        if name == novel:
            held_out.extend(materialised)
    overall = summarise(everything)
    withheld = summarise(held_out)
    return {
        **overall,
        "ece_novel": withheld["ece"],
        "accuracy_novel": withheld["accuracy"],
        "questions_novel": withheld["questions"],
    }
