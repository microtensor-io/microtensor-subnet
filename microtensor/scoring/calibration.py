from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from microtensor.core.constants import ACCURACY_DECIMALS
from microtensor.scoring.metrics import (
    decision_brier,
    decision_judgements,
    decision_level_errors,
    decision_skill,
)

BINS: Final[int] = 10
CORRECT_AT: Final[float] = 0.5
GENERATED: Final[str] = "sequence"


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


def sequence_confidence(logprobs: Sequence[float]) -> float:
    return math.exp(math.fsum(logprobs) / len(logprobs)) if logprobs else 0.0


def generation_summary(rows: Iterable[tuple[float, float]]) -> dict[str, Any]:
    found = [(min(max(float(c), 0.0), 1.0), float(score)) for c, score in rows]
    judged = [(confidence, score >= CORRECT_AT) for confidence, score in found]
    squared = [(confidence - (1.0 if ok else 0.0)) ** 2 for confidence, ok in judged]
    return {
        "questions": len(judged),
        "accuracy": round(sum(1 for _, ok in judged if ok) / len(judged), ACCURACY_DECIMALS)
        if judged
        else 0.0,
        "brier_quality": round(1.0 - math.fsum(squared) / len(squared), ACCURACY_DECIMALS)
        if squared
        else 0.0,
        "ece": expected_calibration_error(judged),
        "level_mae": None,
        "reliability": reliability(judged),
        "prompt_format": GENERATED,
    }


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
    partitions: Mapping[str, Iterable[tuple[Any, Any, Any]]], novel: str
) -> dict[str, Any]:
    everything: list[tuple[Any, Any, Any]] = []
    held_out: list[tuple[Any, Any, Any]] = []
    for name, entries in partitions.items():
        materialised = list(entries)
        everything.extend(materialised)
        if name == novel:
            held_out.extend(materialised)
    overall = summarise((output, gold) for output, gold, _ in everything)
    withheld = summarise((output, gold) for output, gold, _ in held_out)
    return {
        **overall,
        "brier_skill": round(decision_skill(everything), ACCURACY_DECIMALS),
        "ece_novel": withheld["ece"],
        "accuracy_novel": withheld["accuracy"],
        "questions_novel": withheld["questions"],
    }
