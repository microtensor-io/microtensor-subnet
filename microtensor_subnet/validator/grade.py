from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from microtensor_subnet.scoring.sql import admissible, rows_key, run_queries

SQL_EXECUTION: Final[str] = "sqlite_execution"
EXACT_TEXT: Final[str] = "exact_text"


class GradeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Grade:
    metric: float
    success: bool
    reason: str = ""


def _candidate_sql(output: Any) -> str:
    if isinstance(output, Mapping):
        for key in ("sql", "query", "answer"):
            if key in output:
                return str(output[key])
        return ""
    return str(output if output is not None else "")


def grade_sql(output: Any, label: Mapping[str, Any], databases: Mapping[str, bytes]) -> Grade:
    reference = admissible(str(label.get("sql") or label.get("reference_sql") or ""))
    if not reference:
        raise GradeError("the item label carries no reference sql")
    candidate = admissible(_candidate_sql(output))
    if not candidate:
        return Grade(0.0, False, "not a single SELECT statement")
    states = [str(s) for s in label.get("databases") or label.get("blobs") or []]
    if not states:
        raise GradeError("the item label names no database states")
    for sha in states:
        database = databases.get(sha)
        if database is None:
            raise GradeError(f"database {sha} is not in the bundle")
        ran = run_queries(database, [reference, candidate])
        if "rows" not in ran[0]:
            raise GradeError(f"the reference query failed on {sha}: {ran[0].get('error')}")
        if "rows" not in ran[1]:
            return Grade(0.0, False, str(ran[1].get("error", "query failed"))[:200])
        if rows_key(ran[0]["rows"]) != rows_key(ran[1]["rows"]):
            return Grade(0.0, False, "result differs")
    return Grade(1.0, True)


def grade_exact(output: Any, label: Mapping[str, Any]) -> Grade:
    expected = str(label.get("answer", "")).strip().lower()
    actual = _candidate_sql(output).strip().lower()
    return Grade(1.0, True) if expected and actual == expected else Grade(0.0, False, "mismatch")


def grade(
    grader: str, output: Any, label: Mapping[str, Any], databases: Mapping[str, bytes]
) -> Grade:
    if grader == SQL_EXECUTION:
        return grade_sql(output, label, databases)
    if grader == EXACT_TEXT:
        return grade_exact(output, label)
    raise GradeError(f"unknown grader {grader!r}")


def pass_rate(grades: Sequence[Grade]) -> float:
    return sum(1 for g in grades if g.success) / len(grades) if grades else 0.0
