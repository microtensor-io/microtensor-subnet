from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from microtensor_subnet.scoring.sql import admissible, rows_key, run_queries

SQL_EXECUTION: Final[str] = "sqlite_execution"
EXACT_TEXT: Final[str] = "exact_text"
RUBRIC_TEXT: Final[str] = "rubric_text"
WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
FENCE = re.compile(r"^```[A-Za-z]*\s*|\s*```$")


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


def _parsed(output: Any) -> Any:
    if isinstance(output, Mapping):
        return output
    text = FENCE.sub("", str(output if output is not None else "").strip())
    try:
        return json.loads(text)
    except ValueError:
        return text


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str):
        return isinstance(actual, str) and actual.strip() == expected.strip()
    return bool(actual == expected)


def _sentence_case(text: str, proper: Sequence[str]) -> bool:
    words = WORD.findall(text)
    if not words or not words[0][0].isupper():
        return False
    allowed = {str(p).lower() for p in proper}
    for word in words[1:]:
        titled = len(word) > 1 and word[0].isupper() and word[1:].islower()
        if titled and word.lower() not in allowed:
            return False
    return True


def _bounded(value: int, bounds: Any) -> bool:
    low, high = [*list(bounds), None, None][:2]
    return (low is None or value >= int(low)) and (high is None or value <= int(high))


def grade_rubric(output: Any, label: Mapping[str, Any]) -> Grade:
    rules = label.get("rubric")
    if not isinstance(rules, Mapping) or not rules:
        raise GradeError("the item label carries no rubric")
    parsed = _parsed(output)
    wants_json = rules.get("format") == "json"
    if wants_json and not isinstance(parsed, Mapping):
        return Grade(0.0, False, "not a JSON object")
    field = str(rules.get("field") or "")
    if isinstance(parsed, Mapping):
        text = str(parsed.get(field, "")) if field else json.dumps(parsed, sort_keys=True)
    else:
        text = str(parsed)
    checks: list[tuple[bool, str]] = []
    if field and isinstance(parsed, Mapping):
        checks.append((isinstance(parsed.get(field), str), f"{field} is missing"))
    for key, expected in (rules.get("equals") or {}).items():
        actual = parsed.get(key) if isinstance(parsed, Mapping) else None
        checks.append((_same(actual, expected), f"{key} differs"))
    for key, expected in (rules.get("set_equals") or {}).items():
        actual = parsed.get(key) if isinstance(parsed, Mapping) else None
        same = isinstance(actual, list) and {str(a).strip() for a in actual} == {
            str(e).strip() for e in expected
        }
        checks.append((same, f"{key} differs"))
    if "text_equals" in rules:
        checks.append((text.strip() == str(rules["text_equals"]).strip(), "text differs"))
    if "words" in rules:
        checks.append((_bounded(len(text.split()), rules["words"]), "word count out of bounds"))
    if "chars" in rules:
        checks.append((_bounded(len(text.strip()), rules["chars"]), "length out of bounds"))
    lowered = text.lower()
    if rules.get("include_any"):
        hit = any(str(k).lower() in lowered for k in rules["include_any"])
        checks.append((hit, "misses every key term"))
    for banned in rules.get("exclude") or []:
        checks.append((str(banned).lower() not in lowered, f"contains {banned!r}"))
    if rules.get("case") == "sentence":
        checks.append((_sentence_case(text, rules.get("proper") or []), "not sentence case"))
    if not checks:
        raise GradeError("the rubric names no checks")
    failed = [reason for ok, reason in checks if not ok]
    metric = (len(checks) - len(failed)) / len(checks)
    return Grade(metric, not failed, failed[0] if failed else "")


def grade(
    grader: str, output: Any, label: Mapping[str, Any], databases: Mapping[str, bytes]
) -> Grade:
    if grader == SQL_EXECUTION:
        return grade_sql(output, label, databases)
    if grader == EXACT_TEXT:
        return grade_exact(output, label)
    if grader == RUBRIC_TEXT:
        return grade_rubric(output, label)
    raise GradeError(f"unknown grader {grader!r}")


def pass_rate(grades: Sequence[Grade]) -> float:
    return sum(1 for g in grades if g.success) / len(grades) if grades else 0.0
