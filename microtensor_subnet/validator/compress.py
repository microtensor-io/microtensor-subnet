from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

GRADER: Final[str] = "context_compression"
FIELDS: Final[dict[str, type]] = {
    "summary": str,
    "files_touched": list,
    "open_tasks": list,
    "errors": list,
    "decisions": list,
}
EXTENSIONS: Final[str] = (
    "py|pyi|js|ts|tsx|jsx|go|rs|java|rb|c|h|cc|cpp|hpp|toml|cfg|ini|yaml|yml|json|md|rst|txt|sh"
)
NESTED_PATH = re.compile(rf"(?<![\w/.-])((?:[\w.-]+/)+[\w-]+\.(?:{EXTENSIONS}))(?![\w-])")
FILE_NAME = re.compile(rf"(?<![\w/.-])([a-z0-9_][a-z0-9_-]*\.(?:{EXTENSIONS}))(?![\w-])")
ERROR = re.compile(r"\b([A-Za-z_][\w.]*(?:Error|Exception))\b")
TICKED = re.compile(r"`([^`\s]{2,80})`")
CALL = re.compile(r"\b([A-Za-z_][\w.]*)\(")
CODE_MARKS = re.compile(r"[_./()]|[a-z][A-Z]")
ALLOWED: Final[frozenset[str]] = frozenset(
    {
        "github",
        "gitlab",
        "python",
        "python3",
        "pytest",
        "pip",
        "pip3",
        "npm",
        "node.js",
        "nodejs",
        "docker",
        "linux",
        "ubuntu",
        "macos",
        "windows",
        "json",
        "yaml",
        "toml",
        "readme.md",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "requirements.txt",
        "conftest.py",
        "git",
        "bash",
        "conda",
        "tox",
        "venv",
        "print",
        "len",
        "str",
        "int",
        "dict",
        "list",
    }
)
KINDS: Final[tuple[str, ...]] = ("read", "edit", "test", "run", "install", "other")
CHOICE = re.compile(r"^\(?([a-d])(?:\)|\.|:|\s|$)")
CHOICE_NAMED = re.compile(r"\b(?:option|answer|choice)\s*\(?([a-d])\b")
FENCE = re.compile(r"^```[A-Za-z]*\s*|\s*```$")
BOUNDARY_BEFORE: Final[str] = r"(?<![\w.-])"
LOOSE_BEFORE: Final[str] = r"(?<![\w-])"
BOUNDARY_AFTER: Final[str] = r"(?![\w-])"
MAX_ANSWER_WORDS: Final[int] = 8


@dataclass(frozen=True, slots=True)
class Judge:
    count_tokens: Callable[[str], int]
    answer: Callable[[str, Sequence[str]], list[str]]
    pass_threshold: float = 0.8
    min_output_tokens: int = 300
    max_output_share: float = 0.1


@dataclass(frozen=True, slots=True)
class Verdict:
    score: float
    success: bool
    reason: str


def normalise(text: str) -> str:
    cleaned = text.replace("\\", "/")
    for mark in ("`", '"', "'"):
        cleaned = cleaned.replace(mark, " ")
    return " ".join(cleaned.lower().split())


def mentioned(normalised_text: str, alias: str, *, loose: bool = False) -> bool:
    before = LOOSE_BEFORE if loose else BOUNDARY_BEFORE
    pattern = before + re.escape(normalise(alias)) + BOUNDARY_AFTER
    return re.search(pattern, normalised_text) is not None


def parse(output: Any) -> Mapping[str, Any] | None:
    if isinstance(output, Mapping):
        return output
    text = FENCE.sub("", str(output if output is not None else "").strip())
    try:
        found = json.loads(text)
    except ValueError:
        return None
    return found if isinstance(found, Mapping) else None


def valid(found: Mapping[str, Any] | None) -> bool:
    if found is None or set(found) != set(FIELDS):
        return False
    for key, kind in FIELDS.items():
        value = found[key]
        if not isinstance(value, kind):
            return False
        if isinstance(value, list) and not all(isinstance(entry, str) for entry in value):
            return False
    return True


def values_text(found: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in FIELDS:
        value = found.get(key)
        parts += value if isinstance(value, list) else [str(value or "")]
    return "\n".join(str(p) for p in parts)


def identifiers(text: str) -> set[str]:
    found: set[str] = set()
    for pattern in (NESTED_PATH, FILE_NAME, ERROR, CALL):
        found.update(match for match in pattern.findall(text) if len(match) > 2)
    found.update(
        match for match in TICKED.findall(text) if len(match) > 2 and CODE_MARKS.search(match)
    )
    return {name for name in found if normalise(name).rstrip("(") not in ALLOWED}


def recall(text: str, facts: Sequence[Mapping[str, Any]]) -> float:
    if not facts:
        return 1.0
    haystack = normalise(text)
    kept = 0
    for fact in facts:
        if any(mentioned(haystack, alias) for alias in fact.get("aliases") or []):
            kept += 1
    return kept / len(facts)


def hallucination(text: str, session: str) -> float:
    found = identifiers(text)
    if not found:
        return 0.0
    haystack = normalise(session)
    invented = sum(1 for name in found if not mentioned(haystack, name, loose=True))
    return invented / len(found)


def _words(said: str) -> list[str]:
    return re.sub(r"[^a-z0-9/._ -]", " ", said).split()


def probe_score(probe: Mapping[str, Any], answer: str) -> float:
    said = normalise(answer)
    if len(said.split()) > MAX_ANSWER_WORDS:
        return 0.0
    kind = str(probe.get("type") or "exact")
    expected = probe.get("answers")
    if kind == "word":
        words = re.findall(r"[a-z0-9]+", said)
        return 1.0 if words and isinstance(expected, list) and words[0] in expected else 0.0
    if kind == "choice":
        found = CHOICE.match(said) or CHOICE_NAMED.search(said)
        return 1.0 if found and isinstance(expected, list) and found.group(1) in expected else 0.0
    if kind == "next_step":
        truth = expected if isinstance(expected, Mapping) else {}
        named = [w for w in _words(said) if w in KINDS]
        kind_ok = bool(named) and named[0] == truth.get("kind")
        files = list(truth.get("file") or [])
        file_ok = any(mentioned(said, f) for f in files) if files else kind_ok
        if kind_ok and file_ok:
            return 1.0
        return 0.5 if kind_ok or (files and file_ok) else 0.0
    aliases = expected if isinstance(expected, list) else []
    return 1.0 if any(mentioned(said, alias) for alias in aliases) else 0.0


def continuation(
    found: Mapping[str, Any], probes: Sequence[Mapping[str, Any]], judge: Judge
) -> float:
    if not probes:
        return 1.0
    summary = json.dumps(dict(found), ensure_ascii=False, indent=1)
    answers = judge.answer(summary, [str(p.get("question", "")) for p in probes])
    scores = [probe_score(p, a) for p, a in zip(probes, answers, strict=False)]
    return sum(scores) / len(probes)


def grade(
    output: Any, label: Mapping[str, Any], task: Mapping[str, Any], judge: Judge | None
) -> Verdict:
    if judge is None:
        raise ValueError("context compression needs the probe agent and the tokenizer")
    found = parse(output)
    if not valid(found) or found is None:
        return Verdict(0.0, False, "not a JSON object with the five fields")
    raw = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
    session = str(task.get("user", ""))
    input_tokens = int(label.get("input_tokens") or 0) or judge.count_tokens(
        str(task.get("system", "")) + session
    )
    budget = int(label.get("budget_tokens") or 0) or max(
        judge.min_output_tokens, int(judge.max_output_share * input_tokens)
    )
    spent = judge.count_tokens(raw)
    if spent > budget:
        return Verdict(0.0, False, f"{spent} output tokens, over {budget}")
    text = values_text(found)
    kept = recall(text, label.get("facts") or [])
    invented = hallucination(text, session)
    carried = continuation(found, label.get("probes") or [], judge)
    score = (1 - invented) * (0.5 * kept + 0.5 * carried)
    reason = f"recall {kept:.2f}, invented {invented:.2f}, continuation {carried:.2f}"
    return Verdict(round(score, 6), score >= judge.pass_threshold, reason)
