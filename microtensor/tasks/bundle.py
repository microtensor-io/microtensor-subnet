from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from microtensor.core.hashing import canonical_hash

"""Bundle verification, independent of where the bundle came from.

The generator that used to sit beside this is gone: task data is uploaded by
the operator now, and uploaded data needs verifying in ways generated data
did not. The checks themselves are unchanged.
"""

MIN_HIDDEN_TESTS: Final[int] = 6
LEAK_CHECK_MIN_CHARS: Final[int] = 12


def verify_bundle(directory: Path, track: str = "code") -> list[str]:
    problems: list[str] = []
    corpus_path = directory / f"{track}.jsonl"
    tests_path = directory / f"{track}.tests.jsonl"
    if not corpus_path.is_file():
        return [f"{corpus_path.name} is missing"]
    if not tests_path.is_file():
        return [f"{tests_path.name} is missing"]

    tests: dict[str, dict[str, Any]] = {}
    for line in tests_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            tests[str(row["ref"])] = row

    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ref = str(row["ref"])
        declared = row.get("inputs", {}).get("tests_digest", "")
        entry = tests.get(ref)
        if entry is None:
            problems.append(f"{ref}: no hidden tests")
            continue
        if declared:
            actual = canonical_hash(
                {"ref": ref, "entry_point": entry["entry_point"], "tests": entry["tests"]}
            )
            if declared != actual:
                problems.append(f"{ref}: tests_digest does not match the tests file")
        if len(entry.get("tests", [])) < MIN_HIDDEN_TESTS:
            problems.append(f"{ref}: fewer than {MIN_HIDDEN_TESTS} hidden tests")
        prompt = str(row.get("prompt", ""))
        for case in entry.get("tests", []):
            for arg in case.get("args", []):
                rendered = repr(arg)
                if len(rendered) >= LEAK_CHECK_MIN_CHARS and rendered in prompt:
                    problems.append(f"{ref}: hidden test input appears in the prompt")
    return problems
