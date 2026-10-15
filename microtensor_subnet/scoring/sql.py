from __future__ import annotations

import base64
import binascii
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Final

TIMEOUT_SECONDS: Final[float] = 20.0
QUERY_SECONDS: Final[float] = 8.0
ROW_CAP: Final[int] = 5000
MEMORY_BYTES: Final[int] = 512 * 1024 * 1024
MAX_DB_BYTES: Final[int] = 64 * 1024 * 1024
DECIMALS: Final[int] = 6

_FENCE = re.compile(r"```[a-zA-Z]*")
_LABEL = re.compile(r"^\s*(?:sql|sqlite|query|answer)\s*:\s*", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(attach|detach|pragma|vacuum|load_extension|readfile|writefile)\b", re.IGNORECASE
)
_ALLOWED_FIRST: Final[frozenset[str]] = frozenset({"select", "with"})
_CHILD: Final[Path] = Path(__file__).with_name("sql_child.py")


def clean_sql(text: Any) -> str:
    body = _FENCE.sub("\n", str(text if text is not None else "")).strip()
    body = _LABEL.sub("", body)
    first, _, _ = body.partition(";")
    return first.strip()


def admissible(sql: str) -> str:
    statement = clean_sql(sql)
    if not statement:
        return ""
    head = statement.split(None, 1)[0].lower()
    if head not in _ALLOWED_FIRST or _FORBIDDEN.search(statement):
        return ""
    return statement


def gold_case(gold: Any) -> tuple[bytes, str] | None:
    if isinstance(gold, str):
        try:
            gold = json.loads(gold)
        except ValueError:
            return None
    if not isinstance(gold, dict):
        return None
    cases = gold.get("tests")
    case: Any = gold
    if isinstance(cases, list | tuple) and cases and isinstance(cases[0], dict):
        case = cases[0]
    blob = case.get("db", case.get("database"))
    reference = case.get("sql", case.get("query", case.get("reference")))
    if not isinstance(blob, str) or not isinstance(reference, str) or not reference.strip():
        return None
    try:
        raw = base64.b64decode(blob, validate=True)
    except (ValueError, binascii.Error):
        return None
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError):
            return None
    if not raw.startswith(b"SQLite format 3\x00") or len(raw) > MAX_DB_BYTES:
        return None
    return raw, reference


def _normal(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        rounded = round(value, DECIMALS)
        return int(rounded) if rounded.is_integer() else rounded
    if isinstance(value, dict) and "__bytes__" in value:
        return ("bytes", str(value["__bytes__"]))
    return value


def rows_key(rows: list[list[Any]]) -> Counter[tuple[Any, ...]]:
    return Counter(tuple(_normal(v) for v in row) for row in rows)


def run_queries(database: bytes, queries: list[str]) -> list[dict[str, Any]]:
    workdir = tempfile.mkdtemp(prefix="mt-sql-")
    path = os.path.join(workdir, "task.sqlite")
    try:
        with open(path, "wb") as handle:
            handle.write(database)
        os.chmod(path, 0o400)
        request = json.dumps(
            {
                "db": path,
                "queries": queries,
                "row_cap": ROW_CAP,
                "seconds": QUERY_SECONDS,
                "memory_bytes": MEMORY_BYTES,
                "cpu_seconds": int(TIMEOUT_SECONDS),
            }
        )
        try:
            proc = subprocess.run(  # noqa: S603
                [sys.executable, "-I", str(_CHILD)],
                input=request,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
                cwd=workdir,
                env={"PATH": os.environ.get("PATH", "")},
            )
        except subprocess.TimeoutExpired:
            return [{"error": "timeout"} for _ in queries]
        if proc.returncode != 0:
            return [
                {"error": f"child exited {proc.returncode}: {proc.stderr[-200:]}"} for _ in queries
            ]
        try:
            answer = json.loads(proc.stdout)
        except ValueError:
            return [{"error": "child answered with something that is not JSON"} for _ in queries]
        results = answer.get("results") if isinstance(answer, dict) else None
        if not isinstance(results, list) or len(results) != len(queries):
            return [{"error": "child answered the wrong number of queries"} for _ in queries]
        return [dict(r) if isinstance(r, dict) else {"error": "malformed"} for r in results]
    finally:
        for name in os.listdir(workdir):
            with_path = os.path.join(workdir, name)
            os.chmod(with_path, 0o600)
            os.unlink(with_path)
        os.rmdir(workdir)


def execution_match(output: Any, gold: Any) -> float:
    case = gold_case(gold)
    if case is None:
        return 0.0
    database, reference = case
    candidate = admissible(str(output if output is not None else ""))
    expected = admissible(reference)
    if not candidate or not expected:
        return 0.0
    ran = run_queries(database, [expected, candidate])
    if any("rows" not in r for r in ran):
        return 0.0
    return 1.0 if rows_key(ran[0]["rows"]) == rows_key(ran[1]["rows"]) else 0.0
