from __future__ import annotations

import json
import sqlite3
import sys
import time
from typing import Any

RESULT_CAP_BYTES = 1_000_000

ALLOWED_ACTIONS = frozenset(
    {
        getattr(sqlite3, "SQLITE_SELECT", 21),
        getattr(sqlite3, "SQLITE_READ", 20),
        getattr(sqlite3, "SQLITE_FUNCTION", 31),
        getattr(sqlite3, "SQLITE_RECURSIVE", 33),
    }
)


def _limit(memory_bytes: int, cpu_seconds: int) -> None:
    try:
        import resource
    except ImportError:
        return
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _cell(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    return value


def _run(connection: sqlite3.Connection, sql: str, row_cap: int, deadline: float) -> dict[str, Any]:
    def watch() -> int:
        return 1 if time.monotonic() > deadline else 0

    connection.set_progress_handler(watch, 10_000)
    try:
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(row_cap + 1)
    except sqlite3.Error as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        connection.set_progress_handler(None, 0)
    if len(rows) > row_cap:
        return {"error": f"more than {row_cap} rows"}
    payload = [[_cell(v) for v in row] for row in rows]
    if len(json.dumps(payload)) > RESULT_CAP_BYTES:
        return {"error": f"result larger than {RESULT_CAP_BYTES} bytes"}
    return {"rows": payload}


def main() -> int:
    request = json.loads(sys.stdin.read())
    _limit(int(request.get("memory_bytes", 512 * 1024 * 1024)), int(request.get("cpu_seconds", 10)))
    row_cap = int(request.get("row_cap", 5000))
    budget = float(request.get("seconds", 10.0))
    path = str(request["db"])
    results: list[dict[str, Any]] = []
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, timeout=1.0)
    try:
        connection.execute("PRAGMA query_only = 1")
        connection.set_authorizer(
            lambda action, *_: (
                sqlite3.SQLITE_OK if action in ALLOWED_ACTIONS else sqlite3.SQLITE_DENY
            )
        )
        for sql in request.get("queries", []):
            results.append(_run(connection, str(sql), row_cap, time.monotonic() + budget))
    finally:
        connection.close()
    sys.stdout.write(json.dumps({"results": results}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
