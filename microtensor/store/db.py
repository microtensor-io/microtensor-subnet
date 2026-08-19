from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from microtensor.store.schema import MIGRATIONS, SCHEMA_VERSION


class StoreError(RuntimeError):
    pass


class Database:
    """A migrated sqlite connection.

    The migration set is a parameter rather than a fixed import so the
    coordinator can carry its own schema in its own file. The two stores hold
    unrelated rows and must never share a database: a coordinator runs no
    measurement and a worker publishes no settlement.
    """

    def __init__(
        self,
        path: Path | str,
        migrations: Sequence[tuple[int, Sequence[str]]] = MIGRATIONS,
        schema_version: int = SCHEMA_VERSION,
    ) -> None:
        self.path = Path(path)
        self._migrations = migrations
        self._schema_version = schema_version
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path), isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def user_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> int:
        current = self.user_version()
        if current > self._schema_version:
            raise StoreError(
                f"state was written by a newer build (schema {current} > "
                f"{self._schema_version}); upgrade rather than downgrading its state"
            )
        for version, statements in self._migrations:
            if version <= current:
                continue
            with self.transaction():
                for statement in statements:
                    self._conn.execute(statement)
                self._conn.execute(f"PRAGMA user_version={version}")
            current = version
        return current

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.executemany(sql, [tuple(r) for r in rows])

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, tuple(params)).fetchall())

    def one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            row: sqlite3.Row | None = self._conn.execute(sql, tuple(params)).fetchone()
            return row

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
