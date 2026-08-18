from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final[int] = 2

MIGRATIONS: Final[tuple[tuple[int, tuple[str, ...]], ...]] = (
    (
        1,
        (
            """
            CREATE TABLE IF NOT EXISTS rounds (
                round_index   INTEGER PRIMARY KEY,
                seed_block    INTEGER NOT NULL,
                block_hash    TEXT    NOT NULL DEFAULT '',
                close_block   INTEGER NOT NULL,
                config_hash   TEXT    NOT NULL DEFAULT '',
                anchored_at   REAL,
                opened_at     REAL    NOT NULL,
                closed_at     REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS assignments (
                round_index   INTEGER NOT NULL,
                worker_hotkey TEXT    NOT NULL,
                system_digest TEXT    NOT NULL,
                track         TEXT    NOT NULL,
                hardware_class TEXT   NOT NULL,
                miner_hotkey  TEXT    NOT NULL DEFAULT '',
                PRIMARY KEY (round_index, worker_hotkey, system_digest)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS assignments_by_system
                ON assignments (round_index, system_digest)
            """,
            """
            CREATE TABLE IF NOT EXISTS reports (
                round_index    INTEGER NOT NULL,
                worker_hotkey  TEXT    NOT NULL,
                system_digest  TEXT    NOT NULL,
                quality        REAL    NOT NULL,
                quality_rotating REAL  NOT NULL DEFAULT 0,
                quality_fixed  REAL    NOT NULL DEFAULT 0,
                resolve_rate   REAL    NOT NULL DEFAULT 1,
                expected_ms    REAL    NOT NULL DEFAULT 0,
                expected_j     REAL,
                envelope       TEXT    NOT NULL DEFAULT '{}',
                ablation       TEXT,
                device_profile TEXT    NOT NULL DEFAULT '',
                conforming     INTEGER NOT NULL DEFAULT 0,
                engine_version TEXT    NOT NULL DEFAULT '',
                corpus_version TEXT    NOT NULL DEFAULT '',
                fault          TEXT,
                signature      TEXT    NOT NULL DEFAULT '',
                body_hash      TEXT    NOT NULL DEFAULT '',
                received_at    REAL    NOT NULL,
                PRIMARY KEY (round_index, worker_hotkey, system_digest)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS reports_by_system
                ON reports (round_index, system_digest)
            """,
            """
            CREATE TABLE IF NOT EXISTS settlements (
                round_index   INTEGER PRIMARY KEY,
                config_hash   TEXT    NOT NULL DEFAULT '',
                corpus_version TEXT   NOT NULL DEFAULT '',
                reports_root  TEXT    NOT NULL DEFAULT '',
                payload       TEXT    NOT NULL,
                signature     TEXT    NOT NULL DEFAULT '',
                published_at  REAL    NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS divergences (
                round_index   INTEGER NOT NULL,
                system_digest TEXT    NOT NULL,
                worker_hotkey TEXT    NOT NULL,
                reported      REAL    NOT NULL,
                reconciled    REAL,
                kind          TEXT    NOT NULL,
                noted_at      REAL    NOT NULL,
                PRIMARY KEY (round_index, system_digest, worker_hotkey, kind)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS reputation (
                worker_hotkey TEXT PRIMARY KEY,
                agreed        INTEGER NOT NULL DEFAULT 0,
                diverged      INTEGER NOT NULL DEFAULT 0,
                streak        INTEGER NOT NULL DEFAULT 0,
                advisory      INTEGER NOT NULL DEFAULT 0,
                last_round    INTEGER NOT NULL DEFAULT 0
            )
            """,
        ),
    ),
    (
        2,
        (
            """
            CREATE TABLE IF NOT EXISTS catalogue (
                round_index    INTEGER NOT NULL,
                system_digest  TEXT    NOT NULL,
                miner_hotkey   TEXT    NOT NULL,
                uid            INTEGER NOT NULL,
                track          TEXT    NOT NULL,
                hardware_class TEXT    NOT NULL,
                committed_at   INTEGER NOT NULL DEFAULT 0,
                rounds_observed INTEGER NOT NULL DEFAULT 0,
                stale_rounds   INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (round_index, system_digest)
            )
            """,
        ),
    ),
)
