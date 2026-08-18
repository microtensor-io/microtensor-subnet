from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final[int] = 3

MIGRATIONS: Final[tuple[tuple[int, tuple[str, ...]], ...]] = (
    (
        1,
        (
            """
            CREATE TABLE IF NOT EXISTS rounds (
                round_index   INTEGER PRIMARY KEY,
                seed_block    INTEGER NOT NULL,
                block_hash    TEXT    NOT NULL DEFAULT '',
                status        TEXT    NOT NULL DEFAULT 'open',
                reason        TEXT    NOT NULL DEFAULT '',
                started_at    REAL    NOT NULL,
                settled_at    REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS submissions (
                round_index     INTEGER NOT NULL,
                hotkey          TEXT    NOT NULL,
                track           TEXT    NOT NULL,
                hardware_class  TEXT    NOT NULL,
                manifest_digest TEXT    NOT NULL,
                artifact_digest TEXT    NOT NULL DEFAULT '',
                source          TEXT    NOT NULL,
                accepted        INTEGER NOT NULL DEFAULT 0,
                reason          TEXT    NOT NULL DEFAULT '',
                PRIMARY KEY (round_index, hotkey)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                round_index     INTEGER NOT NULL,
                track           TEXT    NOT NULL,
                hardware_class  TEXT    NOT NULL,
                hotkey          TEXT    NOT NULL,
                artifact_digest TEXT    NOT NULL,
                admitted        INTEGER NOT NULL,
                gate_reason     TEXT    NOT NULL DEFAULT '',
                score_rotating  REAL    NOT NULL DEFAULT 0,
                score_fixed     REAL    NOT NULL DEFAULT 0,
                score_combined  REAL    NOT NULL DEFAULT 0,
                n_rotating      INTEGER NOT NULL DEFAULT 0,
                n_fixed         INTEGER NOT NULL DEFAULT 0,
                corpus_version  TEXT    NOT NULL DEFAULT '',
                envelope        TEXT    NOT NULL DEFAULT '',
                PRIMARY KEY (round_index, track, hardware_class, hotkey)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holders (
                track           TEXT    NOT NULL,
                hardware_class  TEXT    NOT NULL,
                rank            INTEGER NOT NULL,
                hotkey          TEXT    NOT NULL,
                since_round     INTEGER NOT NULL,
                PRIMARY KEY (track, hardware_class, rank)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS observations (
                track           TEXT    NOT NULL,
                hardware_class  TEXT    NOT NULL,
                hotkey          TEXT    NOT NULL,
                artifact_digest TEXT    NOT NULL DEFAULT '',
                rounds_observed INTEGER NOT NULL DEFAULT 0,
                stale_rounds    INTEGER NOT NULL DEFAULT 0,
                last_round      INTEGER NOT NULL DEFAULT -1,
                PRIMARY KEY (track, hardware_class, hotkey)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS weights (
                round_index INTEGER NOT NULL,
                hotkey      TEXT    NOT NULL,
                weight      REAL    NOT NULL,
                PRIMARY KEY (round_index, hotkey)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS evaluations_by_hotkey
                ON evaluations (hotkey, round_index)
            """,
            """
            CREATE INDEX IF NOT EXISTS weights_by_round
                ON weights (round_index)
            """,
        ),
    ),
    (
        2,
        (
            """
            ALTER TABLE evaluations ADD COLUMN resolve_rate REAL NOT NULL DEFAULT 1
            """,
            """
            ALTER TABLE evaluations ADD COLUMN expected_ms REAL NOT NULL DEFAULT 0
            """,
            """
            ALTER TABLE evaluations ADD COLUMN expected_j REAL NOT NULL DEFAULT 0
            """,
            """
            ALTER TABLE evaluations ADD COLUMN front_only_score REAL NOT NULL DEFAULT 0
            """,
            """
            ALTER TABLE evaluations ADD COLUMN system_digest TEXT NOT NULL DEFAULT ''
            """,
        ),
    ),
    (
        3,
        (
            """
            CREATE TABLE IF NOT EXISTS contributions (
                round_index     INTEGER NOT NULL,
                hotkey          TEXT    NOT NULL,
                track           TEXT    NOT NULL,
                hardware_class  TEXT    NOT NULL,
                role            TEXT    NOT NULL,
                baseline_digest TEXT    NOT NULL DEFAULT '',
                value           INTEGER,
                share           REAL,
                reason          TEXT    NOT NULL DEFAULT '',
                PRIMARY KEY (round_index, hotkey, role)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS contributions_by_round
                ON contributions (round_index, track, hardware_class)
            """,
        ),
    ),
)
