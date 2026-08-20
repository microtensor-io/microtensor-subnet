from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final[int] = 4

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
    (
        3,
        (
            """
            CREATE TABLE IF NOT EXISTS metagraph (
                hotkey       TEXT    PRIMARY KEY,
                uid          INTEGER NOT NULL,
                round_index  INTEGER NOT NULL DEFAULT 0
            )
            """,
        ),
    ),
    (
        4,
        (
            """
            CREATE TABLE IF NOT EXISTS telemetry_events (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                hotkey         TEXT    NOT NULL,
                round_index    INTEGER NOT NULL,
                phase          TEXT    NOT NULL,
                role           TEXT,
                epoch          INTEGER,
                step           INTEGER,
                loss           REAL,
                throughput     REAL,
                mfu            REAL,
                elapsed_s      INTEGER NOT NULL DEFAULT 0,
                eta_s          INTEGER,
                base_model     TEXT,
                note           TEXT,
                emitted_block  INTEGER NOT NULL DEFAULT 0,
                received_at    INTEGER NOT NULL DEFAULT 0
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS ix_telemetry_events_round
                ON telemetry_events (hotkey, round_index)
            """,
            """
            CREATE TABLE IF NOT EXISTS telemetry_state (
                hotkey         TEXT    NOT NULL,
                round_index    INTEGER NOT NULL,
                phase          TEXT    NOT NULL,
                role           TEXT,
                last_epoch     INTEGER,
                loss           REAL,
                throughput     REAL,
                mfu            REAL,
                elapsed_s      INTEGER NOT NULL DEFAULT 0,
                eta_s          INTEGER,
                note           TEXT,
                last_block     INTEGER NOT NULL DEFAULT 0,
                first_seen     INTEGER NOT NULL DEFAULT 0,
                updated_at     INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (hotkey, round_index)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS telemetry_hardware (
                hotkey              TEXT    NOT NULL,
                round_index         INTEGER NOT NULL,
                gpu_name            TEXT    NOT NULL DEFAULT '',
                gpu_count           INTEGER NOT NULL DEFAULT 0,
                vram_total_mb       INTEGER NOT NULL DEFAULT 0,
                cpu_count           INTEGER NOT NULL DEFAULT 0,
                ram_total_mb        INTEGER NOT NULL DEFAULT 0,
                bandwidth_up_mbps   REAL,
                bandwidth_down_mbps REAL,
                framework           TEXT    NOT NULL DEFAULT '',
                emitted_block       INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (hotkey, round_index)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS frontier_snapshots (
                round_index    INTEGER NOT NULL,
                track          TEXT    NOT NULL,
                hardware_class TEXT    NOT NULL,
                system_digest  TEXT    NOT NULL,
                quality_q      REAL    NOT NULL DEFAULT 0,
                cost_q         REAL    NOT NULL DEFAULT 0,
                hv_exclusive   REAL    NOT NULL DEFAULT 0,
                rank_by_cost   INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (round_index, track, hardware_class, system_digest)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS frontier_summary (
                round_index         INTEGER NOT NULL,
                track               TEXT    NOT NULL,
                hardware_class      TEXT    NOT NULL,
                member_count        INTEGER NOT NULL DEFAULT 0,
                hv_total            REAL    NOT NULL DEFAULT 0,
                best_quality        REAL    NOT NULL DEFAULT 0,
                lowest_cost         REAL    NOT NULL DEFAULT 0,
                median_resolve_rate REAL    NOT NULL DEFAULT 0,
                PRIMARY KEY (round_index, track, hardware_class)
            )
            """,
        ),
    ),
)
