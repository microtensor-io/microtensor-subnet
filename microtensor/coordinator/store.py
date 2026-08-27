from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from microtensor.coordinator.collect import Divergence
from microtensor.coordinator.report import CostBlock, QualityBlock, Report
from microtensor.coordinator.reputation import Standing
from microtensor.coordinator.schema import MIGRATIONS, SCHEMA_VERSION
from microtensor.coordinator.settle import Entry, Settlement
from microtensor.core.protocol import Fault
from microtensor.store.db import Database


class CoordinatorStore:
    """Rounds, assignments, reports, settlements.

    Reports are kept permanently rather than until settlement. They are the
    audit trail: the first time a miner disputes a score, the reports are the
    only thing that can answer, and they cannot be reconstructed afterwards.
    """

    def __init__(self, path: Path | str) -> None:
        self.db = Database(path, migrations=MIGRATIONS, schema_version=SCHEMA_VERSION)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> CoordinatorStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open_round(
        self,
        round_index: int,
        *,
        seed_block: int,
        close_block: int,
        block_hash: str = "",
        config_hash: str = "",
    ) -> None:
        self.db.execute(
            """
            INSERT INTO rounds (round_index, seed_block, block_hash, close_block,
                                config_hash, opened_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (round_index) DO UPDATE SET
                seed_block = excluded.seed_block,
                block_hash = excluded.block_hash,
                close_block = excluded.close_block,
                config_hash = excluded.config_hash
            """,
            (round_index, seed_block, block_hash, close_block, config_hash, time.time()),
        )

    def open_submissions(
        self,
        round_index: int,
        *,
        start_block: int,
        close_block: int,
        end_block: int,
        config_hash: str = "",
    ) -> None:
        """Open a round for submissions dynamically, with no seed yet.

        The operator's action sets the clock: start is now, close and end are
        offsets from it. The seed is the close block's hash, unknown until the
        window ends, so it is left blank and filled when the round is frozen.
        """
        self.db.execute(
            """
            INSERT INTO rounds (round_index, seed_block, block_hash, close_block,
                                start_block, end_block, phase, config_hash, opened_at)
            VALUES (?, ?, '', ?, ?, ?, 'submissions', ?, ?)
            ON CONFLICT (round_index) DO UPDATE SET
                close_block = excluded.close_block,
                start_block = excluded.start_block,
                end_block = excluded.end_block,
                phase = excluded.phase,
                config_hash = excluded.config_hash
            """,
            (
                round_index, close_block, close_block, start_block,
                end_block, config_hash, time.time(),
            ),
        )

    def freeze_round(self, round_index: int, *, block_hash: str) -> None:
        """Move a round from submissions to evaluation, recording the seed."""
        self.db.execute(
            "UPDATE rounds SET block_hash = ?, phase = 'evaluation', closed_at = ? "
            "WHERE round_index = ?",
            (block_hash, time.time(), round_index),
        )

    def next_round_index(self) -> int:
        row = self.latest_round()
        return (int(row["round_index"]) + 1) if row else 0

    def mark_anchored(self, round_index: int) -> None:
        self.db.execute(
            "UPDATE rounds SET anchored_at = ? WHERE round_index = ?",
            (time.time(), round_index),
        )

    def round(self, round_index: int) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM rounds WHERE round_index = ?", (round_index,))
        return dict(row) if row else None

    def latest_round(self) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM rounds ORDER BY round_index DESC LIMIT 1")
        return dict(row) if row else None

    def record_assignment(
        self,
        round_index: int,
        assignment: Mapping[str, Sequence[str]],
        systems: Mapping[str, tuple[str, str, str]],
    ) -> None:
        rows = [
            (
                round_index,
                worker,
                digest,
                systems.get(digest, ("", "", ""))[0],
                systems.get(digest, ("", "", ""))[1],
                systems.get(digest, ("", "", ""))[2],
            )
            for digest, workers in assignment.items()
            for worker in workers
        ]
        if not rows:
            return
        self.db.execute("DELETE FROM assignments WHERE round_index = ?", (round_index,))
        self.db.executemany(
            """
            INSERT INTO assignments (round_index, worker_hotkey, system_digest,
                                     track, hardware_class, miner_hotkey)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (round_index, worker_hotkey, system_digest) DO NOTHING
            """,
            rows,
        )

    def extend_assignment(
        self,
        round_index: int,
        assignment: Mapping[str, Sequence[str]],
        systems: Mapping[str, tuple[str, str, str]],
    ) -> int:
        rows = [
            (
                round_index,
                worker,
                digest,
                systems.get(digest, ("", "", ""))[0],
                systems.get(digest, ("", "", ""))[1],
                systems.get(digest, ("", "", ""))[2],
            )
            for digest, workers in assignment.items()
            for worker in workers
        ]
        if not rows:
            return 0
        before = self.expected_reports(round_index)
        self.db.executemany(
            """
            INSERT INTO assignments (round_index, worker_hotkey, system_digest,
                                     track, hardware_class, miner_hotkey)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (round_index, worker_hotkey, system_digest) DO NOTHING
            """,
            rows,
        )
        return self.expected_reports(round_index) - before

    def unassigned(self, round_index: int) -> tuple[str, ...]:
        rows = self.db.query(
            """
            SELECT c.system_digest FROM catalogue c
            LEFT JOIN assignments a
              ON a.round_index = c.round_index AND a.system_digest = c.system_digest
            WHERE c.round_index = ? AND a.system_digest IS NULL
            ORDER BY c.system_digest
            """,
            (round_index,),
        )
        return tuple(str(r["system_digest"]) for r in rows)

    def round_workers(self, round_index: int) -> tuple[str, ...]:
        rows = self.db.query(
            """
            SELECT DISTINCT worker_hotkey FROM assignments
            WHERE round_index = ? ORDER BY worker_hotkey
            """,
            (round_index,),
        )
        return tuple(str(r["worker_hotkey"]) for r in rows)

    def round_replication(self, round_index: int) -> int:
        row = self.db.one(
            """
            SELECT MAX(n) AS d FROM (
                SELECT COUNT(*) AS n FROM assignments
                WHERE round_index = ? GROUP BY system_digest
            )
            """,
            (round_index,),
        )
        return int(row["d"]) if row and row["d"] else 0

    def assignment_for(self, round_index: int, worker_hotkey: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            """
            SELECT system_digest, track, hardware_class, miner_hotkey
            FROM assignments WHERE round_index = ? AND worker_hotkey = ?
            ORDER BY system_digest
            """,
            (round_index, worker_hotkey),
        )
        return [dict(r) for r in rows]

    def assigned_digests(self, round_index: int, worker_hotkey: str) -> tuple[str, ...]:
        return tuple(a["system_digest"] for a in self.assignment_for(round_index, worker_hotkey))

    def expected_reports(self, round_index: int) -> int:
        row = self.db.one(
            "SELECT COUNT(*) AS n FROM assignments WHERE round_index = ?", (round_index,)
        )
        return int(row["n"]) if row else 0

    def full_assignment(self, round_index: int) -> dict[str, tuple[str, ...]]:
        rows = self.db.query(
            """
            SELECT system_digest, worker_hotkey FROM assignments
            WHERE round_index = ? ORDER BY system_digest, worker_hotkey
            """,
            (round_index,),
        )
        out: dict[str, list[str]] = {}
        for row in rows:
            out.setdefault(str(row["system_digest"]), []).append(str(row["worker_hotkey"]))
        return {k: tuple(v) for k, v in out.items()}

    def record_catalogue(self, round_index: int, catalogue: Mapping[str, Entry]) -> None:
        """Store which systems this round covers and who owns them."""
        rows = [
            (
                round_index,
                e.system_digest,
                e.miner_hotkey,
                e.uid,
                e.track,
                e.hardware_class,
                e.committed_at,
                e.rounds_observed,
                e.stale_rounds,
            )
            for e in catalogue.values()
        ]
        if not rows:
            return
        self.db.executemany(
            """
            INSERT INTO catalogue (round_index, system_digest, miner_hotkey, uid,
                                   track, hardware_class, committed_at,
                                   rounds_observed, stale_rounds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (round_index, system_digest) DO UPDATE SET
                miner_hotkey = excluded.miner_hotkey,
                uid = excluded.uid,
                track = excluded.track,
                hardware_class = excluded.hardware_class,
                committed_at = excluded.committed_at,
                rounds_observed = excluded.rounds_observed,
                stale_rounds = excluded.stale_rounds
            """,
            rows,
        )

    def record_telemetry(self, events: Sequence[Mapping[str, Any]]) -> None:
        """Append events and roll the current state forward in one transaction.

        Kept away from reports and settlements, which are permanent, small and
        load bearing. Telemetry is the opposite of all three: high volume, low
        value per row, and safe to lose.
        """
        if not events:
            return

        from microtensor.coordinator.telemetry import state_from

        now = int(time.time())
        with self.db.transaction():
            self.db.executemany(
                """
                INSERT INTO telemetry_events (
                    hotkey, round_index, phase, role, epoch, step, loss, throughput,
                    mfu, elapsed_s, eta_s, base_model, note, emitted_block, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e["hotkey"],
                        e["round_index"],
                        e["phase"],
                        e.get("role"),
                        e.get("epoch"),
                        e.get("step"),
                        e.get("loss"),
                        e.get("throughput"),
                        e.get("mfu"),
                        e.get("elapsed_s", 0),
                        e.get("eta_s"),
                        e.get("base_model"),
                        e.get("note"),
                        e.get("emitted_block", 0),
                        now,
                    )
                    for e in events
                ],
            )

            by_miner: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
            for event in events:
                by_miner.setdefault((event["hotkey"], event["round_index"]), []).append(event)

            for group in by_miner.values():
                state = state_from(group)
                if state is None:
                    continue
                self.db.execute(
                    """
                    INSERT INTO telemetry_state (
                        hotkey, round_index, phase, role, last_epoch, loss, throughput,
                        mfu, elapsed_s, eta_s, note, last_block, first_seen, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(hotkey, round_index) DO UPDATE SET
                        phase = excluded.phase,
                        role = excluded.role,
                        last_epoch = excluded.last_epoch,
                        loss = excluded.loss,
                        throughput = excluded.throughput,
                        mfu = excluded.mfu,
                        elapsed_s = excluded.elapsed_s,
                        eta_s = excluded.eta_s,
                        note = excluded.note,
                        last_block = MAX(telemetry_state.last_block, excluded.last_block),
                        updated_at = excluded.updated_at
                    """,
                    (
                        state["hotkey"],
                        state["round_index"],
                        state["phase"],
                        state.get("role"),
                        state.get("last_epoch"),
                        state.get("loss"),
                        state.get("throughput"),
                        state.get("mfu"),
                        state.get("elapsed_s", 0),
                        state.get("eta_s"),
                        state.get("note"),
                        state.get("last_block", 0),
                        now,
                        now,
                    ),
                )

    def record_hardware(self, report: Mapping[str, Any]) -> None:
        with self.db.transaction():
            self.db.execute(
                """
                INSERT INTO telemetry_hardware (
                    hotkey, round_index, gpu_name, gpu_count, vram_total_mb, cpu_count,
                    ram_total_mb, bandwidth_up_mbps, bandwidth_down_mbps, framework,
                    emitted_block
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hotkey, round_index) DO UPDATE SET
                    gpu_name = excluded.gpu_name,
                    gpu_count = excluded.gpu_count,
                    vram_total_mb = excluded.vram_total_mb,
                    cpu_count = excluded.cpu_count,
                    ram_total_mb = excluded.ram_total_mb,
                    bandwidth_up_mbps = excluded.bandwidth_up_mbps,
                    bandwidth_down_mbps = excluded.bandwidth_down_mbps,
                    framework = excluded.framework,
                    emitted_block = excluded.emitted_block
                """,
                (
                    report["hotkey"],
                    report["round_index"],
                    report.get("gpu_name", ""),
                    report.get("gpu_count", 0),
                    report.get("vram_total_mb", 0),
                    report.get("cpu_count", 0),
                    report.get("ram_total_mb", 0),
                    report.get("bandwidth_up_mbps"),
                    report.get("bandwidth_down_mbps"),
                    report.get("framework", ""),
                    report.get("emitted_block", 0),
                ),
            )

    def telemetry_state(self, round_index: int) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.query(
                "SELECT * FROM telemetry_state WHERE round_index = ? ORDER BY hotkey",
                (round_index,),
            )
        ]

    def telemetry_for(self, hotkey: str, limit: int = 500) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.query(
                """
                SELECT * FROM telemetry_events WHERE hotkey = ?
                ORDER BY emitted_block DESC, id DESC LIMIT ?
                """,
                (hotkey, limit),
            )
        ]

    def hardware_for(self, round_index: int) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.query(
                "SELECT * FROM telemetry_hardware WHERE round_index = ? ORDER BY hotkey",
                (round_index,),
            )
        ]

    def prune_telemetry(self, older_than_days: int = 30) -> int:
        cutoff = int(time.time()) - older_than_days * 86400
        with self.db.transaction():
            cursor = self.db.execute(
                "DELETE FROM telemetry_events WHERE received_at < ?", (cutoff,)
            )
        return int(cursor.rowcount or 0)

    def record_frontier(
        self,
        round_index: int,
        snapshots: Sequence[Mapping[str, Any]],
        summaries: Sequence[Mapping[str, Any]],
    ) -> None:
        """Keep the frontier as settlement computed it.

        Written once per round because history that is not captured cannot be
        reconstructed: recomputing three hundred rounds to draw one line is the
        alternative, and it gets worse every round.
        """
        if not snapshots and not summaries:
            return

        with self.db.transaction():
            self.db.executemany(
                """
                INSERT INTO frontier_snapshots (
                    round_index, track, hardware_class, system_digest,
                    quality_q, cost_q, hv_exclusive, rank_by_cost
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(round_index, track, hardware_class, system_digest) DO UPDATE SET
                    quality_q = excluded.quality_q,
                    cost_q = excluded.cost_q,
                    hv_exclusive = excluded.hv_exclusive,
                    rank_by_cost = excluded.rank_by_cost
                """,
                [
                    (
                        round_index,
                        s["track"],
                        s["hardware_class"],
                        s["system_digest"],
                        s.get("quality_q", 0.0),
                        s.get("cost_q", 0.0),
                        s.get("hv_exclusive", 0.0),
                        s.get("rank_by_cost", 0),
                    )
                    for s in snapshots
                ],
            )
            self.db.executemany(
                """
                INSERT INTO frontier_summary (
                    round_index, track, hardware_class, member_count, hv_total,
                    best_quality, lowest_cost, median_resolve_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(round_index, track, hardware_class) DO UPDATE SET
                    member_count = excluded.member_count,
                    hv_total = excluded.hv_total,
                    best_quality = excluded.best_quality,
                    lowest_cost = excluded.lowest_cost,
                    median_resolve_rate = excluded.median_resolve_rate
                """,
                [
                    (
                        round_index,
                        s["track"],
                        s["hardware_class"],
                        s.get("member_count", 0),
                        s.get("hv_total", 0.0),
                        s.get("best_quality", 0.0),
                        s.get("lowest_cost", 0.0),
                        s.get("median_resolve_rate", 0.0),
                    )
                    for s in summaries
                ],
            )

    def frontier_history(self, track: str = "", hardware_class: str = "") -> list[dict[str, Any]]:
        if track and hardware_class:
            rows = self.db.query(
                """
                SELECT * FROM frontier_summary WHERE track = ? AND hardware_class = ?
                ORDER BY round_index
                """,
                (track, hardware_class),
            )
        else:
            rows = self.db.query("SELECT * FROM frontier_summary ORDER BY round_index")
        return [dict(row) for row in rows]

    def record_metagraph(self, round_index: int, uid_by_hotkey: Mapping[str, int]) -> None:
        """Keep the uid map the round was opened against.

        Settling happens in a separate invocation from opening, and the reserved
        hotkey is a validator rather than a miner, so it appears in no catalogue
        row. Without this the settle path would have to reach for chain again to
        learn a uid it already had in hand.
        """
        if not uid_by_hotkey:
            return
        with self.db.transaction():
            self.db.executemany(
                """
                INSERT INTO metagraph (hotkey, uid, round_index)
                VALUES (?, ?, ?)
                ON CONFLICT(hotkey) DO UPDATE SET
                    uid = excluded.uid,
                    round_index = excluded.round_index
                """,
                [(h, int(u), round_index) for h, u in sorted(uid_by_hotkey.items())],
            )

    def uids(self) -> dict[str, int]:
        return {
            str(r["hotkey"]): int(r["uid"])
            for r in self.db.query("SELECT hotkey, uid FROM metagraph")
        }

    def catalogue(self, round_index: int) -> dict[str, Entry]:
        rows = self.db.query(
            "SELECT * FROM catalogue WHERE round_index = ? ORDER BY system_digest",
            (round_index,),
        )
        return {
            str(r["system_digest"]): Entry(
                system_digest=str(r["system_digest"]),
                miner_hotkey=str(r["miner_hotkey"]),
                uid=int(r["uid"]),
                track=str(r["track"]),
                hardware_class=str(r["hardware_class"]),
                quality=0.0,
                expected_ms=0.0,
                committed_at=int(r["committed_at"]),
                rounds_observed=int(r["rounds_observed"]),
                stale_rounds=int(r["stale_rounds"]),
            )
            for r in rows
        }

    def observations(self, before: int) -> dict[str, tuple[int, int]]:
        """How many rounds each miner has been seen for, from prior catalogues."""
        rows = self.db.query(
            """
            SELECT miner_hotkey,
                   COUNT(*) AS seen,
                   COUNT(DISTINCT system_digest) AS distinct_systems
            FROM catalogue WHERE round_index < ?
            GROUP BY miner_hotkey
            """,
            (before,),
        )
        return {
            str(r["miner_hotkey"]): (
                int(r["seen"]) + 1,
                max(int(r["seen"]) - int(r["distinct_systems"]), 0),
            )
            for r in rows
        }

    def record_report(self, report: Report) -> None:
        self.db.execute(
            """
            INSERT INTO reports (
                round_index, worker_hotkey, system_digest, quality,
                quality_rotating, quality_fixed, resolve_rate, expected_ms,
                expected_j, envelope, components, ablation, device_profile,
                conforming, engine_version, corpus_version, fault, signature,
                body_hash, received_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (round_index, worker_hotkey, system_digest) DO NOTHING
            """,
            (
                report.round_index,
                report.worker_hotkey,
                report.system_digest,
                report.quality.combined,
                report.quality.rotating,
                report.quality.fixed,
                report.resolve_rate,
                report.cost.expected_ms,
                report.cost.expected_j,
                json.dumps(report.envelope, sort_keys=True),
                json.dumps(report.components, sort_keys=True),
                json.dumps(report.ablation, sort_keys=True) if report.ablation else None,
                report.device_profile,
                int(report.conforming),
                report.engine_version,
                report.corpus_version,
                report.fault.value if report.fault else None,
                report.signature,
                report.digest(),
                time.time(),
            ),
        )

    def has_report(self, round_index: int, worker_hotkey: str, system_digest: str) -> bool:
        row = self.db.one(
            """
            SELECT 1 FROM reports
            WHERE round_index = ? AND worker_hotkey = ? AND system_digest = ?
            """,
            (round_index, worker_hotkey, system_digest),
        )
        return row is not None

    def reporters(self, round_index: int, system_digest: str) -> tuple[str, ...]:
        rows = self.db.query(
            "SELECT worker_hotkey FROM reports WHERE round_index = ? AND system_digest = ?",
            (round_index, system_digest),
        )
        return tuple(str(r["worker_hotkey"]) for r in rows)

    def report_count(self, round_index: int) -> int:
        row = self.db.one("SELECT COUNT(*) AS n FROM reports WHERE round_index = ?", (round_index,))
        return int(row["n"]) if row else 0

    def reports_by_system(self, round_index: int) -> dict[str, list[Report]]:
        rows = self.db.query(
            "SELECT * FROM reports WHERE round_index = ? ORDER BY system_digest, worker_hotkey",
            (round_index,),
        )
        out: dict[str, list[Report]] = {}
        for row in rows:
            out.setdefault(str(row["system_digest"]), []).append(_report_of(row))
        return out

    def report_digests(self, round_index: int) -> tuple[str, ...]:
        rows = self.db.query(
            "SELECT body_hash FROM reports WHERE round_index = ? ORDER BY body_hash",
            (round_index,),
        )
        return tuple(str(r["body_hash"]) for r in rows)

    def reports_payload(self, round_index: int) -> list[dict[str, Any]]:
        """Every report as it was signed, so the settlement is recomputable."""
        rows = self.db.query(
            "SELECT * FROM reports WHERE round_index = ? ORDER BY system_digest, worker_hotkey",
            (round_index,),
        )
        payload = []
        for row in rows:
            report = _report_of(row)
            body = report.body()
            body["signature"] = report.signature
            payload.append(body)
        return payload

    def record_divergences(self, round_index: int, divergences: Sequence[Divergence]) -> None:
        if not divergences:
            return
        self.db.executemany(
            """
            INSERT INTO divergences (round_index, system_digest, worker_hotkey,
                                     reported, reconciled, kind, noted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (round_index, system_digest, worker_hotkey, kind) DO UPDATE SET
                reported = excluded.reported,
                reconciled = excluded.reconciled
            """,
            [
                (
                    round_index,
                    d.system_digest,
                    d.worker_hotkey,
                    d.reported,
                    d.reconciled,
                    d.kind,
                    time.time(),
                )
                for d in divergences
            ],
        )

    def divergence_rate(self, round_index: int) -> float:
        total = self.report_count(round_index)
        if not total:
            return 0.0
        row = self.db.one(
            "SELECT COUNT(*) AS n FROM divergences WHERE round_index = ?", (round_index,)
        )
        return (int(row["n"]) if row else 0) / total

    def standing(self, hotkey: str) -> Standing:
        row = self.db.one("SELECT * FROM reputation WHERE worker_hotkey = ?", (hotkey,))
        if row is None:
            return Standing(hotkey=hotkey)
        return Standing(
            hotkey=hotkey,
            agreed=int(row["agreed"]),
            diverged=int(row["diverged"]),
            streak=int(row["streak"]),
            advisory=bool(row["advisory"]),
            last_round=int(row["last_round"]),
        )

    def save_standing(self, standing: Standing) -> None:
        self.db.execute(
            """
            INSERT INTO reputation (worker_hotkey, agreed, diverged, streak,
                                    advisory, last_round)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (worker_hotkey) DO UPDATE SET
                agreed = excluded.agreed,
                diverged = excluded.diverged,
                streak = excluded.streak,
                advisory = excluded.advisory,
                last_round = excluded.last_round
            """,
            (
                standing.hotkey,
                standing.agreed,
                standing.diverged,
                standing.streak,
                int(standing.advisory),
                standing.last_round,
            ),
        )

    def standings(self) -> list[Standing]:
        rows = self.db.query("SELECT * FROM reputation ORDER BY worker_hotkey")
        return [
            Standing(
                hotkey=str(r["worker_hotkey"]),
                agreed=int(r["agreed"]),
                diverged=int(r["diverged"]),
                streak=int(r["streak"]),
                advisory=bool(r["advisory"]),
                last_round=int(r["last_round"]),
            )
            for r in rows
        ]

    def advisory(self) -> tuple[str, ...]:
        rows = self.db.query(
            "SELECT worker_hotkey FROM reputation WHERE advisory = 1 ORDER BY worker_hotkey"
        )
        return tuple(str(r["worker_hotkey"]) for r in rows)

    def publish(self, settlement: Settlement) -> None:
        self.db.execute(
            """
            INSERT INTO settlements (round_index, config_hash, corpus_version,
                                     reports_root, payload, signature, published_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (round_index) DO UPDATE SET
                config_hash = excluded.config_hash,
                corpus_version = excluded.corpus_version,
                reports_root = excluded.reports_root,
                payload = excluded.payload,
                signature = excluded.signature,
                published_at = excluded.published_at
            """,
            (
                settlement.round_index,
                settlement.config_hash,
                settlement.corpus_version,
                settlement.reports_root,
                json.dumps(settlement.body(), sort_keys=True),
                settlement.signature,
                time.time(),
            ),
        )

    def newest_settlement(self) -> dict[str, Any] | None:
        row = self.db.one(
            "SELECT payload, signature FROM settlements ORDER BY round_index DESC LIMIT 1"
        )
        if row is None:
            return None
        body: dict[str, Any] = json.loads(str(row["payload"]))
        body["signature"] = str(row["signature"])
        return body

    def settlement(self, round_index: int) -> dict[str, Any] | None:
        row = self.db.one(
            "SELECT payload, signature FROM settlements WHERE round_index = ?",
            (round_index,),
        )
        if row is None:
            return None
        body: dict[str, Any] = json.loads(str(row["payload"]))
        body["signature"] = str(row["signature"])
        return body


def _report_of(row: Any) -> Report:
    fault = row["fault"]
    return Report(
        round_index=int(row["round_index"]),
        worker_hotkey=str(row["worker_hotkey"]),
        system_digest=str(row["system_digest"]),
        quality=QualityBlock(
            rotating=float(row["quality_rotating"]),
            fixed=float(row["quality_fixed"]),
            combined=float(row["quality"]),
        ),
        resolve_rate=float(row["resolve_rate"]),
        cost=CostBlock(
            expected_ms=float(row["expected_ms"]),
            expected_j=None if row["expected_j"] is None else float(row["expected_j"]),
        ),
        envelope=json.loads(str(row["envelope"] or "{}")),
        components=json.loads(str(row["components"] or "{}")),
        ablation=json.loads(str(row["ablation"])) if row["ablation"] else None,
        device_profile=str(row["device_profile"]),
        conforming=bool(row["conforming"]),
        engine_version=str(row["engine_version"]),
        corpus_version=str(row["corpus_version"]),
        fault=Fault(str(fault)) if fault else None,
        signature=str(row["signature"]),
    )
