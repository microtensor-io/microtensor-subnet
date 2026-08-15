from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

from microtensor.core.protocol import Evaluation
from microtensor.scoring.schedule import Candidate
from microtensor.store.db import Database

OPEN = "open"
SETTLED = "settled"
ABSTAINED = "abstained"

COUNT_QUERIES: Final[dict[str, str]] = {
    "rounds": "SELECT COUNT(*) AS n FROM rounds",
    "submissions": "SELECT COUNT(*) AS n FROM submissions",
    "evaluations": "SELECT COUNT(*) AS n FROM evaluations",
    "holders": "SELECT COUNT(*) AS n FROM holders",
    "weights": "SELECT COUNT(*) AS n FROM weights",
}


class ValidatorState:
    def __init__(self, path: Path | str) -> None:
        self.db = Database(path)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> ValidatorState:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open_round(self, round_index: int, seed_block: int, block_hash: str = "") -> None:
        self.db.execute(
            """
            INSERT INTO rounds (round_index, seed_block, block_hash, status, started_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (round_index) DO UPDATE SET
                seed_block = excluded.seed_block,
                block_hash = excluded.block_hash
            """,
            (round_index, seed_block, block_hash, OPEN, time.time()),
        )

    def finish_round(self, round_index: int, status: str, reason: str = "") -> None:
        if status not in (SETTLED, ABSTAINED):
            raise ValueError(f"a round finishes settled or abstained, not {status!r}")
        self.db.execute(
            "UPDATE rounds SET status = ?, reason = ?, settled_at = ? WHERE round_index = ?",
            (status, reason, time.time(), round_index),
        )

    def round_status(self, round_index: int) -> str | None:
        row = self.db.one("SELECT status FROM rounds WHERE round_index = ?", (round_index,))
        return str(row["status"]) if row else None

    def last_settled_round(self) -> int | None:
        row = self.db.one(
            "SELECT MAX(round_index) AS r FROM rounds WHERE status = ?", (SETTLED,)
        )
        return int(row["r"]) if row and row["r"] is not None else None

    def is_settled(self, round_index: int) -> bool:
        return self.round_status(round_index) == SETTLED

    def record_submission(
        self,
        round_index: int,
        hotkey: str,
        track: str,
        hardware_class: str,
        manifest_digest: str,
        source: str,
        *,
        artifact_digest: str = "",
        accepted: bool = False,
        reason: str = "",
    ) -> None:
        self.db.execute(
            """
            INSERT INTO submissions (
                round_index, hotkey, track, hardware_class,
                manifest_digest, artifact_digest, source, accepted, reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (round_index, hotkey) DO UPDATE SET
                track = excluded.track,
                hardware_class = excluded.hardware_class,
                manifest_digest = excluded.manifest_digest,
                artifact_digest = excluded.artifact_digest,
                source = excluded.source,
                accepted = excluded.accepted,
                reason = excluded.reason
            """,
            (
                round_index,
                hotkey,
                track,
                hardware_class,
                manifest_digest,
                artifact_digest,
                source,
                int(accepted),
                reason,
            ),
        )

    def rejected_submissions(self, round_index: int) -> list[tuple[str, str]]:
        rows = self.db.query(
            "SELECT hotkey, reason FROM submissions WHERE round_index = ? AND accepted = 0",
            (round_index,),
        )
        return [(str(r["hotkey"]), str(r["reason"])) for r in rows]

    def record_evaluation(self, round_index: int, evaluation: Evaluation) -> None:
        envelope = (
            json.dumps(asdict(evaluation.measured), sort_keys=True)
            if evaluation.measured is not None
            else ""
        )
        self.db.execute(
            """
            INSERT INTO evaluations (
                round_index, track, hardware_class, hotkey, artifact_digest,
                admitted, gate_reason, score_rotating, score_fixed, score_combined,
                n_rotating, n_fixed, corpus_version, envelope
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (round_index, track, hardware_class, hotkey) DO UPDATE SET
                artifact_digest = excluded.artifact_digest,
                admitted = excluded.admitted,
                gate_reason = excluded.gate_reason,
                score_rotating = excluded.score_rotating,
                score_fixed = excluded.score_fixed,
                score_combined = excluded.score_combined,
                n_rotating = excluded.n_rotating,
                n_fixed = excluded.n_fixed,
                corpus_version = excluded.corpus_version,
                envelope = excluded.envelope
            """,
            (
                round_index,
                evaluation.track,
                evaluation.hardware_class,
                evaluation.hotkey,
                evaluation.artifact_digest,
                int(evaluation.gate.admitted),
                evaluation.gate.reason,
                evaluation.score_rotating,
                evaluation.score_fixed,
                evaluation.score_combined,
                evaluation.n_rotating,
                evaluation.n_fixed,
                evaluation.corpus_version,
                envelope,
            ),
        )

    def evaluations(
        self, round_index: int, track: str, hardware_class: str
    ) -> list[dict[str, Any]]:
        rows = self.db.query(
            """
            SELECT * FROM evaluations
            WHERE round_index = ? AND track = ? AND hardware_class = ?
            ORDER BY score_combined DESC, hotkey ASC
            """,
            (round_index, track, hardware_class),
        )
        return [dict(r) for r in rows]

    def observe(
        self,
        track: str,
        hardware_class: str,
        hotkey: str,
        artifact_digest: str,
        round_index: int,
    ) -> None:
        row = self.db.one(
            """
            SELECT artifact_digest, rounds_observed, stale_rounds, last_round
            FROM observations WHERE track = ? AND hardware_class = ? AND hotkey = ?
            """,
            (track, hardware_class, hotkey),
        )

        if row is None:
            observed, stale = 1, 0
        elif int(row["last_round"]) == round_index:
            return
        elif str(row["artifact_digest"]) == artifact_digest:
            observed, stale = int(row["rounds_observed"]) + 1, int(row["stale_rounds"]) + 1
        else:
            observed, stale = int(row["rounds_observed"]) + 1, 0

        self.db.execute(
            """
            INSERT INTO observations (
                track, hardware_class, hotkey, artifact_digest,
                rounds_observed, stale_rounds, last_round
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (track, hardware_class, hotkey) DO UPDATE SET
                artifact_digest = excluded.artifact_digest,
                rounds_observed = excluded.rounds_observed,
                stale_rounds = excluded.stale_rounds,
                last_round = excluded.last_round
            """,
            (track, hardware_class, hotkey, artifact_digest, observed, stale, round_index),
        )

    def observation(self, track: str, hardware_class: str, hotkey: str) -> tuple[int, int]:
        row = self.db.one(
            """
            SELECT rounds_observed, stale_rounds FROM observations
            WHERE track = ? AND hardware_class = ? AND hotkey = ?
            """,
            (track, hardware_class, hotkey),
        )
        if row is None:
            return 0, 0
        return int(row["rounds_observed"]), int(row["stale_rounds"])

    def candidates(self, round_index: int, track: str, hardware_class: str) -> list[Candidate]:
        built: list[Candidate] = []
        for row in self.evaluations(round_index, track, hardware_class):
            if not row["admitted"]:
                continue
            observed, stale = self.observation(track, hardware_class, str(row["hotkey"]))
            built.append(
                Candidate(
                    hotkey=str(row["hotkey"]),
                    score=float(row["score_combined"]),
                    rounds_observed=observed,
                    stale_rounds=stale,
                )
            )
        return built

    def holders(self, track: str, hardware_class: str) -> list[str]:
        rows = self.db.query(
            "SELECT hotkey FROM holders WHERE track = ? AND hardware_class = ? ORDER BY rank ASC",
            (track, hardware_class),
        )
        return [str(r["hotkey"]) for r in rows]

    def _held_since(self, track: str, hardware_class: str) -> dict[tuple[int, str], int]:
        rows = self.db.query(
            "SELECT rank, hotkey, since_round FROM holders WHERE track = ? AND hardware_class = ?",
            (track, hardware_class),
        )
        return {(int(r["rank"]), str(r["hotkey"])): int(r["since_round"]) for r in rows}

    def set_holders(
        self, track: str, hardware_class: str, ranked: Sequence[str], round_index: int
    ) -> None:
        held_since = self._held_since(track, hardware_class)
        with self.db.transaction():
            self.db.execute(
                "DELETE FROM holders WHERE track = ? AND hardware_class = ?",
                (track, hardware_class),
            )
            self.db.executemany(
                """
                INSERT INTO holders (track, hardware_class, rank, hotkey, since_round)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        track,
                        hardware_class,
                        rank,
                        hotkey,
                        held_since.get((rank, hotkey), round_index),
                    )
                    for rank, hotkey in enumerate(ranked)
                ],
            )

    def held_since(self, track: str, hardware_class: str, hotkey: str) -> int | None:
        for (_, held), since in self._held_since(track, hardware_class).items():
            if held == hotkey:
                return since
        return None

    def record_weights(self, round_index: int, weights: Mapping[str, float]) -> None:
        with self.db.transaction():
            self.db.execute("DELETE FROM weights WHERE round_index = ?", (round_index,))
            self.db.executemany(
                "INSERT INTO weights (round_index, hotkey, weight) VALUES (?, ?, ?)",
                [(round_index, hotkey, float(w)) for hotkey, w in weights.items() if w > 0.0],
            )

    def last_weights(self) -> dict[str, float]:
        row = self.db.one("SELECT MAX(round_index) AS r FROM weights")
        if row is None or row["r"] is None:
            return {}
        rows = self.db.query(
            "SELECT hotkey, weight FROM weights WHERE round_index = ?", (int(row["r"]),)
        )
        return {str(r["hotkey"]): float(r["weight"]) for r in rows}

    def weights_at(self, round_index: int) -> dict[str, float]:
        rows = self.db.query(
            "SELECT hotkey, weight FROM weights WHERE round_index = ?", (round_index,)
        )
        return {str(r["hotkey"]): float(r["weight"]) for r in rows}

    def history(self, hotkey: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.query(
            """
            SELECT round_index, track, hardware_class, admitted, gate_reason, score_combined
            FROM evaluations WHERE hotkey = ?
            ORDER BY round_index DESC LIMIT ?
            """,
            (hotkey, limit),
        )
        return [dict(r) for r in rows]

    def summary(self) -> dict[str, int]:
        counted: dict[str, int] = {}
        for name, sql in COUNT_QUERIES.items():
            row = self.db.one(sql)
            counted[name] = int(row["n"]) if row else 0
        return counted
