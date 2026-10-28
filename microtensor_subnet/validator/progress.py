from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol, TypeVar

from microtensor_subnet.scoring import pools

log = logging.getLogger("microtensor_subnet.validator.progress")

QUEUED: Final[str] = "queued"
FETCHING: Final[str] = "fetching"
CHECKING: Final[str] = "checking"
SCORING: Final[str] = "scoring"
SENT: Final[str] = "sent"
CLEARED: Final[str] = "cleared"
REFUSED: Final[str] = "refused"
STAGES: Final[tuple[str, ...]] = (
    QUEUED,
    FETCHING,
    CHECKING,
    pools.SLM,
    pools.HARNESS,
    pools.ESCALATION,
    pools.SYSTEM,
    SCORING,
    SENT,
    CLEARED,
    REFUSED,
)
MINER_PHASE: Final[str] = "miner"
BASELINE_PHASE: Final[str] = "baseline"
PHASES: Final[frozenset[str]] = frozenset({MINER_PHASE, BASELINE_PHASE})
THROTTLE_SECONDS: Final[float] = 15.0
QUEUE_LIMIT: Final[int] = 256
NOTE_LIMIT: Final[int] = 300
RESULT_TIMEOUT_SECONDS: Final[float] = 15.0
SETTLE_SECONDS: Final[float] = 4 * RESULT_TIMEOUT_SECONDS

T = TypeVar("T")


class Progress(Protocol):
    def __call__(self, done: int, total: int, detail: Mapping[str, Any] | None = None) -> None: ...


def stage_detail(stage: str, detail: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not detail:
        return None
    floor = detail.get("floor")
    if (
        stage == pools.SYSTEM
        and isinstance(floor, (int, float))
        and not isinstance(floor, bool)
        and 0.0 <= float(floor) <= 1.0
    ):
        return {"floor": float(floor)}
    phase = detail.get("phase")
    if stage == pools.HARNESS and phase in PHASES:
        return {"phase": str(phase)}
    return None


@dataclass(slots=True)
class Tally:
    progress: Progress | None
    total: int
    steps: int
    done: int = 0
    shown: int = -1
    detail: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._emit()

    def over(self, rows: Iterable[T]) -> Iterator[T]:
        for row in rows:
            yield row
            self.advance(1)

    def advance(self, count: int) -> None:
        self.done += count
        self._emit()

    def mark(self, **detail: Any) -> None:
        self.detail = dict(detail)
        self.shown = -1
        self._emit()

    def _emit(self) -> None:
        if self.progress is None:
            return
        total = max(0, self.total)
        shown = total if self.done >= self.steps else self.done * total // self.steps
        if shown != self.shown:
            self.shown = shown
            if self.detail is None:
                self.progress(shown, total)
            else:
                self.progress(shown, total, self.detail)


class Reporter:
    def __init__(
        self, client: Any = None, arena_id: str = "", epoch_index: int = 0, *, enabled: bool = True
    ) -> None:
        self._client: Any = client if enabled else None
        self._arena_id = arena_id
        self._epoch_index = epoch_index
        self._pending: deque[dict[str, Any]] = deque(maxlen=QUEUE_LIMIT)
        self._results: dict[str, dict[str, Any]] = {}
        self._sending: set[str] = set()
        self._settled: set[str] = set()
        self._last: dict[str, tuple[str, Any, float]] = {}
        self._wake = threading.Condition()
        self._worker: threading.Thread | None = None
        self._closed = False

    def post(
        self,
        miner: str,
        stage: str,
        done: int = 0,
        total: int = 0,
        note: str = "",
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        if self._client is None or stage not in STAGES:
            return
        try:
            self._enqueue(
                miner, stage, max(0, done), max(0, done, total), note, stage_detail(stage, detail)
            )
        except Exception as exc:
            log.debug("progress for %s not queued: %s", miner, exc)

    def tracker(self, miner: str, stage: str) -> Progress:
        def emit(done: int, total: int, detail: Mapping[str, Any] | None = None) -> None:
            self.post(miner, stage, done, total, detail=detail)

        return emit

    def publish(self, miner: str, body: dict[str, Any]) -> None:
        if self._client is None:
            return
        with self._wake:
            if self._closed or miner in self._settled:
                return
            self._results[miner] = body
            self._start()
            self._wake.notify_all()

    def settle(self, miner: str, timeout: float = SETTLE_SECONDS) -> None:
        with self._wake:
            self._settled.add(miner)
            self._results.pop(miner, None)
            self._wake.wait_for(lambda: miner not in self._sending, timeout=timeout)

    def close(self) -> None:
        with self._wake:
            self._closed = True
            self._wake.notify_all()

    def _start(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._drain, name="progress", daemon=True)
            self._worker.start()

    def _enqueue(
        self, miner: str, stage: str, done: int, total: int, note: str, detail: Any
    ) -> None:
        now = time.monotonic()
        with self._wake:
            if self._closed:
                return
            last = self._last.get(miner)
            if (
                last
                and last[0] == stage
                and last[1] == detail
                and done < total
                and now - last[2] < THROTTLE_SECONDS
            ):
                return
            self._last[miner] = (stage, detail, now)
            body: dict[str, Any] = {
                "arena_id": self._arena_id,
                "epoch_index": self._epoch_index,
                "miner": miner,
                "stage": stage,
                "done": done,
                "total": total,
                "note": note[:NOTE_LIMIT],
            }
            if detail:
                body["detail"] = detail
            self._pending.append(body)
            self._start()
            self._wake.notify_all()

    def _drain(self) -> None:
        while True:
            with self._wake:
                self._wake.wait_for(
                    lambda: bool(self._pending) or bool(self._results) or self._closed
                )
                if self._results:
                    miner = next(iter(self._results))
                    result = self._results.pop(miner)
                    self._sending.add(miner)
                elif self._pending:
                    miner, result = "", self._pending.popleft()
                else:
                    return
            if miner:
                self._send_result(miner, result)
                continue
            try:
                self._client.report_progress(result)
            except Exception as exc:
                log.debug("progress for %s not sent: %s", result["miner"], exc)

    def _send_result(self, miner: str, body: dict[str, Any]) -> None:
        try:
            self._client.submit_artifact_result(body, timeout=RESULT_TIMEOUT_SECONDS)
        except Exception as exc:
            log.info("partial result for %s not sent: %s", miner, exc)
        finally:
            with self._wake:
                self._sending.discard(miner)
                self._wake.notify_all()
