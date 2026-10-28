from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Final, TypeVar

log = logging.getLogger("microtensor_subnet.validator.progress")

QUEUED: Final[str] = "queued"
FETCHING: Final[str] = "fetching"
CHECKING: Final[str] = "checking"
SCORING: Final[str] = "scoring"
SENT: Final[str] = "sent"
REFUSED: Final[str] = "refused"
THROTTLE_SECONDS: Final[float] = 15.0
QUEUE_LIMIT: Final[int] = 256
NOTE_LIMIT: Final[int] = 300

Progress = Callable[[int, int], None]
T = TypeVar("T")


@dataclass(slots=True)
class Tally:
    progress: Progress | None
    total: int
    done: int = 0

    def __post_init__(self) -> None:
        self._emit()

    def over(self, rows: Iterable[T]) -> Iterator[T]:
        for row in rows:
            yield row
            self.done += 1
            self._emit()

    def _emit(self) -> None:
        if self.progress is not None:
            self.progress(self.done, self.total)


class Reporter:
    def __init__(
        self, client: Any = None, arena_id: str = "", epoch_index: int = 0, *, enabled: bool = True
    ) -> None:
        self._client: Any = client if enabled else None
        self._arena_id = arena_id
        self._epoch_index = epoch_index
        self._pending: deque[dict[str, Any]] = deque(maxlen=QUEUE_LIMIT)
        self._last: dict[str, tuple[str, float]] = {}
        self._wake = threading.Condition()
        self._worker: threading.Thread | None = None
        self._closed = False

    def post(self, miner: str, stage: str, done: int = 0, total: int = 0, note: str = "") -> None:
        if self._client is None:
            return
        try:
            self._enqueue(miner, stage, max(0, done), max(0, done, total), note)
        except Exception as exc:
            log.debug("progress for %s not queued: %s", miner, exc)

    def tracker(self, miner: str, stage: str) -> Progress:
        return lambda done, total: self.post(miner, stage, done, total)

    def close(self) -> None:
        with self._wake:
            self._closed = True
            self._wake.notify_all()

    def _enqueue(self, miner: str, stage: str, done: int, total: int, note: str) -> None:
        now = time.monotonic()
        with self._wake:
            if self._closed:
                return
            last = self._last.get(miner)
            if last and last[0] == stage and done < total and now - last[1] < THROTTLE_SECONDS:
                return
            self._last[miner] = (stage, now)
            self._pending.append(
                {
                    "arena_id": self._arena_id,
                    "epoch_index": self._epoch_index,
                    "miner": miner,
                    "stage": stage,
                    "done": done,
                    "total": total,
                    "note": note[:NOTE_LIMIT],
                }
            )
            if self._worker is None:
                self._worker = threading.Thread(target=self._drain, name="progress", daemon=True)
                self._worker.start()
            self._wake.notify_all()

    def _drain(self) -> None:
        while True:
            with self._wake:
                self._wake.wait_for(lambda: bool(self._pending) or self._closed)
                if not self._pending:
                    return
                body = self._pending.popleft()
            try:
                self._client.report_progress(body)
            except Exception as exc:
                log.debug("progress for %s not sent: %s", body["miner"], exc)
