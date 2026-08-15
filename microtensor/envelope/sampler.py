from __future__ import annotations

import os
import threading
import time
from types import TracebackType

from microtensor.core.constants import PROFILE_SAMPLE_INTERVAL_MS


def _psutil_reader() -> object | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid())


def _statm_rss() -> int:
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            resident = int(fh.read().split()[1])
    except (OSError, IndexError, ValueError):
        return 0
    return resident * os.sysconf("SC_PAGE_SIZE")


def read_rss_bytes() -> int:
    process = _psutil_reader()
    if process is not None:
        try:
            return int(process.memory_info().rss)
        except Exception:
            return 0
    return _statm_rss()


class ResidentSampler:
    def __init__(self, interval_ms: int = PROFILE_SAMPLE_INTERVAL_MS) -> None:
        if interval_ms < 1:
            raise ValueError("sampling interval must be at least one millisecond")
        self._interval = interval_ms / 1000.0
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def samples(self) -> tuple[int, ...]:
        return tuple(self._samples)

    @property
    def peak_bytes(self) -> int:
        return max(self._samples, default=0)

    @property
    def count(self) -> int:
        return len(self._samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = read_rss_bytes()
            if value:
                self._samples.append(value)
            self._stop.wait(self._interval)

    def start(self) -> ResidentSampler:
        if self._thread is not None:
            raise RuntimeError("sampler is already running")
        self._samples.append(read_rss_bytes())
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> int:
        if self._thread is None:
            return self.peak_bytes
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._interval * 10))
        self._thread = None
        self._samples.append(read_rss_bytes())
        return self.peak_bytes

    def mark(self) -> int:
        value = read_rss_bytes()
        if value:
            self._samples.append(value)
        return value

    def __enter__(self) -> ResidentSampler:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


def sample_for(seconds: float, interval_ms: int = PROFILE_SAMPLE_INTERVAL_MS) -> tuple[int, ...]:
    sampler = ResidentSampler(interval_ms)
    sampler.start()
    time.sleep(seconds)
    sampler.stop()
    return sampler.samples
