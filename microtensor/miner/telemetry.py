from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from microtensor.core.protocol import Role

log = logging.getLogger("microtensor.miner.telemetry")

FLUSH_SECONDS = 60.0
FLUSH_EVENTS = 20
BUFFER_MAX = 500
POST_TIMEOUT_SECONDS = 5


class Phase(str, Enum):
    REGISTERED = "registered"
    TRAINING = "training"
    PACKAGING = "packaging"
    UPLOADING = "uploading"
    COMMITTING = "committing"
    SUBMITTED = "submitted"
    FAILED = "failed"


TERMINAL = frozenset({Phase.SUBMITTED, Phase.FAILED})


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    hotkey: str
    round_index: int
    phase: Phase
    emitted_block: int
    role: Role | None = None
    epoch: int | None = None
    step: int | None = None
    loss: float | None = None
    throughput: float | None = None
    elapsed_s: int = 0
    eta_s: int | None = None
    base_model: str | None = None
    mfu: float | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phase"] = self.phase.value
        payload["role"] = self.role.value if self.role else None
        return payload


@dataclass(frozen=True, slots=True)
class HardwareReport:
    """What the miner says it is training on.

    Self reported and treated as such everywhere it surfaces. A participant can
    claim any accelerator, which is fine for a composition chart and is why this
    never gates admission, never enters a certificate and never touches a score.
    The only hardware figure that carries weight is the envelope a validator
    measures on its own certified machine.
    """

    hotkey: str
    round_index: int
    emitted_block: int
    gpu_name: str = ""
    gpu_count: int = 0
    vram_total_mb: int = 0
    cpu_count: int = 0
    ram_total_mb: int = 0
    bandwidth_up_mbps: float | None = None
    bandwidth_down_mbps: float | None = None
    framework: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Buffer:
    """A bounded window of events waiting to be sent.

    Bounded rather than growing: a coordinator that is unreachable for an hour
    must cost a fixed amount of memory, not an unbounded one. When it is full
    the oldest event goes, because the newest describes where the run actually
    is and the oldest is the one a chart can most afford to lose.
    """

    limit: int = BUFFER_MAX
    events: list[dict[str, Any]] = field(default_factory=list)
    dropped: int = 0

    def add(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        while len(self.events) > self.limit:
            self.events.pop(0)
            self.dropped += 1

    def take(self) -> list[dict[str, Any]]:
        taken, self.events = self.events, []
        return taken

    def __len__(self) -> int:
        return len(self.events)


class TelemetryClient:
    """Posts events to the coordinator without ever getting in training's way.

    Everything runs on a daemon thread behind a queue that never blocks the
    caller. A dropped batch is a gap in a chart; a telemetry client that stalls
    a training loop costs a participant the round, so the two failure modes are
    not close to equivalent and the code is shaped around the second one.
    """

    def __init__(
        self,
        base_url: str,
        wallet: Any = None,
        *,
        flush_seconds: float = FLUSH_SECONDS,
        flush_events: int = FLUSH_EVENTS,
        limit: int = BUFFER_MAX,
        transport: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.wallet = wallet
        self.flush_seconds = flush_seconds
        self.flush_events = flush_events
        self._transport = transport
        self._buffer = Buffer(limit=limit)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=limit * 2)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._sent = 0
        self._failed = 0

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    @property
    def stats(self) -> dict[str, int]:
        return {"sent": self._sent, "failed": self._failed, "dropped": self._buffer.dropped}

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="mt-telemetry", daemon=True)
        self._thread.start()

    def emit(self, event: TelemetryEvent) -> None:
        """Hand off an event. Never raises, never blocks, never waits."""
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(event.to_dict())
        except queue.Full:
            self._buffer.dropped += 1

    def hardware(self, report: HardwareReport) -> None:
        if not self.enabled:
            return
        self._post("/v1/telemetry/hardware", {"hardware": report.to_dict()})

    def stop(self, timeout: float = 3.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        self._thread.join(timeout=timeout)
        self._thread = None

    def _run(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            timeout = max(0.1, self.flush_seconds - (time.monotonic() - last))
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                item = None
            else:
                if item is None:
                    break
                self._buffer.add(item)

            due = len(self._buffer) >= self.flush_events
            elapsed = (time.monotonic() - last) >= self.flush_seconds
            if self._buffer and (due or elapsed):
                self._flush()
                last = time.monotonic()

        self._drain()
        self._flush()

    def _drain(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not None:
                self._buffer.add(item)

    def _flush(self) -> None:
        events = self._buffer.take()
        if not events:
            return
        if self._post("/v1/telemetry", {"events": events}):
            self._sent += len(events)
        else:
            self._failed += len(events)

    def _post(self, path: str, body: dict[str, Any]) -> bool:
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()

        if self._transport is not None:
            try:
                self._transport(path, raw)
                return True
            except Exception as exc:
                log.debug("telemetry transport refused %s: %s", path, exc)
                return False

        url = self.base_url + path
        if urllib.parse.urlparse(url).scheme not in ("http", "https"):
            return False

        try:
            request = urllib.request.Request(  # noqa: S310 - scheme checked above
                url, data=raw, method="POST", headers=self._headers(path, raw)
            )
            with urllib.request.urlopen(request, timeout=POST_TIMEOUT_SECONDS):  # noqa: S310
                return True
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            log.debug("telemetry not delivered: %s", exc)
            return False

    def _headers(self, path: str, raw: bytes) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.wallet is None:
            return headers

        from microtensor.chain.wallet import hotkey_address, sign_bytes
        from microtensor.coordinator.api import (
            HOTKEY_HEADER,
            SIGNATURE_HEADER,
            TIMESTAMP_HEADER,
            signing_bytes,
        )

        stamp = f"{time.time():.6f}"
        try:
            headers[HOTKEY_HEADER] = hotkey_address(self.wallet)
            headers[TIMESTAMP_HEADER] = stamp
            headers[SIGNATURE_HEADER] = sign_bytes(
                self.wallet, signing_bytes("POST", path, stamp, raw)
            )
        except Exception as exc:
            log.debug("telemetry could not be signed: %s", exc)
        return headers


def probe_hardware(hotkey: str, round_index: int, block: int) -> HardwareReport:
    """What this host looks like, as far as it can tell.

    Every field degrades to empty rather than guessing. A wrong figure here is
    worse than a missing one, because a missing one reads as unknown and a wrong
    one reads as fact.
    """
    import os

    report: dict[str, Any] = {
        "hotkey": hotkey,
        "round_index": round_index,
        "emitted_block": block,
        "cpu_count": os.cpu_count() or 0,
    }

    try:
        import torch

        report["framework"] = f"torch {torch.__version__}"
        if torch.cuda.is_available():
            report["gpu_count"] = torch.cuda.device_count()
            report["gpu_name"] = torch.cuda.get_device_name(0)
            total = sum(
                torch.cuda.get_device_properties(i).total_memory
                for i in range(torch.cuda.device_count())
            )
            report["vram_total_mb"] = int(total / 1024**2)
    except Exception as exc:
        log.debug("no accelerator detected: %s", exc)

    try:
        import psutil

        report["ram_total_mb"] = int(psutil.virtual_memory().total / 1024**2)
    except Exception as exc:
        log.debug("total memory not readable: %s", exc)

    return HardwareReport(**report)


def tflops_for(gpu_name: str, gpu_count: int = 1) -> float | None:
    """Peak throughput for a named accelerator, from a table in this repository.

    Derived rather than reported so a participant cannot inflate it and so the
    figure means the same thing across the network. An accelerator absent from
    the table yields nothing rather than an estimate.
    """
    if not gpu_name or gpu_count < 1:
        return None

    name = gpu_name.upper().replace("NVIDIA ", "").replace("GEFORCE ", "").strip()
    for key, value in ACCELERATORS.items():
        if key in name:
            return round(value * gpu_count, 1)
    return None


"""Half precision peak, in TFLOPS, without sparsity."""
ACCELERATORS: dict[str, float] = {
    "H200": 989.0,
    "H100": 989.0,
    "A100": 312.0,
    "L40S": 362.0,
    "L40": 181.0,
    "A6000": 155.0,
    "RTX 5090": 419.0,
    "RTX 4090": 165.2,
    "RTX 4080": 97.5,
    "RTX 3090": 71.0,
    "RTX 3080": 59.5,
    "A40": 149.7,
    "V100": 125.0,
    "T4": 65.0,
}


def batch(events: Sequence[TelemetryEvent]) -> dict[str, Any]:
    return {"events": [event.to_dict() for event in events]}
