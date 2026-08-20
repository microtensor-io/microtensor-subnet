from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from microtensor.core.protocol import Role
from microtensor.miner.telemetry import (
    HardwareReport,
    Phase,
    TelemetryClient,
    TelemetryEvent,
    probe_hardware,
)

log = logging.getLogger("microtensor.miner.daemon")

TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    Phase.REGISTERED: frozenset({Phase.TRAINING, Phase.FAILED}),
    Phase.TRAINING: frozenset({Phase.PACKAGING, Phase.FAILED}),
    Phase.PACKAGING: frozenset({Phase.UPLOADING, Phase.FAILED}),
    Phase.UPLOADING: frozenset({Phase.COMMITTING, Phase.FAILED}),
    Phase.COMMITTING: frozenset({Phase.SUBMITTED, Phase.FAILED}),
    Phase.SUBMITTED: frozenset(),
    Phase.FAILED: frozenset(),
}


class DaemonError(RuntimeError):
    pass


@runtime_checkable
class TrainingHook(Protocol):
    """What the daemon needs from a participant's training code.

    Deliberately three calls rather than a framework. The daemon does not own
    the training loop and should not: a participant's loop is theirs, and the
    only thing this needs from it is to be told where it has got to.
    """

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None: ...

    def on_complete(self, artifacts: dict[Role, Path]) -> None: ...

    def on_failure(self, reason: str) -> None: ...


@dataclass(slots=True)
class Reporter:
    """The hook implementation the daemon hands to training code.

    Everything here is a hand off to the telemetry client, which never blocks.
    A participant calling this from inside a hot loop must not pay for it.
    """

    hotkey: str
    round_index: int
    client: TelemetryClient
    block_of: Callable[[], int]
    role: Role | None = None
    base_model: str = ""
    started: float = field(default_factory=time.monotonic)
    epochs_total: int | None = None
    artifacts: dict[Role, Path] = field(default_factory=dict)
    failure: str = ""

    @property
    def elapsed_s(self) -> int:
        return int(time.monotonic() - self.started)

    def _block(self) -> int:
        try:
            return int(self.block_of())
        except Exception as exc:
            log.debug("block height unavailable for telemetry: %s", exc)
            return 0

    def on_epoch_end(self, epoch: int, metrics: dict[str, float] | Mapping[str, float]) -> None:
        eta = None
        if self.epochs_total and epoch > 0:
            per_epoch = self.elapsed_s / epoch
            eta = int(per_epoch * max(0, self.epochs_total - epoch))

        self.client.emit(
            TelemetryEvent(
                hotkey=self.hotkey,
                round_index=self.round_index,
                phase=Phase.TRAINING,
                emitted_block=self._block(),
                role=self.role,
                epoch=epoch,
                step=_int(metrics.get("step")),
                loss=_float(metrics.get("loss")),
                throughput=_float(metrics.get("throughput")),
                mfu=_float(metrics.get("mfu")),
                elapsed_s=self.elapsed_s,
                eta_s=eta,
                base_model=self.base_model or None,
            )
        )

    def on_complete(self, artifacts: dict[Role, Path]) -> None:
        self.artifacts = dict(artifacts)

    def on_failure(self, reason: str) -> None:
        self.failure = reason


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class Daemon:
    """Trains, then submits, without anyone typing a command.

    The chain commitment stays with the miner throughout. A commitment is a
    signed extrinsic, so only this hotkey can produce one; nothing here moves
    that, it only removes the need for a person to be awake when it happens.
    """

    hotkey: str
    round_index: int
    client: TelemetryClient
    block_of: Callable[[], int]
    phase: Phase = Phase.REGISTERED
    reporter: Reporter | None = None

    def __post_init__(self) -> None:
        if self.reporter is None:
            self.reporter = Reporter(
                hotkey=self.hotkey,
                round_index=self.round_index,
                client=self.client,
                block_of=self.block_of,
            )

    def _block(self) -> int:
        try:
            return int(self.block_of())
        except Exception:
            return 0

    def enter(self, phase: Phase, note: str = "") -> None:
        """Move to the next phase and say so.

        Refuses a transition the machine does not allow, because a daemon that
        silently skips packaging and reports `submitted` is worse than one that
        stops: the first is a lie in the telemetry, the second is a bug report.
        """
        allowed = TRANSITIONS.get(self.phase, frozenset())
        if phase not in allowed:
            raise DaemonError(f"cannot move from {self.phase.value} to {phase.value}")

        self.phase = phase
        elapsed = self.reporter.elapsed_s if self.reporter else 0
        self.client.emit(
            TelemetryEvent(
                hotkey=self.hotkey,
                round_index=self.round_index,
                phase=phase,
                emitted_block=self._block(),
                elapsed_s=elapsed,
                note=note or None,
            )
        )
        log.info("round %d: %s%s", self.round_index, phase.value, f" ({note})" if note else "")

    def report_hardware(self) -> HardwareReport:
        """Sent once, on the way into training.

        Hardware does not change mid run, so it rides the transition rather than
        every epoch. Self reported, and nothing downstream may treat it as more
        than that.
        """
        report = probe_hardware(self.hotkey, self.round_index, self._block())
        self.client.hardware(report)
        return report

    def fail(self, reason: str) -> None:
        if self.phase in (Phase.SUBMITTED, Phase.FAILED):
            return
        self.phase = Phase.FAILED
        self.client.emit(
            TelemetryEvent(
                hotkey=self.hotkey,
                round_index=self.round_index,
                phase=Phase.FAILED,
                emitted_block=self._block(),
                elapsed_s=self.reporter.elapsed_s if self.reporter else 0,
                note=reason,
            )
        )
        log.error("round %d failed: %s", self.round_index, reason)

    def run(self, train: Callable[[TrainingHook], None], submit: Callable[[], Any]) -> bool:
        """Train, then submit, reporting every transition.

        `submit` is whatever `mt miner ship` already does: package, upload, log
        provenance, commit. It is passed in rather than reimplemented, so the
        unattended path and the manual one cannot drift apart.
        """
        assert self.reporter is not None

        try:
            self.enter(Phase.TRAINING)
            self.report_hardware()
            train(self.reporter)

            if self.reporter.failure:
                self.fail(self.reporter.failure)
                return False

            self.enter(Phase.PACKAGING)
            self.enter(Phase.UPLOADING)
            self.enter(Phase.COMMITTING)
            submit()
            self.enter(Phase.SUBMITTED)
            return True
        except DaemonError:
            raise
        except Exception as exc:
            self.fail(str(exc))
            return False
        finally:
            self.client.stop()
