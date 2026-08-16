from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from microtensor.core.protocol import Fault
from microtensor.harness.limits import (
    Limits,
    UnsupportedPlatform,
    apply,
    children_cpu_seconds,
    cpu_seconds_used,
    peak_rss_bytes,
    pin_threads,
    sandbox_available,
)

TERMINATE_GRACE_SECONDS = 2.0


STARVED_CPU_FRACTION = 0.25


@dataclass(frozen=True, slots=True)
class JailResult:
    completed: bool
    value: Any = None
    error: str = ""
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    peak_rss_bytes: int = 0
    timed_out: bool = False
    exit_code: int | None = None
    sandboxed: bool = True
    cpu_budget: float = 0.0

    @property
    def ok(self) -> bool:
        return self.completed and not self.error

    @property
    def killed(self) -> bool:
        return self.timed_out or (self.exit_code is not None and self.exit_code < 0)

    @property
    def starved(self) -> bool:
        return (
            self.timed_out
            and self.sandboxed
            and self.cpu_budget > 0.0
            and self.cpu_seconds < STARVED_CPU_FRACTION * self.cpu_budget
        )

    @property
    def fault(self) -> Fault | None:
        if self.ok:
            return None
        if self.starved:
            return Fault.INFRASTRUCTURE
        if self.timed_out or self.exit_code is not None:
            return Fault.ARTIFACT
        return Fault.INFRASTRUCTURE


def _entry(
    conn: Any, target: Callable[..., Any], args: tuple[Any, ...], limits: Limits | None
) -> None:
    try:
        pin_threads()
        if limits is not None:
            apply(limits)
    except BaseException as exc:
        conn.send(("infrastructure", f"{type(exc).__name__}: {exc}", 0.0, 0))
        conn.close()
        return

    try:
        value = target(*args)
        conn.send(("ok", value, _cpu(), _rss()))
    except MemoryError:
        conn.send(("artifact", "exhausted the memory ceiling", _cpu(), _rss()))
    except BaseException as exc:
        kind = "infrastructure" if _is_infrastructure(exc) else "artifact"
        conn.send((kind, f"{type(exc).__name__}: {exc}", _cpu(), _rss()))
    finally:
        conn.close()


def _is_infrastructure(exc: BaseException) -> bool:
    from microtensor.harness.registry import EngineUnavailable

    return isinstance(exc, (EngineUnavailable, ImportError, UnsupportedPlatform))


def _cpu() -> float:
    try:
        return cpu_seconds_used()
    except UnsupportedPlatform:
        return 0.0


def _rss() -> int:
    try:
        return peak_rss_bytes()
    except UnsupportedPlatform:
        return 0


def run_jailed(
    target: Callable[..., Any],
    *args: Any,
    limits: Limits,
    allow_unsandboxed: bool = False,
    start_method: str = "spawn",
) -> JailResult:
    sandboxed = sandbox_available()
    if not sandboxed and not allow_unsandboxed:
        raise UnsupportedPlatform(
            "refusing to execute an untrusted artifact without resource limits; "
            "run the validator on Linux or pass allow_unsandboxed for local development"
        )

    context: Any = mp.get_context(start_method)
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_entry,
        args=(child, target, args, limits if sandboxed else None),
        daemon=True,
    )

    cpu_before = children_cpu_seconds()
    started = time.monotonic()
    process.start()
    child.close()

    payload: tuple[Any, ...] | None = None
    if parent.poll(limits.wall_seconds):
        try:
            payload = parent.recv()
        except EOFError:
            payload = None

    timed_out = payload is None and process.is_alive()
    _stop(process)
    elapsed = time.monotonic() - started
    exit_code = process.exitcode
    parent.close()
    consumed = max(0.0, children_cpu_seconds() - cpu_before)

    if payload is None:
        return JailResult(
            completed=False,
            error=(
                f"exceeded the {limits.wall_seconds}s wall backstop"
                if timed_out
                else f"worker died without a result (exit {exit_code})"
            ),
            cpu_seconds=consumed,
            wall_seconds=elapsed,
            timed_out=timed_out,
            exit_code=None if timed_out else exit_code,
            sandboxed=sandboxed,
            cpu_budget=float(limits.cpu_seconds),
        )

    kind, value, cpu, rss = payload
    if kind == "ok":
        return JailResult(
            completed=True,
            value=value,
            cpu_seconds=cpu,
            wall_seconds=elapsed,
            peak_rss_bytes=rss,
            exit_code=exit_code,
            sandboxed=sandboxed,
            cpu_budget=float(limits.cpu_seconds),
        )

    return JailResult(
        completed=False,
        error=str(value),
        cpu_seconds=max(float(cpu), consumed),
        wall_seconds=elapsed,
        peak_rss_bytes=rss,
        exit_code=exit_code if kind == "artifact" else None,
        sandboxed=sandboxed,
        cpu_budget=float(limits.cpu_seconds),
    )


def _burn() -> None:
    value = 0
    while True:
        value = (value + 1) % 1_000_003


SPAWN_ALLOWANCE_SECONDS = 5.0


def cpu_limit_binds(probe_seconds: int = 1, slack: float = 2.0) -> bool:
    if not sandbox_available():
        return False
    result = run_jailed(
        _burn,
        limits=Limits(
            cpu_seconds=probe_seconds,
            wall_seconds=probe_seconds * 10,
            rss_bytes=1 << 30,
        ),
    )
    return (
        result.killed
        and not result.timed_out
        and result.wall_seconds <= probe_seconds * slack + SPAWN_ALLOWANCE_SECONDS
    )


def _stop(process: Any) -> None:
    if not process.is_alive():
        process.join()
        return
    process.terminate()
    process.join(TERMINATE_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join()
