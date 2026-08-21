from __future__ import annotations

from typing import Any

_completed: list[Any] = []


def reset() -> None:
    """Start a run with nothing recorded.

    The jail spawns a fresh process per run so this is already empty, but a
    reset keeps the buffer honest under allow_unsandboxed, where the target
    runs in-process and would otherwise inherit the previous run's work.
    """
    _completed.clear()


def record(item: Any) -> None:
    """Note one finished unit of work.

    Called from the task loop rather than collected at the end, because the
    point is to have something to hand back when the loop does not reach the
    end. A miner that answered 150 of 200 tasks before its cpu budget ran out
    forfeits the 50 it did not reach, not the 150 it did.
    """
    _completed.append(item)


def collected() -> list[Any]:
    return list(_completed)
