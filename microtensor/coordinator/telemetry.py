from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from microtensor.core.constants import TELEMETRY_HEARTBEAT_BLOCKS

log = logging.getLogger("microtensor.coordinator.telemetry")

MAX_EVENTS_PER_POST = 200
RETAIN_DAYS = 30

PHASES = frozenset(
    {
        "registered",
        "training",
        "packaging",
        "uploading",
        "committing",
        "submitted",
        "failed",
    }
)
ROLES = frozenset({"front", "router", "specialist"})

UNKNOWN_PHASE = "the event names a phase this build does not know"
TOO_MANY = "too many events in one post"


class TelemetryRejected(ValueError):
    pass


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clean(raw: Mapping[str, Any], hotkey: str) -> dict[str, Any]:
    """One event, in the shape the store holds.

    The hotkey comes from the verified signature rather than the body, so a
    caller cannot report progress on somebody else's behalf. Everything else is
    coerced or nulled: telemetry is observational, so a malformed field is worth
    dropping rather than refusing the batch it arrived in.
    """
    phase = str(raw.get("phase", ""))
    if phase not in PHASES:
        raise TelemetryRejected(f"{UNKNOWN_PHASE}: {phase!r}")

    role = str(raw.get("role") or "") or None
    if role is not None and role not in ROLES:
        role = None

    return {
        "hotkey": hotkey,
        "round_index": _count(raw.get("round_index")) or 0,
        "phase": phase,
        "role": role,
        "epoch": _count(raw.get("epoch")),
        "step": _count(raw.get("step")),
        "loss": _number(raw.get("loss")),
        "throughput": _number(raw.get("throughput")),
        "mfu": _number(raw.get("mfu")),
        "elapsed_s": _count(raw.get("elapsed_s")) or 0,
        "eta_s": _count(raw.get("eta_s")),
        "base_model": str(raw.get("base_model") or "") or None,
        "note": str(raw.get("note") or "") or None,
        "emitted_block": _count(raw.get("emitted_block")) or 0,
    }


def clean_batch(events: Sequence[Mapping[str, Any]], hotkey: str) -> list[dict[str, Any]]:
    if len(events) > MAX_EVENTS_PER_POST:
        raise TelemetryRejected(f"{TOO_MANY}: {len(events)} over {MAX_EVENTS_PER_POST}")
    return [clean(event, hotkey) for event in events]


def clean_hardware(raw: Mapping[str, Any], hotkey: str) -> dict[str, Any]:
    """One hardware report, coerced.

    Self reported, and nothing downstream is allowed to treat it otherwise. It
    exists so the network can be described, not so a participant can be ranked.
    """
    return {
        "hotkey": hotkey,
        "round_index": _count(raw.get("round_index")) or 0,
        "gpu_name": str(raw.get("gpu_name") or "")[:64],
        "gpu_count": _count(raw.get("gpu_count")) or 0,
        "vram_total_mb": _count(raw.get("vram_total_mb")) or 0,
        "cpu_count": _count(raw.get("cpu_count")) or 0,
        "ram_total_mb": _count(raw.get("ram_total_mb")) or 0,
        "bandwidth_up_mbps": _number(raw.get("bandwidth_up_mbps")),
        "bandwidth_down_mbps": _number(raw.get("bandwidth_down_mbps")),
        "framework": str(raw.get("framework") or "")[:64],
        "emitted_block": _count(raw.get("emitted_block")) or 0,
    }


def state_from(events: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The row that describes where a miner currently is.

    Taken from the latest event by emitted block, so a batch arriving out of
    order still resolves to the furthest point the miner reported rather than
    whichever event happened to be last in the list.
    """
    if not events:
        return None

    latest = max(events, key=lambda e: (int(e.get("emitted_block") or 0), int(e.get("epoch") or 0)))
    now = int(time.time())
    return {
        "hotkey": latest["hotkey"],
        "round_index": latest["round_index"],
        "phase": latest["phase"],
        "role": latest.get("role"),
        "last_epoch": latest.get("epoch"),
        "loss": latest.get("loss"),
        "throughput": latest.get("throughput"),
        "mfu": latest.get("mfu"),
        "elapsed_s": latest.get("elapsed_s") or 0,
        "eta_s": latest.get("eta_s"),
        "note": latest.get("note"),
        "last_block": latest.get("emitted_block") or 0,
        "updated_at": now,
    }


def silent(
    states: Sequence[Mapping[str, Any]],
    block: int,
    committed: frozenset[str] | set[str],
    *,
    threshold: int = TELEMETRY_HEARTBEAT_BLOCKS,
) -> dict[str, int]:
    """Who has gone quiet without committing, and how long ago they last spoke.

    A commitment ends the check. A participant who finished training, committed
    on chain and then lost the box has a real artifact sitting in a store that
    any validator can fetch and measure, and discarding it over a dead heartbeat
    would throw away the work for a reason unrelated to the work.

    Measured in blocks rather than epochs, because an epoch count only exists in
    what a miner reported and a silent miner reports nothing. Blocks are chain
    derived, so every party computes the same answer and the decision can be
    checked afterwards.
    """
    dropped: dict[str, int] = {}
    for state in states:
        hotkey = str(state.get("hotkey", ""))
        if not hotkey or hotkey in committed:
            continue
        last = int(state.get("last_block") or 0)
        if block - last >= threshold:
            dropped[hotkey] = last
    return dict(sorted(dropped.items()))
