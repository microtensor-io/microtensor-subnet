from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, TypeVar, cast

from microtensor_subnet.validator.sandbox.slm import MEMORY_SHARE

log = logging.getLogger("microtensor_subnet.validator.rerun.engine")

DEFAULT_MAX_MODEL_LEN: Final[int] = 8192
HANG_GRACE_SECONDS: Final[float] = 30.0
MEMORY_SIGNS: Final[tuple[str, ...]] = (
    "out of memory",
    "outofmemory",
    "cannot allocate memory",
    "no available memory",
    "max seq len",
)
T = TypeVar("T")


class EngineError(RuntimeError):
    pass


class EngineHang(EngineError):
    pass


def out_of_memory(text: str) -> bool:
    found = text.lower()
    return any(sign in found for sign in MEMORY_SIGNS)


def bounded(work: Callable[[], T], seconds: float) -> T:
    if not seconds:
        return work()
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = work()
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise EngineHang(f"no answer within {seconds:.0f} s")
    if "error" in box:
        raise box["error"]
    return cast(T, box["value"])


@dataclass(frozen=True, slots=True)
class Generated:
    text: str
    ids: tuple[int, ...]
    prompt_tokens: int
    ttft_ms: float
    tpot_ms: float
    total_ms: float


def memory_share(definition: Mapping[str, Any], card_bytes: int, configured: float) -> float:
    if configured:
        return configured
    budgets = definition.get("budgets") or {}
    cap = int(budgets.get("memory_bytes") or budgets.get("vram_bytes") or 0) if budgets else 0
    if cap and card_bytes:
        return max(0.1, min(MEMORY_SHARE, cap / card_bytes))
    return MEMORY_SHARE


def full_length(definition: Mapping[str, Any]) -> int:
    budgets = definition.get("budgets") or {}
    if isinstance(budgets, Mapping) and budgets.get("max_input_tokens"):
        return int(budgets["max_input_tokens"]) + int(budgets.get("max_output_tokens") or 0)
    return DEFAULT_MAX_MODEL_LEN
