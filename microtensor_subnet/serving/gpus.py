from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

QUERY: Final[tuple[str, ...]] = (
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
    "--format=csv,noheader,nounits",
)
QUERY_SECONDS: Final[float] = 20.0
ALL: Final[str] = "all"
MB_PER_GB: Final[int] = 1024

Runner = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class Card:
    device: str
    id: str
    name: str = ""
    memory_mb: int = 0
    memory_used_mb: int = 0
    utilisation: float = 0.0
    driver: str = ""

    @property
    def memory_gb(self) -> float:
        return self.memory_mb / MB_PER_GB

    def announced(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "slot": int(self.device) if self.device.isdigit() else 0,
            "name": self.name,
            "memory_mb": self.memory_mb,
            "driver": self.driver,
        }

    def reported(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory_used_mb": self.memory_used_mb,
            "utilisation": round(self.utilisation, 4),
        }


def _int(value: str) -> int:
    try:
        return int(float(value.strip()))
    except ValueError:
        return 0


def _row(line: str) -> Card | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 7 or not parts[0].isdigit():
        return None
    index, uuid, name, total, used, busy, driver = parts[:7]
    return Card(
        device=index,
        id=uuid or f"gpu:{index}",
        name=name,
        memory_mb=_int(total),
        memory_used_mb=_int(used),
        utilisation=min(1.0, max(0.0, _int(busy) / 100.0)),
        driver=driver,
    )


def detect(run: Runner = subprocess.run) -> list[Card]:
    try:
        found = run(
            list(QUERY),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=QUERY_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if found.returncode != 0:
        return []
    cards = [_row(line) for line in str(found.stdout).splitlines() if line.strip()]
    return [card for card in cards if card is not None]


def wanted_devices(spec: str) -> tuple[str, ...]:
    text = str(spec or "").strip().lower()
    if not text or text == ALL:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def choose(cards: Sequence[Card], spec: str) -> list[Card]:
    wanted = wanted_devices(spec)
    if not wanted:
        return list(cards)
    by_device = {card.device: card for card in cards}
    return [by_device[device] for device in wanted if device in by_device]


def stand_ins(spec: str, gpu_gb: float, *, worker: str = "") -> list[Card]:
    devices = wanted_devices(spec) or ("0",)
    prefix = worker or "gpu"
    return [
        Card(device=device, id=f"{prefix}:{device}", memory_mb=int(gpu_gb * MB_PER_GB))
        for device in devices
    ]


def refresh(cards: Sequence[Card], run: Runner = subprocess.run) -> list[Card]:
    seen = {card.id: card for card in detect(run)}
    return [seen.get(card.id, card) for card in cards]


def smallest_gb(cards: Sequence[Card]) -> float:
    sizes = [card.memory_gb for card in cards if card.memory_mb > 0]
    return min(sizes) if sizes else 0.0
