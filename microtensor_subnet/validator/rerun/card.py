from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from microtensor_subnet.core.profiles import PUBLISHED_CARDS


@dataclass(frozen=True, slots=True)
class Card:
    memory_bytes: int = 0
    name: str = ""

    def measures_speed(self, profile: str) -> bool:
        return any(wanted in self.name for wanted in PUBLISHED_CARDS.get(profile, ()))


def card_from(health: Mapping[str, Any]) -> Card:
    return Card(int(health.get("memory_total_bytes") or 0), str(health.get("gpu_name") or ""))
