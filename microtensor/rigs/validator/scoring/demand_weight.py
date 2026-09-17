from __future__ import annotations

import json
import os
import time
from pathlib import Path

from microtensor.rigs.validator.data.tiers import Tier

ALPHA = 0.01
STATIC_PRIOR: dict[str, float] = {
    Tier.FLAGSHIP.value: 1.0,
    Tier.PROFESSIONAL.value: 0.6,
    Tier.STANDARD.value: 0.35,
    Tier.ENTRY.value: 0.15,
}
MIN_WEIGHT = 0.01


class DemandWeights:
    def __init__(self, path: Path | None = None, alpha: float = ALPHA) -> None:
        self.path = path
        self.alpha = alpha
        self.values: dict[str, float] = {}
        self.updated_at: dict[str, float] = {}
        self.load()

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.values = {str(k): float(v) for k, v in (data.get("values") or {}).items()}
        self.updated_at = {str(k): float(v) for k, v in (data.get("updated_at") or {}).items()}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps({"values": self.values, "updated_at": self.updated_at}, indent=1),
            encoding="utf-8",
        )
        os.replace(temp, self.path)

    def weight(self, key: str, tier: Tier | None = None) -> float:
        if key in self.values:
            return max(MIN_WEIGHT, self.values[key])
        if tier is not None:
            return STATIC_PRIOR.get(tier.value, MIN_WEIGHT)
        return STATIC_PRIOR.get(key, MIN_WEIGHT)

    def update(self, key: str, revenue_per_gpu_hour: float, tier: Tier | None = None) -> float:
        previous = self.weight(key, tier)
        sample = max(0.0, revenue_per_gpu_hour)
        value = previous + self.alpha * (sample - previous)
        self.values[key] = max(MIN_WEIGHT, value)
        self.updated_at[key] = time.time()
        return self.values[key]

    def update_many(self, revenue: dict[str, float], tiers: dict[str, Tier] | None = None) -> None:
        for key, sample in revenue.items():
            self.update(key, sample, (tiers or {}).get(key))
        self.save()

    def snapshot(self) -> dict[str, float]:
        merged = dict(STATIC_PRIOR)
        merged.update({k: max(MIN_WEIGHT, v) for k, v in self.values.items()})
        return merged
