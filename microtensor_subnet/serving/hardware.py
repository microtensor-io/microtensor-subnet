from __future__ import annotations

from typing import Final

SERVING_CLASSES: Final[dict[str, float]] = {
    "gpu_24gb": 24.0,
    "gpu_80gb": 80.0,
    "gpu_8x80gb": 640.0,
}
SLACK_GB: Final[float] = 2.0


def class_of(gpu_gb: float) -> str:
    fits = [
        (size, name) for name, size in SERVING_CLASSES.items() if float(gpu_gb) + SLACK_GB >= size
    ]
    return max(fits)[1] if fits else ""


def satisfies(have: str, need: str) -> bool:
    if not need:
        return True
    if have not in SERVING_CLASSES or need not in SERVING_CLASSES:
        return False
    return SERVING_CLASSES[have] >= SERVING_CLASSES[need]
