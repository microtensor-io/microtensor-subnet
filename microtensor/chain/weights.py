from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from microtensor.core.constants import MECHANISM_VERSION

U16_MAX: Final[int] = 65535


@dataclass(frozen=True, slots=True)
class WeightVector:
    uids: tuple[int, ...]
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.uids) != len(self.values):
            raise ValueError("weight vector must pair one value with each uid")
        if len(set(self.uids)) != len(self.uids):
            raise ValueError("weight vector must not repeat a uid")
        if any(uid < 0 for uid in self.uids):
            raise ValueError("uids must not be negative")
        if any(not 0 <= value <= U16_MAX for value in self.values):
            raise ValueError(f"weights must fit in u16, got {self.values}")

    def __len__(self) -> int:
        return len(self.uids)

    @property
    def total(self) -> int:
        return sum(self.values)

    @property
    def is_empty(self) -> bool:
        return not self.uids

    def as_lists(self) -> tuple[list[int], list[int]]:
        return list(self.uids), list(self.values)

    def as_fractions(self) -> dict[int, float]:
        total = self.total
        if total <= 0:
            return {}
        return {uid: value / total for uid, value in zip(self.uids, self.values, strict=True)}


def version_key(version: str = MECHANISM_VERSION) -> int:
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"mechanism version {version!r} must look like major.minor.patch")
    major, minor, patch = (int(p) for p in parts)
    return major * 10_000 + minor * 100 + patch


def quantise_weights(weights: Mapping[int, float], *, total: int = U16_MAX) -> WeightVector:
    positive = {int(uid): float(w) for uid, w in weights.items() if w > 0.0}
    if not positive or total <= 0:
        return WeightVector((), ())

    mass = sum(positive.values())
    uids = sorted(positive)
    exact = [positive[uid] / mass * total for uid in uids]
    floors = [int(value) for value in exact]

    shortfall = total - sum(floors)
    order = sorted(
        range(len(uids)),
        key=lambda i: (-(exact[i] - floors[i]), uids[i]),
    )
    for i in order[:shortfall]:
        floors[i] += 1

    kept = [(uids[i], floors[i]) for i in range(len(uids)) if floors[i] > 0]
    return WeightVector(tuple(u for u, _ in kept), tuple(v for _, v in kept))


def restrict_to_metagraph(weights: Mapping[int, float], size: int) -> dict[int, float]:
    return {uid: w for uid, w in weights.items() if 0 <= uid < size and w > 0.0}
