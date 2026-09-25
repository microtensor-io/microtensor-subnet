from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

GIB: Final[int] = 1024**3
DEFAULT_SLOTS: Final[int] = 4
MIN_SLOTS: Final[int] = 1
KEEP_MARGIN: Final[float] = 1.5


class PlanError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Offer:
    model: str
    artifact_digest: str
    archive_repo: str = ""
    track: str = ""
    hardware_class: str = ""
    size_bytes: int = 0
    peak_rss_bytes: int = 0
    servable: bool = True
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    wanted: float = 0.0
    cold: bool = False
    operators_online: int = 0
    requests: int = 0

    @classmethod
    def from_wire(cls, row: Mapping[str, Any]) -> Offer:
        return cls(
            model=str(row.get("model", "")),
            artifact_digest=str(row.get("artifact_digest", "")),
            archive_repo=str(row.get("archive_repo", "")),
            track=str(row.get("track", "")),
            hardware_class=str(row.get("hardware_class", "")),
            size_bytes=int(row.get("size_bytes") or 0),
            peak_rss_bytes=int(row.get("peak_rss_bytes") or 0),
            servable=bool(row.get("servable", True)),
            ttft_ms=float(row.get("ttft_ms") or 0.0),
            tpot_ms=float(row.get("tpot_ms") or 0.0),
            wanted=float(row.get("wanted") or 0.0),
            cold=bool(row.get("cold", False)),
            operators_online=int(row.get("operators_online") or 0),
            requests=int(row.get("requests") or 0),
        )

    @property
    def footprint(self) -> int:
        return max(self.peak_rss_bytes, self.size_bytes)


@dataclass(frozen=True, slots=True)
class Capacity:
    disk_bytes: int
    memory_bytes: int
    slots: int
    max_models: int = 0

    def __post_init__(self) -> None:
        if self.disk_bytes <= 0 or self.memory_bytes <= 0:
            raise PlanError("a planner needs a disk and a memory budget above zero")
        if self.slots < MIN_SLOTS:
            raise PlanError(f"a planner needs at least {MIN_SLOTS} concurrent slot")
        if self.max_models < 0:
            raise PlanError("a model ceiling cannot be negative")

    @property
    def ceiling(self) -> int:
        return self.max_models or self.slots


@dataclass(frozen=True, slots=True)
class Policy:
    tracks: frozenset[str] = frozenset()
    classes: frozenset[str] = frozenset()
    exclude: frozenset[str] = frozenset()

    def allows(self, offer: Offer) -> bool:
        if offer.model in self.exclude:
            return False
        if self.tracks and offer.track not in self.tracks:
            return False
        return not (self.classes and offer.hardware_class not in self.classes)


@dataclass(frozen=True, slots=True)
class Holding:
    model: str
    artifact_digest: str
    concurrency: int


@dataclass(frozen=True, slots=True)
class Plan:
    hold: tuple[Holding, ...] = ()
    fetch: tuple[Offer, ...] = ()
    drop: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    disk_bytes: int = 0
    memory_bytes: int = 0
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(h.model for h in self.hold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hold": [
                {
                    "model": h.model,
                    "artifact_digest": h.artifact_digest,
                    "concurrency": h.concurrency,
                }
                for h in self.hold
            ],
            "fetch": [o.model for o in self.fetch],
            "drop": list(self.drop),
            "skipped": list(self.skipped),
            "disk_bytes": self.disk_bytes,
            "memory_bytes": self.memory_bytes,
            "reasons": dict(self.reasons),
        }


def rank(offers: Iterable[Offer], held: Mapping[str, str]) -> list[Offer]:
    def key(offer: Offer) -> tuple[float, int, int, str]:
        score = offer.wanted * (KEEP_MARGIN if offer.model in held else 1.0)
        return (-score, 0 if offer.cold else 1, offer.footprint, offer.model)

    return sorted(offers, key=key)


def _slots(capacity: Capacity, chosen: int) -> list[int]:
    if chosen <= 0:
        return []
    base = max(MIN_SLOTS, capacity.slots // chosen)
    shares = [base] * chosen
    left = capacity.slots - base * chosen
    index = 0
    while left > 0:
        shares[index % chosen] += 1
        left -= 1
        index += 1
    return shares


def build(
    offers: Sequence[Mapping[str, Any]] | Sequence[Offer],
    capacity: Capacity,
    *,
    held: Mapping[str, str] | None = None,
    policy: Policy | None = None,
    measured: Mapping[str, float] | None = None,
) -> Plan:
    rules = policy or Policy()
    holding = dict(held or {})
    timings = dict(measured or {})

    rows = [o if isinstance(o, Offer) else Offer.from_wire(o) for o in offers]
    reasons: dict[str, str] = {}
    usable: list[Offer] = []

    for offer in rows:
        if not offer.model or not offer.artifact_digest:
            continue
        if not offer.servable:
            reasons[offer.model] = "the network publishes no serving envelope for it"
            continue
        if not rules.allows(offer):
            reasons[offer.model] = "your policy excludes it"
            continue
        if offer.footprint > capacity.memory_bytes:
            reasons[offer.model] = "it does not fit in memory"
            continue
        if offer.size_bytes > capacity.disk_bytes:
            reasons[offer.model] = "it does not fit on disk"
            continue
        seen = timings.get(offer.model)
        if seen is not None and offer.tpot_ms and seen > offer.tpot_ms:
            reasons[offer.model] = (
                f"this machine takes {seen:.0f} ms a token and the envelope allows "
                f"{offer.tpot_ms:.0f}"
            )
            continue
        usable.append(offer)

    chosen: list[Offer] = []
    disk = memory = 0
    for offer in rank(usable, holding):
        if len(chosen) >= capacity.ceiling:
            reasons.setdefault(offer.model, "you are already holding as many models as you allow")
            continue
        if disk + offer.size_bytes > capacity.disk_bytes:
            reasons.setdefault(offer.model, "no disk left after the models above it")
            continue
        if memory + offer.footprint > capacity.memory_bytes:
            reasons.setdefault(offer.model, "no memory left after the models above it")
            continue
        chosen.append(offer)
        disk += offer.size_bytes
        memory += offer.footprint

    shares = _slots(capacity, len(chosen))
    hold = tuple(
        Holding(model=offer.model, artifact_digest=offer.artifact_digest, concurrency=share)
        for offer, share in zip(chosen, shares, strict=True)
    )
    wanted_now = {offer.model: offer.artifact_digest for offer in chosen}

    fetch = tuple(offer for offer in chosen if holding.get(offer.model) != offer.artifact_digest)
    drop = tuple(sorted(model for model in holding if model not in wanted_now))

    return Plan(
        hold=hold,
        fetch=fetch,
        drop=drop,
        skipped=tuple(sorted(reasons)),
        disk_bytes=disk,
        memory_bytes=memory,
        reasons=reasons,
    )


def changed(plan: Plan, held: Mapping[str, str]) -> bool:
    return {h.model: h.artifact_digest for h in plan.hold} != dict(held)
