from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

KEEP_MARGIN: Final[float] = 1.5


class PlanError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Offer:
    model: str
    artifact_digest: str
    arena: str = ""
    task: str = ""
    hardware_class: str = ""
    vram_gb: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    servable: bool = True
    wanted: float = 0.0
    cold: bool = False
    operators_online: int = 0
    requests: int = 0

    @classmethod
    def from_wire(cls, row: Mapping[str, Any]) -> Offer:
        envelope = dict(row.get("envelope") or {})
        return cls(
            model=str(row.get("model") or row.get("id") or ""),
            artifact_digest=str(row.get("artifact_digest", "")),
            arena=str(row.get("arena", "")),
            task=str(row.get("task", "")),
            hardware_class=str(row.get("hardware_class", "")),
            vram_gb=float(envelope.get("vram_gb") or 0.0),
            ttft_ms=float(envelope.get("ttft_ms") or row.get("ttft_ms") or 0.0),
            tpot_ms=float(envelope.get("tpot_ms") or row.get("tpot_ms") or 0.0),
            servable=bool(row.get("servable", True)),
            wanted=float(row.get("wanted") or 0.0),
            cold=bool(row.get("cold", False)),
            operators_online=int(row.get("operators_online") or 0),
            requests=int(row.get("requests") or 0),
        )


@dataclass(frozen=True, slots=True)
class Capacity:
    gpus: tuple[str, ...]
    gpu_gb: float
    disk_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.gpus:
            raise PlanError("a planner needs at least one GPU to hold a system on")
        if self.gpu_gb <= 0:
            raise PlanError("a planner needs the memory of a GPU in GB")

    @property
    def ceiling(self) -> int:
        return len(self.gpus)


@dataclass(frozen=True, slots=True)
class Policy:
    arenas: frozenset[str] = frozenset()
    exclude: frozenset[str] = frozenset()

    def allows(self, offer: Offer) -> bool:
        if offer.model in self.exclude:
            return False
        return not (self.arenas and offer.arena not in self.arenas)


@dataclass(frozen=True, slots=True)
class Holding:
    model: str
    artifact_digest: str
    gpu: str
    concurrency: int = 1


@dataclass(frozen=True, slots=True)
class Plan:
    hold: tuple[Holding, ...] = ()
    fetch: tuple[Offer, ...] = ()
    drop: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
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
                    "gpu": h.gpu,
                    "concurrency": h.concurrency,
                }
                for h in self.hold
            ],
            "fetch": [o.model for o in self.fetch],
            "drop": list(self.drop),
            "skipped": list(self.skipped),
            "reasons": dict(self.reasons),
        }


def rank(offers: Iterable[Offer], held: Mapping[str, str]) -> list[Offer]:
    def key(offer: Offer) -> tuple[float, int, float, str]:
        score = offer.wanted * (KEEP_MARGIN if offer.model in held else 1.0)
        return (-score, 0 if offer.cold else 1, offer.vram_gb, offer.model)

    return sorted(offers, key=key)


def build(
    offers: Sequence[Mapping[str, Any]] | Sequence[Offer],
    capacity: Capacity,
    *,
    held: Mapping[str, str] | None = None,
    policy: Policy | None = None,
    pinned: Mapping[str, str] | None = None,
) -> Plan:
    rules = policy or Policy()
    holding = dict(held or {})
    rows = [o if isinstance(o, Offer) else Offer.from_wire(o) for o in offers]
    reasons: dict[str, str] = {}
    usable: list[Offer] = []
    for offer in rows:
        if not offer.model or not offer.artifact_digest:
            continue
        if pinned is not None and pinned.get(offer.model) != offer.artifact_digest:
            if offer.model in pinned:
                reasons[offer.model] = "the catalogue moved past the pin you asked for"
            continue
        if not offer.servable:
            reasons[offer.model] = "the network publishes no serving envelope for it"
            continue
        if not rules.allows(offer):
            reasons[offer.model] = "your policy excludes it"
            continue
        if offer.vram_gb > capacity.gpu_gb:
            reasons[offer.model] = (
                f"it needs {offer.vram_gb:.0f} GB of GPU memory, yours have {capacity.gpu_gb:.0f}"
            )
            continue
        usable.append(offer)

    chosen: list[Offer] = []
    for offer in rank(usable, holding):
        if len(chosen) >= capacity.ceiling:
            reasons.setdefault(offer.model, "every GPU is already holding a system")
            continue
        chosen.append(offer)

    hold = tuple(
        Holding(model=offer.model, artifact_digest=offer.artifact_digest, gpu=gpu)
        for offer, gpu in zip(chosen, capacity.gpus, strict=False)
    )
    wanted_now = {offer.model: offer.artifact_digest for offer in chosen}
    fetch = tuple(offer for offer in chosen if holding.get(offer.model) != offer.artifact_digest)
    drop = tuple(sorted(model for model in holding if model not in wanted_now))
    return Plan(hold=hold, fetch=fetch, drop=drop, skipped=tuple(sorted(reasons)), reasons=reasons)


def changed(plan: Plan, held: Mapping[str, str]) -> bool:
    return {h.model: h.artifact_digest for h in plan.hold} != dict(held)
