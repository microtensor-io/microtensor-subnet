from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from microtensor_subnet.serving import hardware
from microtensor_subnet.serving.engines import LLAMA, SGLANG, EngineError, backend_for

KEEP_MARGIN: Final[float] = 1.5
CERTIFIED: Final[str] = "certified"
GENERAL: Final[str] = "general"
SYSTEM: Final[str] = "system"
MODEL: Final[str] = "model"


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
    kind: str = CERTIFIED
    format: str = ""
    engine: str = ""
    context: int = 0
    concurrency: int = 1
    verifiable: bool = True
    family: str = ""
    licence: str = ""
    licence_url: str = ""

    @property
    def general(self) -> bool:
        return self.kind == GENERAL

    @property
    def serves_as(self) -> str:
        return MODEL if self.general else SYSTEM

    @classmethod
    def from_wire(cls, row: Mapping[str, Any]) -> Offer:
        envelope = dict(row.get("envelope") or {})
        kind = str(row.get("kind") or CERTIFIED)
        artifact_format = str(row.get("format") or "").strip().lower()
        raw = row.get("engine")
        engine = str(raw.get("backend") or "") if isinstance(raw, Mapping) else ""
        if not engine and artifact_format:
            try:
                engine = backend_for(artifact_format)
            except EngineError:
                engine = ""
        stated = row.get("verifiable")
        verifiable = (
            stated
            if isinstance(stated, bool)
            else (bool(row.get("verifiable_on")) if kind == GENERAL else True)
        )
        return cls(
            model=str(row.get("model") or row.get("id") or ""),
            artifact_digest=str(row.get("artifact_digest", "")),
            arena=str(row.get("arena", "")),
            task=str(row.get("task", "")),
            hardware_class=str(row.get("hardware_class", "")),
            vram_gb=float(envelope.get("vram_gb") or row.get("vram_gb") or 0.0),
            ttft_ms=float(envelope.get("ttft_ms") or row.get("ttft_ms") or 0.0),
            tpot_ms=float(envelope.get("tpot_ms") or row.get("tpot_ms") or 0.0),
            servable=bool(row.get("servable", True)),
            wanted=float(row.get("wanted") or 0.0),
            cold=bool(row.get("cold", False)),
            operators_online=int(row.get("operators_online") or 0),
            requests=int(row.get("requests") or 0),
            kind=kind,
            format=artifact_format,
            engine=engine,
            context=int(row.get("context") or 0),
            concurrency=max(1, int(row.get("concurrency") or 1)),
            verifiable=verifiable,
            family=str(row.get("family") or "").strip().lower(),
            licence=str(row.get("licence") or ""),
            licence_url=str(row.get("licence_url") or ""),
        )


@dataclass(frozen=True, slots=True)
class Capacity:
    gpus: tuple[str, ...]
    gpu_gb: float
    disk_bytes: int = 0
    hardware_class: str = ""
    images: frozenset[str] = frozenset({SGLANG, LLAMA})

    def __post_init__(self) -> None:
        if not self.gpus:
            raise PlanError("a planner needs at least one GPU to hold a system on")
        if self.gpu_gb <= 0:
            raise PlanError("a planner needs the memory of a GPU in GB")
        if not self.hardware_class:
            object.__setattr__(self, "hardware_class", hardware.class_of(self.gpu_gb))

    @property
    def ceiling(self) -> int:
        return len(self.gpus)


@dataclass(frozen=True, slots=True)
class Policy:
    arenas: frozenset[str] = frozenset()
    exclude: frozenset[str] = frozenset()
    kinds: frozenset[str] = frozenset()
    include: frozenset[str] = frozenset()
    families: frozenset[str] = frozenset()
    accept_licences: bool = False

    def allows(self, offer: Offer) -> bool:
        if offer.model in self.exclude:
            return False
        if self.include and offer.model not in self.include:
            return False
        if self.kinds and offer.serves_as not in self.kinds:
            return False
        if offer.general:
            return not (self.families and offer.family not in self.families)
        return not (self.arenas and offer.arena not in self.arenas)


@dataclass(frozen=True, slots=True)
class Holding:
    model: str
    artifact_digest: str
    gpu: str
    concurrency: int = 1
    kind: str = CERTIFIED


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
                    "kind": h.kind,
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


def unrunnable(offer: Offer, capacity: Capacity) -> str:
    if not offer.general:
        return ""
    if not offer.engine:
        return f"no engine serves its format {offer.format or '(unnamed)'}"
    if offer.engine not in capacity.images:
        return f"no {offer.engine} image is configured on this host"
    if not hardware.satisfies(capacity.hardware_class, offer.hardware_class):
        held = capacity.hardware_class or "below every serving class"
        return f"it needs a {offer.hardware_class} host, this one is {held}"
    return ""


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
        if offer.general and not rules.accept_licences:
            reasons[offer.model] = (
                f"its licence is not accepted: {offer.licence or 'unnamed'} "
                f"({offer.licence_url or 'no link'}); pass --accept-licences"
            )
            continue
        blocked = unrunnable(offer, capacity)
        if blocked:
            reasons[offer.model] = blocked
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
            reasons.setdefault(offer.model, "every GPU is already holding a system or a model")
            continue
        chosen.append(offer)

    hold = tuple(
        Holding(
            model=offer.model,
            artifact_digest=offer.artifact_digest,
            gpu=gpu,
            concurrency=offer.concurrency if offer.general else 1,
            kind=offer.kind,
        )
        for offer, gpu in zip(chosen, capacity.gpus, strict=False)
    )
    wanted_now = {offer.model: offer.artifact_digest for offer in chosen}
    fetch = tuple(offer for offer in chosen if holding.get(offer.model) != offer.artifact_digest)
    drop = tuple(sorted(model for model in holding if model not in wanted_now))
    return Plan(hold=hold, fetch=fetch, drop=drop, skipped=tuple(sorted(reasons)), reasons=reasons)


def changed(plan: Plan, held: Mapping[str, str]) -> bool:
    return {h.model: h.artifact_digest for h in plan.hold} != dict(held)
