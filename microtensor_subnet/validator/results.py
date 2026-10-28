from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from microtensor_subnet.core.reports import PoolScore


@dataclass(slots=True)
class ItemRun:
    item_id: str
    family: str
    output: Any
    success: bool
    metric: float
    reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    cost_micros: int = 0
    escalated: bool = False
    probability: float | None = None
    answered: bool = True


@dataclass(slots=True)
class SlmRun:
    runs: list[ItemRun]
    twins: list[ItemRun]
    ttft_p95_ms: float = 0.0
    tpot_p95_ms: float = 0.0
    vram_peak_bytes: int = 0
    parameters: int = 0

    @property
    def pass_rate(self) -> float:
        return sum(1 for r in self.runs if r.success) / len(self.runs) if self.runs else 0.0

    @property
    def twin_rate(self) -> float:
        return sum(1 for r in self.twins if r.success) / len(self.twins) if self.twins else 0.0

    def drafts(self) -> dict[str, tuple[str, bool]]:
        return {r.item_id: (str(r.output or ""), r.success) for r in self.runs}


@dataclass(slots=True)
class Condition:
    name: str
    runs: list[ItemRun]
    tokens: int
    added_ms: list[float] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return sum(1 for r in self.runs if r.success) / len(self.runs) if self.runs else 0.0


@dataclass(slots=True)
class HarnessRun:
    miner: Condition
    baselines: list[Condition]
    limits_ok: bool = True
    limits_reason: str = ""

    def best_baseline(self) -> Condition:
        return max(self.baselines, key=lambda c: (c.pass_rate, -c.tokens))


@dataclass(slots=True)
class SourceRun:
    source: str
    probabilities: list[float]
    labels: list[int]
    latencies_ms: list[float] = field(default_factory=list)


@dataclass(slots=True)
class RouterRun:
    sources: list[SourceRun]
    latency_p95_ms: float = 0.0
    loaded: bool = True
    reason: str = ""


@dataclass(slots=True)
class EscalationRun:
    repairs: list[ItemRun]
    redo_pass_rate: float
    redo_cost_micros: int

    @property
    def repair_cost_micros(self) -> int:
        return sum(r.cost_micros for r in self.repairs)


@dataclass(slots=True)
class FloorRun:
    floor: float
    runs: list[ItemRun]

    @property
    def pass_rate(self) -> float:
        return sum(1 for r in self.runs if r.success) / len(self.runs) if self.runs else 0.0

    @property
    def cost_micros(self) -> int:
        return sum(r.cost_micros for r in self.runs)

    @property
    def latencies(self) -> list[float]:
        return [r.latency_ms for r in self.runs]


@dataclass(slots=True)
class SystemRun:
    floors: list[FloorRun]
    reference_cost_micros: int
    reference_latency_p95_ms: float
    reference_items: int = 0


@dataclass(slots=True)
class ReferenceRun:
    system_answers: dict[str, ItemRun]
    redo_answers: dict[str, ItemRun]
    cost_micros: int
    latency_p95_ms: float

    @property
    def redo_pass_rate(self) -> float:
        rows = list(self.redo_answers.values())
        return sum(1 for r in rows if r.success) / len(rows) if rows else 0.0

    @property
    def redo_cost_micros(self) -> int:
        return sum(r.cost_micros for r in self.redo_answers.values())


@dataclass(slots=True)
class ArtifactOutcome:
    hotkey: str
    slm: SlmRun | None = None
    base_runs: dict[str, SlmRun] = field(default_factory=dict)
    harness: HarnessRun | None = None
    router: RouterRun | None = None
    escalation: EscalationRun | None = None
    system: SystemRun | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, PoolScore] = field(default_factory=dict)
    receipts: list[dict[str, Any]] = field(default_factory=list)
    failure: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
