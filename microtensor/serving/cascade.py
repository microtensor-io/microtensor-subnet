from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("microtensor.serving.cascade")

FRONT = "front"
ROUTER = "router"
SPECIALIST = "specialist"
ROLES = (FRONT, ROUTER, SPECIALIST)

RESOLVE = "resolve"
ESCALATE = "escalate"


class CascadeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Component:
    role: str
    artifact_digest: str
    path: str = ""
    base_model: str = ""

    @classmethod
    def from_wire(cls, role: str, raw: Mapping[str, Any]) -> Component:
        return cls(
            role=role,
            artifact_digest=str(raw.get("artifact_digest", "")),
            path=str(raw.get("path", "")),
            base_model=str(raw.get("base_model", "")),
        )


@dataclass(frozen=True, slots=True)
class System:
    front: Component
    router: Component | None = None
    specialist: Component | None = None
    router_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.router is None) != (self.specialist is None):
            raise CascadeError("a router and a specialist are declared together or not at all")

    @property
    def single(self) -> bool:
        return self.router is None and self.specialist is None

    @property
    def components(self) -> tuple[Component, ...]:
        return tuple(c for c in (self.front, self.router, self.specialist) if c is not None)

    @property
    def engines(self) -> tuple[Component, ...]:
        return tuple(c for c in (self.front, self.specialist) if c is not None)

    @classmethod
    def alone(cls, artifact_digest: str) -> System:
        return cls(front=Component(role=FRONT, artifact_digest=artifact_digest))

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> System:
        raw = manifest.get("system")
        if not isinstance(raw, Mapping):
            return cls.alone(str(manifest.get("artifact_digest", "")))

        front = raw.get("front")
        if not isinstance(front, Mapping):
            raise CascadeError("a system names a front component")
        router = raw.get("router")
        specialist = raw.get("specialist")
        return cls(
            front=Component.from_wire(FRONT, front),
            router=Component.from_wire(ROUTER, router) if isinstance(router, Mapping) else None,
            specialist=(
                Component.from_wire(SPECIALIST, specialist)
                if isinstance(specialist, Mapping)
                else None
            ),
            router_features=tuple(str(f) for f in raw.get("router_features", ())),
        )


def _entropy(logits: Sequence[float]) -> float:
    if not logits:
        return 0.0
    top = max(logits)
    weights = [math.exp(v - top) for v in logits]
    total = sum(weights)
    if total <= 0:
        return 0.0
    found = 0.0
    for weight in weights:
        share = weight / total
        if share > 0:
            found -= share * math.log(share)
    return found


def features(
    *,
    logprobs: Sequence[float],
    entropies: Sequence[float],
    output_tokens: int,
    input_tokens: int,
    schema_valid: bool = True,
) -> dict[str, float]:
    total = float(sum(logprobs))
    return {
        "seq_logprob": total,
        "seq_logprob_norm": total / len(logprobs) if logprobs else 0.0,
        "output_tokens": float(output_tokens),
        "mean_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
        "max_entropy": max(entropies) if entropies else 0.0,
        "schema_valid": 1.0 if schema_valid else 0.0,
        "input_tokens": float(input_tokens),
    }


def features_of(answer: Mapping[str, Any], prompt_tokens: int) -> dict[str, float]:
    logprobs = [float(v) for v in answer.get("logprobs") or []]
    entropies = [float(v) for v in answer.get("entropies") or []]
    if not entropies:
        rows = answer.get("logit_rows") or []
        entropies = [_entropy([float(v) for v in row]) for row in rows]
    completion = answer.get("completion_tokens") or []
    return features(
        logprobs=logprobs,
        entropies=entropies,
        output_tokens=len(completion),
        input_tokens=prompt_tokens,
        schema_valid=bool(answer.get("schema_valid", True)),
    )


@dataclass(slots=True)
class Router:
    features: tuple[str, ...]
    decide: Any = None
    path: Path | None = None

    def choose(self, found: Mapping[str, float]) -> str:
        if self.decide is None:
            return RESOLVE
        missing = [name for name in self.features if name not in found]
        if missing:
            raise CascadeError(f"the router reads features nobody computed: {missing}")
        chosen = self.decide(found)
        value = getattr(chosen, "value", chosen)
        return ESCALATE if str(value) == ESCALATE else RESOLVE


def load_router(path: Path, declared: Sequence[str]) -> Router:
    from microtensor.harness.engines.router import load_router as load

    found = load(path, tuple(declared))
    return Router(features=tuple(declared), decide=found.decide, path=path)


@dataclass(slots=True)
class Leg:
    role: str
    text: str = ""
    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)
    finish_reason: str = "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "tokens": {
                "prompt": list(self.prompt_tokens),
                "completion": list(self.completion_tokens),
            },
        }


@dataclass(slots=True)
class Answer:
    legs: list[Leg] = field(default_factory=list)
    escalated: bool = False
    router_features: dict[str, float] = field(default_factory=dict)

    @property
    def served(self) -> Leg:
        if not self.legs:
            raise CascadeError("a cascade produced no leg at all")
        return self.legs[-1]

    def to_dict(self) -> dict[str, Any]:
        found = self.served
        return {
            "text": found.text,
            "prompt_tokens": list(found.prompt_tokens),
            "completion_tokens": list(found.completion_tokens),
            "finish_reason": found.finish_reason,
            "escalated": self.escalated,
            "answered_by": found.role,
            "legs": [leg.to_dict() for leg in self.legs],
            "router_features": dict(self.router_features),
        }


def manifest_of(artifact: Path) -> dict[str, Any]:
    found = artifact if artifact.is_dir() else artifact.parent
    manifest = found / "manifest.json"
    if not manifest.exists():
        return {}
    try:
        return dict(json.loads(manifest.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise CascadeError(f"{manifest} is not readable json: {exc}") from exc


def entrypoint_of(artifact: Path) -> Path:
    found = artifact if artifact.is_dir() else artifact.parent
    manifest = manifest_of(artifact)
    named = str((manifest.get("load") or {}).get("entrypoint", ""))
    if named:
        candidate = found / named
        if candidate.exists():
            return candidate
        raise CascadeError(f"{manifest.get('load')} names {named}, which is not in {found}")
    if artifact.is_file():
        return artifact
    for child in sorted(found.glob("*.gguf")):
        return child
    raise CascadeError(f"no entrypoint named and nothing servable under {found}")
