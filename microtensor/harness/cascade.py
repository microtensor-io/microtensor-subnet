from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from microtensor.core.protocol import ArtifactFormat, LoadManifest
from microtensor.harness.contract import Request, Response
from microtensor.harness.engines.router import Decision, Router, decide, features_from, load_router
from microtensor.harness.execute import rebuild_manifest
from microtensor.harness.registry import engine_for, load_builtin


@dataclass(frozen=True, slots=True)
class Leg:
    task_ref: str
    response: Response
    escalated: bool
    front_response: Response
    features: dict[str, float] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return not self.escalated


@dataclass(frozen=True, slots=True)
class CascadeResult:
    legs: tuple[Leg, ...]

    @property
    def resolve_rate(self) -> float:
        if not self.legs:
            return 1.0
        return sum(1 for leg in self.legs if leg.resolved) / len(self.legs)

    @property
    def front_ms(self) -> float:
        if not self.legs:
            return 0.0
        return sum(leg.front_response.total_ms for leg in self.legs) / len(self.legs)

    @property
    def specialist_ms(self) -> float:
        escalated = [leg for leg in self.legs if leg.escalated]
        if not escalated:
            return 0.0
        return sum(leg.response.total_ms for leg in escalated) / len(escalated)

    @property
    def expected_ms(self) -> float:
        """c_f + (1 - rho) * c_g, from timings measured on this run."""
        return self.front_ms + (1.0 - self.resolve_rate) * self.specialist_ms

    def responses(self) -> list[Response]:
        return [leg.response for leg in self.legs]

    def front_responses(self) -> list[Response]:
        return [leg.front_response for leg in self.legs]


def _load(artifact: Path, payload: dict[str, Any]) -> tuple[Any, LoadManifest]:
    manifest = rebuild_manifest(payload)
    engine = engine_for(manifest.format)
    engine.load(artifact, manifest)
    return engine, manifest


def _prompt_tokens(request: Request) -> int:
    return len(request.prompt.split())


def run_cascade(
    front_path: str,
    front_manifest: dict[str, Any],
    requests: Sequence[Request],
    router_path: str = "",
    router_features: Sequence[str] = (),
    specialist_path: str = "",
    specialist_manifest: dict[str, Any] | None = None,
) -> list[Leg]:
    """Run every task through the front, route on validator-computed features,
    and answer the escalated remainder with the specialist.

    A system with no router and no specialist reduces to the single-artifact
    path exactly, which is what keeps the pre-system behaviour intact.
    """
    load_builtin()

    router: Router | None = None
    if router_path:
        router = load_router(Path(router_path), tuple(router_features))

    front, _ = _load(Path(front_path), front_manifest)
    legs: list[Leg] = []

    specialist = None
    try:
        for request in requests:
            try:
                response = front.generate(request)
            except Exception as exc:
                response = Response.failed(request.task_ref, str(exc))

            if router is None or not response.ok:
                legs.append(
                    Leg(
                        task_ref=request.task_ref,
                        response=response,
                        escalated=False,
                        front_response=response,
                    )
                )
                continue

            features = features_from(response, prompt_tokens=_prompt_tokens(request))
            if decide(router, features) is Decision.RESOLVE:
                legs.append(
                    Leg(
                        task_ref=request.task_ref,
                        response=response,
                        escalated=False,
                        front_response=response,
                        features=features,
                    )
                )
                continue

            if specialist is None:
                if not specialist_path or specialist_manifest is None:
                    raise ValueError("the router escalated but no specialist was supplied")
                specialist, _ = _load(Path(specialist_path), specialist_manifest)

            try:
                escalated = specialist.generate(request)
            except Exception as exc:
                escalated = Response.failed(request.task_ref, str(exc))

            legs.append(
                Leg(
                    task_ref=request.task_ref,
                    response=escalated,
                    escalated=True,
                    front_response=response,
                    features=features,
                )
            )
    finally:
        front.unload()
        if specialist is not None:
            specialist.unload()

    return legs


def single_format(payload: dict[str, Any]) -> ArtifactFormat:
    return ArtifactFormat(payload["format"])
