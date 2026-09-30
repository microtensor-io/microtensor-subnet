from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from microtensor.core.protocol import ArtifactFormat, LoadManifest
from microtensor.core.tracks import ANSWER_MODES, DECIDE, GENERATE, Decoding
from microtensor.harness import progress


class EngineError(RuntimeError):
    pass


class EngineLoadError(EngineError):
    pass


class EngineTimeout(EngineError):
    pass


class UnsupportedArtifact(EngineError):
    pass


@dataclass(frozen=True, slots=True)
class Request:
    task_ref: str
    prompt: str
    inputs: dict[str, Any] = field(default_factory=dict)
    max_output_tokens: int = 512
    decoding: Decoding = Decoding.GREEDY
    seed: int = 0
    nonce: str = ""
    chat: bool = False
    mode: str = GENERATE

    def __post_init__(self) -> None:
        if not self.task_ref:
            raise ValueError("every request must carry a task reference")
        if self.mode not in ANSWER_MODES:
            raise ValueError(f"request mode {self.mode!r} is not one of {sorted(ANSWER_MODES)}")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.decoding is Decoding.SEEDED and self.seed == 0:
            raise ValueError("seeded decoding requires a non-zero seed")


@dataclass(frozen=True, slots=True)
class Response:
    task_ref: str
    output: Any = None
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    output_tokens: int = 0
    peak_rss_bytes: int = 0
    error: str = ""
    logprobs: tuple[float, ...] = ()
    entropies: tuple[float, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def tokens_per_second(self) -> float:
        elapsed = self.total_ms - self.ttft_ms
        if elapsed <= 0.0 or self.output_tokens <= 1:
            return 0.0
        return (self.output_tokens - 1) / (elapsed / 1000.0)

    @classmethod
    def failed(cls, task_ref: str, error: str, total_ms: float = 0.0) -> Response:
        return cls(
            task_ref=task_ref,
            error=error or "unspecified engine failure",
            total_ms=total_ms,
        )


@runtime_checkable
class Engine(Protocol):
    format: ArtifactFormat

    def load(self, artifact: Path, manifest: LoadManifest) -> None: ...

    def generate(self, request: Request) -> Response: ...

    def unload(self) -> None: ...


@dataclass(frozen=True, slots=True)
class EngineInfo:
    format: ArtifactFormat
    name: str
    # This adapter's own version, not the library's. The two were printed as
    # one string, so `mt inspect engines` read "llama-cpp 0.2.0" while the
    # pinned dependency was 0.3.35 — and the library version is exactly what
    # decides which architectures load at all.
    version: str
    deterministic: bool
    notes: str = ""
    # The installed library, resolved when asked rather than written down.
    runtime: str = ""


def supports_decide(engine: Any) -> bool:
    return callable(getattr(engine, "decide", None))


def answer(engine: Any, request: Request) -> Response:
    if request.mode != DECIDE:
        response: Response = engine.generate(request)
        return response
    if not supports_decide(engine):
        return Response.failed(
            request.task_ref,
            f"the {getattr(engine, 'format', 'unknown')} engine cannot answer a decision task",
        )
    decided: Response = engine.decide(request)
    return decided


def batch(requests: Sequence[Request], engine: Engine) -> list[Response]:
    progress.reset()
    responses: list[Response] = []
    for request in requests:
        try:
            response = answer(engine, request)
        except EngineError as exc:
            response = Response.failed(request.task_ref, str(exc))
        responses.append(response)
        progress.record(response)
    return responses
