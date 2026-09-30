from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from microtensor.core.hashing import canonical_hash, canonical_json

TRACE_VERSION: Final[str] = "t1"
STEP_KINDS: Final[frozenset[str]] = frozenset({"tool", "hook", "error", "retry"})

Verifier = Callable[[str, bytes, str], bool]


class TraceError(ValueError):
    pass


def _fields(
    raw: Any, what: str, required: frozenset[str], optional: frozenset[str]
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TraceError(f"{what} is not an object")
    keys = set(raw)
    missing = sorted(required - keys)
    if missing:
        raise TraceError(f"{what} is missing {', '.join(missing)}")
    unknown = sorted(keys - required - optional)
    if unknown:
        raise TraceError(f"{what} carries fields the format does not define: {', '.join(unknown)}")
    return dict(raw)


def _ms(value: Any, what: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise TraceError(f"{what} must be a finite, non negative number of milliseconds")
    return number


def _count(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TraceError(f"{what} must be a non negative whole number")
    return value


@dataclass(frozen=True, slots=True)
class SmallAnswer:
    output: Any
    confidence: float
    tokens: tuple[int, ...]
    prompt_tokens: int
    ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "confidence": self.confidence,
            "tokens": list(self.tokens),
            "prompt_tokens": self.prompt_tokens,
            "ms": self.ms,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> SmallAnswer:
        body = _fields(
            raw,
            "the small model answer",
            frozenset({"output", "confidence", "tokens", "prompt_tokens", "ms"}),
            frozenset(),
        )
        confidence = float(body["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise TraceError("the small model confidence must lie in [0, 1]")
        tokens = body["tokens"]
        if not isinstance(tokens, list) or not tokens:
            raise TraceError("the small model answer lists no tokens")
        return cls(
            output=body["output"],
            confidence=confidence,
            tokens=tuple(_count(t, "a small model token") for t in tokens),
            prompt_tokens=_count(body["prompt_tokens"], "the small model prompt tokens"),
            ms=_ms(body["ms"], "the small model time"),
        )


@dataclass(frozen=True, slots=True)
class RouterDecision:
    features: dict[str, float]
    escalate: bool
    at_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {"features": dict(self.features), "escalate": self.escalate, "at_ms": self.at_ms}

    @classmethod
    def from_dict(cls, raw: Any) -> RouterDecision:
        body = _fields(
            raw, "the router decision", frozenset({"features", "escalate", "at_ms"}), frozenset()
        )
        features = body["features"]
        if not isinstance(features, Mapping) or not features:
            raise TraceError("the router decision carries no features")
        values: dict[str, float] = {}
        for name, value in features.items():
            number = float(value)
            if not math.isfinite(number):
                raise TraceError(f"router feature {name!r} is not a finite number")
            values[str(name)] = number
        if not isinstance(body["escalate"], bool):
            raise TraceError("the router decision must say escalate true or false")
        return cls(
            features=values,
            escalate=body["escalate"],
            at_ms=_ms(body["at_ms"], "the router decision time"),
        )


@dataclass(frozen=True, slots=True)
class HarnessStep:
    kind: str
    name: str
    at_ms: float
    ms: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "at_ms": self.at_ms,
            "ms": self.ms,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> HarnessStep:
        body = _fields(
            raw,
            "a harness step",
            frozenset({"kind", "name", "at_ms", "ms"}),
            frozenset({"detail"}),
        )
        if body["kind"] not in STEP_KINDS:
            raise TraceError(f"harness step kind must be one of {sorted(STEP_KINDS)}")
        return cls(
            kind=str(body["kind"]),
            name=str(body["name"]),
            at_ms=_ms(body["at_ms"], "a harness step start"),
            ms=_ms(body["ms"], "a harness step time"),
            detail=str(body.get("detail", "")),
        )


@dataclass(frozen=True, slots=True)
class EscalationAnswer:
    model: str
    revision: str
    output: Any
    prompt_tokens: int
    completion_tokens: int
    ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "revision": self.revision,
            "output": self.output,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ms": self.ms,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> EscalationAnswer:
        body = _fields(
            raw,
            "the escalation answer",
            frozenset({"model", "revision", "output", "prompt_tokens", "completion_tokens", "ms"}),
            frozenset(),
        )
        return cls(
            model=str(body["model"]),
            revision=str(body["revision"]),
            output=body["output"],
            prompt_tokens=_count(body["prompt_tokens"], "the escalation prompt tokens"),
            completion_tokens=_count(body["completion_tokens"], "the escalation completion tokens"),
            ms=_ms(body["ms"], "the escalation time"),
        )


@dataclass(frozen=True, slots=True)
class Trace:
    round_index: int
    task_ref: str
    system_digest: str
    hotkey: str
    small: SmallAnswer
    router: RouterDecision
    final: Any
    total_ms: float
    steps: tuple[HarnessStep, ...] = ()
    escalation: EscalationAnswer | None = None
    version: str = TRACE_VERSION
    signature: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.version != TRACE_VERSION:
            raise TraceError(f"trace version {self.version!r} is not {TRACE_VERSION!r}")
        if not self.task_ref or not self.system_digest or not self.hotkey:
            raise TraceError("a trace names its task, its system and the miner hotkey")
        if self.router.escalate != (self.escalation is not None):
            raise TraceError(
                "the router decision and the escalation disagree: an escalated request "
                "carries the escalation answer, and only an escalated one does"
            )
        if self.router.at_ms > self.total_ms:
            raise TraceError("the router decided after the request finished")
        late = [s.name for s in self.steps if s.at_ms > self.total_ms]
        if late:
            raise TraceError(f"harness steps start after the request finished: {late}")

    @property
    def escalated(self) -> bool:
        return self.escalation is not None

    def body(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "round_index": self.round_index,
            "task_ref": self.task_ref,
            "system_digest": self.system_digest,
            "hotkey": self.hotkey,
            "small": self.small.to_dict(),
            "router": self.router.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "escalation": self.escalation.to_dict() if self.escalation else None,
            "final": self.final,
            "total_ms": self.total_ms,
        }

    def signed_message(self) -> bytes:
        return canonical_json(self.body())

    def digest(self) -> str:
        return canonical_hash(self.body())

    def signed_with(self, signature: str) -> Trace:
        return Trace(
            round_index=self.round_index,
            task_ref=self.task_ref,
            system_digest=self.system_digest,
            hotkey=self.hotkey,
            small=self.small,
            router=self.router,
            final=self.final,
            total_ms=self.total_ms,
            steps=self.steps,
            escalation=self.escalation,
            version=self.version,
            signature=signature,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "signature": self.signature}

    @classmethod
    def from_dict(cls, raw: Any) -> Trace:
        body = _fields(
            raw,
            "the trace",
            frozenset(
                {
                    "version",
                    "round_index",
                    "task_ref",
                    "system_digest",
                    "hotkey",
                    "small",
                    "router",
                    "steps",
                    "escalation",
                    "final",
                    "total_ms",
                    "signature",
                }
            ),
            frozenset(),
        )
        steps = body["steps"]
        if not isinstance(steps, list):
            raise TraceError("the trace steps must be a list")
        try:
            return cls(
                round_index=_count(body["round_index"], "the round index"),
                task_ref=str(body["task_ref"]),
                system_digest=str(body["system_digest"]),
                hotkey=str(body["hotkey"]),
                small=SmallAnswer.from_dict(body["small"]),
                router=RouterDecision.from_dict(body["router"]),
                final=body["final"],
                total_ms=_ms(body["total_ms"], "the request time"),
                steps=tuple(HarnessStep.from_dict(step) for step in steps),
                escalation=EscalationAnswer.from_dict(body["escalation"])
                if body["escalation"] is not None
                else None,
                version=str(body["version"]),
                signature=str(body["signature"]),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, TraceError):
                raise
            raise TraceError(f"the trace is malformed: {exc}") from exc


def read(raw: Any, verify: Verifier) -> Trace:
    trace = Trace.from_dict(raw)
    if not trace.signature:
        raise TraceError("the trace is unsigned")
    if not verify(trace.hotkey, trace.signed_message(), trace.signature):
        raise TraceError("the trace signature does not verify against the miner hotkey")
    return trace
