from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureClass(str, Enum):
    SSH_TRANSPORT = "SSH_TRANSPORT"
    AGENT_CRASH = "AGENT_CRASH"
    CHALLENGE_REJECT = "CHALLENGE_REJECT"
    SPEC_MISMATCH = "SPEC_MISMATCH"
    UNAUTHORISED_WORK = "UNAUTHORISED_WORK"
    NESTED_CONTAINER = "NESTED_CONTAINER"
    DUPLICATE_UUID = "DUPLICATE_UUID"
    VERSION_STALE = "VERSION_STALE"


MEANINGS: dict[FailureClass, str] = {
    FailureClass.SSH_TRANSPORT: "could not reach the agent",
    FailureClass.AGENT_CRASH: "agent died during the check",
    FailureClass.CHALLENGE_REJECT: "wrong answer, or too slow",
    FailureClass.SPEC_MISMATCH: "hardware is not what was declared",
    FailureClass.UNAUTHORISED_WORK: "something on the card that is not ours",
    FailureClass.NESTED_CONTAINER: "agent is not on a real host",
    FailureClass.DUPLICATE_UUID: "this card is enrolled elsewhere",
    FailureClass.VERSION_STALE: "agent behind the minimum",
}


@dataclass(frozen=True)
class Failure:
    kind: FailureClass
    detail: str = ""
    seed: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "class": self.kind.value,
            "meaning": MEANINGS[self.kind],
            "detail": self.detail,
            "seed": self.seed,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class Verdict:
    check: str
    failure: Failure | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return self.failure is None

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "check": self.check,
            "passed": self.passed,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "evidence": dict(self.evidence),
        }
        if self.failure is not None:
            body["failure"] = self.failure.payload()
        return body


def worst(verdicts: list[Verdict]) -> Failure | None:
    order = list(FailureClass)
    failures = [v.failure for v in verdicts if v.failure is not None]
    if not failures:
        return None
    return min(failures, key=lambda f: order.index(f.kind))
