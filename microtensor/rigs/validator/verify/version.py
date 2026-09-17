from __future__ import annotations

import re
from typing import Any

from microtensor.rigs.protocol.failures import Failure, FailureClass, Verdict
from microtensor.rigs.validator.session.client import AgentClient, AgentError

NUMBERS = re.compile(r"\d+")


def parse(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in NUMBERS.findall(text or "")[:3])


def stale(agent_version: str, minimum: str) -> bool:
    floor = parse(minimum)
    if not floor:
        return False
    return parse(agent_version) < floor


def check(agent_version: str, minimum: str, seed: str = "") -> Verdict:
    evidence: dict[str, Any] = {
        "agent_version": agent_version,
        "minimum": minimum,
        "summary": agent_version or "unknown",
    }
    if not agent_version:
        return Verdict(
            "version",
            failure=Failure(FailureClass.VERSION_STALE, "the agent reports no version", seed=seed),
            evidence=evidence,
        )
    if stale(agent_version, minimum):
        return Verdict(
            "version",
            failure=Failure(
                FailureClass.VERSION_STALE,
                f"agent {agent_version} is below the minimum {minimum}",
                seed=seed,
            ),
            evidence=evidence,
        )
    return Verdict("version", evidence=evidence)


async def fetch(
    agents: AgentClient, rig: dict[str, Any], minimum: str, seed: str = ""
) -> tuple[Verdict, dict[str, Any]]:
    try:
        answer = await agents.version(rig)
    except AgentError as exc:
        return Verdict(
            "version", failure=exc.failure(seed), evidence={"summary": exc.detail[:120]}
        ), {}
    verdict = check(str(answer.get("version", "") or ""), minimum, seed)
    return verdict, answer
