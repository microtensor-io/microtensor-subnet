from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from microtensor.core.escalation import EscalationModel
from microtensor.core.protocol import Role
from microtensor.core.system import SystemManifest
from microtensor.core.trace import Trace
from microtensor.harness.contract import Response
from microtensor.harness.engines.router import Decision, Router, decide, features_from, load_router
from microtensor.harness.sdk import EscalationCall, Runtime, confidence_of, engine_small

MARGIN_TOLERANCE: Final[float] = 0.5
CONFIDENCE_TOLERANCE: Final[float] = 0.02
FEATURE_TOLERANCE: Final[float] = 0.05
SAMPLE: Final[int] = 8


class Replayer(Protocol):
    def replay(
        self, prompt: str, tokens: Sequence[int], chat: bool
    ) -> tuple[list[float], list[float], list[float]]: ...

    def prompt_tokens(self, prompt: str, chat: bool) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class Verdict:
    certified: bool
    checked: int
    reasons: tuple[str, ...]


def sample(traces: Sequence[Trace], seed: str, size: int = SAMPLE) -> list[Trace]:
    return sorted(
        traces, key=lambda t: hashlib.sha256(f"{seed}:{t.task_ref}".encode()).hexdigest()
    )[:size]


def check_small(engine: Replayer, trace: Trace, chat: bool) -> tuple[str, list[float], list[float]]:
    margins, logprobs, entropies = engine.replay(trace.small.prompt, trace.small.tokens, chat)
    worst = max(margins, default=0.0)
    if worst > MARGIN_TOLERANCE:
        return (
            f"{trace.task_ref}: the small model's tokens are not what the archived model "
            f"produces (worst margin {worst:.2f} logits)",
            logprobs,
            entropies,
        )
    if len(engine.prompt_tokens(trace.small.prompt, chat)) != trace.small.prompt_tokens:
        return (
            f"{trace.task_ref}: the small model's prompt length is misreported",
            logprobs,
            entropies,
        )
    if abs(confidence_of(logprobs) - trace.small.confidence) > CONFIDENCE_TOLERANCE:
        return (
            f"{trace.task_ref}: the reported confidence is not the small model's own",
            logprobs,
            entropies,
        )
    return "", logprobs, entropies


def check_router(
    router: Router,
    declared: Sequence[str],
    trace: Trace,
    logprobs: Sequence[float],
    entropies: Sequence[float],
) -> str:
    response = Response(
        task_ref=trace.task_ref,
        output=trace.small.output,
        output_tokens=len(trace.small.tokens),
        logprobs=tuple(logprobs),
        entropies=tuple(entropies),
    )
    found = features_from(response, prompt_tokens=trace.small.prompt_tokens)
    features = {name: float(found.get(name, 0.0)) for name in declared}
    for name, value in features.items():
        claimed = trace.router.features.get(name)
        if claimed is None or not math.isclose(value, claimed, abs_tol=FEATURE_TOLERANCE):
            return f"{trace.task_ref}: router feature {name} was not computed from the small model"
    if (decide(router, features) is Decision.ESCALATE) != trace.router.escalate:
        return f"{trace.task_ref}: the router decision does not follow from its declared rule"
    return ""


def check_escalation(
    system: SystemManifest, trace: Trace, allowlist: Mapping[str, EscalationModel]
) -> str:
    if trace.escalation is None:
        return ""
    key = f"{trace.escalation.model}@{trace.escalation.revision}"
    if system.escalation is None or key != system.escalation.key:
        return f"{trace.task_ref}: escalated to a model the system did not declare"
    if key not in allowlist:
        return f"{trace.task_ref}: escalated to a model that is not on the allowlist"
    return ""


def replay_archive(
    artifact: Path,
    system: SystemManifest,
    engine: Any,
    trace: Trace,
    task_prompt: str,
    inputs: Mapping[str, Any],
    chat: bool,
) -> str:
    if system.harness is None or system.escalation is None:
        return "the archived system is incomplete"
    escalated = trace.escalation

    def escalate(_: str) -> EscalationCall:
        if escalated is None:
            raise RuntimeError("the archived run escalated where the live run did not")
        return EscalationCall(
            output=str(escalated.output),
            prompt_tokens=escalated.prompt_tokens,
            completion_tokens=escalated.completion_tokens,
            ms=0.0,
        )

    runtime = Runtime(
        artifact / system.harness.path,
        load_router(artifact / system.locate(Role.ROUTER), system.router_features),
        system.router_features,
        engine_small(engine, chat=chat, max_output_tokens=max(1, len(trace.small.tokens) + 1)),
        escalate,
        escalation=system.escalation,
        system_digest=trace.system_digest,
        hotkey=trace.hotkey,
    )
    try:
        again = runtime.run(trace.round_index, trace.task_ref, task_prompt, inputs)
    except Exception as exc:
        return f"{trace.task_ref}: the archived system could not run: {exc}"
    if again.small.prompt != trace.small.prompt:
        return f"{trace.task_ref}: the archived harness renders a different prompt"
    if tuple(again.small.tokens) != tuple(trace.small.tokens):
        return f"{trace.task_ref}: the archived small model answers differently"
    if again.router.escalate != trace.router.escalate:
        return f"{trace.task_ref}: the archived router decides differently"
    if again.final != trace.final:
        return f"{trace.task_ref}: the archived system returns a different final answer"
    return ""


def verify_system(
    artifact: Path,
    system: SystemManifest,
    engine: Any,
    traces: Sequence[Trace],
    tasks: Mapping[str, tuple[str, Mapping[str, Any]]],
    *,
    seed: str,
    allowlist: Mapping[str, EscalationModel],
    chat: bool,
    size: int = SAMPLE,
) -> Verdict:
    if system.router is None:
        return Verdict(certified=False, checked=0, reasons=("the system declares no router",))
    router = load_router(artifact / system.locate(Role.ROUTER), system.router_features)
    reasons: list[str] = []
    chosen = sample(traces, seed, size)
    for trace in chosen:
        reason, logprobs, entropies = check_small(engine, trace, chat)
        reason = reason or check_router(router, system.router_features, trace, logprobs, entropies)
        reason = reason or check_escalation(system, trace, allowlist)
        if not reason and trace.task_ref in tasks:
            prompt, inputs = tasks[trace.task_ref]
            reason = replay_archive(artifact, system, engine, trace, prompt, inputs, chat)
        if reason:
            reasons.append(reason)
    return Verdict(certified=not reasons, checked=len(chosen), reasons=tuple(reasons))


def jailed_verify(
    artifact: str,
    system: dict[str, Any],
    weights: str,
    load: dict[str, Any],
    traces: list[dict[str, Any]],
    tasks: dict[str, tuple[str, dict[str, Any]]],
    seed: str,
    allowlist: list[dict[str, Any]],
    chat: bool,
    size: int = SAMPLE,
) -> dict[str, Any]:
    from microtensor.core.escalation import allowlist as escalation_allowlist
    from microtensor.harness.engines.gguf import GgufEngine
    from microtensor.harness.execute import rebuild_manifest

    engine = GgufEngine()
    engine.load(Path(weights), rebuild_manifest(load))
    try:
        verdict = verify_system(
            Path(artifact),
            SystemManifest.from_dict(system),
            engine,
            [Trace.from_dict(raw) for raw in traces],
            tasks,
            seed=seed,
            allowlist=escalation_allowlist(allowlist),
            chat=chat,
            size=size,
        )
    finally:
        engine.unload()
    return {
        "certified": verdict.certified,
        "checked": verdict.checked,
        "reasons": list(verdict.reasons),
    }
