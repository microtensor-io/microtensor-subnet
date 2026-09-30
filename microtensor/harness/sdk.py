from __future__ import annotations

import importlib.util
import json
import math
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from microtensor.core.system import EscalationRef
from microtensor.core.trace import (
    EscalationAnswer,
    HarnessStep,
    RouterDecision,
    SmallAnswer,
    Trace,
)
from microtensor.harness.contract import Request, Response
from microtensor.harness.engines.router import Decision, Router, decide, features_from
from microtensor.harness.package import load

SDK_VERSION: Final[str] = "1.0.0"
INPUT_SLOT: Final[str] = "{input}"


class HarnessRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SmallCall:
    output: str
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    entropies: tuple[float, ...]
    prompt_tokens: int
    ms: float


@dataclass(frozen=True, slots=True)
class EscalationCall:
    output: str
    prompt_tokens: int
    completion_tokens: int
    ms: float


SmallModel = Callable[[str], SmallCall]
EscalationModel = Callable[[str], EscalationCall]


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"mt_harness_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise HarnessRuntimeError(f"harness file {path.name} cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise HarnessRuntimeError(f"harness file {path.name} defines no run function")
    return module


def confidence_of(logprobs: Sequence[float]) -> float:
    return math.exp(math.fsum(logprobs) / len(logprobs)) if logprobs else 0.0


class Runtime:
    def __init__(
        self,
        package: Path,
        router: Router,
        features: Sequence[str],
        small: SmallModel,
        escalate: EscalationModel,
        *,
        escalation: EscalationRef,
        system_digest: str,
        hotkey: str,
    ) -> None:
        spec = load(package)
        self.prompts = {
            str(name): (package / str(rel)).read_text(encoding="utf-8")
            for name, rel in dict(spec["prompts"]).items()
        }
        self.tools = {
            str(entry["name"]): _module(package / str(entry["path"]))
            for entry in spec.get("tools") or []
        }
        self.hooks = {
            str(event): _module(package / str(rel))
            for event, rel in dict(spec.get("hooks") or {}).items()
        }
        self.router = router
        self.features = tuple(features)
        self.small = small
        self.escalate = escalate
        self.escalation = escalation
        self.system_digest = system_digest
        self.hotkey = hotkey

    def render(self, prompt: str) -> str:
        template = self.prompts.get("task", INPUT_SLOT)
        body = template.replace(INPUT_SLOT, prompt) if INPUT_SLOT in template else template + prompt
        system = self.prompts.get("system", "")
        return f"{system}\n\n{body}" if system else body

    def _hook(
        self, event: str, payload: dict[str, Any], steps: list[HarnessStep], started: float
    ) -> dict[str, Any]:
        module = self.hooks.get(event)
        if module is None:
            return payload

        def tool(name: str, argument: Any) -> Any:
            if name not in self.tools:
                raise HarnessRuntimeError(f"the harness calls undeclared tool {name!r}")
            at = (time.perf_counter() - started) * 1000.0
            try:
                result = self.tools[name].run(argument)
            except Exception as exc:
                steps.append(_step("error", name, at, started, f"{type(exc).__name__}: {exc}"))
                raise
            steps.append(_step("tool", name, at, started))
            return result

        at = (time.perf_counter() - started) * 1000.0
        try:
            found = module.run(dict(payload), tool)
        except Exception as exc:
            steps.append(_step("error", event, at, started, f"{type(exc).__name__}: {exc}"))
            return payload
        steps.append(_step("hook", event, at, started))
        return dict(found) if isinstance(found, Mapping) else payload

    def run(self, round_index: int, task_ref: str, prompt: str, inputs: Mapping[str, Any]) -> Trace:
        started = time.perf_counter()
        steps: list[HarnessStep] = []
        payload = self._hook("before", {"prompt": prompt, "inputs": dict(inputs)}, steps, started)
        rendered = self.render(str(payload.get("prompt", prompt)))

        small = self.small(rendered)
        response = Response(
            task_ref=task_ref,
            output=small.output,
            output_tokens=len(small.tokens),
            logprobs=small.logprobs,
            entropies=small.entropies,
        )
        found = features_from(response, prompt_tokens=small.prompt_tokens)
        features = {name: float(found.get(name, 0.0)) for name in self.features}
        decided_at = (time.perf_counter() - started) * 1000.0
        escalate = decide(self.router, features) is Decision.ESCALATE

        final: Any = small.output
        escalation: EscalationAnswer | None = None
        if escalate:
            answer = self.escalate(rendered)
            escalation = EscalationAnswer(
                model=self.escalation.model,
                revision=self.escalation.revision,
                output=answer.output,
                prompt_tokens=answer.prompt_tokens,
                completion_tokens=answer.completion_tokens,
                ms=answer.ms,
            )
            final = answer.output

        final = self._hook("after", {"output": final}, steps, started).get("output", final)
        return Trace(
            round_index=round_index,
            task_ref=task_ref,
            system_digest=self.system_digest,
            hotkey=self.hotkey,
            small=SmallAnswer(
                output=small.output,
                confidence=confidence_of(small.logprobs),
                tokens=small.tokens,
                prompt_tokens=small.prompt_tokens,
                ms=small.ms,
            ),
            router=RouterDecision(features=features, escalate=escalate, at_ms=decided_at),
            final=final,
            total_ms=(time.perf_counter() - started) * 1000.0,
            steps=tuple(steps),
            escalation=escalation,
        )


def _step(kind: str, name: str, at: float, started: float, detail: str = "") -> HarnessStep:
    return HarnessStep(
        kind=kind,
        name=name,
        at_ms=at,
        ms=max(0.0, (time.perf_counter() - started) * 1000.0 - at),
        detail=detail,
    )


def engine_small(engine: Any, *, chat: bool, max_output_tokens: int = 512) -> SmallModel:
    def call(prompt: str) -> SmallCall:
        started = time.perf_counter()
        response = engine.generate(
            Request(task_ref="small", prompt=prompt, max_output_tokens=max_output_tokens, chat=chat)
        )
        if not response.ok:
            raise HarnessRuntimeError(f"the small model failed: {response.error}")
        text = str(response.output)
        return SmallCall(
            output=text,
            tokens=tuple(engine.tokenize(text)),
            logprobs=tuple(response.logprobs),
            entropies=tuple(response.entropies),
            prompt_tokens=len(engine.tokenize(prompt)),
            ms=(time.perf_counter() - started) * 1000.0,
        )

    return call


def openai_escalation(url: str, model: str, *, max_tokens: int = 1024) -> EscalationModel:
    endpoint = url.rstrip("/") + "/v1/chat/completions"
    if not endpoint.startswith(("http://", "https://")):
        raise HarnessRuntimeError(f"the escalation endpoint {url!r} is not http or https")

    def call(prompt: str) -> EscalationCall:
        started = time.perf_counter()
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
            }
        ).encode()
        request = urllib.request.Request(  # noqa: S310
            endpoint, data=body, headers={"content-type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=600) as answer:  # noqa: S310
            data = json.loads(answer.read().decode("utf-8"))
        usage = data.get("usage") or {}
        return EscalationCall(
            output=str(data["choices"][0]["message"]["content"]),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            ms=(time.perf_counter() - started) * 1000.0,
        )

    return call
