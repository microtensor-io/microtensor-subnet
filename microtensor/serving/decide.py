from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microtensor.core.tracks import DECIDE, DECISION_PROMPT_VERSION
from microtensor.harness import decision_prompt


class DecideError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Prepared:
    questions: tuple[decision_prompt.Question, ...]
    texts: tuple[str, ...]
    answer_ids: tuple[tuple[int, ...], ...]


def prepare(
    spec: Any,
    render: Callable[[list[dict[str, str]]], str],
    token_id: Callable[[str], int],
) -> Prepared:
    try:
        context, questions = decision_prompt.parse(spec)
    except decision_prompt.DecisionError as exc:
        raise DecideError(str(exc)) from exc
    texts = tuple(render(decision_prompt.messages(context, question)) for question in questions)
    ids = tuple(tuple(token_id(text) for text in question.answers) for question in questions)
    return Prepared(questions=questions, texts=texts, answer_ids=ids)


def assemble(prepared: Prepared, scores: Sequence[Mapping[int, float]]) -> dict[str, Any]:
    if len(scores) != len(prepared.questions):
        raise DecideError("the engine answered a different number of questions than it was asked")
    answers: dict[str, Any] = {}
    for question, wanted, found in zip(
        prepared.questions, prepared.answer_ids, scores, strict=True
    ):
        missing = [token for token in wanted if token not in found]
        if missing:
            raise DecideError(f"the engine returned no score for answer tokens {missing}")
        shares = decision_prompt.to_grid(
            decision_prompt.softmax([float(found[token]) for token in wanted])
        )
        answers[question.name] = decision_prompt.build_answer(question, shares)
    return {"answers": answers, "prompt_format": DECISION_PROMPT_VERSION}


def input_tokens(prepared: Prepared, tokenize: Callable[[str], Sequence[int]]) -> int:
    rows = [list(tokenize(text)) for text in prepared.texts]
    cut = decision_prompt.common_prefix(rows)
    return cut + sum(len(row) - cut for row in rows)


@dataclass(frozen=True, slots=True)
class Tools:
    render: Callable[[list[dict[str, str]]], str]
    token_id: Callable[[str], int]
    tokenize: Callable[[str], Sequence[int]]


def hf_tools(directory: Path) -> Tools:
    import json

    from tokenizers import Tokenizer

    from microtensor.harness import chat_template

    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    template = str(config.get("chat_template") or "")
    if not template:
        raise DecideError("this model ships no chat template, which a decision needs")

    def render(messages: list[dict[str, str]]) -> str:
        found = chat_template.render(template, messages)
        if found is None:
            raise DecideError("the chat template could not render a decision prompt")
        return found

    def tokenize(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False).ids)

    def token_id(text: str) -> int:
        ids = tokenize(text)
        if len(ids) != 1:
            raise DecideError(f"this model splits the answer {text!r} into {len(ids)} tokens")
        return ids[0]

    return Tools(render=render, token_id=token_id, tokenize=tokenize)


class SGLangDecider:
    def __init__(self, engine: Any, tools: Tools) -> None:
        self._engine = engine
        self._tools = tools

    async def __call__(self, spec: Any) -> tuple[dict[str, Any], int]:
        prepared = prepare(spec, self._tools.render, self._tools.token_id)
        scores = await self._engine.decide(prepared.texts, prepared.answer_ids)
        return assemble(prepared, scores), input_tokens(prepared, self._tools.tokenize)


class GgufDecider:
    def __init__(self, weights: Path, context: int = 4096) -> None:
        self._weights = weights
        self._context = context
        self._engine: Any = None
        self._lock = asyncio.Lock()

    def _load(self) -> Any:
        from microtensor.core.protocol import ArtifactFormat, LoadManifest
        from microtensor.harness.engines.gguf import GgufEngine

        engine = GgufEngine()
        engine.load(
            self._weights,
            LoadManifest(
                format=ArtifactFormat.GGUF,
                quantization="",
                entrypoint=self._weights.name,
                max_input={"tokens": self._context},
            ),
        )
        return engine

    def _decide(self, spec: Any) -> tuple[dict[str, Any], int]:
        from microtensor.harness.contract import Request

        if self._engine is None:
            self._engine = self._load()
        context = str((spec or {}).get("context", "")) if isinstance(spec, Mapping) else ""
        answered = self._engine.decide(
            Request(
                task_ref="serve",
                prompt=context or "decision",
                inputs={"decision": spec},
                chat=True,
                mode=DECIDE,
            )
        )
        if not answered.ok:
            raise DecideError(answered.error)
        parsed_context, questions = decision_prompt.parse(spec)
        rows = [self._engine._decision_tokens(parsed_context, q) for q in questions]
        cut = decision_prompt.common_prefix(rows)
        return dict(answered.output), cut + sum(len(row) - cut for row in rows)

    async def __call__(self, spec: Any) -> tuple[dict[str, Any], int]:
        async with self._lock:
            return await asyncio.to_thread(self._decide, spec)


def decider_for(artifact: Path, engine: Any) -> Any:
    if artifact.is_file() and artifact.suffix == ".gguf":
        return GgufDecider(artifact)
    if artifact.is_dir():
        weights = sorted(artifact.glob("*.gguf"))
        if weights and not (artifact / "tokenizer.json").is_file():
            return GgufDecider(weights[0])
        if (artifact / "tokenizer.json").is_file() and callable(getattr(engine, "decide", None)):
            return SGLangDecider(engine, hf_tools(artifact))
    return None
