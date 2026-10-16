from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol

import httpx

from microtensor_subnet.serving import verify
from microtensor_subnet.serving.engines import LLAMA, SGLANG
from microtensor_subnet.serving.verify import VerificationError

THREADS: Final[int] = 1
SEED: Final[int] = 0
GPU_LAYERS: Final[int] = 0
DEFAULT_CONTEXT: Final[int] = 4096
LLAMA_CONTEXT: Final[int] = 8192
GENERATE_PATH: Final[str] = "/generate"
PREFILL_SECONDS: Final[float] = 600.0
CONNECT_SECONDS: Final[float] = 10.0


class Prefill(Protocol):
    async def margins(
        self, prompt_ids: Sequence[int], completion_ids: Sequence[int]
    ) -> list[float]: ...


def _llama() -> Any:
    import llama_cpp

    return llama_cpp


def open_artifact(
    path: Path,
    *,
    context: int = DEFAULT_CONTEXT,
    threads: int = THREADS,
    gpu_layers: int = GPU_LAYERS,
) -> Any:
    weights = Path(path)
    if not weights.is_file():
        raise VerificationError(f"no artifact at {weights}")
    try:
        llama_cpp = _llama()
    except ImportError as exc:
        raise VerificationError("llama-cpp-python is required to prefill gguf weights") from exc
    try:
        return llama_cpp.Llama(
            model_path=str(weights),
            n_ctx=context,
            n_threads=threads,
            n_threads_batch=threads,
            n_gpu_layers=gpu_layers,
            seed=SEED,
            logits_all=True,
            embedding=False,
            verbose=False,
        )
    except Exception as exc:
        raise VerificationError(f"the artifact could not be opened: {exc}") from exc


def aligned_rows(scores: Any, prompt_length: int, completion_length: int) -> list[list[float]]:
    if prompt_length < 1:
        raise VerificationError("a prompt must carry at least one token to condition on")
    if completion_length < 1:
        return []
    first = prompt_length - 1
    last = first + completion_length
    total = len(scores)
    if last > total:
        raise VerificationError(
            f"the sequence needs {last} logit rows and the engine returned {total}; "
            "the context is probably shorter than prompt plus completion"
        )
    return [list(scores[index]) for index in range(first, last)]


def canonical_logits(
    model: Any, prompt_tokens: list[int], completion_tokens: list[int]
) -> list[list[float]]:
    if not completion_tokens:
        return []
    if not prompt_tokens:
        raise VerificationError("a prompt must carry at least one token to condition on")
    sequence = list(prompt_tokens) + list(completion_tokens)
    try:
        model.reset()
        model.eval(sequence)
    except Exception as exc:
        raise VerificationError(f"the artifact could not evaluate the sequence: {exc}") from exc
    scores = getattr(model, "scores", None)
    if scores is None:
        raise VerificationError("the engine exposed no logits; it was not opened with logits_all")
    return aligned_rows(scores, len(prompt_tokens), len(completion_tokens))


class LlamaPrefill:
    def __init__(
        self,
        weights: Path,
        *,
        context: int = DEFAULT_CONTEXT,
        gpu_layers: int = GPU_LAYERS,
        threads: int = THREADS,
    ) -> None:
        self.weights = Path(weights)
        self.context = max(1, int(context))
        self.gpu_layers = int(gpu_layers)
        self.threads = max(1, int(threads))
        self._model: Any = None
        self._lock = threading.Lock()

    def _open(self) -> Any:
        if self._model is None:
            self._model = open_artifact(
                self.weights,
                context=self.context,
                threads=self.threads,
                gpu_layers=self.gpu_layers,
            )
        return self._model

    def _margins(self, prompt_ids: Sequence[int], completion_ids: Sequence[int]) -> list[float]:
        prompt = [int(token) for token in prompt_ids]
        completion = [int(token) for token in completion_ids]
        if not completion:
            return []
        if len(prompt) + len(completion) > self.context:
            raise VerificationError(
                f"{len(prompt) + len(completion)} tokens do not fit the prefill context "
                f"of {self.context}"
            )
        with self._lock:
            rows = canonical_logits(self._open(), prompt, completion)
        return verify.margins(rows, completion)

    async def margins(
        self, prompt_ids: Sequence[int], completion_ids: Sequence[int]
    ) -> list[float]:
        return await asyncio.to_thread(self._margins, prompt_ids, completion_ids)


def _entry(raw: Any) -> tuple[float | None, int]:
    if not isinstance(raw, Sequence) or isinstance(raw, str) or len(raw) < 2:
        raise VerificationError("the engine returned a logprob entry of an unknown shape")
    value = raw[0]
    try:
        return (None if value is None else float(value)), int(raw[1])
    except (TypeError, ValueError) as exc:
        raise VerificationError("a logprob entry is not a number and a token id") from exc


def logprob_margins(answer: Any, completion_ids: Sequence[int]) -> list[float]:
    if isinstance(answer, list) and len(answer) == 1:
        answer = answer[0]
    meta = answer.get("meta_info") if isinstance(answer, Mapping) else None
    if not isinstance(meta, Mapping):
        raise VerificationError("the engine answered without meta_info")
    tokens = meta.get("input_token_logprobs")
    tops = meta.get("input_top_logprobs")
    wanted = [int(token) for token in completion_ids]
    count = len(wanted)
    if not count:
        return []
    if not isinstance(tokens, list) or not isinstance(tops, list):
        raise VerificationError("the engine returned no input logprobs; was return_logprob set")
    if len(tokens) < count or len(tops) < count:
        raise VerificationError(
            f"the engine scored {min(len(tokens), len(tops))} positions for a completion of {count}"
        )
    chosen: list[float] = []
    top: list[float] = []
    for raw, best, token in zip(tokens[-count:], tops[-count:], wanted, strict=True):
        value, ident = _entry(raw)
        if value is None or ident != token:
            raise VerificationError("the engine scored other tokens than the completion")
        if not isinstance(best, list) or not best:
            raise VerificationError("the engine returned no top logprob for a position")
        leader, _ = _entry(best[0])
        if leader is None:
            raise VerificationError("the engine returned an empty top logprob")
        chosen.append(value)
        top.append(leader)
    return verify.margins_from_logprobs(chosen, top)


class SGLangPrefill:
    def __init__(
        self,
        url: str,
        *,
        timeout: float = PREFILL_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.client = client

    @staticmethod
    def body(prompt_ids: Sequence[int], completion_ids: Sequence[int]) -> dict[str, Any]:
        prompt = [int(token) for token in prompt_ids]
        return {
            "input_ids": prompt + [int(token) for token in completion_ids],
            "sampling_params": {"max_new_tokens": 0},
            "return_logprob": True,
            "logprob_start_len": len(prompt) - 1,
            "top_logprobs_num": 1,
        }

    async def _post(self, payload: Mapping[str, Any]) -> httpx.Response:
        if self.client is not None:
            return await self.client.post(self.url + GENERATE_PATH, json=payload)
        timeout = httpx.Timeout(self.timeout, connect=CONNECT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as http:
            return await http.post(self.url + GENERATE_PATH, json=payload)

    async def margins(
        self, prompt_ids: Sequence[int], completion_ids: Sequence[int]
    ) -> list[float]:
        if not completion_ids:
            return []
        if not prompt_ids:
            raise VerificationError("a prompt must carry at least one token to condition on")
        try:
            found = await self._post(self.body(prompt_ids, completion_ids))
        except httpx.HTTPError as exc:
            raise VerificationError(f"the prefill engine could not be reached: {exc}") from exc
        if found.status_code != 200:
            raise VerificationError(
                f"the prefill engine answered {found.status_code}: {found.text[-300:]}"
            )
        try:
            answer = found.json()
        except ValueError as exc:
            raise VerificationError("the prefill engine answered with something not json") from exc
        return logprob_margins(answer, completion_ids)


def prefill_for(
    backend: str,
    *,
    url: str = "",
    weights: Path | None = None,
    context: int = DEFAULT_CONTEXT,
    gpu_layers: int = GPU_LAYERS,
) -> Prefill:
    if backend == SGLANG:
        if not url:
            raise VerificationError("an sglang prefill needs the url of a running engine")
        return SGLangPrefill(url)
    if backend == LLAMA:
        if weights is None:
            raise VerificationError("a llama.cpp prefill needs the gguf weights it loads")
        return LlamaPrefill(weights, context=context, gpu_layers=gpu_layers)
    raise VerificationError(f"no prefill backend {backend!r}; known backends are sglang, llama")
