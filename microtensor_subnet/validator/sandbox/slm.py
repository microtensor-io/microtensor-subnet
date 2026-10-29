from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import httpx

from microtensor_subnet.artifact import contract
from microtensor_subnet.core import profiles

TIMEOUT_SECONDS: Final[float] = 300.0
MODEL_NAME: Final[str] = "slm"
MEMORY_SHARE: Final[float] = 0.9
SHARED_MEMORY_SHARE: Final[float] = 0.75
BATCH_INVARIANT_ENV: Final[str] = "VLLM_BATCH_INVARIANT"
ID_PREFIX: Final[str] = "token_id:"


class SlmError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_ms: float
    finish_reason: str
    output_tokens: tuple[int, ...] = ()
    logprobs: tuple[float, ...] = ()

    @property
    def tpot_ms(self) -> float:
        produced = max(0, self.completion_tokens - 1)
        return (self.total_ms - self.ttft_ms) / produced if produced else 0.0


def _ids_from(choice: Mapping[str, Any]) -> list[int]:
    listed = choice.get("token_ids")
    if isinstance(listed, list):
        return [int(t) for t in listed]
    logprobs = choice.get("logprobs") or {}
    found: list[int] = []
    for token in (logprobs.get("tokens") if isinstance(logprobs, Mapping) else None) or []:
        text = str(token)
        if text.startswith(ID_PREFIX) and text[len(ID_PREFIX) :].isdigit():
            found.append(int(text[len(ID_PREFIX) :]))
    return found


def _logprobs_from(choice: Mapping[str, Any]) -> list[float]:
    logprobs = choice.get("logprobs") or {}
    values = logprobs.get("token_logprobs") if isinstance(logprobs, Mapping) else None
    return [float(v) for v in values or [] if v is not None]


@dataclass(slots=True)
class SlmClient:
    base_url: str
    timeout: float = TIMEOUT_SECONDS

    def ready(self) -> bool:
        try:
            with httpx.Client(timeout=10.0) as http:
                response = http.get(self.base_url.rstrip("/") + "/v1/models")
        except httpx.HTTPError:
            return False
        return bool(response.status_code == 200)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: Sequence[str] = (),
        seed: int | None = None,
        logprobs: int = 0,
    ) -> Generation:
        body: dict[str, Any] = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if stop:
            body["stop"] = list(stop)
        if seed is not None:
            body["seed"] = int(seed)
        if logprobs:
            body["logprobs"] = int(logprobs)
            body["return_token_ids"] = True
            body["return_tokens_as_token_ids"] = True
        started = time.perf_counter()
        first: float | None = None
        text: list[str] = []
        pieces = 0
        usage: dict[str, Any] = {}
        finish = ""
        ids: list[int] = []
        probs: list[float] = []
        try:
            with (
                httpx.Client(timeout=self.timeout) as http,
                http.stream(
                    "POST", self.base_url.rstrip("/") + contract.SLM_COMPLETIONS, json=body
                ) as response,
            ):
                if response.status_code != 200:
                    raise SlmError(f"the model server answered {response.status_code}")
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    if chunk.get("usage"):
                        usage = dict(chunk["usage"])
                    for choice in chunk.get("choices") or []:
                        piece = str(choice.get("text") or "")
                        if piece and first is None:
                            first = time.perf_counter()
                        if piece:
                            pieces += 1
                        text.append(piece)
                        ids += _ids_from(choice)
                        probs += _logprobs_from(choice)
                        if choice.get("finish_reason"):
                            finish = str(choice["finish_reason"])
        except httpx.HTTPError as exc:
            raise SlmError(f"the model server could not be reached: {exc}") from exc
        ended = time.perf_counter()
        return Generation(
            text="".join(text),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or pieces),
            ttft_ms=((first or ended) - started) * 1000,
            total_ms=(ended - started) * 1000,
            finish_reason=finish,
            output_tokens=tuple(ids),
            logprobs=tuple(probs),
        )


def llama_command(model_file: str, *, threads: int, max_model_len: int) -> list[str]:
    return [
        "-m",
        model_file,
        "--host",
        "0.0.0.0",  # noqa: S104
        "--port",
        "8000",
        "--alias",
        MODEL_NAME,
        "-t",
        str(threads),
        "-tb",
        str(threads),
        "-c",
        str(max_model_len),
        "-np",
        "1",
    ]


def vllm_command(
    model_dir: str,
    *,
    max_model_len: int,
    quantization: str = "",
    memory_share: float = MEMORY_SHARE,
    max_seqs: int = 0,
    dtype: str = "",
    logprobs: int = 0,
    prefix_caching: bool | None = None,
    tensor_parallel: int = 1,
) -> list[str]:
    argv = [
        "--model",
        model_dir,
        "--served-model-name",
        MODEL_NAME,
        "--port",
        "8000",
        "--host",
        "0.0.0.0",  # noqa: S104
        "--max-model-len",
        str(max_model_len),
        "--seed",
        "0",
        "--disable-log-requests",
        "--gpu-memory-utilization",
        str(memory_share),
    ]
    if quantization:
        argv += ["--quantization", quantization]
    if max_seqs:
        argv += ["--max-num-seqs", str(max_seqs)]
    if dtype:
        argv += ["--dtype", dtype]
    if logprobs:
        argv += ["--max-logprobs", str(logprobs)]
    if prefix_caching is False:
        argv.append("--no-enable-prefix-caching")
    elif prefix_caching is True:
        argv.append("--enable-prefix-caching")
    if tensor_parallel > 1:
        argv += ["--tensor-parallel-size", str(tensor_parallel)]
    return argv


def engine_command(
    model_dir: str,
    *,
    profile: str,
    max_model_len: int,
    memory_share: float = MEMORY_SHARE,
    quantization: str = "",
) -> list[str]:
    return vllm_command(
        model_dir,
        max_model_len=max_model_len,
        quantization=quantization,
        memory_share=memory_share,
        max_seqs=profiles.max_num_seqs(profile),
        dtype=profiles.dtype_of(profile),
        logprobs=1,
        prefix_caching=False,
        tensor_parallel=profiles.tensor_parallel(profile),
    )


def engine_env(profile: str) -> dict[str, str]:
    env = {"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1"}
    if profiles.batch_invariant(profile):
        env[BATCH_INVARIANT_ENV] = "1"
    return env
