from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import httpx

from microtensor_subnet.artifact import contract

TIMEOUT_SECONDS: Final[float] = 300.0
MODEL_NAME: Final[str] = "slm"
MEMORY_SHARE: Final[float] = 0.9
SHARED_MEMORY_SHARE: Final[float] = 0.75


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

    @property
    def tpot_ms(self) -> float:
        produced = max(0, self.completion_tokens - 1)
        return (self.total_ms - self.ttft_ms) / produced if produced else 0.0


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

    def generate(self, prompt: str, *, max_tokens: int, stop: Sequence[str] = ()) -> Generation:
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
        started = time.perf_counter()
        first: float | None = None
        text: list[str] = []
        usage: dict[str, Any] = {}
        finish = ""
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
                        text.append(piece)
                        if choice.get("finish_reason"):
                            finish = str(choice["finish_reason"])
        except httpx.HTTPError as exc:
            raise SlmError(f"the model server could not be reached: {exc}") from exc
        ended = time.perf_counter()
        return Generation(
            text="".join(text),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            ttft_ms=((first or ended) - started) * 1000,
            total_ms=(ended - started) * 1000,
            finish_reason=finish,
        )


def vllm_command(
    model_dir: str,
    *,
    max_model_len: int,
    quantization: str = "",
    memory_share: float = MEMORY_SHARE,
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
    return argv
