from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from microtensor.serving.agent import DEFAULT_MAX_TOKENS, AgentError, Engine, _httpx

log = logging.getLogger("microtensor.serving.engines")

LLAMA: Final[str] = "llama"
SGLANG: Final[str] = "sglang"

GGUF: Final[str] = "gguf"
AWQ: Final[str] = "awq"
FP8: Final[str] = "fp8"
SAFETENSORS: Final[str] = "safetensors"

DEFAULT_CONTEXT: Final[int] = 4096
DEFAULT_CHUNKED_PREFILL: Final[int] = 8192
HEALTH_TIMEOUT: Final[float] = 5.0


class EngineError(AgentError):
    pass


@dataclass(frozen=True, slots=True)
class Launch:
    backend: str
    argv: tuple[str, ...]


def _llama_argv(
    binary: str,
    weights: Path,
    port: int,
    *,
    context: int,
    concurrency: int,
    threads: int,
    gpu_layers: int,
    share: float,
) -> tuple[str, ...]:
    argv = [
        binary,
        "--model",
        str(weights),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context),
        "--parallel",
        str(max(1, concurrency)),
    ]
    if threads:
        argv += ["--threads", str(threads)]
    if gpu_layers:
        argv += ["--n-gpu-layers", str(gpu_layers)]
    return tuple(argv)


def _sglang_argv(
    binary: str,
    weights: Path,
    port: int,
    *,
    context: int,
    concurrency: int,
    threads: int,
    gpu_layers: int,
    share: float,
    quantization: str = "",
) -> tuple[str, ...]:
    argv = [
        binary,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(weights),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--context-length",
        str(context),
        "--max-running-requests",
        str(max(1, concurrency)),
        "--chunked-prefill-size",
        str(DEFAULT_CHUNKED_PREFILL),
        "--mem-fraction-static",
        f"{share:.3f}",
        "--enable-cache-report",
    ]
    if quantization:
        argv += ["--quantization", quantization]
    return tuple(argv)


QUANTIZATION: Final[dict[str, str]] = {AWQ: "awq_marlin"}


def backend_for(artifact_format: str) -> str:
    found = (artifact_format or "").strip().lower()
    if not found or found == GGUF:
        return LLAMA
    if found in (AWQ, FP8, SAFETENSORS):
        return SGLANG
    raise EngineError(f"nothing serves {found!r}; known formats are gguf, awq, fp8, safetensors")


def launch(
    artifact_format: str,
    weights: Path,
    port: int,
    *,
    binary: str = "",
    context: int = DEFAULT_CONTEXT,
    concurrency: int = 8,
    threads: int = 0,
    gpu_layers: int = -1,
    share: float = 0.85,
) -> Launch:
    backend = backend_for(artifact_format)
    found = (artifact_format or "").strip().lower()
    if backend == LLAMA:
        return Launch(
            backend=backend,
            argv=_llama_argv(
                binary or "llama-server",
                weights,
                port,
                context=context,
                concurrency=concurrency,
                threads=threads,
                gpu_layers=gpu_layers,
                share=share,
            ),
        )
    return Launch(
        backend=backend,
        argv=_sglang_argv(
            binary or "python",
            weights,
            port,
            context=context,
            concurrency=concurrency,
            threads=threads,
            gpu_layers=gpu_layers,
            share=share,
            quantization=QUANTIZATION.get(found, ""),
        ),
    )


def share_for(models: int, headroom: float = 0.08) -> float:
    if models <= 0:
        raise EngineError("a card holding no models has no share to give")
    return max(0.05, round((1.0 - headroom) / models, 3))


class SGLangEngine(Engine):
    @staticmethod
    def _body(request: Mapping[str, Any], *, stream: bool) -> dict[str, Any]:
        prompt = str(request.get("prompt", ""))
        if not prompt:
            raise EngineError("the gateway sent a request with no prompt")
        sampling: dict[str, Any] = {
            "max_new_tokens": int(request.get("max_tokens", DEFAULT_MAX_TOKENS)),
            "temperature": float(request.get("temperature", 0.0)),
        }
        for ours, theirs in (("top_p", "top_p"), ("top_k", "top_k"), ("stop", "stop")):
            if request.get(ours) is not None:
                sampling[theirs] = request[ours]
        return {
            "text": prompt,
            "sampling_params": sampling,
            "return_logprob": True,
            "logprob_start_len": 0,
            "stream": stream,
        }

    @staticmethod
    def _tokens(rows: Sequence[Any] | None) -> list[int]:
        found: list[int] = []
        for row in rows or ():
            if isinstance(row, Sequence) and len(row) >= 2 and isinstance(row[1], int):
                found.append(int(row[1]))
        return found

    @staticmethod
    def _logprobs(rows: Sequence[Any] | None) -> list[float]:
        found: list[float] = []
        for row in rows or ():
            if isinstance(row, Sequence) and row and isinstance(row[0], int | float):
                found.append(float(row[0]))
        return found

    @classmethod
    def _answer(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        meta = dict(payload.get("meta_info") or {})
        finish = meta.get("finish_reason")
        kind = str((finish or {}).get("type", "stop")) if isinstance(finish, Mapping) else "stop"
        return {
            "text": str(payload.get("text", "")),
            "prompt_tokens": cls._tokens(meta.get("input_token_logprobs")),
            "completion_tokens": cls._tokens(meta.get("output_token_logprobs")),
            "logprobs": cls._logprobs(meta.get("output_token_logprobs")),
            "finish_reason": "length" if kind == "length" else "stop",
        }

    async def generate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        httpx = _httpx()
        async with httpx.AsyncClient(base_url=self.url, timeout=self.timeout) as client:
            found = await client.post("/generate", json=self._body(request, stream=False))
            found.raise_for_status()
            payload = found.json()
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        return self._answer(payload)

    async def stream(
        self, request: Mapping[str, Any], on_delta: Callable[[str], Awaitable[None]]
    ) -> dict[str, Any]:
        httpx = _httpx()
        seen = ""
        payload: dict[str, Any] = {}
        async with (
            httpx.AsyncClient(base_url=self.url, timeout=self.timeout) as client,
            client.stream("POST", "/generate", json=self._body(request, stream=True)) as found,
        ):
            found.raise_for_status()
            async for line in found.aiter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if not body or body == "[DONE]":
                    continue
                try:
                    piece = json.loads(body)
                except ValueError:
                    continue
                whole = str(piece.get("text", ""))
                if whole.startswith(seen) and len(whole) > len(seen):
                    await on_delta(whole[len(seen) :])
                    seen = whole
                payload = piece
        return self._answer(payload)

    async def ready(self) -> bool:
        httpx = _httpx()
        try:
            async with httpx.AsyncClient(base_url=self.url, timeout=HEALTH_TIMEOUT) as client:
                found = await client.get("/get_model_info")
            return bool(found.status_code == 200)
        except Exception:
            return False
