from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

import httpx

from microtensor_subnet.validator.sandbox.docker import ContainerSpec

LLAMA: Final[str] = "llama"
SGLANG: Final[str] = "sglang"
BACKENDS: Final[tuple[str, ...]] = (SGLANG, LLAMA)

GGUF: Final[str] = "gguf"
AWQ: Final[str] = "awq"
FP8: Final[str] = "fp8"
SAFETENSORS: Final[str] = "safetensors"

SGLANG_IMAGE: Final[str] = "lmsysorg/sglang:v0.5.2-cu126"
LLAMA_IMAGE: Final[str] = "ghcr.io/ggml-org/llama.cpp:server-cuda"

SERVED_NAME: Final[str] = "model"
ENGINE_PORT: Final[int] = 8000
MODEL_MOUNT: Final[str] = "/model"
CHAT_PATH: Final[str] = "/v1/chat/completions"
HEALTH_PATH: Final[str] = "/health"
CHUNKED_PREFILL: Final[int] = 8192
GPU_LAYERS: Final[int] = 999
HEALTH_TIMEOUT: Final[float] = 5.0
CONNECT_TIMEOUT: Final[float] = 10.0
SGLANG_SHM: Final[str] = "16g"
QUANTIZATION: Final[dict[str, str]] = {AWQ: "awq_marlin"}

Delta = dict[str, Any]
OnDelta = Callable[[Delta], Awaitable[None]]
ClientFactory = Callable[[], httpx.AsyncClient]


class EngineError(RuntimeError):
    pass


def backend_for(artifact_format: str) -> str:
    found = (artifact_format or "").strip().lower()
    if found == GGUF:
        return LLAMA
    if found in (AWQ, FP8, SAFETENSORS):
        return SGLANG
    raise EngineError(
        f"nothing serves the format {found!r}; known formats are gguf, awq, fp8, safetensors"
    )


def inside(path: str) -> str:
    found = PurePosixPath(path.replace("\\", "/"))
    if not path or found.is_absolute() or ".." in found.parts:
        raise EngineError(f"{path!r} is not a path inside the model folder")
    return found.as_posix()


def _extra(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        argv: list[str] = []
        for key, value in raw.items():
            if value is False or value is None:
                continue
            argv.append(str(key))
            if value is not True:
                argv.append(str(value))
        return tuple(argv)
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return tuple(str(item) for item in raw)
    raise EngineError("the engine block's extra arguments must be a list or a mapping")


@dataclass(frozen=True, slots=True)
class EngineSpec:
    backend: str
    tool_call_parser: str = ""
    reasoning_parser: str = ""
    quantization: str = ""
    chat_template: str = ""
    extra: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise EngineError(f"unknown engine {self.backend!r}; known engines are sglang, llama")
        if self.chat_template:
            inside(self.chat_template)

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> EngineSpec:
        raw = entry.get("engine")
        engine: dict[str, Any] = dict(raw) if isinstance(raw, Mapping) else {}
        artifact_format = str(entry.get("format") or "").strip().lower()
        backend = str(engine.get("backend") or "").strip().lower() or backend_for(artifact_format)
        quantization = str(engine.get("quantization") or "")
        if not quantization and backend == SGLANG:
            quantization = QUANTIZATION.get(artifact_format, "")
        manifest = entry.get("manifest")
        template = manifest.get("template") if isinstance(manifest, Mapping) else ""
        return cls(
            backend=backend,
            tool_call_parser=str(engine.get("tool_call_parser") or ""),
            reasoning_parser=str(engine.get("reasoning_parser") or ""),
            quantization=quantization,
            chat_template=str(engine.get("chat_template") or template or ""),
            extra=_extra(engine.get("extra")),
        )


def sglang_argv(
    spec: EngineSpec, *, context: int, concurrency: int, mem_fraction: float
) -> list[str]:
    argv = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_MOUNT,
        "--served-model-name",
        SERVED_NAME,
        "--host",
        "0.0.0.0",  # noqa: S104
        "--port",
        str(ENGINE_PORT),
        "--max-running-requests",
        str(max(1, concurrency)),
        "--mem-fraction-static",
        f"{mem_fraction:.3f}",
        "--chunked-prefill-size",
        str(CHUNKED_PREFILL),
        "--random-seed",
        "0",
    ]
    if context > 0:
        argv += ["--context-length", str(context)]
    if spec.quantization:
        argv += ["--quantization", spec.quantization]
    if spec.tool_call_parser:
        argv += ["--tool-call-parser", spec.tool_call_parser]
    if spec.reasoning_parser:
        argv += ["--reasoning-parser", spec.reasoning_parser]
    if spec.chat_template:
        argv += ["--chat-template", f"{MODEL_MOUNT}/{inside(spec.chat_template)}"]
    return argv + list(spec.extra)


def llama_argv(spec: EngineSpec, weights: str, *, context: int, parallel: int) -> list[str]:
    slots = max(1, parallel)
    argv = [
        "--model",
        weights,
        "--alias",
        SERVED_NAME,
        "--host",
        "0.0.0.0",  # noqa: S104
        "--port",
        str(ENGINE_PORT),
        "--parallel",
        str(slots),
        "--n-gpu-layers",
        str(GPU_LAYERS),
        "--jinja",
        "--reasoning-format",
        "deepseek" if spec.reasoning_parser else "none",
    ]
    if context > 0:
        argv += ["--ctx-size", str(context * slots)]
    if spec.chat_template:
        argv += ["--chat-template-file", f"{MODEL_MOUNT}/{inside(spec.chat_template)}"]
    return argv + list(spec.extra)


def container_spec(
    spec: EngineSpec,
    *,
    name: str,
    image: str,
    network: str,
    model_dir: Path,
    gpu: str,
    context: int,
    concurrency: int,
    mem_fraction: float,
    weights: str = "",
) -> ContainerSpec:
    if spec.backend == LLAMA:
        if not weights:
            raise EngineError("llama.cpp needs the gguf file it loads")
        command = llama_argv(
            spec, f"{MODEL_MOUNT}/{inside(weights)}", context=context, parallel=concurrency
        )
    else:
        command = sglang_argv(
            spec, context=context, concurrency=concurrency, mem_fraction=mem_fraction
        )
    return ContainerSpec(
        name=name,
        image=image,
        network=network,
        command=tuple(command),
        env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        volumes=((str(model_dir), MODEL_MOUNT, "ro"),),
        gpus=(gpu,),
        shm_size=SGLANG_SHM if spec.backend == SGLANG else "",
        ipc_none=False,
    )


@dataclass(frozen=True, slots=True)
class Chat:
    content: str
    reasoning: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float


def _event(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        found = json.loads(payload)
    except ValueError:
        return None
    return found if isinstance(found, dict) else None


def _deltas(delta: Mapping[str, Any]) -> list[Delta]:
    out: list[Delta] = []
    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        out.append({"reasoning": reasoning})
    content = delta.get("content")
    if isinstance(content, str) and content:
        out.append({"content": content})
    calls: list[dict[str, Any]] = []
    for position, raw in enumerate(delta.get("tool_calls") or []):
        if not isinstance(raw, Mapping):
            continue
        function = raw.get("function")
        named = function if isinstance(function, Mapping) else {}
        arguments = named.get("arguments")
        calls.append(
            {
                "index": raw["index"] if isinstance(raw.get("index"), int) else position,
                "id": str(raw.get("id") or ""),
                "name": str(named.get("name") or ""),
                "arguments": arguments
                if isinstance(arguments, str)
                else (json.dumps(arguments, ensure_ascii=False) if arguments else ""),
            }
        )
    if calls:
        out.append({"tool_calls": calls})
    return out


def _gather(
    delta: Delta, content: list[str], reasoning: list[str], calls: dict[int, dict[str, Any]]
) -> None:
    if "content" in delta:
        content.append(str(delta["content"]))
    if "reasoning" in delta:
        reasoning.append(str(delta["reasoning"]))
    for piece in delta.get("tool_calls") or []:
        held = calls.setdefault(
            int(piece["index"]),
            {"index": int(piece["index"]), "id": "", "name": "", "arguments": ""},
        )
        if piece["id"] and not held["id"]:
            held["id"] = piece["id"]
        if piece["name"] and not held["name"]:
            held["name"] = piece["name"]
        held["arguments"] += piece["arguments"]


def _failure(event: Mapping[str, Any]) -> str:
    if event.get("object") == "error":
        return str(event.get("message") or "the engine failed")
    error = event.get("error")
    if isinstance(error, Mapping):
        return str(error.get("message") or "the engine failed")
    return str(error) if error else ""


class ChatEngine:
    def __init__(self, url: str, *, timeout: float, client: ClientFactory | None = None) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._client: ClientFactory = client or self._default

    def _default(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=CONNECT_TIMEOUT))

    @staticmethod
    def body(request: Mapping[str, Any]) -> dict[str, Any]:
        messages = request.get("messages")
        if (
            not isinstance(messages, Sequence)
            or isinstance(messages, str)
            or not messages
            or not all(isinstance(m, Mapping) for m in messages)
        ):
            raise EngineError("a chat request carries its messages as a list of objects")
        out: dict[str, Any] = {
            "model": SERVED_NAME,
            "messages": [dict(m) for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        tools = request.get("tools")
        if tools:
            out["tools"] = [dict(t) for t in tools]
            if request.get("tool_choice") is not None:
                out["tool_choice"] = request["tool_choice"]
        for key in ("max_tokens", "temperature", "top_p"):
            if request.get(key) is not None:
                out[key] = request[key]
        stop = request.get("stop")
        if stop:
            out["stop"] = [stop] if isinstance(stop, str) else [str(s) for s in stop]
        if request.get("response_format"):
            out["response_format"] = request["response_format"]
        return out

    async def chat(self, request: Mapping[str, Any], on_delta: OnDelta) -> Chat:
        payload = self.body(request)
        started = time.perf_counter()
        first: float | None = None
        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        finish = ""
        try:
            async with (
                self._client() as http,
                http.stream("POST", self.url + CHAT_PATH, json=payload) as found,
            ):
                if found.status_code != 200:
                    detail = (await found.aread()).decode("utf-8", "replace")
                    raise EngineError(f"the engine answered {found.status_code}: {detail[-300:]}")
                async for line in found.aiter_lines():
                    if line.strip() == "data: [DONE]":
                        break
                    event = _event(line)
                    if event is None:
                        continue
                    failure = _failure(event)
                    if failure:
                        raise EngineError(f"the engine failed mid stream: {failure[:300]}")
                    if isinstance(event.get("usage"), Mapping):
                        usage = dict(event["usage"])
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") if isinstance(choice, Mapping) else None
                        for piece in _deltas(delta if isinstance(delta, Mapping) else {}):
                            if first is None:
                                first = time.perf_counter()
                            _gather(piece, content, reasoning, calls)
                            await on_delta(piece)
                        if isinstance(choice, Mapping) and choice.get("finish_reason"):
                            finish = str(choice["finish_reason"])
        except httpx.HTTPError as exc:
            raise EngineError(f"the engine could not be reached: {exc}") from exc
        ended = time.perf_counter()
        return Chat(
            content="".join(content),
            reasoning="".join(reasoning),
            tool_calls=[calls[index] for index in sorted(calls)],
            finish_reason=finish or ("tool_calls" if calls else "stop"),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            ttft_ms=((first or ended) - started) * 1000,
        )

    async def ready(self) -> bool:
        try:
            async with self._client() as http:
                found = await http.get(self.url + HEALTH_PATH, timeout=HEALTH_TIMEOUT)
        except httpx.HTTPError:
            return False
        return bool(found.status_code == 200)
