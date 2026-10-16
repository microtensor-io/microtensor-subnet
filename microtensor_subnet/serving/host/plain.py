from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, NamedTuple

import httpx

from microtensor_subnet.serving import engines, markers
from microtensor_subnet.serving.engines import ChatEngine, EngineSpec
from microtensor_subnet.serving.host.fetch import FetchedModel
from microtensor_subnet.serving.tokens import TokenError, Tokeniser
from microtensor_subnet.serving.wire.session import OnChunk
from microtensor_subnet.validator.sandbox.docker import Docker

log = logging.getLogger("microtensor_subnet.serving.host.plain")

CONTEXT_EXCEEDED: Final[str] = "context_length_exceeded"
TOOLS_UNSUPPORTED: Final[str] = "tools_not_supported"
IMAGES_UNSUPPORTED: Final[str] = "images_not_supported"
ENGINE_SECONDS: Final[float] = 1800.0
POLL_SECONDS: Final[float] = 5.0
IMAGE_PARTS: Final[frozenset[str]] = frozenset({"image", "image_url", "input_image"})


class PlainError(RuntimeError):
    pass


class ClampError(PlainError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(slots=True)
class PlainHeld:
    model: str
    artifact_digest: str
    entry: dict[str, Any]
    spec: EngineSpec
    engine: ChatEngine
    tokeniser: Tokeniser
    model_dir: Path
    gpu: str
    family: str = ""
    context: int = 0
    max_output: int = 0
    concurrency: int = 1
    tool_calling: bool = False
    vision: bool = False
    container: str = ""
    network: str = ""
    files: dict[str, str] = field(default_factory=dict)


class Clamped(NamedTuple):
    request: dict[str, Any]
    prompt_ids: list[int]


def _vision(entry: Mapping[str, Any]) -> bool:
    raw = entry.get("modalities")
    if isinstance(raw, Mapping):
        raw = raw.get("input") or []
    if isinstance(raw, str):
        raw = [raw]
    return any(str(kind).lower() == "image" for kind in raw or [])


def _tokeniser(fetched: FetchedModel, spec: EngineSpec) -> Tokeniser:
    if spec.chat_template and spec.chat_template not in fetched.files:
        raise PlainError("the engine's chat template is not pinned by the model manifest")
    return Tokeniser.load(fetched.model_dir, template=spec.chat_template, gguf_file=fetched.weights)


def _held(
    fetched: FetchedModel,
    entry: Mapping[str, Any],
    spec: EngineSpec,
    url: str,
    *,
    gpu: str,
    tokeniser: Tokeniser,
    container: str = "",
    network: str = "",
) -> PlainHeld:
    return PlainHeld(
        model=fetched.model,
        artifact_digest=fetched.artifact_digest,
        entry=dict(entry),
        spec=spec,
        engine=ChatEngine(url, timeout=ENGINE_SECONDS),
        tokeniser=tokeniser,
        model_dir=fetched.model_dir,
        gpu=gpu,
        family=str(entry.get("family") or "").strip().lower(),
        context=int(entry.get("context") or 0),
        max_output=int(entry.get("max_output") or 0),
        concurrency=max(1, int(entry.get("concurrency") or 1)),
        tool_calling=bool(entry.get("tool_calling")),
        vision=_vision(entry),
        container=container,
        network=network,
        files=dict(fetched.files),
    )


def wait_ready(docker: Docker, name: str, url: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not docker.running(name):
            raise PlainError(f"the engine exited: {docker.logs(name, 40)[-300:]}")
        try:
            with httpx.Client(timeout=engines.HEALTH_TIMEOUT) as http:
                if http.get(url + engines.HEALTH_PATH).status_code == 200:
                    return
        except httpx.HTTPError:
            pass
        time.sleep(POLL_SECONDS)
    raise PlainError("the engine did not become ready")


def open_model(
    docker: Docker,
    fetched: FetchedModel,
    entry: Mapping[str, Any],
    *,
    gpu: str,
    images: Mapping[str, str],
    mem_fraction: float,
    ready_seconds: float,
) -> PlainHeld:
    spec = EngineSpec.from_entry(entry)
    image = images.get(spec.backend, "")
    if not image:
        raise PlainError(f"no {spec.backend} image is configured on this host")
    tokeniser = _tokeniser(fetched, spec)
    key = secrets.token_hex(4)
    name = f"mt-model-{key}"
    network = f"mt-net-{key}"
    docker.create_network(network)
    try:
        docker.start(
            engines.container_spec(
                spec,
                name=name,
                image=image,
                network=network,
                model_dir=fetched.model_dir,
                gpu=gpu,
                context=int(entry.get("context") or 0),
                concurrency=max(1, int(entry.get("concurrency") or 1)),
                mem_fraction=mem_fraction,
                weights=fetched.weights,
            )
        )
        url = f"http://{docker.address(name, network)}:{engines.ENGINE_PORT}"
        wait_ready(docker, name, url, ready_seconds)
    except Exception:
        docker.kill(name)
        docker.remove_network(network)
        raise
    return _held(
        fetched, entry, spec, url, gpu=gpu, tokeniser=tokeniser, container=name, network=network
    )


def attach_model(
    fetched: FetchedModel, entry: Mapping[str, Any], url: str, *, gpu: str
) -> PlainHeld:
    spec = EngineSpec.from_entry(entry)
    return _held(
        fetched, entry, spec, url.rstrip("/"), gpu=gpu, tokeniser=_tokeniser(fetched, spec)
    )


def close_model(docker: Docker, held: PlainHeld) -> None:
    if held.container:
        docker.kill(held.container)
    if held.network:
        docker.remove_network(held.network)


def _images(messages: Sequence[Any]) -> bool:
    for message in messages:
        parts = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(parts, list) and any(
            isinstance(part, Mapping) and str(part.get("type")) in IMAGE_PARTS for part in parts
        ):
            return True
    return False


def clamp(held: PlainHeld, request: Mapping[str, Any]) -> Clamped:
    asked = dict(request)
    messages = list(asked.get("messages") or [])
    tools = list(asked.get("tools") or []) or None
    if tools and not held.tool_calling:
        raise ClampError(TOOLS_UNSUPPORTED, f"{held.model} does not call tools")
    if not held.vision and _images(messages):
        raise ClampError(IMAGES_UNSUPPORTED, f"{held.model} reads text only")
    ids = held.tokeniser.prompt_ids(messages, tools)
    room = held.context - len(ids) if held.context else None
    wanted = int(asked.get("max_tokens") or 0)
    if wanted:
        max_tokens = min(wanted, held.max_output) if held.max_output else wanted
    else:
        max_tokens = held.max_output
        if room is not None:
            max_tokens = min(max_tokens, room) if max_tokens else room
    if room is not None and (room < 1 or max_tokens > room):
        raise ClampError(
            CONTEXT_EXCEEDED,
            f"the prompt is {len(ids)} tokens and {max_tokens} are asked for output; "
            f"{held.model} holds {held.context}",
        )
    if max_tokens:
        asked["max_tokens"] = max_tokens
    return Clamped(asked, ids)


def completion_ids(held: PlainHeld, found: engines.Chat) -> list[int]:
    if held.family not in markers.FAMILIES:
        return []
    try:
        return held.tokeniser.completion_ids(
            held.family,
            reasoning=found.reasoning,
            content=found.content,
            tool_calls=found.tool_calls,
        )
    except (markers.MarkerError, TokenError) as exc:
        log.debug("no completion ids for %s: %s", held.model, exc)
        return []


async def chat(
    held: PlainHeld, request: Mapping[str, Any], on_chunk: OnChunk, *, timeout: float
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        clamped = await asyncio.to_thread(clamp, held, request)
    except ClampError as exc:
        return {"error": str(exc)}
    sent = time.perf_counter()
    found = await asyncio.wait_for(held.engine.chat(clamped.request, on_chunk), timeout)
    completion = await asyncio.to_thread(completion_ids, held, found)
    return {
        "content": found.content,
        "reasoning": found.reasoning,
        "tool_calls": found.tool_calls,
        "finish_reason": found.finish_reason,
        "usage": {
            "prompt_tokens": found.prompt_tokens or len(clamped.prompt_ids),
            "completion_tokens": found.completion_tokens or len(completion),
        },
        "tokens": {"prompt": clamped.prompt_ids, "completion": completion},
        "artifact_digest": held.artifact_digest,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "ttft_ms": (sent - started) * 1000 + found.ttft_ms,
    }
