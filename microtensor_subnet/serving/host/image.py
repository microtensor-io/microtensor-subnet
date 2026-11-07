from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import httpx

from microtensor_subnet.validator.sandbox.docker import ContainerSpec, Docker

log = logging.getLogger("microtensor_subnet.serving.host.image")

SHIM_PORT: Final[int] = 8800
PORT_BASE: Final[int] = 9900
ANY: Final[str] = "0.0.0.0"  # noqa: S104
RUN_PATH: Final[str] = "/v1/run"
HEALTH_PATH: Final[str] = "/health"
POLL_SECONDS: Final[float] = 5.0
DEFAULT_CONCURRENCY: Final[int] = 8
LOOPBACK: Final[str] = "127.0.0.1"


class ImageError(RuntimeError):
    pass


@dataclass(slots=True)
class ImageHeld:
    model: str
    artifact_digest: str
    image_digest: str
    entry: dict[str, Any]
    gpu: str
    slot: int
    container: str
    network: str
    url: str


def certified_image(entry: Mapping[str, Any]) -> tuple[str, str]:
    images = entry.get("images")
    found = images.get("certified") if isinstance(images, Mapping) else None
    if not isinstance(found, Mapping):
        return "", ""
    image = str(found.get("image") or "").strip()
    digest = str(found.get("digest") or "").strip().lower()
    return (image, digest) if image and digest else ("", "")


def wait_ready(docker: Docker, name: str, url: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not docker.running(name):
            raise ImageError(f"the certified image exited: {docker.logs(name, 40)[-300:]}")
        try:
            with httpx.Client(timeout=POLL_SECONDS) as http:
                if http.get(url + HEALTH_PATH).status_code == 200:
                    return
        except httpx.HTTPError:
            pass
        time.sleep(POLL_SECONDS)
    raise ImageError("the certified image did not become ready")


def open_image(
    docker: Docker,
    entry: Mapping[str, Any],
    *,
    gpu: str,
    slot: int,
    ready_seconds: float,
    shm_size: str = "8g",
    workers: int = DEFAULT_CONCURRENCY,
) -> ImageHeld:
    image, digest = certified_image(entry)
    if not image:
        raise ImageError(f"{entry.get('model', '?')} has no certified image")
    reference = docker.ensure_digest(image, digest)
    key = secrets.token_hex(4)
    name = f"mt-cert-{key}"
    network = f"mt-net-{key}"
    port = PORT_BASE + slot
    docker.create_network(network)
    try:
        docker.start(
            ContainerSpec(
                name=name,
                image=reference,
                network=network,
                env={
                    "HF_HUB_OFFLINE": "1",
                    "VLLM_NO_USAGE_STATS": "1",
                    "MT_SHIM_BIND": ANY,
                    "MT_SHIM_PORT": str(SHIM_PORT),
                    "MT_SHIM_WORKERS": str(max(1, workers)),
                },
                gpus=(gpu,),
                hardened=True,
                shm_size=shm_size,
                ipc_none=False,
                ports=((f"{LOOPBACK}:{port}", SHIM_PORT),),
            )
        )
        url = f"http://{LOOPBACK}:{port}"
        wait_ready(docker, name, url, ready_seconds)
    except Exception:
        docker.kill(name)
        docker.remove_network(network)
        raise
    return ImageHeld(
        model=str(entry.get("model") or entry.get("id") or ""),
        artifact_digest=str(entry.get("artifact_digest") or ""),
        image_digest=digest,
        entry=dict(entry),
        gpu=gpu,
        slot=slot,
        container=name,
        network=network,
        url=url,
    )


def close_image(docker: Docker, held: ImageHeld) -> None:
    docker.kill(held.container)
    docker.remove_network(held.network)


def answer_image(held: ImageHeld, request: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as http:
            response = http.post(held.url + RUN_PATH, json=dict(request))
    except httpx.TimeoutException:
        return {"error": "the certified image did not answer in time"}
    except httpx.HTTPError as exc:
        return {"error": f"the certified image is unreachable: {exc.__class__.__name__}"}
    if response.status_code != 200:
        return {"error": f"the certified image answered {response.status_code}"}
    try:
        found = response.json()
    except ValueError:
        return {"error": "the certified image answered with something not json"}
    if not isinstance(found, Mapping):
        return {"error": "the certified image answered with something not an object"}
    answer = dict(found)
    answer.setdefault("latency_ms", (time.perf_counter() - started) * 1000)
    return answer
