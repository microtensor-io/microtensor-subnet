from __future__ import annotations

import base64
import logging
import time
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx

import microtensor_subnet
from microtensor_subnet.artifact import contract
from microtensor_subnet.scoring.stats import percentile
from microtensor_subnet.validator.rerun.engine import (
    DEFAULT_MAX_MODEL_LEN,
    EngineError,
    EngineHang,
    Generated,
)
from microtensor_subnet.validator.sandbox import Limits, compat
from microtensor_subnet.validator.sandbox.docker import pinned
from microtensor_subnet.validator.sandbox.slm import (
    MEMORY_SHARE,
    SlmError,
    engine_command,
    engine_env,
)

log = logging.getLogger("microtensor_subnet.validator.rerun.pod")

POD_PORT: Final[int] = 8010
REQUEST_SECONDS: Final[float] = 60.0
REQUEST_ATTEMPTS: Final[int] = 12
POLL_SECONDS: Final[float] = 1.0
PREPARE_SECONDS: Final[float] = 2700.0
READY_SECONDS: Final[float] = 900.0
CALL_SECONDS: Final[float] = 600.0
GRACE_SECONDS: Final[float] = 60.0
BAD_CARD_TEXT: Final[str] = "CUDA unknown error"
ROUTER_SECONDS: Final[float] = 900.0
SERVER_MODULE: Final[str] = "microtensor_subnet.validator.sandbox.router_server"
CODE_FILES: Final[tuple[str, ...]] = (
    "artifact/contract.py",
    "validator/sandbox/router_server.py",
)
PACKAGES: Final[tuple[str, ...]] = ("", "artifact", "validator", "validator/sandbox")
START_CODE: Final[str] = (
    "import base64,os,zlib;"
    "exec(compile(zlib.decompress(base64.b64decode(os.environ['MT_POD_WORKER'])),"
    "'podworker','exec'))"
)


class PodLost(RuntimeError):
    pass


class BadCard(PodLost):
    pass


class BadWeights(RuntimeError):
    pass


class PodClient:
    def __init__(self, base_url: str, token: str, *, timeout: float = REQUEST_SECONDS) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        failure = ""
        for attempt in range(REQUEST_ATTEMPTS):
            try:
                with httpx.Client(timeout=self.timeout) as http:
                    response = http.request(
                        method,
                        self.base_url + path,
                        json=dict(body) if body is not None else None,
                        headers={"Authorization": f"Bearer {self.token}"},
                    )
            except httpx.HTTPError as exc:
                failure = f"{type(exc).__name__}: {str(exc)[:160]}"
            else:
                if response.status_code < 500 and not self._from_proxy(response):
                    return self._answer(response, path)
                failure = f"{path} answered {response.status_code}"
            time.sleep(min(15.0, 2.0 * (attempt + 1)))
        raise PodLost(f"the pod could not be reached: {failure}")

    @staticmethod
    def _from_proxy(response: httpx.Response) -> bool:
        if response.status_code != 404:
            return False
        try:
            return not isinstance(response.json(), Mapping)
        except ValueError:
            return True

    @staticmethod
    def _answer(response: httpx.Response, path: str) -> Any:
        if response.status_code >= 400:
            try:
                detail = str(response.json().get("detail") or "")
            except ValueError:
                detail = response.text[:200]
            raise EngineError(f"{path} answered {response.status_code}: {detail}")
        try:
            return response.json()
        except ValueError as exc:
            raise PodLost(f"{path} answered with something not JSON") from exc

    def wait_task(self, key: str, seconds: float) -> dict[str, Any]:
        deadline = time.monotonic() + seconds
        pause = 0.25
        while time.monotonic() < deadline:
            found = self.request("GET", f"/tasks/{key}")
            if str(found.get("state")) != "running":
                return dict(found)
            time.sleep(pause)
            pause = min(POLL_SECONDS, pause * 1.5)
        raise EngineHang(f"the pod's task did not finish within {seconds:.0f} s")

    def health(self) -> dict[str, Any]:
        return dict(self.request("GET", "/health"))

    def cleanup(self) -> dict[str, Any]:
        return dict(self.request("POST", "/cleanup", {}))

    def prepare(
        self,
        *,
        kind: str,
        name: str,
        files: Mapping[str, str] | None = None,
        hub: Mapping[str, str] | None = None,
    ) -> None:
        body: dict[str, Any] = {"kind": kind, "name": name}
        if files is not None:
            body["files"] = {str(k): str(v) for k, v in files.items()}
        if hub is not None:
            body["hub"] = dict(hub)
        started = self.request("POST", "/prepare", body)
        found = self.wait_task(str(started["task"]), PREPARE_SECONDS)
        state = str(found.get("state"))
        if state == "hash":
            raise BadWeights(str(found.get("reason") or "the weights do not match their hash"))
        if state != "done":
            raise EngineError(f"the download failed: {found.get('reason')}")


@dataclass(slots=True)
class RemoteEngine:
    client: PodClient
    kind: str
    name: str
    profile: str
    max_model_len: int
    share: float = MEMORY_SHARE
    ready_seconds: float = READY_SECONDS
    call_seconds: float = CALL_SECONDS
    vram_peak_bytes: int = 0
    started: bool = False
    model_dir: str = ""

    def _body(self) -> dict[str, Any]:
        body = {
            "kind": self.kind,
            "name": self.name,
            "argv": engine_command(
                compat.GUEST,
                profile=self.profile,
                max_model_len=self.max_model_len,
                memory_share=self.share,
            ),
            "env": engine_env(self.profile),
        }
        if self.model_dir:
            body["model_dir"] = self.model_dir
        return body

    def _wait(self) -> None:
        deadline = time.monotonic() + self.ready_seconds
        while time.monotonic() < deadline:
            found = self.client.request("GET", "/engine")
            state = str(found.get("state"))
            if state == "ready":
                self.started = True
                return
            if state in ("exited", "stuck", "none"):
                note = str(found.get("note") or state)
                if BAD_CARD_TEXT in note:
                    raise BadCard(f"this pod's card can't be used: {BAD_CARD_TEXT}")
                raise EngineError(f"the engine exited: {note}")
            time.sleep(POLL_SECONDS)
        raise EngineError("the engine did not become ready")

    def start(self) -> None:
        self.client.request("POST", "/engine/start", self._body())
        self._wait()

    def stop(self) -> None:
        if not self.started:
            return
        self.started = False
        try:
            self.client.request("POST", "/engine/stop", {})
        except (PodLost, EngineError) as exc:
            log.warning("the engine was not stopped: %s", exc)

    def restart(self) -> None:
        self.client.request("POST", "/engine/restart", {})
        self._wait()

    def generate(
        self, prompt: str, *, max_tokens: int, stop: Sequence[str], seed: int
    ) -> Generated:
        started = self.client.request(
            "POST",
            "/calls",
            {
                "prompt": prompt,
                "max_tokens": int(max_tokens),
                "stop": [str(s) for s in stop],
                "seed": int(seed),
                "limit_seconds": self.call_seconds,
            },
        )
        found = self.client.wait_task(str(started["task"]), self.call_seconds + GRACE_SECONDS)
        state = str(found.get("state"))
        if state == "hang":
            raise EngineHang(str(found.get("reason") or "the engine hung"))
        if state != "done":
            raise SlmError(str(found.get("reason") or f"the call ended as {state}"))
        return Generated(
            text=str(found.get("text") or ""),
            ids=tuple(int(t) for t in found.get("ids") or []),
            prompt_tokens=int(found.get("prompt_tokens") or 0),
            ttft_ms=float(found.get("ttft_ms") or 0.0),
            tpot_ms=float(found.get("tpot_ms") or 0.0),
            total_ms=float(found.get("total_ms") or 0.0),
        )

    def sample_vram(self) -> int:
        found = self.client.request("GET", "/engine")
        self.vram_peak_bytes = max(self.vram_peak_bytes, int(found.get("peak_bytes") or 0))
        return self.vram_peak_bytes


def remote_engine(
    client: PodClient,
    *,
    kind: str,
    name: str,
    profile: str,
    max_model_len: int | None,
    share: float,
    call_seconds: float = 0.0,
    model_dir: str = "",
) -> RemoteEngine:
    return RemoteEngine(
        client=client,
        kind=kind,
        name=name,
        profile=profile,
        max_model_len=max_model_len or DEFAULT_MAX_MODEL_LEN,
        share=share,
        call_seconds=call_seconds or CALL_SECONDS,
        model_dir=model_dir,
    )


def router_code() -> dict[str, str]:
    source = Path(microtensor_subnet.__file__).parent
    found = {f"microtensor_subnet/{name}/__init__.py".replace("//", "/"): "" for name in PACKAGES}
    for name in CODE_FILES:
        found[f"microtensor_subnet/{name}"] = base64.b64encode((source / name).read_bytes()).decode(
            "ascii"
        )
    return found


def measure_router(
    client: PodClient,
    *,
    name: str,
    config: Mapping[str, Any],
    limits: Limits,
    pairs: Sequence[tuple[str, str]],
) -> tuple[float | None, str]:
    if not pairs:
        return None, "the transcript holds no drafts for the router"
    block = config.get("router") if isinstance(config.get("router"), Mapping) else {}
    wanted = int((block or {}).get("max_input_tokens") or 4096)
    started = client.request(
        "POST",
        "/router/measure",
        {
            "kind": "miner",
            "name": name,
            "module": SERVER_MODULE,
            "code": router_code(),
            "pairs": [{"query": q, "draft": d} for q, d in pairs],
            "max_input_tokens": min(wanted, limits.router_max_input_tokens),
            "max_parameters": min(limits.router_max_parameters, contract.ROUTER_MAX_PARAMETERS),
        },
    )
    found = client.wait_task(str(started["task"]), ROUTER_SECONDS)
    if str(found.get("state")) != "done":
        return None, f"router failed: {found.get('reason')}"[:300]
    return percentile([float(v) for v in found.get("latencies") or []], 0.95), ""


def worker_payload() -> str:
    source = Path(__file__).with_name("podworker.py").read_bytes()
    return base64.b64encode(zlib.compress(source, 9)).decode("ascii")


def engine_image(definition: Mapping[str, Any]) -> str:
    block = definition.get("evaluation") or {}
    images = block.get("images") if isinstance(block, Mapping) else None
    engine = images.get("engine") if isinstance(images, Mapping) else None
    if not isinstance(engine, Mapping) or not engine.get("image") or not engine.get("digest"):
        raise EngineError("the arena definition pins no engine image")
    return pinned(str(engine["image"]), str(engine["digest"]))
