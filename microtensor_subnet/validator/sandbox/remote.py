from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final
from urllib.parse import urlparse

import httpx

from microtensor_subnet.artifact import contract
from microtensor_subnet.validator.sandbox.slm import MODEL_NAME, TIMEOUT_SECONDS

log = logging.getLogger("microtensor_subnet.validator.sandbox.remote")

REQUEST_SECONDS: Final[float] = 60.0
REQUEST_ATTEMPTS: Final[int] = 5
POLL_SECONDS: Final[float] = 1.0
PREPARE_SECONDS: Final[float] = 2700.0
CALL_LIMIT_SECONDS: Final[float] = TIMEOUT_SECONDS - 30.0
GRACE_SECONDS: Final[float] = 15.0
MINER_KIND: Final[str] = "miner"
MODELS_PATH: Final[str] = "/v1/models"
MAX_BODY: Final[int] = 32 * 1024 * 1024
READY: Final[str] = "ready"
RUNNING: Final[str] = "running"
DONE: Final[str] = "done"
ENDED_ENGINE: Final[frozenset[str]] = frozenset({"exited", "stuck", "none"})


class RemoteEngineError(RuntimeError):
    pass


class RemoteEngineHang(RemoteEngineError):
    pass


@dataclass(slots=True)
class RemoteEngine:
    base_url: str
    token: str = field(repr=False)
    timeout: float = REQUEST_SECONDS

    def request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        failure = ""
        for attempt in range(REQUEST_ATTEMPTS):
            try:
                with httpx.Client(timeout=self.timeout) as http:
                    response = http.request(
                        method,
                        self.base_url.rstrip("/") + path,
                        json=dict(body) if body is not None else None,
                        headers={"Authorization": f"Bearer {self.token}"},
                    )
            except httpx.HTTPError as exc:
                failure = f"{type(exc).__name__}: {str(exc)[:160]}"
            else:
                if response.status_code < 500:
                    return self._answer(response, path)
                failure = f"{path} answered {response.status_code}"
            time.sleep(min(15.0, 2.0 * (attempt + 1)))
        raise RemoteEngineError(f"the remote engine could not be reached: {failure}")

    @staticmethod
    def _answer(response: httpx.Response, path: str) -> Any:
        if response.status_code >= 400:
            raise RemoteEngineError(
                f"{path} answered {response.status_code}: {response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise RemoteEngineError(f"{path} answered with something not JSON") from exc

    def wait_task(self, key: str, seconds: float) -> dict[str, Any]:
        deadline = time.monotonic() + seconds
        pause = 0.25
        while time.monotonic() < deadline:
            found = self.request("GET", f"/tasks/{key}")
            if str(found.get("state")) != RUNNING:
                return dict(found)
            time.sleep(pause)
            pause = min(POLL_SECONDS, pause * 1.5)
        raise RemoteEngineHang(f"the remote task did not finish within {seconds:.0f} s")

    def health(self) -> dict[str, Any]:
        return dict(self.request("GET", "/health"))

    def engine_state(self) -> dict[str, Any]:
        return dict(self.request("GET", "/engine"))

    def prepare(self, name: str, files: Mapping[str, str]) -> None:
        started = self.request(
            "POST", "/prepare", {"kind": MINER_KIND, "name": name, "files": dict(files)}
        )
        found = self.wait_task(str(started["task"]), PREPARE_SECONDS)
        if str(found.get("state")) != DONE:
            raise RemoteEngineError(f"the model was not fetched: {found.get('reason')}")

    def start(
        self, name: str, argv: Sequence[str], env: Mapping[str, str], ready_seconds: float
    ) -> None:
        self.request(
            "POST",
            "/engine/start",
            {"kind": MINER_KIND, "name": name, "argv": list(argv), "env": dict(env)},
        )
        deadline = time.monotonic() + ready_seconds
        while time.monotonic() < deadline:
            found = self.engine_state()
            state = str(found.get("state"))
            if state == READY:
                return
            if state in ENDED_ENGINE:
                raise RemoteEngineError(f"the engine ended: {found.get('note') or state}")
            time.sleep(POLL_SECONDS)
        raise RemoteEngineError("the engine did not become ready")

    def ready(self) -> bool:
        return str(self.engine_state().get("state")) == READY

    def generate(
        self, prompt: str, *, max_tokens: int, stop: Sequence[str], seed: int | None
    ) -> dict[str, Any]:
        started = self.request(
            "POST",
            "/calls",
            {
                "prompt": prompt,
                "max_tokens": int(max_tokens),
                "stop": [str(s) for s in stop],
                "seed": seed,
                "limit_seconds": CALL_LIMIT_SECONDS,
            },
        )
        found = self.wait_task(str(started["task"]), CALL_LIMIT_SECONDS + GRACE_SECONDS)
        state = str(found.get("state"))
        if state == "hang":
            raise RemoteEngineHang(str(found.get("reason") or "the engine hung"))
        if state != DONE:
            raise RemoteEngineError(str(found.get("reason") or f"the call ended as {state}"))
        return found

    def cleanup(self) -> dict[str, Any]:
        return dict(self.request("POST", "/cleanup", {}))


def completion(found: Mapping[str, Any], max_tokens: int) -> dict[str, Any]:
    ids = [int(t) for t in found.get("ids") or []]
    prompt_tokens = int(found.get("prompt_tokens") or 0)
    produced = int(found.get("completion_tokens") or len(ids))
    limit_reached = bool(max_tokens) and produced >= max_tokens
    return {
        "id": "remote",
        "object": "text_completion",
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "text": str(found.get("text") or ""),
                "token_ids": ids,
                "finish_reason": str(found.get("finish_reason") or "")
                or ("length" if limit_reached else "stop"),
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": produced,
            "total_tokens": prompt_tokens + produced,
        },
    }


def _stops(body: Mapping[str, Any]) -> list[str]:
    stop = body.get("stop") or []
    return [stop] if isinstance(stop, str) else [str(s) for s in stop]


class EngineBridge:
    def __init__(self, engine: RemoteEngine, host: str = "127.0.0.1", port: int = 0) -> None:
        self.engine = engine
        self._server = ThreadingHTTPServer((host, port), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{int(port)}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                log.debug("engine bridge: " + fmt, *args)

            def _send(self, status: int, raw: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _json(self, status: int, payload: Mapping[str, Any]) -> None:
                self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

            def do_GET(self) -> None:
                if urlparse(self.path).path != MODELS_PATH:
                    self._json(404, {"error": "no such route"})
                    return
                try:
                    ready = bridge.engine.ready()
                except RemoteEngineError as exc:
                    self._json(503, {"error": str(exc)[:300]})
                    return
                if not ready:
                    self._json(503, {"error": "the engine is not ready"})
                    return
                self._json(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})

            def do_POST(self) -> None:
                if urlparse(self.path).path != contract.SLM_COMPLETIONS:
                    self._json(404, {"error": "no such route"})
                    return
                length = int(self.headers.get("content-length") or 0)
                if length > MAX_BODY:
                    self._json(413, {"error": "body too large"})
                    return
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    prompt = str(body["prompt"])
                    max_tokens = int(body.get("max_tokens") or 0)
                    seed = body.get("seed")
                except (KeyError, TypeError, ValueError, AttributeError):
                    self._json(400, {"error": "malformed request"})
                    return
                try:
                    found = bridge.engine.generate(
                        prompt,
                        max_tokens=max_tokens,
                        stop=_stops(body),
                        seed=int(seed) if seed is not None else None,
                    )
                except RemoteEngineHang as exc:
                    self._json(504, {"error": str(exc)[:300]})
                    return
                except RemoteEngineError as exc:
                    self._json(502, {"error": str(exc)[:300]})
                    return
                answer = completion(found, max_tokens)
                if not body.get("stream"):
                    self._json(200, answer)
                    return
                stream = f"data: {json.dumps(answer)}\n\ndata: [DONE]\n\n".encode()
                self._send(200, stream, "text/event-stream")

        return Handler
