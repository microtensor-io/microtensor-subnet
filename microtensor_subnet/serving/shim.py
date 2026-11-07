from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

import httpx

from microtensor_subnet.artifact import contract
from microtensor_subnet.artifact.validate import bare_hex, weights_sha256
from microtensor_subnet.core.hashing import digest_file
from microtensor_subnet.serving.host.answer import Held, answer
from microtensor_subnet.validator.sandbox import (
    GATEWAY_PORT,
    MAIN_PORT,
    Sandbox,
    gateway,
    router_server,
)
from microtensor_subnet.validator.sandbox.services import ArtifactSession, Services
from microtensor_subnet.validator.sandbox.slm import vllm_command

log = logging.getLogger("microtensor_subnet.serving.shim")

ROOT: Final[str] = "/opt/system"
BUILD: Final[str] = "build.json"
MANIFEST: Final[str] = "manifest.json"
ROOT_ENV: Final[str] = "MT_SYSTEM_DIR"
BIND_ENV: Final[str] = "MT_SHIM_BIND"
PORT_ENV: Final[str] = "MT_SHIM_PORT"
WORKERS_ENV: Final[str] = "MT_SHIM_WORKERS"
DEFAULT_WORKERS: Final[int] = 8
DEFAULT_MAX_SEQS: Final[int] = 16
LOOPBACK: Final[str] = "127.0.0.1"
ENGINE_PORT: Final[int] = 8000
ROUTER_PORT: Final[int] = 8011
SERVICES_PORT: Final[int] = 9701
SHIM_PORT: Final[int] = 8800
SESSION_KEY: Final[str] = "shim"
READY_SECONDS: Final[float] = 1800.0
REQUEST_SECONDS: Final[float] = 120.0
MAX_BODY: Final[int] = 8 * 1024 * 1024
RUN_PATH: Final[str] = "/v1/run"
CHAT_PATH: Final[str] = "/v1/chat/completions"
HEALTH_PATH: Final[str] = "/health"
METRICS_PATH: Final[str] = "/metrics"
WEIGHTS_MISMATCH: Final[int] = 3


class ShimError(RuntimeError):
    pass


@dataclass(slots=True)
class Metrics:
    ok: int = 0
    failed: int = 0
    in_flight: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def text(self) -> str:
        with self.lock:
            rows = [
                f'mt_shim_requests_total{{outcome="ok"}} {self.ok}',
                f'mt_shim_requests_total{{outcome="error"}} {self.failed}',
                f"mt_shim_in_flight {self.in_flight}",
                f"mt_shim_ttft_ms_sum {self.ttft_ms:.1f}",
                f"mt_shim_total_ms_sum {self.total_ms:.1f}",
            ]
        return "\n".join(rows) + "\n"


def read_build(root: Path) -> dict[str, Any]:
    path = root / BUILD
    if not path.is_file():
        path = root / MANIFEST
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ShimError(f"the build record {path} is unreadable: {exc}") from exc
    if not isinstance(found, Mapping):
        raise ShimError("the build record is not an object")
    return dict(found)


def weights_of(slm_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(slm_dir).as_posix()): digest_file(path)
        for path in sorted(slm_dir.rglob("*"))
        if path.is_file()
    }


def check_weights(build: Mapping[str, Any], slm_dir: Path) -> None:
    expected = bare_hex(str(build.get("weights_sha256") or build.get("weights_hash") or ""))
    if not expected:
        raise ShimError("the build record names no weights hash")
    files = weights_of(slm_dir)
    listed = build.get("slm_files")
    if isinstance(listed, Mapping):
        for name, digest in listed.items():
            if bare_hex(files.get(str(name), "")) != bare_hex(str(digest)):
                raise ShimError(f"the weights hash of {name} differs from the build record")
    found = weights_sha256(files)
    if found != expected:
        raise ShimError(f"the weights hash is {found[:12]}, the build record says {expected[:12]}")


def engine_argv(
    build: Mapping[str, Any], slm_dir: Path, workers: int = DEFAULT_WORKERS
) -> list[str]:
    engine = build.get("engine")
    flags = list(engine.get("flags") or []) if isinstance(engine, Mapping) else []
    if flags:
        return [str(flag) for flag in flags]
    return vllm_command(
        str(slm_dir),
        max_model_len=int(build.get("max_model_len") or 8192),
        max_seqs=int(build.get("max_seqs") or max(DEFAULT_MAX_SEQS, workers)),
        dtype=str(build.get("dtype") or ""),
        prefix_caching=False,
    )


def engine_env(build: Mapping[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "TRANSFORMERS_OFFLINE": "1"})
    engine = build.get("engine")
    extra = engine.get("env") if isinstance(engine, Mapping) else None
    if isinstance(extra, Mapping):
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def _wait(url: str, seconds: float, alive: Any = None) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if alive is not None and alive.poll() is not None:
            raise ShimError(f"a process exited with {alive.returncode} before it was ready")
        try:
            with httpx.Client(timeout=5.0) as http:
                if http.get(url).status_code < 500:
                    return
        except httpx.HTTPError:
            pass
        time.sleep(3.0)
    raise ShimError(f"{url} never became ready")


def start_engine(
    build: Mapping[str, Any], slm_dir: Path, workers: int = DEFAULT_WORKERS
) -> subprocess.Popen[bytes]:
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    argv += engine_argv(build, slm_dir, workers)
    process = subprocess.Popen(argv, env=engine_env(build))  # noqa: S603
    _wait(f"http://{LOOPBACK}:{ENGINE_PORT}/health", READY_SECONDS, process)
    return process


def start_router(root: Path, build: Mapping[str, Any]) -> str:
    router_dir = root / "artifact" / contract.ROUTER_DIR
    if not (router_dir / "config.json").is_file():
        return ""
    config = dict(build.get("config") or {})
    wanted = int((config.get("router") or {}).get("max_input_tokens") or 4096)
    model = router_server.RouterModel(router_dir, wanted, "cpu")
    server = ThreadingHTTPServer((LOOPBACK, ROUTER_PORT), router_server.build(model))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://{LOOPBACK}:{ROUTER_PORT}"


def start_gateway(services_port: int, key: str = SESSION_KEY, port: int = GATEWAY_PORT) -> str:
    handler = gateway.build(f"http://{LOOPBACK}:{services_port}", key)
    server = ThreadingHTTPServer((LOOPBACK, port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://{LOOPBACK}:{port}"


def spawn_harness(root: Path, base: str, port: int = MAIN_PORT) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env.update(
        {
            contract.ENV_PORT: str(port),
            contract.ENV_SLM_URL: base + "/slm",
            contract.ENV_ROUTER_URL: base + "/router",
            contract.ENV_TOOLS_URL: base,
            contract.ENV_RELAY_URL: base + "/relay",
            "HOME": "/tmp",  # noqa: S108
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    main = root / "artifact" / contract.MAIN_FILE
    return subprocess.Popen(  # noqa: S603
        [sys.executable, str(main)], env=env, cwd=str(main.parent)
    )


def entry_of(build: Mapping[str, Any]) -> dict[str, Any]:
    engine = build.get("engine")
    digest = str(engine.get("digest") or "") if isinstance(engine, Mapping) else ""
    return {
        "model": str(build.get("model") or f"mt/{build.get('arena', '')}"),
        "artifact_digest": str(build.get("artifact_digest") or build.get("artifact_sha") or ""),
        "arena": str(build.get("arena") or ""),
        "slm_prompt": str(build.get("slm_prompt") or os.environ.get("MT_SLM_PROMPT", "")),
        "grader": str(build.get("grader") or os.environ.get("MT_GRADER", "")),
        "envelope": dict(build.get("envelope") or {}),
        "image_digest": str(build.get("image_digest") or digest),
    }


@dataclass(slots=True)
class Shim:
    root: Path
    build: dict[str, Any]
    metrics: Metrics = field(default_factory=Metrics)
    held: Held | None = None
    engine: subprocess.Popen[bytes] | None = None
    harnesses: list[subprocess.Popen[bytes]] = field(default_factory=list)
    services: Services | None = None
    request_seconds: float = REQUEST_SECONDS
    workers: int = DEFAULT_WORKERS
    idle: queue.Queue[Held] = field(default_factory=queue.Queue)

    @property
    def ready(self) -> bool:
        return (
            self.held is not None
            and self.engine is not None
            and self.engine.poll() is None
            and bool(self.harnesses)
            and all(process.poll() is None for process in self.harnesses)
        )

    def start(self) -> None:
        slm_dir = self.root / "slm"
        check_weights(self.build, slm_dir)
        self.engine = start_engine(self.build, slm_dir, self.workers)
        router_url = start_router(self.root, self.build)
        self.services = Services(LOOPBACK, SERVICES_PORT)
        self.services.start()
        entry = entry_of(self.build)
        built = [self.add_worker(index, router_url, entry) for index in range(self.workers)]
        for index, process in enumerate(self.harnesses):
            _wait(f"http://{LOOPBACK}:{MAIN_PORT + index}/health", READY_SECONDS, process)
        for held in built:
            self.idle.put(held)
        self.held = built[0]
        log.info("certified image ready for %s with %d workers", entry["model"], self.workers)

    def add_worker(self, index: int, router_url: str, entry: dict[str, Any]) -> Held:
        if self.services is None:
            raise ShimError("the services are not started")
        key = f"{SESSION_KEY}-{index}"
        session = ArtifactSession(
            sample_database=b"",
            fault_plan=None,
            slm_url=f"http://{LOOPBACK}:{ENGINE_PORT}",
            router_url=router_url,
            desk=None,
        )
        self.services.attach(key, session)
        base = start_gateway(self.services.port, key, GATEWAY_PORT + index)
        self.harnesses.append(spawn_harness(self.root, base, MAIN_PORT + index))
        sandbox = Sandbox(
            key=key,
            network="",
            slm_container="",
            gateway_container="",
            main_container="",
            router_process=None,
            slm_url=session.slm_url,
            router_url=router_url,
            main_url=f"http://{LOOPBACK}:{MAIN_PORT + index}",
            session=session,
            gpu="0",
        )
        return Held(
            model=entry["model"],
            artifact_digest=entry["artifact_digest"],
            entry=entry,
            sandbox=sandbox,
            gpu="0",
            slot=index,
        )

    def run(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self.held is None:
            return {"error": "the system is still starting"}
        started = time.perf_counter()
        try:
            worker = self.idle.get(timeout=self.request_seconds)
        except queue.Empty:
            return {"error": "every worker stayed busy"}
        with self.metrics.lock:
            self.metrics.in_flight += 1
        try:
            found = answer(worker, request, timeout=self.request_seconds)
        finally:
            self.idle.put(worker)
            elapsed = (time.perf_counter() - started) * 1000
            with self.metrics.lock:
                self.metrics.in_flight -= 1
                self.metrics.total_ms += elapsed
        with self.metrics.lock:
            if found.get("error"):
                self.metrics.failed += 1
            else:
                self.metrics.ok += 1
                self.metrics.ttft_ms += float(found.get("latency_ms") or elapsed)
        found.setdefault("ttft_ms", float(found.get("latency_ms") or elapsed))
        found["artifact_digest"] = self.held.artifact_digest
        found["image_digest"] = str(self.held.entry.get("image_digest") or "")
        return found

    def chat(self, body: Mapping[str, Any]) -> tuple[int, bytes]:
        asked = dict(body)
        asked["temperature"] = 0.0
        asked.pop("top_p", None)
        asked["seed"] = 0
        try:
            with httpx.Client(timeout=self.request_seconds) as http:
                response = http.post(f"http://{LOOPBACK}:{ENGINE_PORT}{CHAT_PATH}", json=asked)
        except httpx.HTTPError as exc:
            detail = {"error": f"engine unreachable: {exc.__class__.__name__}"}
            return 502, json.dumps(detail).encode("utf-8")
        return response.status_code, response.content

    def stop(self) -> None:
        for process in (*self.harnesses, self.engine):
            if process is not None and process.poll() is None:
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=30)
        if self.services is not None:
            self.services.stop()


def build_handler(shim: Shim) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            log.debug("shim: " + fmt, *args)

        def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"))

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == HEALTH_PATH:
                if shim.ready:
                    self._json(200, {"status": "ok", "model": shim.held.model if shim.held else ""})
                else:
                    self._json(503, {"status": "starting"})
                return
            if path == METRICS_PATH:
                self._send(200, shim.metrics.text().encode("utf-8"), "text/plain; version=0.0.4")
                return
            self._json(404, {"error": "no such route"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            length = int(self.headers.get("content-length") or 0)
            if length > MAX_BODY:
                self._json(413, {"error": "body too large"})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._json(400, {"error": "the body is not json"})
                return
            if not isinstance(body, Mapping):
                self._json(400, {"error": "the body is not an object"})
                return
            if path == RUN_PATH:
                self._json(200, shim.run(body))
                return
            if path == CHAT_PATH:
                status, content = shim.chat(body)
                self._send(status, content)
                return
            self._json(404, {"error": "no such route"})

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mt-shim")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get(ROOT_ENV, ROOT)))
    parser.add_argument("--listen", default=os.environ.get(BIND_ENV, LOOPBACK))
    parser.add_argument("--port", type=int, default=int(os.environ.get(PORT_ENV, SHIM_PORT)))
    parser.add_argument("--request-seconds", type=float, default=REQUEST_SECONDS)
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get(WORKERS_ENV) or DEFAULT_WORKERS)
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        shim = Shim(
            args.root,
            read_build(args.root),
            request_seconds=args.request_seconds,
            workers=max(1, args.workers),
        )
    except ShimError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    server = ThreadingHTTPServer((args.listen, args.port), build_handler(shim))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        shim.start()
    except ShimError as exc:
        print(str(exc), file=sys.stderr)
        shim.stop()
        return WEIGHTS_MISMATCH if "weights hash" in str(exc) else 2
    try:
        while shim.ready:
            time.sleep(5.0)
    except KeyboardInterrupt:
        pass
    finally:
        shim.stop()
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
