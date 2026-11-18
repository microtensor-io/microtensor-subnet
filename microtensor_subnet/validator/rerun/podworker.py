from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final

MAX_BODY: Final[int] = 32 * 1024 * 1024
MAX_ACTIVE_CALLS: Final[int] = 64
MAX_KEPT_TASKS: Final[int] = 2048
ENGINE_PORT: Final[int] = 8000
ROUTER_PORT: Final[int] = 9800
READY_SECONDS: Final[float] = 900.0
ROUTER_READY_SECONDS: Final[float] = 600.0
STOP_GRACE_SECONDS: Final[float] = 15.0
CHUNK: Final[int] = 1024 * 1024
FETCH_ATTEMPTS: Final[int] = 6
CLEAN_GPU_BYTES: Final[int] = 1536 * 1024 * 1024
MIB: Final[int] = 1024**2
ID_PREFIX: Final[str] = "token_id:"
ENGINE_SEED_MASK: Final[int] = 2**63 - 1
MODEL_NAME: Final[str] = "slm"
NOTE_CHARS: Final[int] = 30000
USER_AGENT: Final[str] = "microtensor-pod"
HEX64: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
REPO: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
ENV_KEYS: Final[re.Pattern[str]] = re.compile(r"^(VLLM_|HF_HUB_|CUDA_VISIBLE)[A-Z0-9_]*$")
SYSTEM_DIR: Final[Path] = Path("/opt/system")


class Refused(Exception):
    def __init__(self, detail: str, status: int = 400) -> None:
        super().__init__(detail)
        self.status = status


class BadFile(RuntimeError):
    pass


def safe_relative(name: str) -> str:
    text = str(name)
    parts = text.split("/")
    if not text.strip() or text.startswith("/") or "\\" in text or ".." in parts or "" in parts:
        raise BadFile(f"unsafe file name {text[:120]!r}")
    return text


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, part: Path) -> None:
    failure = "the download did not finish"
    for attempt in range(FETCH_ATTEMPTS):
        offset = part.stat().st_size if part.is_file() else 0
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = bool(offset) and response.status == 206
                expected = int(response.headers.get("Content-Length") or 0)
                written = 0
                with part.open("ab" if resumed else "wb") as handle:
                    while chunk := response.read(CHUNK):
                        handle.write(chunk)
                        written += len(chunk)
            if not expected or written >= expected:
                return
            failure = f"{url} ended after {written} of {expected} bytes"
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                part.unlink(missing_ok=True)
                failure = "the server refused the resume"
            elif exc.code in (401, 403, 404):
                part.unlink(missing_ok=True)
                raise BadFile(f"{url} answered {exc.code}") from exc
            else:
                failure = f"{url} answered {exc.code}"
        except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
            failure = f"{type(exc).__name__}: {str(exc)[:160]}"
        time.sleep(min(30.0, 3.0 * (attempt + 1)))
    raise OSError(failure)


def fetch_url(url: str, target: Path, sha: str = "") -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and (not sha or file_sha(target) == sha):
        return
    part = target.with_name(target.name + ".part")
    for _ in range(2):
        download(url, part)
        if not sha or file_sha(part) == sha:
            os.replace(part, target)
            return
        part.unlink(missing_ok=True)
    raise BadFile(f"{target.name} does not match its sha256")


def kill_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=STOP_GRACE_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=STOP_GRACE_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            pass


def spawn(argv: list[str], env: Mapping[str, str], log_path: Path) -> subprocess.Popen[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        return subprocess.Popen(  # noqa: S603
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **env},
            start_new_session=os.name == "posix",
        )


def tail(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - NOTE_CHARS))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def gpu_query(field: str) -> str:
    try:
        found = subprocess.run(  # noqa: S603
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = found.stdout.strip().splitlines()
    return lines[0].strip() if found.returncode == 0 and lines else ""


def gpu_bytes(field: str) -> int:
    value = gpu_query(field)
    return int(value) * MIB if value.isdigit() else 0


def ids_from(choice: Mapping[str, Any]) -> list[int]:
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


def generate(body: Mapping[str, Any], limit: float) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": MODEL_NAME,
        "prompt": str(body["prompt"]),
        "max_tokens": int(body["max_tokens"]),
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": 1,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
    }
    stop = [str(s) for s in body.get("stop") or []]
    if stop:
        request["stop"] = stop
    if body.get("seed") is not None:
        request["seed"] = int(body["seed"]) & ENGINE_SEED_MASK
    started = time.perf_counter()
    deadline = started + limit
    first: float | None = None
    text: list[str] = []
    ids: list[int] = []
    pieces = 0
    usage: dict[str, Any] = {}
    finish = ""
    connection = http.client.HTTPConnection("127.0.0.1", ENGINE_PORT, timeout=limit)
    try:
        connection.request(
            "POST",
            "/v1/completions",
            body=json.dumps(request).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read(600).decode("utf-8", "replace")
            return {"state": "error", "reason": f"the engine answered {response.status}: {detail}"}
        while True:
            raw = response.readline()
            if not raw:
                break
            if time.perf_counter() > deadline:
                return {"state": "hang", "reason": f"no answer within {limit:.0f} s"}
            line = raw.decode("utf-8", "replace").strip()
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
                ids += ids_from(choice)
                if choice.get("finish_reason"):
                    finish = str(choice["finish_reason"])
    except TimeoutError:
        return {"state": "hang", "reason": f"no answer within {limit:.0f} s"}
    except (OSError, http.client.HTTPException, ValueError) as exc:
        return {"state": "error", "reason": f"the engine could not be reached: {exc}"[:400]}
    finally:
        connection.close()
    ended = time.perf_counter()
    total_ms = (ended - started) * 1000
    ttft_ms = ((first or ended) - started) * 1000
    produced = int(usage.get("completion_tokens") or pieces)
    return {
        "state": "done",
        "text": "".join(text),
        "ids": ids,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": produced,
        "ttft_ms": ttft_ms,
        "tpot_ms": (total_ms - ttft_ms) / (produced - 1) if produced > 1 else 0.0,
        "total_ms": total_ms,
        "finish_reason": finish,
    }


class Pod:
    def __init__(self, home: Path, mirror: str) -> None:
        self.home = home
        self.mirror = mirror.rstrip("/")
        self.jobs = home / "jobs"
        self.kept = home / "reference"
        self.lock = threading.Lock()
        self.tasks: dict[str, dict[str, Any]] = {}
        self.active_calls = 0
        self.engine: subprocess.Popen[bytes] | None = None
        self.engine_state = "none"
        self.engine_body: dict[str, Any] | None = None
        self.engine_generation = 0
        self.router: subprocess.Popen[bytes] | None = None
        self.peak_bytes = 0
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.kept.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._sample, daemon=True).start()

    def _sample(self) -> None:
        while True:
            if self.engine_state in ("starting", "ready"):
                self.peak_bytes = max(self.peak_bytes, gpu_bytes("memory.used"))
            time.sleep(2.0)

    def _task(self, state: str = "running", **extra: Any) -> str:
        key = uuid.uuid4().hex
        with self.lock:
            while len(self.tasks) >= MAX_KEPT_TASKS:
                self.tasks.pop(next(iter(self.tasks)))
            self.tasks[key] = {"state": state, **extra}
        return key

    def _finish(self, key: str, **values: Any) -> None:
        with self.lock:
            if key in self.tasks:
                self.tasks[key].update(values)

    def task(self, key: str) -> dict[str, Any]:
        with self.lock:
            found = self.tasks.get(key)
            return dict(found) if found is not None else {}

    def directory(self, kind: str, name: str) -> Path:
        if not NAME.match(name) or name in (".", ".."):
            raise Refused("the name is not valid")
        if kind == "reference":
            return self.kept / name
        if kind == "miner":
            return self.jobs / name
        raise Refused("kind must be miner or reference")

    def prepare(self, body: Mapping[str, Any]) -> str:
        into = self.directory(str(body.get("kind", "")), str(body.get("name", "")))
        files = body.get("files")
        hub = body.get("hub")
        if isinstance(files, Mapping):
            for name, sha in files.items():
                safe_relative(str(name))
                if not HEX64.match(str(sha).lower()):
                    raise Refused("every file needs a sha256")
        elif isinstance(hub, Mapping):
            if not REPO.match(str(hub.get("repo", ""))):
                raise Refused("the repo is not valid")
        else:
            raise Refused("send files or hub")
        key = self._task(path=str(into))
        threading.Thread(target=self._prepare, args=(key, into, files, hub), daemon=True).start()
        return key

    def _prepare(self, key: str, into: Path, files: Any, hub: Any) -> None:
        try:
            into.mkdir(parents=True, exist_ok=True)
            if files is not None:
                for name, sha in files.items():
                    fetch_url(
                        f"{self.mirror}/v2/mirror/{str(sha).lower()}",
                        into / safe_relative(str(name)),
                        str(sha).lower(),
                    )
            else:
                self._prepare_hub(into, hub)
        except BadFile as exc:
            self._finish(key, state="hash", reason=str(exc)[:400])
        except Exception as exc:
            self._finish(key, state="failed", reason=f"{type(exc).__name__}: {exc}"[:400])
        else:
            self._finish(key, state="done")

    def _prepare_hub(self, into: Path, hub: Mapping[str, Any]) -> None:
        repo = str(hub["repo"])
        revision = re.sub(r"[^A-Za-z0-9._-]", "", str(hub.get("revision") or "main"))
        marker = into / ".complete"
        if marker.is_file():
            return
        request = urllib.request.Request(
            f"https://huggingface.co/api/models/{repo}/revision/{revision}",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            listed = json.loads(response.read().decode("utf-8"))
        for row in listed.get("siblings") or []:
            name = safe_relative(str(row.get("rfilename", "")))
            fetch_url(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", into / name)
        marker.write_text("ok", encoding="utf-8")

    def baked_model(self, wanted: str) -> Path:
        model = Path(wanted)
        if not model.is_absolute() or SYSTEM_DIR not in model.parents:
            raise Refused("a baked model must live under the system folder")
        if not model.is_dir():
            raise Refused("this image holds no baked model there", 409)
        return model

    def start_engine(self, body: Mapping[str, Any]) -> dict[str, Any]:
        baked = str(body.get("model_dir") or "")
        if baked:
            model = self.baked_model(baked)
        else:
            model = self.directory(str(body.get("kind", "")), str(body.get("name", "")))
            if not model.is_dir():
                raise Refused("that model has not been prepared", 409)
        argv = [str(a) for a in body.get("argv") or []]
        if "--model" not in argv or argv.index("--model") + 1 >= len(argv):
            raise Refused("the engine arguments name no model")
        argv[argv.index("--model") + 1] = str(model)
        env = {
            str(k): str(v) for k, v in dict(body.get("env") or {}).items() if ENV_KEYS.match(str(k))
        }
        command = shlex.split(os.environ.get("MT_POD_ENGINE", "vllm serve"))
        self.stop_engine()
        with self.lock:
            self.engine_generation += 1
            generation = self.engine_generation
            self.engine_body = dict(body)
            self.engine_state = "starting"
            self.peak_bytes = 0
        process = spawn([*command, *argv], env, self.home / "engine.log")
        with self.lock:
            self.engine = process
        threading.Thread(target=self._watch_engine, args=(process, generation), daemon=True).start()
        return self.engine_view()

    def _watch_engine(self, process: subprocess.Popen[bytes], generation: int) -> None:
        deadline = time.monotonic() + READY_SECONDS
        while time.monotonic() < deadline:
            with self.lock:
                if generation != self.engine_generation:
                    return
            if process.poll() is not None:
                self._set_engine(generation, "exited")
                return
            try:
                connection = http.client.HTTPConnection("127.0.0.1", ENGINE_PORT, timeout=5)
                connection.request("GET", "/v1/models")
                status = connection.getresponse().status
                connection.close()
            except (OSError, http.client.HTTPException):
                status = 0
            if status == 200:
                self._set_engine(generation, "ready")
                break
            time.sleep(2.0)
        else:
            self._set_engine(generation, "stuck")
            return
        while process.poll() is None:
            with self.lock:
                if generation != self.engine_generation:
                    return
            time.sleep(2.0)
        self._set_engine(generation, "exited")

    def _set_engine(self, generation: int, state: str) -> None:
        with self.lock:
            if generation == self.engine_generation:
                self.engine_state = state

    def stop_engine(self) -> dict[str, Any]:
        with self.lock:
            process = self.engine
            self.engine = None
            self.engine_generation += 1
            self.engine_state = "none"
        if process is not None:
            kill_tree(process)
        return self.engine_view()

    def restart_engine(self) -> dict[str, Any]:
        with self.lock:
            body = self.engine_body
        if body is None:
            raise Refused("no engine has been started", 409)
        return self.start_engine(body)

    def engine_view(self) -> dict[str, Any]:
        with self.lock:
            state = self.engine_state
        note = tail(self.home / "engine.log") if state in ("exited", "stuck") else ""
        return {"state": state, "note": note, "peak_bytes": self.peak_bytes}

    def start_call(self, body: Mapping[str, Any]) -> str:
        if not isinstance(body.get("prompt"), str) or not isinstance(body.get("max_tokens"), int):
            raise Refused("a call needs a prompt and max_tokens")
        with self.lock:
            if self.active_calls >= MAX_ACTIVE_CALLS:
                raise Refused("too many calls are running", 429)
            self.active_calls += 1
        limit = max(1.0, min(float(body.get("limit_seconds") or 600.0), 3600.0))
        key = self._task()
        threading.Thread(target=self._call, args=(key, dict(body), limit), daemon=True).start()
        return key

    def _call(self, key: str, body: dict[str, Any], limit: float) -> None:
        try:
            found = generate(body, limit)
        except Exception as exc:
            found = {"state": "error", "reason": f"{type(exc).__name__}: {exc}"[:400]}
        finally:
            with self.lock:
                self.active_calls -= 1
        self._finish(key, **found)

    def measure_router(self, body: Mapping[str, Any]) -> str:
        model = self.directory(str(body.get("kind", "")), str(body.get("name", "")))
        pairs = body.get("pairs")
        code = body.get("code")
        if not model.is_dir():
            raise Refused("that router has not been prepared", 409)
        if not isinstance(pairs, list) or not isinstance(code, Mapping):
            raise Refused("send pairs and code")
        key = self._task()
        threading.Thread(
            target=self._measure, args=(key, model, pairs, code, dict(body)), daemon=True
        ).start()
        return key

    def _measure(
        self, key: str, model: Path, pairs: list[Any], code: Any, body: dict[str, Any]
    ) -> None:
        root = self.jobs / f"router-code-{key[:8]}"
        process: subprocess.Popen[bytes] | None = None
        log = self.home / "router.log"
        try:
            for name, blob in code.items():
                target = root / safe_relative(str(name))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(str(blob)))
            argv = [
                sys.executable,
                "-m",
                str(body.get("module") or "microtensor_subnet.validator.sandbox.router_server"),
                "--model",
                str(model),
                "--port",
                str(ROUTER_PORT),
                "--listen",
                "127.0.0.1",
                "--max-input-tokens",
                str(int(body.get("max_input_tokens") or 4096)),
                "--max-parameters",
                str(int(body.get("max_parameters") or 0)),
                "--device",
                "cuda",
            ]
            process = spawn(argv, {"PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": "0"}, log)
            with self.lock:
                self.router = process
            self._wait_router(process, log)
            latencies = [self._router_call(pair) for pair in pairs]
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"[:400]
            self._finish(key, state="failed", reason=reason, note=tail(log)[-600:])
        else:
            self._finish(key, state="done", latencies=latencies)
        finally:
            if process is not None:
                kill_tree(process)
            with self.lock:
                self.router = None
            shutil.rmtree(root, ignore_errors=True)

    def _wait_router(self, process: subprocess.Popen[bytes], log: Path) -> None:
        deadline = time.monotonic() + ROUTER_READY_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"the router exited: {tail(log)[-400:]}")
            try:
                connection = http.client.HTTPConnection("127.0.0.1", ROUTER_PORT, timeout=5)
                connection.request("GET", "/health")
                status = connection.getresponse().status
                connection.close()
            except (OSError, http.client.HTTPException):
                status = 0
            if status == 200:
                return
            time.sleep(2.0)
        raise RuntimeError("the router did not become ready")

    def _router_call(self, pair: Any) -> float:
        connection = http.client.HTTPConnection("127.0.0.1", ROUTER_PORT, timeout=60)
        try:
            connection.request(
                "POST",
                "/decide",
                body=json.dumps(
                    {"input": {"query": str(pair["query"]), "draft": str(pair["draft"])}}
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            raw = response.read()
            if response.status != 200:
                raise RuntimeError(f"the router answered {response.status}")
            return float(json.loads(raw.decode("utf-8")).get("latency_ms") or 0.0)
        finally:
            connection.close()

    def cleanup(self) -> dict[str, Any]:
        self.stop_engine()
        with self.lock:
            router = self.router
            self.router = None
            self.engine_body = None
            self.tasks.clear()
        if router is not None:
            kill_tree(router)
        for child in list(self.jobs.iterdir()):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        (self.home / "engine.log").unlink(missing_ok=True)
        (self.home / "router.log").unlink(missing_ok=True)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not self.clean()["clean"]:
            time.sleep(2.0)
        return self.clean()

    def clean(self) -> dict[str, Any]:
        problems: list[str] = []
        with self.lock:
            if self.engine is not None or self.engine_state != "none":
                problems.append("an engine is running")
            if self.router is not None:
                problems.append("a router is running")
            if self.active_calls:
                problems.append("calls are running")
        if any(self.jobs.iterdir()):
            problems.append("job files remain")
        used = gpu_query("memory.used")
        if used.isdigit() and int(used) * MIB > CLEAN_GPU_BYTES:
            problems.append(f"the GPU holds {used} MiB")
        return {"clean": not problems, "problems": problems}

    def health(self) -> dict[str, Any]:
        disk = shutil.disk_usage(self.home)
        return {
            "gpu_name": gpu_query("name"),
            "memory_total_bytes": gpu_bytes("memory.total"),
            "memory_used_bytes": gpu_bytes("memory.used"),
            "disk_free_bytes": disk.free,
            "engine": self.engine_view(),
            **self.clean(),
        }


def make_handler(pod: Pod, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send(self, status: int, payload: Mapping[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _authorised(self) -> bool:
            sent = self.headers.get("Authorization", "")
            return bool(token) and hmac.compare_digest(sent, f"Bearer {token}")

        def _body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError as exc:
                raise Refused("bad length") from exc
            if length < 0 or length > MAX_BODY:
                self.close_connection = True
                raise Refused("the body is too large", 413)
            try:
                found = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            except (ValueError, UnicodeDecodeError) as exc:
                raise Refused("the body is not JSON") from exc
            if not isinstance(found, dict):
                raise Refused("the body is a JSON object")
            return found

        def _route(self, method: str) -> tuple[int, dict[str, Any]]:
            path = self.path.split("?", 1)[0].rstrip("/")
            if method == "GET":
                if path == "/health":
                    return 200, pod.health()
                if path == "/engine":
                    return 200, pod.engine_view()
                if path.startswith("/tasks/"):
                    found = pod.task(path[len("/tasks/") :])
                    return (200, found) if found else (404, {"detail": "no such task"})
                return 404, {"detail": "not found"}
            body = self._body()
            if path == "/prepare":
                return 202, {"task": pod.prepare(body)}
            if path == "/engine/start":
                return 202, pod.start_engine(body)
            if path == "/engine/stop":
                return 200, pod.stop_engine()
            if path == "/engine/restart":
                return 202, pod.restart_engine()
            if path == "/calls":
                return 202, {"task": pod.start_call(body)}
            if path == "/router/measure":
                return 202, {"task": pod.measure_router(body)}
            if path == "/cleanup":
                return 200, pod.cleanup()
            return 404, {"detail": "not found"}

        def _serve(self, method: str) -> None:
            if not self._authorised():
                self.close_connection = True
                self._send(401, {"detail": "unauthorised"})
                return
            try:
                status, payload = self._route(method)
            except Refused as exc:
                status, payload = exc.status, {"detail": str(exc)}
            except BadFile as exc:
                status, payload = 400, {"detail": str(exc)}
            except Exception as exc:
                status, payload = 500, {"detail": f"{type(exc).__name__}: {exc}"[:400]}
            self._send(status, payload)

        def do_GET(self) -> None:
            self._serve("GET")

        def do_POST(self) -> None:
            self._serve("POST")

    return Handler


def main() -> int:
    token = os.environ.get("MT_POD_TOKEN", "")
    if not token:
        print("MT_POD_TOKEN is not set", file=sys.stderr)
        return 2
    pod = Pod(
        Path(os.environ.get("MT_POD_HOME", "/workspace/mt")),
        os.environ.get("MT_MIRROR_URL", "https://api.microtensor.cloud"),
    )
    server = ThreadingHTTPServer(
        ("0.0.0.0", int(os.environ.get("MT_POD_PORT", "8010"))),  # noqa: S104
        make_handler(pod, token),
    )
    server.daemon_threads = True

    def stop(*_: Any) -> None:
        pod.stop_engine()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
