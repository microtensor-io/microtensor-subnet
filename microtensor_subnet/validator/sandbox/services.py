from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final
from urllib.parse import urlparse

import httpx

from microtensor_subnet.artifact import contract
from microtensor_subnet.core import transcript
from microtensor_subnet.validator import receipts
from microtensor_subnet.validator.relay import RelayError, TicketDesk
from microtensor_subnet.validator.sandbox import tools

log = logging.getLogger("microtensor_subnet.validator.sandbox.services")

MAX_BODY: Final[int] = 4 * 1024 * 1024
UPSTREAM_SECONDS: Final[float] = 300.0
SLM_PREFIX: Final[str] = "/slm"
ROUTER_PREFIX: Final[str] = "/router"
RELAY_PREFIX: Final[str] = "/relay"
LIVE: Final[str] = "live"
RECORD: Final[str] = "record"
REPLAY: Final[str] = "replay"
MISMATCH_STATUS: Final[int] = 409
ID_PREFIX: Final[str] = "token_id:"
FAULT_STATUS: Final[dict[str, int]] = {"timeout": 504, "error": 500}
FAULT_BODY: Final[dict[str, bytes]] = {
    "timeout": b'{"error": "model timed out"}',
    "error": b'{"error": "internal model failure"}',
}


class ReplayMismatch(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Mismatch:
    item_id: str
    seq: int
    what: str

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "seq": self.seq, "what": self.what}


def completion_body(call: Mapping[str, Any]) -> bytes:
    tokens = list(call.get("output_tokens") or [])
    prompt_tokens = int(call.get("prompt_tokens") or 0)
    payload = {
        "id": f"slm-{int(call.get('seq') or 0)}",
        "object": "text_completion",
        "model": "slm",
        "choices": [
            {
                "index": 0,
                "text": str(call.get("output_text") or ""),
                "finish_reason": str(call.get("finish_reason") or "stop"),
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(tokens),
            "total_tokens": prompt_tokens + len(tokens),
        },
    }
    return json.dumps(payload).encode("utf-8")


def _token_ids(choice: Mapping[str, Any]) -> list[int]:
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


def _logprobs(choice: Mapping[str, Any]) -> list[float]:
    logprobs = choice.get("logprobs") or {}
    values = logprobs.get("token_logprobs") if isinstance(logprobs, Mapping) else None
    return [float(v) for v in values or [] if v is not None]


def slm_record(
    seq: int,
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    spent_ms: float,
    fault: str,
) -> dict[str, Any]:
    choices = response.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], Mapping) else {}
    usage = response.get("usage") or {}
    return transcript.slm_call(
        seq,
        prompt=str(request.get("prompt") or ""),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        max_tokens=int(request.get("max_tokens") or 0),
        stop=[str(s) for s in request.get("stop") or []],
        seed=int(request.get("seed") or 0),
        output_tokens=_token_ids(choice),
        output_text=str(choice.get("text") or ""),
        logprobs=_logprobs(choice),
        finish_reason=str(choice.get("finish_reason") or ""),
        ttft_ms=0.0,
        total_ms=spent_ms,
        fault=fault,
    )


def slm_request_matches(call: Mapping[str, Any], request: Mapping[str, Any]) -> str:
    if call.get("prompt_sha256") != transcript.prompt_sha256(str(request.get("prompt") or "")):
        return "slm prompt differs"
    if int(call.get("max_tokens") or 0) != int(request.get("max_tokens") or 0):
        return "slm max_tokens differs"
    if [str(s) for s in call.get("stop") or []] != [str(s) for s in request.get("stop") or []]:
        return "slm stop differs"
    if int(call.get("seed") or 0) != int(request.get("seed") or 0):
        return "slm seed differs"
    return ""


@dataclass(slots=True)
class Recorder:
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def seq(self) -> int:
        return len(self.calls)

    def add(self, call: dict[str, Any]) -> None:
        self.calls.append(call)


@dataclass(slots=True)
class Replay:
    calls: list[dict[str, Any]]
    server_key: str
    arena_id: str = ""
    epoch_index: int | None = None
    miner: str = ""
    item_id: str = ""
    purpose: str = ""
    cap_micros: int = 0
    position: int = 0
    checking: int = 0
    mismatch: str = ""
    served: int = 0

    @property
    def expected(self) -> int:
        return len(self.calls)

    @property
    def remaining(self) -> int:
        return max(0, len(self.calls) - self.position)

    def verdict(self) -> Mismatch | None:
        if self.mismatch:
            return Mismatch(self.item_id, self.checking, self.mismatch)
        if self.remaining:
            return Mismatch(
                self.item_id,
                self.position,
                f"missing {self.calls[self.position].get('kind')} call",
            )
        return None

    def fail(self, what: str) -> ReplayMismatch:
        if not self.mismatch:
            self.mismatch = what
        return ReplayMismatch(f"{self.item_id} call {self.checking}: {self.mismatch}")

    def take(self, kind: str) -> dict[str, Any]:
        if self.mismatch:
            raise ReplayMismatch(self.mismatch)
        self.checking = self.position
        if self.position >= len(self.calls):
            raise self.fail(f"extra {kind} call")
        call = self.calls[self.position]
        if call.get("kind") != kind:
            raise self.fail(f"expected a {call.get('kind')} call, got {kind}")
        self.position += 1
        self.served += 1
        return dict(call)

    def check_receipt(self, call: Mapping[str, Any]) -> None:
        receipt = call.get("receipt") or {}
        status = int(call.get("status") or 200)
        if status >= 400 and not receipt:
            return
        reason = receipts.check(
            receipt,
            server_public_key=self.server_key,
            arena_id=self.arena_id,
            epoch_index=self.epoch_index,
            miner=self.miner,
            item_id=self.item_id,
            purpose=self.purpose,
            cap_micros=self.cap_micros,
        )
        if reason:
            raise self.fail(reason)


@dataclass(slots=True)
class ArtifactSession:
    sample_database: bytes
    fault_plan: tools.FaultPlan | None
    slm_url: str
    router_url: str
    desk: TicketDesk | None
    databases: dict[str, bytes] = field(default_factory=dict)
    item_databases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    current_item: str = ""
    max_model_calls: int = 0
    model_faults: bool = False
    item_calls: dict[str, int] = field(default_factory=dict)
    item_model_ms: dict[str, float] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    slm_calls: int = 0
    tool_calls: int = 0
    faults_served: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    mode: str = LIVE
    recorder: Recorder | None = None
    replay: Replay | None = None
    server_key: str = ""
    arena_id: str = ""
    epoch_index: int | None = None
    miner: str = ""
    item_seed: int = 0
    item_pool: str = ""
    item_condition: str = ""
    item_request: dict[str, Any] | None = None
    item_tokens_in: int = 0
    item_tokens_out: int = 0
    item_receipts: list[dict[str, Any]] = field(default_factory=list)
    prompts: dict[tuple[str, int], str] = field(default_factory=dict)
    engine_failures: int = 0
    relay_purpose: str = ""
    relay_cap_micros: int = 0

    @property
    def recording(self) -> bool:
        return self.mode == RECORD

    @property
    def replaying(self) -> bool:
        return self.mode == REPLAY

    def database_for(self, item_id: str) -> bytes:
        for sha in self.item_databases.get(item_id, ()):
            found = self.databases.get(sha)
            if found is not None:
                return found
        return self.sample_database

    def register_item(self, item_id: str, database: bytes) -> None:
        sha = hashlib.sha256(database).hexdigest()
        with self.lock:
            self.databases[sha] = database
            self.item_databases[item_id] = (sha,)

    def forget_item(self, item_id: str) -> None:
        with self.lock:
            for sha in self.item_databases.pop(item_id, ()):
                self.databases.pop(sha, None)

    def reset_counters(self) -> None:
        with self.lock:
            self.tokens_in = 0
            self.tokens_out = 0
            self.slm_calls = 0
            self.tool_calls = 0
            self.item_calls.clear()
            self.item_model_ms.clear()

    def model_ms(self, item_id: str) -> float:
        with self.lock:
            return self.item_model_ms.get(item_id, 0.0)

    def admit_model_call(self, item_id: str) -> tuple[bool, int]:
        with self.lock:
            made = self.item_calls.get(item_id, 0)
            if self.max_model_calls and made >= self.max_model_calls:
                return False, made
            self.item_calls[item_id] = made + 1
            return True, made

    def begin_item(
        self,
        item_id: str,
        seed: int = 0,
        calls: Sequence[Mapping[str, Any]] | None = None,
        *,
        pool: str = "",
        condition: str = "",
        request: Mapping[str, Any] | None = None,
        purpose: str = "",
        cap_micros: int = 0,
    ) -> None:
        with self.lock:
            self.current_item = item_id
            self.item_seed = int(seed)
            self.item_pool = pool
            self.item_condition = condition
            self.item_request = dict(request) if request is not None else None
            self.relay_purpose = purpose
            self.relay_cap_micros = int(cap_micros)
            self.item_tokens_in = 0
            self.item_tokens_out = 0
            self.item_receipts = []
            self.item_calls.pop(item_id, None)
            self.item_model_ms.pop(item_id, None)
            if self.recording:
                self.recorder = Recorder()
            if self.replaying:
                self.replay = Replay(
                    calls=[dict(c) for c in calls or []],
                    server_key=self.server_key,
                    arena_id=self.arena_id,
                    epoch_index=self.epoch_index,
                    miner=self.miner,
                    item_id=item_id,
                    purpose=purpose,
                    cap_micros=int(cap_micros),
                )

    def finish_item(self) -> Mismatch | None:
        with self.lock:
            verdict = self.replay.verdict() if self.replay is not None else None
            self.current_item = ""
            self.item_request = None
        return verdict

    def recorded(self, output: Any) -> dict[str, Any]:
        with self.lock:
            calls = list(self.recorder.calls) if self.recorder is not None else []
            return transcript.item_entry(
                item_id=self.current_item,
                pool=self.item_pool,
                condition=self.item_condition,
                seed=self.item_seed,
                calls=calls,
                output=output,
                request=self.item_request,
            )

    def tally(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self.lock:
            self.slm_calls += 1
            self.tokens_in += int(prompt_tokens)
            self.tokens_out += int(completion_tokens)
            self.item_tokens_in += int(prompt_tokens)
            self.item_tokens_out += int(completion_tokens)


def recording_session(
    *,
    sample_database: bytes,
    fault_plan: tools.FaultPlan | None,
    slm_url: str,
    router_url: str,
    desk: TicketDesk | None,
    databases: Mapping[str, bytes] | None = None,
    item_databases: Mapping[str, Sequence[str]] | None = None,
    max_model_calls: int = 0,
    model_faults: bool = False,
) -> ArtifactSession:
    return ArtifactSession(
        sample_database=sample_database,
        fault_plan=fault_plan,
        slm_url=slm_url,
        router_url=router_url,
        desk=desk,
        databases=dict(databases or {}),
        item_databases={k: tuple(v) for k, v in (item_databases or {}).items()},
        max_model_calls=max_model_calls,
        model_faults=model_faults,
        mode=RECORD,
        recorder=Recorder(),
    )


def replay_session(
    transcript_item: Mapping[str, Any] | None = None,
    databases: Mapping[str, bytes] | None = None,
    fault_plan: tools.FaultPlan | None = None,
    server_key: str = "",
    *,
    item_databases: Mapping[str, Sequence[str]] | None = None,
    sample_database: bytes = b"",
    miner_hotkey: str = "",
    arena_id: str = "",
    epoch_index: int | None = None,
    max_model_calls: int = 0,
    model_faults: bool = False,
    purpose: str = "",
    cap_micros: int = 0,
) -> ArtifactSession:
    session = ArtifactSession(
        sample_database=sample_database,
        fault_plan=fault_plan,
        slm_url="",
        router_url="",
        desk=None,
        databases=dict(databases or {}),
        item_databases={k: tuple(v) for k, v in (item_databases or {}).items()},
        max_model_calls=max_model_calls,
        model_faults=model_faults,
        mode=REPLAY,
        server_key=server_key,
        arena_id=arena_id,
        epoch_index=epoch_index,
        miner=miner_hotkey,
    )
    if transcript_item is not None:
        request = transcript_item.get("request")
        session.begin_item(
            str(transcript_item.get("item_id", "")),
            int(transcript_item.get("seed") or 0),
            [dict(c) for c in transcript_item.get("calls") or [] if isinstance(c, Mapping)],
            pool=str(transcript_item.get("pool", "")),
            condition=str(transcript_item.get("condition", "")),
            request=dict(request) if isinstance(request, Mapping) else None,
            purpose=purpose,
            cap_micros=cap_micros,
        )
    return session


class Services:
    def __init__(self, host: str, port: int) -> None:
        self.sessions: dict[str, ArtifactSession] = {}
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer((host, port), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def attach(self, key: str, session: ArtifactSession) -> None:
        with self._lock:
            self.sessions[key] = session

    def detach(self, key: str) -> None:
        with self._lock:
            self.sessions.pop(key, None)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        services = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                log.debug("services: " + fmt, *args)

            def _send(
                self, status: int, body: bytes, content_type: str = "application/json"
            ) -> None:
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status: int, payload: Mapping[str, Any]) -> None:
                self._send(status, json.dumps(payload).encode("utf-8"))

            def _session(self) -> ArtifactSession | None:
                keys = [
                    value
                    for name, value in self.headers.items()
                    if name.lower() == contract.SANDBOX_HEADER
                ]
                if len(keys) != 1 or not keys[0]:
                    return None
                with services._lock:
                    return services.sessions.get(keys[0])

            def _body(self) -> dict[str, Any]:
                length = int(self.headers.get("content-length") or 0)
                if length > MAX_BODY:
                    raise ValueError("body too large")
                raw = self.rfile.read(length) if length else b""
                parsed = json.loads(raw or b"{}")
                return parsed if isinstance(parsed, dict) else {}

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                try:
                    body = self._body()
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
                    return
                session = self._session()
                if session is None:
                    self._json(403, {"error": "unknown sandbox"})
                    return
                try:
                    if path == contract.TOOL_SCHEMA or path == contract.TOOL_QUERY:
                        self._tool(session, path, body)
                    elif path.startswith(SLM_PREFIX):
                        self._slm(session, path[len(SLM_PREFIX) :], body)
                    elif path.startswith(ROUTER_PREFIX):
                        self._router(session, path[len(ROUTER_PREFIX) :], body)
                    elif path == RELAY_PREFIX + contract.RELAY_COMPLETIONS:
                        self._relay(session, body)
                    else:
                        self._json(404, {"error": "no such service"})
                except ReplayMismatch as exc:
                    self._json(MISMATCH_STATUS, {"error": "replay mismatch", "what": str(exc)})
                except Exception as exc:
                    log.warning("service %s failed: %s", path, exc)
                    self._json(500, {"error": "service failure"})

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/health":
                    self._json(200, {"status": "ok"})
                    return
                session = self._session()
                if session is None:
                    self._json(403, {"error": "unknown sandbox"})
                    return
                if path.startswith(SLM_PREFIX) and path.endswith("/models"):
                    self._json(200, {"object": "list", "data": [{"id": "slm", "object": "model"}]})
                    return
                self._json(404, {"error": "no such service"})

            def _serve_tool(self, kind: str, payload: Mapping[str, Any], *, wait: bool) -> None:
                try:
                    status, raw = tools.apply_fault(kind, payload)
                except tools.ToolFault:
                    if wait:
                        threading.Event().wait(UPSTREAM_SECONDS / 10)
                    self._json(504, {"error": "tool timed out"})
                    return
                self._send(status, raw)

            def _tool(self, session: ArtifactSession, path: str, body: dict[str, Any]) -> None:
                with session.lock:
                    session.tool_calls += 1
                    item_id = session.current_item
                database = session.database_for(item_id)
                request = tools.tool_request(path, body)
                payload = tools.serve(database, path, request)
                kind = session.fault_plan.fault_for(item_id) if session.fault_plan else ""
                replay = session.replay
                if replay is not None:
                    expected = replay.take(transcript.TOOL)
                    if expected.get("route") != path:
                        raise replay.fail(f"expected tool {expected.get('route')}, got {path}")
                    if not transcript.canonical_equal(expected.get("request") or {}, request):
                        raise replay.fail("tool request differs")
                    if not transcript.canonical_equal(expected.get("response") or {}, payload):
                        raise replay.fail("tool response differs")
                    if str(expected.get("fault") or "") != kind:
                        raise replay.fail("tool fault differs")
                    self._serve_tool(kind, payload, wait=False)
                    return
                if kind:
                    with session.lock:
                        session.faults_served += 1
                recorder = session.recorder
                if recorder is not None:
                    with session.lock:
                        recorder.add(
                            transcript.tool_call(
                                recorder.seq,
                                route=path,
                                request=request,
                                response=payload,
                                fault=kind,
                            )
                        )
                self._serve_tool(kind, payload, wait=True)

            def _model_fault(self, session: ArtifactSession, item_id: str, made: int) -> str:
                if session.model_faults and session.fault_plan and made == 0:
                    return session.fault_plan.fault_for(item_id)
                return ""

            def _slm(self, session: ArtifactSession, path: str, body: dict[str, Any]) -> None:
                with session.lock:
                    item_id = session.current_item
                    seed = session.item_seed
                body["temperature"] = 0
                body["stream"] = False
                body.setdefault("model", "slm")
                if session.mode != LIVE:
                    body["seed"] = seed
                admitted, made = session.admit_model_call(item_id)
                if not admitted:
                    self._json(429, {"error": "model call limit for this item reached"})
                    return
                fault = self._model_fault(session, item_id, made)
                replay = session.replay
                if replay is not None:
                    self._replay_slm(session, replay, body, fault)
                    return
                if fault in FAULT_STATUS:
                    with session.lock:
                        session.faults_served += 1
                        recorder = session.recorder
                        if recorder is not None:
                            recorder.add(
                                slm_record(recorder.seq, body, {}, spent_ms=0.0, fault=fault)
                            )
                    if fault == "timeout":
                        threading.Event().wait(UPSTREAM_SECONDS / 10)
                    self._send(FAULT_STATUS[fault], FAULT_BODY[fault])
                    return
                if session.recording:
                    body["logprobs"] = max(1, int(body.get("logprobs") or 0))
                    body["return_token_ids"] = True
                    body["return_tokens_as_token_ids"] = True
                started = time.perf_counter()
                with httpx.Client(timeout=UPSTREAM_SECONDS) as http:
                    response = http.post(session.slm_url.rstrip("/") + path, json=body)
                spent = (time.perf_counter() - started) * 1000
                with session.lock:
                    session.item_model_ms[item_id] = session.item_model_ms.get(item_id, 0.0) + spent
                if response.status_code != 200:
                    with session.lock:
                        session.engine_failures += 1
                    self._send(
                        response.status_code,
                        response.content,
                        response.headers.get("content-type", "application/json"),
                    )
                    return
                try:
                    parsed = response.json()
                except ValueError:
                    parsed = {}
                usage = parsed.get("usage") or {} if isinstance(parsed, Mapping) else {}
                session.tally(
                    int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
                )
                raw = response.content
                if session.recording and isinstance(parsed, Mapping):
                    with session.lock:
                        recorder = session.recorder
                        if recorder is not None:
                            call = slm_record(
                                recorder.seq, body, parsed, spent_ms=spent, fault=fault
                            )
                            recorder.add(call)
                            raw = completion_body(call)
                if fault:
                    with session.lock:
                        session.faults_served += 1
                    raw = tools.model_fault(fault, raw)
                self._send(200, raw)

            def _replay_slm(
                self, session: ArtifactSession, replay: Replay, body: dict[str, Any], fault: str
            ) -> None:
                expected = replay.take(transcript.SLM)
                why = slm_request_matches(expected, body)
                if why:
                    raise replay.fail(why)
                if str(expected.get("fault") or "") != fault:
                    raise replay.fail("model fault differs")
                with session.lock:
                    session.prompts[(replay.item_id, int(expected.get("seq") or 0))] = str(
                        body.get("prompt") or ""
                    )
                if fault in FAULT_STATUS:
                    self._send(FAULT_STATUS[fault], FAULT_BODY[fault])
                    return
                session.tally(
                    int(expected.get("prompt_tokens") or 0),
                    len(expected.get("output_tokens") or []),
                )
                raw = completion_body(expected)
                if fault:
                    raw = tools.model_fault(fault, raw)
                self._send(200, raw)

            def _router(self, session: ArtifactSession, path: str, body: dict[str, Any]) -> None:
                replay = session.replay
                if replay is not None:
                    expected = replay.take(transcript.ROUTER)
                    if expected.get("route") != path:
                        raise replay.fail(f"expected router {expected.get('route')}, got {path}")
                    if not transcript.canonical_equal(expected.get("request") or {}, body):
                        raise replay.fail("router request differs")
                    self._send(200, json.dumps(expected.get("response") or {}).encode("utf-8"))
                    return
                if not session.router_url:
                    self._json(404, {"error": "no router"})
                    return
                with httpx.Client(timeout=UPSTREAM_SECONDS) as http:
                    response = http.post(session.router_url.rstrip("/") + path, json=body)
                recorder = session.recorder
                if recorder is not None and response.status_code == 200:
                    try:
                        parsed = response.json()
                    except ValueError:
                        parsed = {}
                    if isinstance(parsed, Mapping):
                        with session.lock:
                            recorder.add(
                                transcript.router_call(
                                    recorder.seq, route=path, request=body, response=parsed
                                )
                            )
                        self._send(200, json.dumps(parsed).encode("utf-8"))
                        return
                self._send(
                    response.status_code,
                    response.content,
                    response.headers.get("content-type", "application/json"),
                )

            def _relay(self, session: ArtifactSession, body: dict[str, Any]) -> None:
                replay = session.replay
                if replay is not None:
                    expected = replay.take(transcript.RELAY)
                    if expected.get("request_sha256") != transcript.canonical_sha256(body):
                        raise replay.fail("relay request differs")
                    replay.check_receipt(expected)
                    receipt = expected.get("receipt")
                    if isinstance(receipt, Mapping) and receipt:
                        with session.lock:
                            session.item_receipts.append(dict(receipt))
                    self._json(int(expected.get("status") or 200), expected.get("response") or {})
                    return
                desk = session.desk
                opaque = self.headers.get(contract.TICKET_HEADER, "") or str(body.get("ticket", ""))
                if desk is None or not opaque:
                    self._relay_answer(
                        session,
                        body,
                        403,
                        {"error": {"code": "refused", "message": "no ticket"}},
                        {},
                    )
                    return
                messages = body.get("messages")
                if not isinstance(messages, list):
                    self._relay_answer(
                        session,
                        body,
                        422,
                        {"error": {"code": "refused", "message": "messages required"}},
                        {},
                    )
                    return
                try:
                    answer = desk.complete(
                        opaque,
                        model=str(body.get("model", "")),
                        messages=[m for m in messages if isinstance(m, dict)],
                        max_tokens=int(body["max_tokens"]) if body.get("max_tokens") else None,
                    )
                except RelayError as exc:
                    self._relay_answer(
                        session,
                        body,
                        402 if exc.code in ("cap", "limit") else 403,
                        {"error": {"code": exc.code, "message": "the frontier call was refused"}},
                        {},
                    )
                    return
                self._relay_answer(
                    session,
                    body,
                    200,
                    {
                        "id": answer.receipt.receipt,
                        "object": "chat.completion",
                        "model": answer.receipt.model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": answer.text},
                                "finish_reason": answer.finish_reason or "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": answer.receipt.tokens_in,
                            "completion_tokens": answer.receipt.tokens_out,
                            "total_tokens": answer.receipt.tokens_in + answer.receipt.tokens_out,
                        },
                    },
                    answer.receipt.body,
                )

            def _relay_answer(
                self,
                session: ArtifactSession,
                request: Mapping[str, Any],
                status: int,
                response: Mapping[str, Any],
                receipt: Mapping[str, Any],
            ) -> None:
                recorder = session.recorder
                if receipt:
                    with session.lock:
                        session.item_receipts.append(dict(receipt))
                if recorder is not None:
                    with session.lock:
                        recorder.add(
                            transcript.relay_call(
                                recorder.seq,
                                request=request,
                                response=response,
                                receipt=receipt,
                                status=status,
                            )
                        )
                self._json(status, response)

        return Handler


SessionFactory = Callable[[], ArtifactSession]
