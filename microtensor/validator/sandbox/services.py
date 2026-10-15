from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final
from urllib.parse import urlparse

import httpx

from microtensor.artifact import contract
from microtensor.validator.relay import RelayError, TicketDesk
from microtensor.validator.sandbox import tools

log = logging.getLogger("microtensor.validator.sandbox.services")

MAX_BODY: Final[int] = 4 * 1024 * 1024
UPSTREAM_SECONDS: Final[float] = 300.0
SLM_PREFIX: Final[str] = "/slm"
ROUTER_PREFIX: Final[str] = "/router"
RELAY_PREFIX: Final[str] = "/relay"


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
    tokens_in: int = 0
    tokens_out: int = 0
    slm_calls: int = 0
    tool_calls: int = 0
    faults_served: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

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
                key = self.headers.get("x-mt-sandbox", "")
                with services._lock:
                    return services.sessions.get(key)

            def _body(self) -> dict[str, Any]:
                length = int(self.headers.get("content-length") or 0)
                if length > MAX_BODY:
                    raise ValueError("body too large")
                raw = self.rfile.read(length) if length else b""
                parsed = json.loads(raw or b"{}")
                return parsed if isinstance(parsed, dict) else {}

            def do_POST(self) -> None:
                session = self._session()
                if session is None:
                    self._json(403, {"error": "unknown sandbox"})
                    return
                path = urlparse(self.path).path
                try:
                    body = self._body()
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
                    return
                try:
                    if path == contract.TOOL_SCHEMA or path == contract.TOOL_QUERY:
                        self._tool(session, path, body)
                    elif path.startswith(SLM_PREFIX):
                        self._forward(
                            session, session.slm_url, path[len(SLM_PREFIX) :], body, slm=True
                        )
                    elif path.startswith(ROUTER_PREFIX):
                        self._forward(session, session.router_url, path[len(ROUTER_PREFIX) :], body)
                    elif path == RELAY_PREFIX + contract.RELAY_COMPLETIONS:
                        self._relay(session, body)
                    else:
                        self._json(404, {"error": "no such service"})
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

            def _tool(self, session: ArtifactSession, path: str, body: dict[str, Any]) -> None:
                with session.lock:
                    session.tool_calls += 1
                    item_id = session.current_item
                database = session.database_for(item_id)
                if path == contract.TOOL_SCHEMA:
                    payload = tools.schema_lookup(database, str(body.get("table", "")))
                else:
                    payload = tools.query_execution(database, str(body.get("sql", "")))
                kind = session.fault_plan.fault_for(item_id) if session.fault_plan else ""
                if kind:
                    with session.lock:
                        session.faults_served += 1
                try:
                    status, raw = tools.apply_fault(kind, payload)
                except tools.ToolFault:
                    threading.Event().wait(UPSTREAM_SECONDS / 10)
                    self._json(504, {"error": "tool timed out"})
                    return
                self._send(status, raw)

            def _forward(
                self,
                session: ArtifactSession,
                base: str,
                path: str,
                body: dict[str, Any],
                *,
                slm: bool = False,
            ) -> None:
                if slm:
                    body["temperature"] = 0
                    body["stream"] = False
                    body.setdefault("model", "slm")
                with httpx.Client(timeout=UPSTREAM_SECONDS) as http:
                    response = http.post(base.rstrip("/") + path, json=body)
                if slm and response.status_code == 200:
                    try:
                        usage = response.json().get("usage") or {}
                        with session.lock:
                            session.slm_calls += 1
                            session.tokens_in += int(usage.get("prompt_tokens") or 0)
                            session.tokens_out += int(usage.get("completion_tokens") or 0)
                    except ValueError:
                        pass
                self._send(
                    response.status_code,
                    response.content,
                    response.headers.get("content-type", "application/json"),
                )

            def _relay(self, session: ArtifactSession, body: dict[str, Any]) -> None:
                desk = session.desk
                opaque = self.headers.get(contract.TICKET_HEADER, "") or str(body.get("ticket", ""))
                if desk is None or not opaque:
                    self._json(403, {"error": {"code": "refused", "message": "no ticket"}})
                    return
                messages = body.get("messages")
                if not isinstance(messages, list):
                    self._json(422, {"error": {"code": "refused", "message": "messages required"}})
                    return
                try:
                    answer = desk.complete(
                        opaque,
                        model=str(body.get("model", "")),
                        messages=[m for m in messages if isinstance(m, dict)],
                        max_tokens=int(body["max_tokens"]) if body.get("max_tokens") else None,
                    )
                except RelayError as exc:
                    self._json(
                        402 if exc.code in ("cap", "limit") else 403,
                        {"error": {"code": exc.code, "message": "the frontier call was refused"}},
                    )
                    return
                self._json(
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
                )

        return Handler


SessionFactory = Callable[[], ArtifactSession]
