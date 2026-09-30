from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final

from microtensor.harness.sdk import Runtime

TASK_PATH: Final[str] = "/v1/systems/{name}/task"
MAX_TASK_BYTES: Final[int] = 8 * 1024 * 1024

Handler = Callable[[Mapping[str, Any]], dict[str, Any]]
Signer = Callable[[Mapping[str, Any]], str]


def system_handler(runtime: Runtime, sign: Signer) -> Handler:
    def handle(frame: Mapping[str, Any]) -> dict[str, Any]:
        trace = runtime.run(
            int(frame["round_index"]),
            str(frame["task_ref"]),
            str(frame.get("prompt", "")),
            dict(frame.get("inputs") or {}),
        )
        return trace.signed_with(sign(trace.body())).to_dict()

    return handle


def wallet_signer(wallet: Any) -> Signer:
    from microtensor.chain.wallet import sign_payload

    def sign(body: Mapping[str, Any]) -> str:
        return sign_payload(wallet, body)

    return sign


def serve_http(
    handlers: Mapping[str, Handler], host: str = "127.0.0.1", port: int = 0
) -> ThreadingHTTPServer:
    class Answer(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            return

        def _reply(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            parts = self.path.strip("/").split("/")
            if len(parts) != 4 or parts[:2] != ["v1", "systems"] or parts[3] != "task":
                self._reply(404, {"error": "no such route"})
                return
            handler = handlers.get(parts[2])
            if handler is None:
                self._reply(404, {"error": f"this host serves no system named {parts[2]!r}"})
                return
            length = int(self.headers.get("content-length") or 0)
            if length <= 0 or length > MAX_TASK_BYTES:
                self._reply(413, {"error": "the task body is empty or too large"})
                return
            try:
                frame = json.loads(self.rfile.read(length))
                self._reply(200, {"trace": handler(frame)})
            except Exception as exc:
                self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    server = ThreadingHTTPServer((host, port), Answer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
