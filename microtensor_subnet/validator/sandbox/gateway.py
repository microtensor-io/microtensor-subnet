from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import httpx

MAX_BODY = 4 * 1024 * 1024
UPSTREAM_SECONDS = 330.0
SANDBOX_HEADER = "x-mt-sandbox"
HOP_HEADERS = frozenset(
    {"connection", "keep-alive", "transfer-encoding", "host", "content-length", SANDBOX_HEADER}
)


def forwarded_headers(incoming: Iterable[tuple[str, str]], sandbox: str) -> dict[str, str]:
    kept = {key: value for key, value in incoming if key.lower() not in HOP_HEADERS}
    kept[SANDBOX_HEADER] = sandbox
    return kept


def build(upstream: str, sandbox: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return None

        def _proxy(self, method: str) -> None:
            length = int(self.headers.get("content-length") or 0)
            if length > MAX_BODY:
                self._reply(413, b'{"error":"body too large"}')
                return
            body = self.rfile.read(length) if length else b""
            path = urlparse(self.path).path
            headers = forwarded_headers(self.headers.items(), sandbox)
            try:
                with httpx.Client(timeout=UPSTREAM_SECONDS) as http:
                    response = http.request(
                        method, upstream.rstrip("/") + path, content=body or None, headers=headers
                    )
            except httpx.HTTPError:
                self._reply(502, json.dumps({"error": "service unavailable"}).encode())
                return
            self._reply(
                response.status_code,
                response.content,
                response.headers.get("content-type", "application/json"),
            )

        def _reply(self, status: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            self._proxy("POST")

        def do_GET(self) -> None:
            if urlparse(self.path).path == "/health":
                self._reply(200, b'{"status":"ok"}')
                return
            self._proxy("GET")

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mt-gateway")
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--sandbox", required=True)
    parser.add_argument("--listen", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.listen, args.port), build(args.upstream, args.sandbox))
    server.daemon_threads = True
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
