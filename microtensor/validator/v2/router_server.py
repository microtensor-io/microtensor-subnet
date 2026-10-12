from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from microtensor.artifact import contract

MAX_BODY = 1024 * 1024


class RouterModel:
    def __init__(self, root: Path, max_input_tokens: int, device: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(root))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(root))
        if int(getattr(self.model.config, "num_labels", 1)) != 1:
            raise ValueError("the router must have exactly one output logit")
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.max_input_tokens = max_input_tokens
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def decide(self, query: str, draft: str) -> float:
        encoded = self.tokenizer(
            query,
            draft,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.no_grad():
            logit = self.model(**encoded).logits.reshape(-1)[0]
        return float(self.torch.sigmoid(logit).item())


def build(model: RouterModel) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return None

        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if urlparse(self.path).path == "/health":
                self._reply(200, {"status": "ok", "parameters": model.parameters})
                return
            self._reply(404, {"error": "no such route"})

        def do_POST(self) -> None:
            if urlparse(self.path).path != contract.ROUTER_DECIDE:
                self._reply(404, {"error": "no such route"})
                return
            length = int(self.headers.get("content-length") or 0)
            if length > MAX_BODY:
                self._reply(413, {"error": "body too large"})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                inputs = body.get("input") or {}
                query = str(inputs.get("query", ""))
                draft = str(inputs.get("draft", ""))
            except (ValueError, AttributeError):
                self._reply(400, {"error": "malformed request"})
                return
            started = time.perf_counter()
            try:
                probability = model.decide(query, draft)
            except Exception as exc:
                self._reply(500, {"error": f"router failed: {type(exc).__name__}"})
                return
            self._reply(
                200,
                {
                    "answers": [
                        {
                            "name": contract.ROUTER_ANSWER,
                            "type": "predicate",
                            "probability": probability,
                        }
                    ],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mt-router-server")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    model = RouterModel(args.model, args.max_input_tokens, args.device)
    if model.parameters > contract.ROUTER_MAX_PARAMETERS:
        print(f"router has {model.parameters} parameters, over the cap", file=sys.stderr)
        return 2
    server = ThreadingHTTPServer((args.listen, args.port), build(model))
    server.daemon_threads = True
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
