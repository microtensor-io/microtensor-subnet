from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

import httpx

from microtensor_subnet.artifact import contract
from microtensor_subnet.core.items import Item
from microtensor_subnet.scoring.stats import percentile
from microtensor_subnet.validator import transcripts
from microtensor_subnet.validator.context import (
    ESCALATION,
    HARNESS,
    SLM,
    SYSTEM,
    EpochContext,
    render_prompt,
)
from microtensor_subnet.validator.relay import REPAIR
from microtensor_subnet.validator.relay import SYSTEM as SYSTEM_PURPOSE
from microtensor_subnet.validator.sandbox import (
    GATEWAY_PORT,
    MAIN_PORT,
    READY_SECONDS,
    Limits,
    SandboxConfig,
    SandboxError,
    call_main,
)
from microtensor_subnet.validator.sandbox import tools as tool_faults
from microtensor_subnet.validator.sandbox.docker import ContainerSpec, Docker

log = logging.getLogger("microtensor_subnet.validator.replay")

FAKETIME: Final[str] = "@2026-01-01 00:00:00 x0"
FAKETIME_LIBRARY: Final[str] = "/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1"
REPAIR_ERRORS: Final[str] = "the draft returned the wrong result"
ROUTES: Final[dict[str, str]] = {
    HARNESS: contract.HARNESS_RUN,
    ESCALATION: contract.ESCALATION_REPAIR,
    SYSTEM: contract.SYSTEM_RUN,
}
SERVICES_MODULE: Final[str] = "microtensor_subnet.validator.sandbox.services"
TAP_SECONDS: Final[float] = 330.0
HOP_HEADERS: Final[frozenset[str]] = frozenset(
    {"connection", "keep-alive", "transfer-encoding", "host", "content-length"}
)
MAX_BODY: Final[int] = 4 * 1024 * 1024

replay_session: Callable[..., Any] = transcripts.resolve(SERVICES_MODULE, "replay_session")


def services_class() -> Any:
    from microtensor_subnet.validator.sandbox import services

    return services.Services


def faketime_library() -> str:
    return str(getattr(contract, "FAKETIME_LIBRARY", FAKETIME_LIBRARY))


class ReplayError(RuntimeError):
    pass


@dataclass(slots=True)
class ReplayResult:
    ok: bool
    reason: str = ""
    items: int = 0
    calls: int = 0
    tokens: dict[str, dict[str, int]] = field(default_factory=dict)
    added_wall_p95_ms: float = 0.0
    first_mismatch: dict[str, Any] | None = None
    added_ms: dict[str, float] = field(default_factory=dict)
    wall_ms: dict[tuple[str, str], float] = field(default_factory=dict)
    prompts: dict[tuple[str, int], str] = field(default_factory=dict)
    relay_cost_micros: int = 0
    item_failures: dict[tuple[str, str], str] = field(default_factory=dict)
    over_calls: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "items": self.items,
            "calls": self.calls,
            "tokens": {name: dict(row) for name, row in sorted(self.tokens.items())},
            "added_wall_p95_ms": round(self.added_wall_p95_ms, 1),
            "first_mismatch": dict(self.first_mismatch) if self.first_mismatch else None,
        }

    def fail(self, item_id: str, seq: int, what: str) -> ReplayResult:
        self.ok = False
        self.reason = f"{what} at {item_id} call {seq}" if item_id else what
        self.first_mismatch = {"item_id": item_id, "seq": seq, "what": what}
        return self

    def count(self, condition: str, tokens_in: int, tokens_out: int) -> None:
        row = self.tokens.setdefault(condition, {"in": 0, "out": 0})
        row["in"] += tokens_in
        row["out"] += tokens_out


@dataclass(slots=True)
class TapState:
    item_id: str = ""
    calls: int = 0
    prompts: dict[tuple[str, int], str] = field(default_factory=dict)


class PromptTap:
    def __init__(self, host: str, port: int, upstream: str) -> None:
        self.upstream = upstream.rstrip("/")
        self._state: dict[str, TapState] = {}
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

    def begin(self, key: str, item_id: str) -> None:
        with self._lock:
            state = self._state.setdefault(key, TapState())
            state.item_id = item_id
            state.calls = 0

    def prompts(self, key: str) -> dict[tuple[str, int], str]:
        with self._lock:
            state = self._state.get(key)
            return dict(state.prompts) if state is not None else {}

    def forget(self, key: str) -> None:
        with self._lock:
            self._state.pop(key, None)

    def _record(self, key: str, path: str, raw: bytes) -> None:
        with self._lock:
            state = self._state.get(key)
            if state is None:
                return
            seq = state.calls
            state.calls += 1
            if not path.endswith(contract.SLM_COMPLETIONS):
                return
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                return
            prompt = body.get("prompt") if isinstance(body, dict) else None
            if isinstance(prompt, str):
                state.prompts[(state.item_id, seq)] = prompt

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        tap = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                log.debug("replay tap: " + fmt, *args)

            def _reply(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _proxy(self, method: str) -> None:
                length = int(self.headers.get("content-length") or 0)
                if length > MAX_BODY:
                    self._reply(413, b'{"error":"body too large"}', "application/json")
                    return
                raw = self.rfile.read(length) if length else b""
                path = urlparse(self.path).path
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
                key = next((v for k, v in headers.items() if k.lower() == "x-mt-sandbox"), "")
                if method == "POST" and key:
                    tap._record(key, path, raw)
                try:
                    with httpx.Client(timeout=TAP_SECONDS) as http:
                        response = http.request(
                            method, tap.upstream + path, content=raw or None, headers=headers
                        )
                except httpx.HTTPError:
                    self._reply(502, b'{"error":"service unavailable"}', "application/json")
                    return
                self._reply(
                    response.status_code,
                    response.content,
                    response.headers.get("content-type", "application/json"),
                )

            def do_POST(self) -> None:
                self._proxy("POST")

            def do_GET(self) -> None:
                if urlparse(self.path).path == "/health":
                    self._reply(200, b'{"status":"ok"}', "application/json")
                    return
                self._proxy("GET")

        return Handler


@dataclass(slots=True)
class Replayer:
    docker: Docker
    config: SandboxConfig
    context: EpochContext
    services: Any
    tap: PromptTap
    host_ip: str
    server_key: str
    replayer_image: str
    reference_harness_dir: Path
    variant_dirs: Sequence[Path]
    request_seconds: float
    caps: Mapping[str, int] = field(default_factory=dict)
    seeds: Mapping[str, int] = field(default_factory=dict)
    drafts: Mapping[str, str] = field(default_factory=dict)
    relay_allowance_micros: int = 0
    max_input_tokens: int = 0
    max_output_tokens: int = 0

    @property
    def limits(self) -> Limits:
        return Limits.from_definition(self.context.definition, self.context.max_model_len)

    def tree_for(self, condition: str, artifact_dir: Path, has: frozenset[str]) -> Path | None:
        if condition == transcripts.CONDITION_BARE:
            return None
        if condition.startswith("B") and condition[1:].isdigit():
            index = int(condition[1:]) - 1
            variants = [self.reference_harness_dir, *self.variant_dirs]
            return variants[index] if 0 <= index < len(variants) else None
        own = bool({"harness", "escalation"} & has)
        return artifact_dir if own else self.reference_harness_dir

    def _gateway(self, key: str, network: str) -> str:
        name = f"mt-rgw-{key}"
        upstream = self.docker.own_address(network) or self.host_ip
        self.docker.start(
            ContainerSpec(
                name=name,
                image=self.config.gateway_image,
                network="bridge",
                entrypoint="python",
                command=(
                    "-m",
                    "microtensor_subnet.validator.sandbox.gateway",
                    "--upstream",
                    f"http://{upstream}:{self.tap.port}",
                    "--sandbox",
                    key,
                    "--port",
                    str(GATEWAY_PORT),
                ),
                cpus=1.0,
                memory="512m",
                pids=64,
                read_only=True,
                hardened=False,
                extra_networks=(network,),
                tmpfs=("/tmp",),  # noqa: S108
            )
        )
        return name

    def _main(self, key: str, tree: Path, network: str, gateway_ip: str) -> tuple[str, str]:
        name = f"mt-rpl-{key}"
        base = f"http://{gateway_ip}:{GATEWAY_PORT}"
        self.docker.start(
            ContainerSpec(
                name=name,
                image=self.replayer_image,
                network=network,
                command=("python", "-X", "utf8", "/artifact/main.py"),
                env={
                    contract.ENV_PORT: str(MAIN_PORT),
                    contract.ENV_SLM_URL: base + "/slm",
                    contract.ENV_ROUTER_URL: base + "/router",
                    contract.ENV_TOOLS_URL: base,
                    contract.ENV_RELAY_URL: base + "/relay",
                    "HOME": "/tmp",  # noqa: S108
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "FAKETIME": FAKETIME,
                    "LD_PRELOAD": faketime_library(),
                    "OMP_NUM_THREADS": "1",
                },
                volumes=((str(tree), "/artifact", "ro"),),
                runtime=self.config.sandbox_runtime,
                cpus=self.config.sandbox_cpus,
                memory=self.config.sandbox_memory,
                pids=self.config.sandbox_pids,
                read_only=True,
                hardened=True,
                user="65534:65534",
                tmpfs=("/tmp",),  # noqa: S108
                ipc_none=True,
            )
        )
        url = f"http://{self.docker.address(name, network)}:{MAIN_PORT}"
        self._wait(name, url)
        return name, url

    def _wait(self, name: str, url: str) -> None:
        deadline = time.monotonic() + READY_SECONDS
        while time.monotonic() < deadline:
            if not self.docker.running(name):
                raise SandboxError(f"main.py exited: {self.docker.logs(name, 40)[-300:]}")
            try:
                with httpx.Client(timeout=5.0) as http:
                    if http.get(url + "/health").status_code < 500:
                        return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
        raise SandboxError("main.py did not start listening")

    def _budget(
        self,
        result: ReplayResult,
        item_id: str,
        condition: str,
        calls: Sequence[Mapping[str, Any]],
        prompts: Mapping[tuple[str, int], str],
        count_tokens: Callable[[str], int] | None,
    ) -> None:
        key = (item_id, condition)
        slm = [c for c in calls if str(c.get("kind", "")) == transcripts.KIND_SLM]
        limit = self.limits.max_model_calls
        if limit and len(slm) > limit:
            result.over_calls.add(item_id)
        for call in slm:
            produced = len(call.get("output_tokens") or [])
            if self.max_output_tokens and produced > self.max_output_tokens:
                result.item_failures.setdefault(
                    key, f"{produced} output tokens, over {self.max_output_tokens}"
                )
            prompt = prompts.get((item_id, int(call.get("seq") or 0)))
            if prompt is None or count_tokens is None or not self.max_input_tokens:
                continue
            counted = count_tokens(prompt)
            if counted > self.max_input_tokens:
                result.item_failures.setdefault(
                    key, f"{counted} prompt tokens, over {self.max_input_tokens}"
                )

    def _direct(
        self,
        row: Mapping[str, Any],
        item: Item,
        seed: int,
        result: ReplayResult,
        count_tokens: Callable[[str], int] | None,
    ) -> str:
        calls = transcripts.calls_of(row)
        if len(calls) != 1 or str(calls[0].get("kind", "")) != transcripts.KIND_SLM:
            return "extra call" if len(calls) > 1 else "missing call"
        call = calls[0]
        prompt = render_prompt(self.context.slm_prompt, item.task)
        if str(call.get("prompt_sha256", "")) != transcripts.prompt_sha256(prompt):
            return "slm prompt differs"
        if int(call.get("max_tokens") or 0) != self.context.max_tokens:
            return "slm max_tokens differs"
        if [str(s) for s in call.get("stop") or []] != list(self.context.stop):
            return "slm stop differs"
        if int(call.get("seed") or 0) != seed:
            return "slm seed differs"
        expected = {"answer": str(call.get("output_text") or "")}
        if transcripts.canonical_json(row.get("output")) != transcripts.canonical_json(expected):
            return "output differs"
        seq = int(call.get("seq") or 0)
        condition = str(row.get("condition", ""))
        result.prompts[(item.item_id, seq)] = prompt
        result.count(
            condition, int(call.get("prompt_tokens") or 0), len(call.get("output_tokens") or [])
        )
        result.calls += 1
        self._budget(result, item.item_id, condition, calls, result.prompts, count_tokens)
        return ""

    def _request_reason(self, row: Mapping[str, Any], item: Item, pool: str, condition: str) -> str:
        request = row.get("request")
        if not isinstance(request, Mapping):
            return "request missing"
        if str(request.get("item_id", "")) != item.item_id:
            return "request item"
        if transcripts.canonical_json(request.get("task")) != transcripts.canonical_json(item.task):
            return "request task"
        if condition == transcripts.CONDITION_REPAIR:
            draft = self.drafts.get(item.item_id)
            if draft is not None and str(request.get("draft", "")) != draft:
                return "request draft"
            if str(request.get("errors", "")) != REPAIR_ERRORS:
                return "request errors"
        floor = transcripts.floor_of(condition)
        if floor is not None:
            raw_floor = request.get("floor")
            if raw_floor is None:
                return "request floor"
            try:
                posted = float(raw_floor)
            except (TypeError, ValueError):
                return "request floor"
            if posted != floor:
                return "request floor"
        return ""

    def replay(
        self,
        transcript: Mapping[str, Any],
        *,
        artifact_dir: Path,
        has: frozenset[str],
        count_tokens: Callable[[str], int] | None = None,
    ) -> ReplayResult:
        result = ReplayResult(ok=True)
        hotkey = str(transcript.get("miner_hotkey", ""))
        known = transcripts.items_by_id(self.context)
        key = secrets.token_hex(4)
        network = f"mt-rnet-{key}"
        item_databases = self.context.items.item_databases()
        gateway = main = ""
        main_url = ""
        current: Path | None = None
        self.docker.create_network(network)
        self.tap.begin(key, "")
        try:
            gateway = self._gateway(key, network)
            gateway_ip = self.docker.address(gateway, network)
            for row in transcripts.items_of(transcript):
                item_id = str(row.get("item_id", ""))
                condition = str(row.get("condition", ""))
                pool = str(row.get("pool", ""))
                item = known.get(item_id)
                if item is None:
                    return result.fail(item_id, 0, "unknown item")
                wanted = int(self.seeds.get(item_id, int(row.get("seed") or 0)))
                if int(row.get("seed") or 0) != wanted:
                    return result.fail(item_id, 0, "seed differs")
                result.items += 1
                if pool == SLM or condition == transcripts.CONDITION_BARE:
                    reason = self._direct(row, item, wanted, result, count_tokens)
                    if reason:
                        return result.fail(item_id, 0, reason)
                    continue
                route = ROUTES.get(pool)
                tree = self.tree_for(condition, artifact_dir, has)
                if route is None or tree is None:
                    return result.fail(item_id, 0, "condition")
                reason = self._request_reason(row, item, pool, condition)
                if reason:
                    return result.fail(item_id, 0, reason)
                if tree != current:
                    if main:
                        self.docker.kill(main)
                    main, main_url = self._main(key, tree, network, gateway_ip)
                    current = tree
                purpose = (
                    REPAIR
                    if condition == transcripts.CONDITION_REPAIR
                    else SYSTEM_PURPOSE
                    if transcripts.floor_of(condition) is not None
                    else ""
                )
                calls = transcripts.calls_of(row)
                session = replay_session(
                    row,
                    self.context.items.databases,
                    tool_faults.FaultPlan(self.context.seed),
                    self.server_key,
                    max_model_calls=self.limits.max_model_calls,
                    model_faults=self.limits.model_faults,
                    sample_database=self.context.sample_database,
                    item_databases=item_databases.get(item_id, ()),
                    arena_id=self.context.arena_id,
                    epoch_index=self.context.epoch_index,
                    miner=hotkey,
                    purpose=purpose,
                    cap_micros=self.caps.get(item_id, 0) if purpose else 0,
                )
                self.services.attach(key, session)
                self.tap.begin(key, item_id)
                request = row.get("request")
                call = call_main(
                    main_url,
                    route,
                    dict(request) if isinstance(request, Mapping) else {},
                    timeout=self.request_seconds,
                )
                verdict = str(session.replay.verdict())
                if verdict:
                    return result.fail(item_id, int(session.replay.served), verdict)
                body: Any = call.body if call.ok else {"error": call.error}
                if transcripts.canonical_json(body) != transcripts.canonical_json(
                    row.get("output")
                ):
                    return result.fail(item_id, len(calls), "output differs")
                result.wall_ms[(item_id, condition)] = call.latency_ms
                if pool == HARNESS and condition == transcripts.CONDITION_MINER:
                    result.added_ms[item_id] = call.latency_ms
                result.count(condition, int(session.tokens_in), int(session.tokens_out))
                result.calls += len(calls)
                result.relay_cost_micros += transcripts.receipt_cost(
                    transcripts.relay_receipts(row)
                )
                prompts = self.tap.prompts(key)
                result.prompts.update(prompts)
                self._budget(result, item_id, condition, calls, prompts, count_tokens)
        except SandboxError as exc:
            return result.fail("", 0, f"sandbox: {exc}")
        finally:
            self.services.detach(key)
            self.tap.forget(key)
            for name in (main, gateway):
                if name:
                    self.docker.kill(name)
            self.docker.remove_network(network)
        if self.relay_allowance_micros and result.relay_cost_micros > self.relay_allowance_micros:
            return result.fail("", 0, "relay allowance exceeded")
        result.added_wall_p95_ms = percentile(list(result.added_ms.values()), 0.95)
        return result


def token_counter(slm_dir: Path) -> Callable[[str], int] | None:
    path = slm_dir / "tokenizer.json"
    if not path.is_file():
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    try:
        tokenizer = Tokenizer.from_file(str(path))
    except Exception as exc:
        log.warning("tokenizer in %s not loaded: %s", slm_dir.name, exc)
        return None
    return lambda text: len(tokenizer.encode(text).ids)
