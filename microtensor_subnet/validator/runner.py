from __future__ import annotations

import gzip
import json
import logging
import os
import secrets
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import httpx

from microtensor_subnet.artifact import contract
from microtensor_subnet.artifact.validate import (
    Manifest,
    ManifestError,
    bare_hex,
    load_manifest,
    manifest_sha256,
    sha256_of,
)
from microtensor_subnet.core import profiles, transcript
from microtensor_subnet.core.items import ItemError, unpack_bundle
from microtensor_subnet.validator import collect, preflight, spotcheck
from microtensor_subnet.validator.client import ARENA_NETWORKS, ServerClient, ServerError
from microtensor_subnet.validator.config import (
    REQUEST_SECONDS,
    SANDBOX_CPUS,
    SANDBOX_MEMORY,
    SANDBOX_PIDS,
    SLM_READY_SECONDS,
)
from microtensor_subnet.validator.context import (
    ESCALATION,
    HARNESS,
    SLM,
    SQL_MAX_TOKENS,
    SQL_STOP,
    SYSTEM,
    render_prompt,
)
from microtensor_subnet.validator.items import (
    TESTNET,
    ReleaseError,
    server_key,
    template_harness,
    verify_document,
)
from microtensor_subnet.validator.relay import (
    REPAIR,
    RETRY_SECONDS,
    RelayClient,
    RelayError,
    TicketDesk,
    retry_pause,
)
from microtensor_subnet.validator.relay import SYSTEM as SYSTEM_PURPOSE
from microtensor_subnet.validator.sandbox import (
    GATEWAY_PORT,
    MAIN_PORT,
    READY_SECONDS,
    Call,
    Limits,
    SandboxHost,
    call_main,
    compat,
    cpu_container,
    host_ip,
    services,
    tools,
)
from microtensor_subnet.validator.sandbox.docker import (
    ContainerSpec,
    Docker,
    DockerError,
    own_container,
)
from microtensor_subnet.validator.sandbox.remote import (
    EngineBridge,
    RemoteEngine,
    RemoteEngineError,
)
from microtensor_subnet.validator.sandbox.slm import (
    MEMORY_SHARE,
    SlmClient,
    SlmError,
    engine_command,
    engine_env,
)

log = logging.getLogger("microtensor_subnet.validator.runner")

INPUTS_FILE: Final[str] = "inputs.jsonl"
DRAFTS_FILE: Final[str] = "reference/drafts.jsonl"
BLOB_PREFIX: Final[str] = "db/"
REFERENCE_PREFIX: Final[str] = "reference/"
REFERENCE_HARNESS: Final[str] = "harness"
REFERENCE_ESCALATION: Final[str] = "escalation"
REFERENCE_ROUTER: Final[str] = "router"
VARIANT_PREFIX: Final[str] = "variant-"
TRANSCRIPT_FILE: Final[str] = "transcript.json"
DRY_TRANSCRIPT_FILE: Final[str] = "transcript.dry.json"
JOURNAL_FILE: Final[str] = "rows.jsonl"
DRY_JOURNAL_FILE: Final[str] = "rows.dry.jsonl"
RECEIPT_FILE: Final[str] = "receipt.json"
MIRRORED: Final[str] = "mirrored"
RUNNING: Final[str] = "running"
UPLOADED: Final[str] = "uploaded"
ERRORS_TEXT: Final[str] = "the draft returned the wrong result"
DEFAULT_CONTEXT: Final[int] = 8192
ENGINE_ATTEMPTS: Final[int] = 3
ROW_ATTEMPTS: Final[int] = 3
RECORD_SECONDS: Final[float] = 600.0
UPLOAD_RETRY_SECONDS: Final[float] = 3 * 24 * 3600.0
UPLOAD_PAUSE_CAP_SECONDS: Final[float] = 300.0
ENGINE_PORT: int = 8000
PROGRESS_SECONDS: Final[float] = 15.0
ENV_RUNNER_DIGEST: Final[str] = "MT_RUNNER_DIGEST"
IMAGE_KINDS: Final[tuple[str, ...]] = ("engine", "runner", "sandbox")
ROUTES: Final[dict[str, str]] = {
    HARNESS: contract.HARNESS_RUN,
    ESCALATION: contract.ESCALATION_REPAIR,
    SYSTEM: contract.SYSTEM_RUN,
}


class RunnerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    home: Path
    arena_id: str
    epoch_index: int
    server_url: str
    relay_url: str
    gpus: tuple[str, ...] = ("0",)
    sandbox_runtime: str = ""
    sandbox_cpus: float = SANDBOX_CPUS
    sandbox_memory: str = SANDBOX_MEMORY
    sandbox_pids: int = SANDBOX_PIDS
    request_seconds: float = REQUEST_SECONDS
    record_seconds: float = RECORD_SECONDS
    slm_ready_seconds: int = SLM_READY_SECONDS
    host_services_port: int = 9702
    slm_memory_share: float = 0.0
    dry_run: bool = False
    item_limit: int = 0
    runner_digest: str = ""
    commitment: str = ""
    vllm_image: str = ""
    llama_image: str = ""
    sandbox_image: str = ""
    gateway_image: str = ""
    engine_url: str = ""
    engine_key: str = field(default="", repr=False)
    for_miner: str = ""

    @property
    def remote(self) -> bool:
        return bool(self.engine_url)

    @property
    def work_dir(self) -> Path:
        return self.home / "work" / self.arena_id / str(self.epoch_index)

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"


@dataclass(frozen=True, slots=True)
class InputItem:
    item_id: str
    pools: tuple[str, ...]
    family: str
    task: dict[str, Any]
    twin_of: str
    seed: int
    databases: tuple[str, ...]


@dataclass(slots=True)
class Inputs:
    seed: str
    bundle_sha256: str
    open_block: int
    close_block: int
    items: list[InputItem]
    databases: dict[str, bytes]
    trees: dict[str, Path]
    drafts: dict[str, str]

    def pool(self, name: str, limit: int = 0) -> list[InputItem]:
        chosen = [item for item in self.items if name in item.pools]
        return chosen[:limit] if limit else chosen

    def originals(self, name: str, limit: int = 0) -> list[InputItem]:
        return [item for item in self.pool(name, limit) if not item.twin_of]

    def twins(self, name: str, limit: int = 0) -> list[InputItem]:
        return [item for item in self.pool(name, limit) if item.twin_of]

    def item_databases(self) -> dict[str, tuple[str, ...]]:
        return {item.item_id: item.databases for item in self.items if item.databases}

    @property
    def variant_dirs(self) -> list[Path]:
        return [
            path for name, path in sorted(self.trees.items()) if name.startswith(VARIANT_PREFIX)
        ]


@dataclass(frozen=True, slots=True)
class Images:
    engine: str
    engine_digest: str
    runner: str
    runner_digest: str
    sandbox: str
    sandbox_digest: str
    replayer: str = ""
    replayer_digest: str = ""
    llama: str = ""
    llama_digest: str = ""

    @classmethod
    def of(cls, definition: Mapping[str, Any]) -> Images:
        block = definition.get("evaluation") or {}
        images = block.get("images") if isinstance(block, Mapping) else None
        if not isinstance(images, Mapping):
            raise RunnerError("the arena definition pins no evaluation images")
        found: dict[str, str] = {}
        for kind in ("engine", "runner", "sandbox", "replayer", "llama"):
            entry = images.get(kind) or {}
            name = str(entry.get("image") or "") if isinstance(entry, Mapping) else ""
            digest = str(entry.get("digest") or "") if isinstance(entry, Mapping) else ""
            if kind in IMAGE_KINDS and (not name or not digest):
                raise RunnerError(f"the arena definition pins no {kind} image digest")
            found[kind] = name
            found[f"{kind}_digest"] = digest
        return cls(**found)

    def digests(self) -> dict[str, str]:
        return {
            "engine": self.engine_digest,
            "runner": self.runner_digest,
            "sandbox": self.sandbox_digest,
        }


@dataclass(frozen=True, slots=True)
class MirrorEntry:
    hotkey: str
    status: str
    reason: str
    profile: str
    artifact_sha: str
    manifest_sha256: str
    weights_sha256: str
    artifact_files: dict[str, str]
    slm_repo: str
    slm_revision: str
    slm_files: dict[str, str]

    @classmethod
    def of(cls, hotkey: str, raw: Mapping[str, Any]) -> MirrorEntry:
        artifact = raw.get("artifact") or {}
        slm = raw.get("slm") or {}
        return cls(
            hotkey=hotkey,
            status=str(raw.get("status") or ""),
            reason=str(raw.get("reason") or ""),
            profile=str(raw.get("profile") or ""),
            artifact_sha=str(raw.get("artifact_sha") or ""),
            manifest_sha256=str(raw.get("manifest_sha256") or "").lower(),
            weights_sha256=str(raw.get("weights_sha256") or "").lower(),
            artifact_files={
                str(k): str(v).lower() for k, v in dict(artifact.get("files") or {}).items()
            },
            slm_repo=str(slm.get("repo") or ""),
            slm_revision=str(slm.get("revision") or ""),
            slm_files={str(k): str(v).lower() for k, v in dict(slm.get("files") or {}).items()},
        )


@dataclass(slots=True)
class Progress:
    client: ServerClient | None
    arena_id: str
    epoch_index: int
    miner: str
    total: int
    done: int = 0
    last_sent: float = 0.0
    last_detail: tuple[str, str] = ("", "")

    def step(self, pool: str, condition: str) -> None:
        self.done += 1
        detail = (pool, condition)
        now = time.monotonic()
        if (
            detail == self.last_detail
            and now - self.last_sent < PROGRESS_SECONDS
            and self.done < self.total
        ):
            return
        self.last_detail = detail
        self.last_sent = now
        self._post(RUNNING, {"pool": pool, "condition": condition})

    def uploaded(self) -> None:
        self.done = self.total
        self._post(UPLOADED, None)

    def _post(self, stage: str, detail: Mapping[str, Any] | None) -> None:
        if self.client is None:
            return
        body: dict[str, Any] = {
            "arena_id": self.arena_id,
            "epoch_index": self.epoch_index,
            "miner": self.miner,
            "stage": stage,
            "done": self.done,
            "total": self.total,
        }
        if detail:
            body["detail"] = dict(detail)
        try:
            self.client.report_progress(body)
        except Exception as exc:
            log.debug("progress not sent: %s", exc)


def document_body(document: Mapping[str, Any] | None, key: str, what: str) -> dict[str, Any]:
    if not document:
        raise RunnerError(f"the server has published no {what} document for this epoch")
    if "body" not in document:
        return dict(document)
    if key and document.get("signature"):
        try:
            verify_document(dict(document), key)
        except ReleaseError as exc:
            raise RunnerError(f"the {what} document is not trusted: {exc}") from exc
    return dict(document.get("body") or {})


def parse_inputs(files: Mapping[str, bytes], document: Mapping[str, Any]) -> Inputs:
    raw = files.get(INPUTS_FILE)
    if raw is None:
        raise RunnerError(f"the inputs bundle has no {INPUTS_FILE}")
    items: list[InputItem] = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("item_id"):
            raise RunnerError("the inputs bundle holds a row without an item id")
        task = row.get("task")
        if not isinstance(task, dict):
            raise RunnerError(f"item {row['item_id']} carries no task")
        pools = row.get("pool") or row.get("pools") or []
        listed = [pools] if isinstance(pools, str) else list(pools)
        blobs = row.get("blobs") or row.get("databases") or task.get("blobs") or []
        if isinstance(blobs, str):
            blobs = [blobs]
        items.append(
            InputItem(
                item_id=str(row["item_id"]),
                pools=tuple(str(p) for p in listed),
                family=str(row.get("family") or row["item_id"]),
                task=dict(task),
                twin_of=str(row.get("twin_of") or ""),
                seed=int(row.get("seed") or 0),
                databases=tuple(str(b) for b in blobs),
            )
        )
    databases = {
        name[len(BLOB_PREFIX) :]: body
        for name, body in files.items()
        if name.startswith(BLOB_PREFIX)
    }
    drafts: dict[str, str] = {}
    for line in files.get(DRAFTS_FILE, b"").decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict) and row.get("item_id"):
                drafts[str(row["item_id"])] = str(row.get("draft") or "")
    return Inputs(
        seed=str(document.get("seed") or ""),
        bundle_sha256=str(document.get("bundle_sha256") or "").lower(),
        open_block=int(document.get("open_block") or 0),
        close_block=int(document.get("close_block") or 0),
        items=items,
        databases=databases,
        trees={},
        drafts=drafts,
    )


def write_trees(files: Mapping[str, bytes], into: Path) -> dict[str, Path]:
    names = {
        name[len(REFERENCE_PREFIX) :].split("/", 1)[0]
        for name in files
        if name.startswith(REFERENCE_PREFIX) and "/" in name[len(REFERENCE_PREFIX) :]
    }
    trees: dict[str, Path] = {}
    for name in sorted(names):
        root = into / name
        root.mkdir(parents=True, exist_ok=True)
        prefix = f"{REFERENCE_PREFIX}{name}/"
        for member, body in files.items():
            if member.startswith(prefix):
                target = root / member[len(prefix) :]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
        trees[name] = root
    return trees


def max_tokens_of(definition: Mapping[str, Any]) -> int:
    block = definition.get("generation") or {}
    found = block.get("max_tokens") if isinstance(block, Mapping) else None
    return int(found or SQL_MAX_TOKENS)


def stop_of(definition: Mapping[str, Any]) -> tuple[str, ...]:
    block = definition.get("generation") or {}
    found = block.get("stop", SQL_STOP) if isinstance(block, Mapping) else SQL_STOP
    return tuple(str(s) for s in found)


def slm_prompt_of(definition: Mapping[str, Any]) -> str:
    return str(
        definition.get("slm_prompt")
        or "Schema:\n{schema}\n\nQuestion: {question}\n\n"
        "Write one SQLite query that answers the question.\nSQL:"
    )


def floors_of(definition: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(float(f) for f in definition.get("floors") or (0.8, 0.9, 0.95, 0.99))


def max_model_len_of(definition: Mapping[str, Any], config: Mapping[str, Any]) -> int:
    budgets = definition.get("budgets") or {}
    if isinstance(budgets, Mapping) and budgets.get("max_input_tokens"):
        return int(budgets["max_input_tokens"]) + int(budgets.get("max_output_tokens") or 0)
    slm = config.get("slm") or {}
    return int((slm.get("max_context") if isinstance(slm, Mapping) else None) or DEFAULT_CONTEXT)


def memory_share_of(definition: Mapping[str, Any], profile: str, override: float) -> float:
    if override:
        return override
    budgets = definition.get("budgets") or {}
    cap = int(budgets.get("memory_bytes") or 0) if isinstance(budgets, Mapping) else 0
    total = profiles.memory_bytes(profile) if not profiles.on_cpu(profile) else 0
    if cap and total:
        return max(0.1, min(MEMORY_SHARE, round(cap / total, 2)))
    return MEMORY_SHARE


def relay_allowance_of(definition: Mapping[str, Any]) -> int:
    block = definition.get("evaluation") or {}
    found = block.get("relay_allowance_micros") if isinstance(block, Mapping) else None
    return int(found or 0)


def artifact_config(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / contract.CONFIG_FILE
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(contract.DEFAULT_CONFIG)
    return dict(parsed) if isinstance(parsed, dict) else dict(contract.DEFAULT_CONFIG)


@dataclass(slots=True)
class Miner:
    hotkey: str
    entry: MirrorEntry
    artifact_dir: Path
    slm_dir: Path
    manifest: Manifest
    config: dict[str, Any]
    parameters: int

    @property
    def has(self) -> frozenset[str]:
        return self.manifest.has


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("pool", "")), str(row.get("condition", "")), str(row.get("item_id", "")))


@dataclass(slots=True)
class Journal:
    path: Path
    identity: dict[str, Any]
    rows: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def opened(cls, path: Path, identity: Mapping[str, Any]) -> Journal:
        journal = cls(path, dict(identity))
        journal._load()
        return journal

    def _load(self) -> None:
        raw = self.path.read_bytes() if self.path.is_file() else b""
        good = 0
        position = raw.find(b"\n")
        start = 0
        while position >= 0:
            try:
                parsed = json.loads(raw[start:position])
            except ValueError:
                break
            if not good:
                if parsed != {"journal": self.identity}:
                    log.warning("%s belongs to other inputs; starting over", self.path.name)
                    break
            elif isinstance(parsed, dict):
                self.rows[row_key(parsed)] = parsed
            else:
                break
            good = start = position + 1
            position = raw.find(b"\n", start)
        if not good:
            self.path.write_bytes(self._line({"journal": self.identity}))
            return
        if good < len(raw):
            with self.path.open("r+b") as handle:
                handle.truncate(good)
        if self.rows:
            log.info("%s holds %d finished rows; they are reused", self.path.name, len(self.rows))

    @staticmethod
    def _line(value: Mapping[str, Any]) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"

    def take(self, pool: str, condition: str, item_id: str) -> dict[str, Any] | None:
        return self.rows.get((pool, condition, item_id))

    def add(self, row: Mapping[str, Any]) -> None:
        self.rows[row_key(row)] = dict(row)
        with self.path.open("ab") as handle:
            handle.write(self._line(row))
            handle.flush()
            os.fsync(handle.fileno())


@dataclass(slots=True)
class Session:
    config: RunnerConfig
    docker: Docker
    definition: dict[str, Any]
    images: Images
    inputs: Inputs
    miner: Miner
    served: services.Services
    host: SandboxHost
    desk: TicketDesk | None
    allowance_micros: int
    progress: Progress
    key: str = ""
    network: str = ""
    engine_container: str = ""
    gateway_container: str = ""
    main_container: str = ""
    router_process: subprocess.Popen[bytes] | None = None
    engine_url: str = ""
    router_url: str = ""
    main_url: str = ""
    current_tree: Path | None = None
    session: services.ArtifactSession | None = None
    entries: list[dict[str, Any]] = field(default_factory=list)
    tickets: dict[tuple[str, str], str] = field(default_factory=dict)
    remote_engine: RemoteEngine | None = None
    bridge: EngineBridge | None = None
    journal: Journal | None = None

    @property
    def limits(self) -> Limits:
        budgets = self.definition.get("budgets") or {}
        max_len = (
            int(budgets["max_input_tokens"]) + int(budgets.get("max_output_tokens") or 0)
            if isinstance(budgets, Mapping) and budgets.get("max_input_tokens")
            else None
        )
        return Limits.from_definition(self.definition, max_len)

    @property
    def profile(self) -> str:
        return self.miner.entry.profile

    def open(self) -> None:
        self.key = secrets.token_hex(4)
        self.network = f"mt-miner-{self.key}"
        self.docker.create_network(self.network)
        try:
            self._engine()
            self._router()
            self.start_gateway()
            self._session()
        except Exception:
            self.close()
            raise

    def _start_remote_engine(self) -> None:
        engine = RemoteEngine(self.config.engine_url, self.config.engine_key)
        self.remote_engine = engine
        try:
            card = engine.health().get("gpu_name") or "an unknown card"
            log.info("the remote engine runs on %s", card)
            engine.prepare(self._remote_name(), self.miner.entry.slm_files)
            self._launch_remote(engine)
        except RemoteEngineError as exc:
            raise RunnerError(f"the remote engine: {exc}") from exc
        self.bridge = EngineBridge(engine)
        self.bridge.start()
        self.engine_url = self.bridge.url

    def _remote_name(self) -> str:
        entry = self.miner.entry
        return f"miner-{(entry.weights_sha256 or entry.artifact_sha)[:32]}"

    def _launch_remote(self, engine: RemoteEngine) -> None:
        engine.start(
            self._remote_name(),
            engine_command(
                compat.GUEST,
                profile=self.profile,
                max_model_len=max_model_len_of(self.definition, self.miner.config),
                memory_share=memory_share_of(
                    self.definition, self.profile, self.config.slm_memory_share
                ),
            ),
            engine_env(self.profile),
            float(self.config.slm_ready_seconds),
        )

    def _engine(self) -> None:
        if self.config.remote:
            self._start_remote_engine()
            return
        name = f"mt-engine-{self.key}"
        max_len = max_model_len_of(self.definition, self.miner.config)
        limits = self.limits
        if profiles.on_cpu(self.profile):
            if not self.images.llama or not self.images.llama_digest:
                raise RunnerError("profile C needs the llama image pinned in the arena definition")
            reference = self.docker.ensure_digest(self.images.llama, self.images.llama_digest)
            if not limits.cpu_threads:
                limits = replace(limits, cpu_threads=profiles.CPU_THREADS)
            self.docker.start(
                cpu_container(name, reference, self.miner.slm_dir, max_len, limits, self.network)
            )
        else:
            parallel = profiles.tensor_parallel(self.profile)
            if len(self.config.gpus) < parallel:
                raise RunnerError(
                    f"profile {self.profile} needs {parallel} GPUs; pass them with --gpus"
                )
            reference = self.docker.ensure_digest(self.images.engine, self.images.engine_digest)
            self.docker.start(
                ContainerSpec(
                    name=name,
                    image=reference,
                    network=self.network,
                    command=engine_command(
                        compat.GUEST,
                        profile=self.profile,
                        max_model_len=max_len,
                        memory_share=memory_share_of(
                            self.definition, self.profile, self.config.slm_memory_share
                        ),
                    ),
                    env=engine_env(self.profile),
                    volumes=compat.plain(self.miner.slm_dir).volumes,
                    gpus=tuple(self.config.gpus[:parallel]),
                    shm_size="8g",
                    ipc_none=False,
                )
            )
        self.engine_container = name
        self.engine_url = f"http://{self.docker.address(name, self.network)}:{ENGINE_PORT}"
        client = SlmClient(self.engine_url)
        deadline = time.monotonic() + self.config.slm_ready_seconds
        while time.monotonic() < deadline:
            if self.docker.exited(name):
                raise RunnerError(f"the engine exited: {self.docker.exit_note(name)}")
            if client.ready():
                return
            time.sleep(5.0)
        detail = self.docker.failure(name)
        raise RunnerError("the engine did not become ready" + (f": {detail}" if detail else ""))

    def _router_dir(self) -> Path | None:
        if "router" in self.miner.has:
            return self.miner.artifact_dir / contract.ROUTER_DIR
        found = self.inputs.trees.get(REFERENCE_ROUTER)
        if found is not None:
            return found
        references = dict(self.definition.get("references") or {})
        repo = str(references.get("router") or "")
        if not repo:
            return None
        try:
            return collect.download(
                repo, str(references.get("router_revision") or "main"), self.config.cache_dir
            )
        except Exception as exc:
            log.warning("the reference router could not be fetched: %s", exc)
            return None

    def _router(self) -> None:
        router_dir = self._router_dir()
        if router_dir is None:
            log.warning("no router to serve; /router/decide answers 404")
            return
        self.router_process, self.router_url = self.host.open_router(
            router_dir,
            slot=0,
            gpu=self.config.gpus[0],
            config=self.miner.config,
            limits=self.limits,
            device="cpu",
        )

    def start_gateway(self) -> None:
        name = f"mt-gw-{self.key}"
        reference = self.docker.ensure_digest(self.images.runner, self.images.runner_digest)
        upstream = self.docker.own_address(self.network) or self.host.host_ip
        self.docker.start(
            ContainerSpec(
                name=name,
                image=reference,
                network="bridge",
                entrypoint="python",
                command=(
                    "-m",
                    "microtensor_subnet.validator.sandbox.gateway",
                    "--upstream",
                    f"http://{upstream}:{self.served.port}",
                    "--sandbox",
                    self.key,
                    "--port",
                    str(GATEWAY_PORT),
                ),
                cpus=1.0,
                memory="512m",
                pids=64,
                read_only=True,
                hardened=False,
                extra_networks=(self.network,),
                tmpfs=("/tmp",),  # noqa: S108
            )
        )
        self.gateway_container = name

    def _session(self) -> None:
        sample_sha = str(self.definition.get("sample_database") or "")
        sample = self.inputs.databases.get(sample_sha) or next(
            iter(self.inputs.databases.values()), b""
        )
        limits = self.limits
        self.session = services.recording_session(
            sample_database=sample,
            fault_plan=tools.FaultPlan(self.inputs.seed),
            slm_url=self.engine_url,
            router_url=self.router_url,
            desk=self.desk,
            databases=self.inputs.databases,
            item_databases=self.inputs.item_databases(),
            max_model_calls=limits.max_model_calls,
            model_faults=limits.model_faults,
        )
        self.served.attach(self.key, self.session)

    def main(self, tree: Path) -> str:
        if tree == self.current_tree and self.main_url:
            return self.main_url
        if self.main_container:
            self.docker.kill(self.main_container)
            self.main_container = ""
        name = f"mt-run-{self.key}"
        reference = self.docker.ensure_digest(self.images.sandbox, self.images.sandbox_digest)
        gateway_ip = self.docker.address(self.gateway_container, self.network)
        base = f"http://{gateway_ip}:{GATEWAY_PORT}"
        self.docker.start(
            ContainerSpec(
                name=name,
                image=reference,
                network=self.network,
                command=("python", "-X", "utf8", "/artifact/main.py"),
                env={
                    contract.ENV_PORT: str(MAIN_PORT),
                    contract.ENV_SLM_URL: base + "/slm",
                    contract.ENV_ROUTER_URL: base + "/router",
                    contract.ENV_TOOLS_URL: base,
                    contract.ENV_RELAY_URL: base + "/relay",
                    "HOME": "/tmp",  # noqa: S108
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "FAKETIME": contract.FAKETIME,
                    "LD_PRELOAD": contract.FAKETIME_LIBRARY,
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
        self.main_container = name
        self.main_url = f"http://{self.docker.address(name, self.network)}:{MAIN_PORT}"
        self.current_tree = tree
        wait_main(self.docker, name, self.main_url)
        return self.main_url

    def stop_main(self) -> None:
        if self.main_container:
            self.docker.kill(self.main_container)
        self.main_container = ""
        self.main_url = ""
        self.current_tree = None

    def revive_engine(self, *, force: bool = False) -> None:
        if self.remote_engine is not None:
            try:
                if force or not self.remote_engine.ready():
                    log.warning("the remote engine is not ready; starting it again")
                    self._launch_remote(self.remote_engine)
            except RemoteEngineError as exc:
                raise RunnerError(f"the remote engine: {exc}") from exc
            return
        if not self.engine_container:
            return
        running = not force and not self.docker.exited(self.engine_container)
        if running and SlmClient(self.engine_url).ready():
            return
        log.warning("the engine is not answering; starting it again")
        self.docker.kill(self.engine_container)
        self.engine_container = ""
        self._engine()
        if self.session is not None:
            self.session.slm_url = self.engine_url

    def close(self) -> None:
        if self.key:
            self.served.detach(self.key)
        for name in (self.main_container, self.gateway_container, self.engine_container):
            if name:
                self.docker.kill(name)
        if self.router_process is not None:
            self.host.close_router(self.router_process)
        if self.network:
            self.docker.remove_network(self.network)
        if self.bridge is not None:
            self.bridge.stop()
            self.bridge = None
        if self.remote_engine is not None:
            try:
                self.remote_engine.cleanup()
            except RemoteEngineError as exc:
                log.warning("the remote engine was not cleaned: %s", exc)
            self.remote_engine = None

    def generate(self, item: InputItem, pool: str, condition: str) -> dict[str, Any]:
        prompt = render_prompt(slm_prompt_of(self.definition), item.task)
        max_tokens = max_tokens_of(self.definition)
        stop = stop_of(self.definition)
        last = ""
        for attempt in range(1, ENGINE_ATTEMPTS + 1):
            client = SlmClient(self.engine_url, timeout=self.config.record_seconds)
            try:
                found = client.generate(
                    prompt, max_tokens=max_tokens, stop=stop, seed=item.seed, logprobs=1
                )
                if found.text and not found.output_tokens:
                    raise SlmError("the engine answered with text but no token ids")
            except SlmError as exc:
                last = str(exc)
                log.warning("%s: engine call %d failed: %s", item.item_id, attempt, exc)
                time.sleep(2.0 * attempt)
                if attempt < ENGINE_ATTEMPTS:
                    self.revive_engine(force=attempt > 1)
                continue
            call = transcript.slm_call(
                0,
                prompt=prompt,
                prompt_tokens=found.prompt_tokens,
                max_tokens=max_tokens,
                stop=stop,
                seed=item.seed,
                output_tokens=found.output_tokens,
                output_text=found.text,
                finish_reason=found.finish_reason,
                ttft_ms=found.ttft_ms,
                total_ms=found.total_ms,
            )
            return transcript.item_entry(
                item_id=item.item_id,
                pool=pool,
                condition=condition,
                seed=item.seed,
                calls=[call],
                output={"answer": found.text},
            )
        raise RunnerError(f"the engine failed {ENGINE_ATTEMPTS} times on {item.item_id}: {last}")

    @property
    def planned_repairs(self) -> int:
        if "escalation" not in self.miner.has:
            return 0
        chosen = self.inputs.pool(ESCALATION, self.config.item_limit)
        return sum(1 for item in chosen if item.item_id in self.inputs.drafts)

    @property
    def cap_micros(self) -> int:
        planned = self.planned_repairs + len(self.inputs.pool(SYSTEM, self.config.item_limit))
        return max(1, self.allowance_micros // max(1, planned))

    def ticket(self, item_id: str, purpose: str, *, renew: bool = False) -> str:
        if self.config.dry_run:
            return ""
        if self.desk is None:
            raise RunnerError(
                f"{item_id}: no relay token, so no {purpose} ticket; nothing uploaded"
            )
        found = self.tickets.get((item_id, purpose))
        if found is None or renew:
            if found is not None:
                self.desk.forget(found)
            found = self._issue(self.desk, item_id, purpose)
            self.tickets[(item_id, purpose)] = found
        return found

    def _issue(self, desk: TicketDesk, item_id: str, purpose: str) -> str:
        deadline = time.monotonic() + RETRY_SECONDS
        attempt = 0
        while True:
            attempt += 1
            try:
                return desk.issue(
                    miner=self.miner.hotkey,
                    item_id=item_id,
                    purpose=purpose,
                    cap_micros=self.cap_micros,
                )
            except RelayError as exc:
                if not exc.transient or time.monotonic() >= deadline:
                    raise RunnerError(
                        f"{item_id}: the relay refused the {purpose} ticket ({exc.code}: {exc}); "
                        "nothing uploaded"
                    ) from exc
                log.warning("%s: no %s ticket yet, retrying: %s", item_id, purpose, exc)
                time.sleep(retry_pause(attempt))

    def run_main(
        self,
        tree: Path,
        route: str,
        item: InputItem,
        pool: str,
        condition: str,
        body: dict[str, Any],
        *,
        purpose: str = "",
    ) -> dict[str, Any]:
        if self.session is None:
            raise RunnerError("the recording session is not open")
        failures = 0
        outages = 0
        outage_ends = 0.0
        renew = False
        while True:
            request = {**body, "seed": item.seed}
            if purpose:
                request["ticket"] = self.ticket(item.item_id, purpose, renew=renew)
            url = self.main(tree)
            self.session.begin_item(
                item.item_id,
                item.seed,
                pool=pool,
                condition=condition,
                request=transcript.compact_request(request),
                purpose=purpose,
                cap_micros=self.cap_micros if purpose else 0,
            )
            call = call_main(url, route, request, timeout=self.config.record_seconds)
            output: Any = call.body if call.ok else {"error": call.error}
            problem, source = self._row_problem(call)
            if not problem:
                entry = self.session.recorded(output)
                self.session.finish_item()
                return entry
            self.session.finish_item()
            renew = True
            self.stop_main()
            where = f"{item.item_id} ({pool} {condition})"
            log.warning("%s has to run again: %s", where, problem)
            if source == services.RELAY_SOURCE:
                outages += 1
                outage_ends = outage_ends or time.monotonic() + RETRY_SECONDS
                if time.monotonic() < outage_ends:
                    time.sleep(retry_pause(outages))
                    continue
            failures += 1
            if failures >= ROW_ATTEMPTS or source == services.RELAY_SOURCE:
                raise RunnerError(
                    f"{where} failed {failures} times: {problem}; nothing uploaded. "
                    "Run the same command to continue from this row"
                )
            time.sleep(2.0 * failures)
            if source == services.ENGINE_SOURCE:
                self.revive_engine(force=failures > 1)

    def _row_problem(self, call: Call) -> tuple[str, str]:
        if call.error == "timeout":
            limit = self.config.record_seconds
            return f"main.py did not answer within {limit:.0f} s", services.SERVICE_SOURCE
        if call.error.startswith("unreachable") or not self.docker.running(self.main_container):
            detail = self.docker.logs(self.main_container, 40)[-300:]
            return f"main.py died: {call.error} {detail}".strip(), services.SERVICE_SOURCE
        return self.session.row_problem() if self.session is not None else ("", "")

    def record(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)
        self.progress.step(str(entry.get("pool", "")), str(entry.get("condition", "")))

    def reused(self, item: InputItem, pool: str, condition: str) -> bool:
        kept = None
        if self.journal is not None:
            kept = self.journal.take(pool, condition, item.item_id)
        if kept is None:
            return False
        self.record(kept)
        return True

    def finish(self, entry: dict[str, Any]) -> None:
        if self.journal is not None:
            self.journal.add(entry)
        self.record(entry)

    def harness_tree(self) -> Path:
        if {"harness", "escalation"} & self.miner.has:
            return self.miner.artifact_dir
        found = self.inputs.trees.get(REFERENCE_HARNESS)
        if found is None:
            raise RunnerError("the inputs bundle carries no reference harness")
        return found

    def reference_trees(self) -> list[Path]:
        harness = self.inputs.trees.get(REFERENCE_HARNESS)
        if harness is None:
            raise RunnerError("the inputs bundle carries no reference harness")
        return [harness, *self.inputs.variant_dirs]

    def plan(self) -> int:
        limit = self.config.item_limit
        slm = len(self.inputs.pool(SLM, limit))
        harness = len(self.inputs.pool(HARNESS, limit))
        conditions = 1 + len(self.reference_trees()) + (1 if "harness" in self.miner.has else 0)
        system = len(self.inputs.pool(SYSTEM, limit)) * len(floors_of(self.definition))
        return slm + harness * conditions + self.planned_repairs + system

    def run_pools(self) -> None:
        limit = self.config.item_limit
        for item in [*self.inputs.originals(SLM, limit), *self.inputs.twins(SLM, limit)]:
            if not self.reused(item, SLM, transcript.MINER):
                self.finish(self.generate(item, SLM, transcript.MINER))
        harness_items = self.inputs.pool(HARNESS, limit)
        if "harness" in self.miner.has:
            for item in harness_items:
                if self.reused(item, HARNESS, transcript.MINER):
                    continue
                self.finish(
                    self.run_main(
                        self.miner.artifact_dir,
                        contract.HARNESS_RUN,
                        item,
                        HARNESS,
                        transcript.MINER,
                        {"item_id": item.item_id, "task": dict(item.task)},
                    )
                )
        for item in harness_items:
            if not self.reused(item, HARNESS, transcript.BARE):
                self.finish(self.generate(item, HARNESS, transcript.BARE))
        for index, tree in enumerate(self.reference_trees(), start=1):
            condition = transcript.reference_condition(index)
            for item in harness_items:
                if self.reused(item, HARNESS, condition):
                    continue
                self.finish(
                    self.run_main(
                        tree,
                        contract.HARNESS_RUN,
                        item,
                        HARNESS,
                        condition,
                        {"item_id": item.item_id, "task": dict(item.task)},
                    )
                )
        if "escalation" in self.miner.has:
            for item in self.inputs.pool(ESCALATION, limit):
                draft = self.inputs.drafts.get(item.item_id)
                if draft is None or self.reused(item, ESCALATION, transcript.REPAIR):
                    continue
                self.finish(
                    self.run_main(
                        self.miner.artifact_dir,
                        contract.ESCALATION_REPAIR,
                        item,
                        ESCALATION,
                        transcript.REPAIR,
                        {
                            "item_id": item.item_id,
                            "task": dict(item.task),
                            "draft": draft,
                            "trace": "",
                            "errors": ERRORS_TEXT,
                        },
                        purpose=REPAIR,
                    )
                )
        system_items = self.inputs.pool(SYSTEM, limit)
        for floor in floors_of(self.definition):
            condition = transcript.system_condition(floor)
            for item in system_items:
                if self.reused(item, SYSTEM, condition):
                    continue
                self.finish(
                    self.run_main(
                        self.harness_tree(),
                        contract.SYSTEM_RUN,
                        item,
                        SYSTEM,
                        condition,
                        {
                            "item_id": item.item_id,
                            "task": dict(item.task),
                            "floor": floor,
                        },
                        purpose=SYSTEM_PURPOSE,
                    )
                )


def wait_main(docker: Docker, name: str, url: str) -> None:
    deadline = time.monotonic() + READY_SECONDS
    while time.monotonic() < deadline:
        if not docker.running(name):
            raise RunnerError(f"main.py exited: {docker.logs(name, 40)[-300:]}")
        try:
            with httpx.Client(timeout=5.0) as http:
                if http.get(url + "/health").status_code < 500:
                    return
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    raise RunnerError("main.py did not start listening")


def load_definition(client: ServerClient, arena_id: str) -> dict[str, Any]:
    try:
        arena = client.arena(arena_id)
    except ServerError as exc:
        raise RunnerError(str(exc)) from exc
    if arena is None:
        raise RunnerError(f"the server has no arena {arena_id}")
    definition = dict(arena.get("definition") or {})
    if not profiles.miner_mode(definition):
        raise RunnerError(f"{arena_id} is evaluated by validators; nothing to run")
    return definition


def mirror_entry(client: ServerClient, config: RunnerConfig, hotkey: str, key: str) -> MirrorEntry:
    try:
        document = client.mirror(config.arena_id, config.epoch_index)
    except ServerError as exc:
        raise RunnerError(str(exc)) from exc
    body = document_body(document, key, "mirror")
    miners = body.get("miners") or {}
    raw = miners.get(hotkey) if isinstance(miners, Mapping) else None
    if not isinstance(raw, Mapping):
        raise RunnerError(f"{hotkey} is not on the mirror for epoch {config.epoch_index}")
    entry = MirrorEntry.of(hotkey, raw)
    if entry.status != MIRRORED:
        raise RunnerError(
            f"the mirror lists this miner as {entry.status or 'unknown'}"
            + (f": {entry.reason}" if entry.reason else "")
        )
    return entry


def add_template_harness(
    client: ServerClient,
    config: RunnerConfig,
    definition: Mapping[str, Any],
    inputs: Inputs,
) -> None:
    if REFERENCE_HARNESS in inputs.trees:
        return
    try:
        arena = client.arena(config.arena_id) or {}
        if arena.get("network") != ARENA_NETWORKS[TESTNET]:
            return
        inputs.trees[REFERENCE_HARNESS] = template_harness(
            str(definition.get("task") or ""), config.cache_dir
        )
    except (ReleaseError, ServerError, OSError) as exc:
        raise RunnerError(str(exc)) from exc


def fetch_inputs(
    client: ServerClient, config: RunnerConfig, key: str, definition: Mapping[str, Any]
) -> Inputs:
    try:
        document = client.inputs(config.arena_id, config.epoch_index)
    except ServerError as exc:
        raise RunnerError(str(exc)) from exc
    body = document_body(document, key, "inputs")
    sha = str(body.get("bundle_sha256") or "").lower()
    cached = config.work_dir / "inputs.tar"
    if cached.is_file() and sha and bare_hex(sha256_of(cached)) == sha:
        blob = cached.read_bytes()
    else:
        try:
            blob = client.inputs_bundle(config.arena_id, config.epoch_index, sha)
        except ServerError as exc:
            raise RunnerError(str(exc)) from exc
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(blob)
    try:
        files = unpack_bundle(blob)
    except (ItemError, OSError) as exc:
        raise RunnerError(f"the inputs bundle cannot be read: {exc}") from exc
    inputs = parse_inputs(files, body)
    inputs.trees = write_trees(files, config.work_dir / "reference")
    add_template_harness(client, config, definition, inputs)
    return inputs


def fetch_miner(
    client: ServerClient,
    config: RunnerConfig,
    definition: Mapping[str, Any],
    entry: MirrorEntry,
) -> Miner:
    allowed = profiles.allowed_profiles(definition)
    if entry.profile not in allowed:
        raise RunnerError(
            f"profile {entry.profile or 'none'} is not allowed in this arena "
            f"({', '.join(allowed) or 'no profiles'})"
        )
    artifact_dir = config.work_dir / "artifact"
    for name, sha in sorted(entry.artifact_files.items()):
        try:
            client.mirror_file(sha, artifact_dir / name)
        except ServerError as exc:
            raise RunnerError(f"artifact file {name}: {exc}") from exc
    try:
        manifest = load_manifest(artifact_dir)
    except ManifestError as exc:
        raise RunnerError(f"the mirrored manifest is invalid: {exc}") from exc
    found = bare_hex(manifest_sha256(artifact_dir))
    if found != entry.manifest_sha256:
        raise RunnerError(f"manifest {found[:12]} is not the mirrored {entry.manifest_sha256[:12]}")
    if not manifest.declares_profile or manifest.profile != entry.profile:
        raise RunnerError("the manifest profile does not match the commitment profile")
    slm_dir = config.cache_dir / "slm" / (entry.weights_sha256 or entry.artifact_sha)
    for name, sha in sorted(entry.slm_files.items()):
        if config.remote and name.endswith(contract.WEIGHT_SUFFIXES):
            continue
        try:
            client.mirror_file(sha, slm_dir / name)
        except ServerError as exc:
            raise RunnerError(f"slm file {name}: {exc}") from exc
    if config.remote:
        return Miner(
            hotkey=entry.hotkey,
            entry=entry,
            artifact_dir=artifact_dir,
            slm_dir=slm_dir,
            manifest=manifest,
            config=artifact_config(artifact_dir),
            parameters=0,
        )
    checked = preflight.check(
        slm_dir,
        profile=entry.profile,
        bounds=preflight.bounds_of(dict(definition)),
        max_len=preflight.max_len_of(dict(definition)),
        formats=preflight.formats_of(dict(definition)),
    )
    if not checked.ok:
        raise RunnerError(f"pre-flight: {checked.message}")
    if checked.weights_sha256 != entry.weights_sha256:
        raise RunnerError("the mirrored weights do not hash to the committed weights_sha256")
    return Miner(
        hotkey=entry.hotkey,
        entry=entry,
        artifact_dir=artifact_dir,
        slm_dir=slm_dir,
        manifest=manifest,
        config=artifact_config(artifact_dir),
        parameters=checked.parameters,
    )


def check_profile(entry: MirrorEntry, gpus: Sequence[str]) -> None:
    problem = profiles.card_problem(entry.profile, gpus)
    if problem:
        raise RunnerError(f"{problem}. The commitment declares {entry.profile}")


def runner_digest(config: RunnerConfig, images: Images) -> str:
    if config.runner_digest and config.runner_digest.lower() != images.runner_digest.lower():
        raise RunnerError(
            f"this runner image is {config.runner_digest}, not the pinned {images.runner_digest}"
        )
    return images.runner_digest


def relay_token(client: ServerClient, arena_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + RETRY_SECONDS
    attempt = 0
    while True:
        attempt += 1
        try:
            return client.miner_relay_token(arena_id)
        except ServerError as exc:
            if not exc.retryable or time.monotonic() >= deadline:
                raise
            log.warning("no miner relay token yet, retrying: %s", exc)
            time.sleep(retry_pause(attempt))


def open_relay(client: ServerClient, config: RunnerConfig) -> tuple[TicketDesk | None, int]:
    try:
        token = client.relay_token() if config.for_miner else relay_token(client, config.arena_id)
    except ServerError as exc:
        log.warning("no miner relay token (%s); escalation and system calls are refused", exc)
        return None, 0
    secret = str(token.get("token") or "")
    if not secret:
        return None, 0
    relay = RelayClient(config.relay_url, secret)
    return TicketDesk(relay, config.arena_id, config.epoch_index), int(
        token.get("allowance_micros") or 0
    )


def add_coordinator_drafts(
    client: ServerClient, config: RunnerConfig, key: str, inputs: Inputs, miner: Miner
) -> None:
    if inputs.drafts or "escalation" not in miner.has or not inputs.pool(ESCALATION):
        return
    found = spotcheck.fetch(client, config.arena_id, config.epoch_index, key, leaves={})
    if found is not None and found.drafts_complete:
        inputs.drafts = dict(found.drafts)
        return
    if config.dry_run:
        log.warning("the reference drafts are not published yet; this dry run has no repair rows")
        return
    raise RunnerError(
        "the coordinator has not finished publishing the reference drafts for this epoch; "
        "run again when it has, because a run without them has no repair rows"
    )


def build_transcript(
    config: RunnerConfig, miner: Miner, images: Images, entries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not config.commitment:
        raise RunnerError("the on-chain commitment string is unknown; it seals the transcript")
    return transcript.Transcript(
        arena_id=config.arena_id,
        epoch=config.epoch_index,
        miner_hotkey=miner.hotkey,
        profile=miner.entry.profile,
        commitment_sha256=transcript.commitment_sha256(config.commitment),
        manifest_sha256=miner.entry.manifest_sha256,
        weights_sha256=miner.entry.weights_sha256,
        images={**images.digests(), "runner": runner_digest(config, images)},
        items=[dict(e) for e in entries],
    ).to_dict()


@dataclass(frozen=True, slots=True)
class Upload:
    transcript_sha256: str
    path: Path
    receipt: dict[str, Any]
    items: int


def send_transcript(
    client: ServerClient, arena_id: str, epoch_index: int, signed: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    compressed = gzip.compress(transcript.encode(signed))
    deadline = time.monotonic() + UPLOAD_RETRY_SECONDS
    attempt = 0
    while True:
        attempt += 1
        try:
            receipt = client.upload_transcript(arena_id, epoch_index, compressed)
            break
        except ServerError as exc:
            if exc.retryable and time.monotonic() < deadline:
                pause = min(UPLOAD_PAUSE_CAP_SECONDS, 2.0 ** min(attempt, 16))
                log.warning("upload %d failed, retrying in %.0f s: %s", attempt, pause, exc)
                time.sleep(pause)
                continue
            raise RunnerError(
                f"upload: {exc}. The transcript is kept at {path}; send it with "
                f"mt miner upload --arena {arena_id} --epoch {epoch_index} --transcript {path}"
            ) from exc
    if str(receipt.get("transcript_sha256") or "").lower() != transcript.transcript_sha256(signed):
        raise RunnerError("the server receipt names another transcript sha256")
    path.with_name(RECEIPT_FILE).write_text(json.dumps(receipt, indent=2), "utf-8")
    return receipt


def upload_saved(path: Path, arena_id: str, epoch_index: int, *, client: ServerClient) -> Upload:
    found = load_transcript(path)
    if str(found.get("arena_id")) != arena_id or str(found.get("epoch")) != str(epoch_index):
        raise RunnerError(f"{path} is for {found.get('arena_id')} epoch {found.get('epoch')}")
    if str(found.get("miner_hotkey")) != client.hotkey:
        raise RunnerError(f"{path} belongs to {found.get('miner_hotkey')}, not this hotkey")
    if not transcript.verify_signature(found):
        raise RunnerError(f"{path} is not signed by {client.hotkey}")
    receipt = send_transcript(client, arena_id, epoch_index, found, path)
    items = found.get("items") or []
    return Upload(transcript.transcript_sha256(found), path, receipt, len(items))


def journal_identity(config: RunnerConfig, miner: Miner, inputs: Inputs) -> dict[str, Any]:
    return {
        "arena_id": config.arena_id,
        "epoch": config.epoch_index,
        "manifest_sha256": miner.entry.manifest_sha256,
        "weights_sha256": miner.entry.weights_sha256,
        "inputs_sha256": inputs.bundle_sha256,
    }


def signed_again(path: Path, body: Mapping[str, Any], wallet: Any) -> dict[str, Any]:
    try:
        held = transcript.decode(path.read_bytes()) if path.is_file() else {}
    except (OSError, transcript.TranscriptError):
        held = {}
    if (
        held
        and transcript.canonical_equal(transcript.unsigned(held), transcript.unsigned(body))
        and transcript.verify_signature(held)
    ):
        return held
    return transcript.sign(body, wallet)


def services_listen() -> str:
    return "0.0.0.0" if own_container() else host_ip()  # noqa: S104


def run(
    config: RunnerConfig, *, client: ServerClient, wallet: Any, docker: Docker | None = None
) -> Upload:
    hotkey = config.for_miner or client.hotkey
    engine = docker or Docker()
    definition = load_definition(client, config.arena_id)
    images = Images.of(definition)
    try:
        key = server_key(client)
    except (ReleaseError, ServerError) as exc:
        raise RunnerError(str(exc)) from exc
    entry = mirror_entry(client, config, hotkey, key)
    if not config.remote:
        check_profile(entry, config.gpus)
    engine.sweep()
    config.work_dir.mkdir(parents=True, exist_ok=True)
    inputs = fetch_inputs(client, config, key, definition)
    miner = fetch_miner(client, config, definition, entry)
    add_coordinator_drafts(client, config, key, inputs, miner)
    desk, allowance = (None, 0) if config.dry_run else open_relay(client, config)
    allowance = (
        relay_allowance_of(definition)
        if config.for_miner
        else (allowance or relay_allowance_of(definition))
    )
    if desk is not None and not allowance:
        raise RunnerError("the arena sets no relay allowance; frontier calls cannot be capped")
    address = host_ip()
    served = services.Services(services_listen(), config.host_services_port)
    served.start()
    host = SandboxHost(config, engine, served, address)
    session = Session(
        config=config,
        docker=engine,
        definition=definition,
        images=images,
        inputs=inputs,
        miner=miner,
        served=served,
        host=host,
        desk=desk,
        allowance_micros=allowance,
        progress=Progress(
            None if config.dry_run else client, config.arena_id, config.epoch_index, hotkey, 0
        ),
        journal=Journal.opened(
            config.work_dir / (DRY_JOURNAL_FILE if config.dry_run else JOURNAL_FILE),
            journal_identity(config, miner, inputs),
        ),
    )
    session.progress.total = session.plan()
    try:
        session.open()
        session.run_pools()
    except DockerError as exc:
        raise RunnerError(str(exc)) from exc
    finally:
        session.close()
        served.stop()
    body = build_transcript(config, miner, images, session.entries)
    if config.for_miner:
        body = {**body, "signed_by": client.hotkey}
    path = config.work_dir / (DRY_TRANSCRIPT_FILE if config.dry_run else TRANSCRIPT_FILE)
    signed = signed_again(path, body, wallet)
    sha = transcript.transcript_sha256(signed)
    saving = path.with_name(path.name + ".part")
    saving.write_bytes(transcript.encode(signed))
    os.replace(saving, path)
    receipt: dict[str, Any] = {}
    if not config.dry_run:
        receipt = send_transcript(client, config.arena_id, config.epoch_index, signed, path)
        session.progress.uploaded()
    return Upload(transcript_sha256=sha, path=path, receipt=receipt, items=len(session.entries))


@dataclass(slots=True)
class ReplayOutcome:
    ok: bool
    reason: str = ""
    items: int = 0
    calls: int = 0
    first_mismatch: dict[str, Any] | None = None

    def fail(self, item_id: str, seq: int, what: str) -> ReplayOutcome:
        self.ok = False
        self.reason = f"{what} at {item_id} call {seq}" if item_id else what
        self.first_mismatch = {"item_id": item_id, "seq": seq, "what": what}
        return self


def load_transcript(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    found = transcript.decode(raw)
    problems = transcript.validate(found)
    if problems:
        raise RunnerError(problems[0])
    return found


def _direct_check(row: Mapping[str, Any], item: InputItem, definition: Mapping[str, Any]) -> str:
    calls = [c for c in row.get("calls") or [] if isinstance(c, Mapping)]
    if len(calls) != 1 or calls[0].get("kind") != transcript.SLM:
        return "extra call" if len(calls) > 1 else "missing call"
    call = calls[0]
    prompt = render_prompt(slm_prompt_of(definition), item.task)
    if call.get("prompt_sha256") != transcript.prompt_sha256(prompt):
        return "prompt"
    if int(call.get("max_tokens") or 0) != max_tokens_of(definition):
        return "max_tokens"
    if [str(s) for s in call.get("stop") or []] != list(stop_of(definition)):
        return "stop"
    if int(call.get("seed") or 0) != item.seed:
        return "seed"
    expected = {"answer": str(call.get("output_text") or "")}
    if not transcript.canonical_equal(row.get("output"), expected):
        return "output"
    return ""


def tree_for(
    condition: str, artifact_dir: Path, has: frozenset[str], inputs: Inputs
) -> Path | None:
    if condition == transcript.BARE:
        return None
    harness = inputs.trees.get(REFERENCE_HARNESS)
    if condition.startswith("B") and condition[1:].isdigit():
        index = int(condition[1:]) - 1
        variants = [harness, *inputs.variant_dirs] if harness is not None else []
        return variants[index] if 0 <= index < len(variants) else None
    if {"harness", "escalation"} & has:
        return artifact_dir
    return harness


def replay(
    config: RunnerConfig,
    *,
    client: ServerClient,
    recorded: Mapping[str, Any],
    artifact_dir: Path | None = None,
    docker: Docker | None = None,
) -> ReplayOutcome:
    engine = docker or Docker()
    definition = load_definition(client, config.arena_id)
    images = Images.of(definition)
    try:
        key = server_key(client)
    except (ReleaseError, ServerError) as exc:
        raise RunnerError(str(exc)) from exc
    config.work_dir.mkdir(parents=True, exist_ok=True)
    inputs = fetch_inputs(client, config, key, definition)
    root = artifact_dir or config.work_dir / "artifact"
    try:
        manifest = load_manifest(root)
    except ManifestError as exc:
        raise RunnerError(f"{root}: {exc}") from exc
    known = {item.item_id: item for item in inputs.items}
    limits = Limits.from_definition(definition, None)
    outcome = ReplayOutcome(ok=True)
    sample_sha = str(definition.get("sample_database") or "")
    session = services.replay_session(
        databases=inputs.databases,
        fault_plan=tools.FaultPlan(inputs.seed),
        server_key=key,
        item_databases=inputs.item_databases(),
        sample_database=inputs.databases.get(sample_sha)
        or next(iter(inputs.databases.values()), b""),
        miner_hotkey=str(recorded.get("miner_hotkey") or ""),
        arena_id=config.arena_id,
        epoch_index=config.epoch_index,
        max_model_calls=limits.max_model_calls,
        model_faults=limits.model_faults,
    )
    image = images.replayer or images.sandbox
    digest = images.replayer_digest or images.sandbox_digest
    served = services.Services(services_listen(), config.host_services_port)
    served.start()
    host = SandboxHost(config, engine, served, host_ip())
    runtime = Session(
        config=config,
        docker=engine,
        definition=definition,
        images=Images(
            engine=images.engine,
            engine_digest=images.engine_digest,
            runner=images.runner,
            runner_digest=images.runner_digest,
            sandbox=image,
            sandbox_digest=digest,
        ),
        inputs=inputs,
        miner=Miner(
            hotkey=str(recorded.get("miner_hotkey") or ""),
            entry=MirrorEntry.of(str(recorded.get("miner_hotkey") or ""), {}),
            artifact_dir=root,
            slm_dir=root,
            manifest=manifest,
            config=artifact_config(root),
            parameters=0,
        ),
        served=served,
        host=host,
        desk=None,
        allowance_micros=0,
        progress=Progress(None, config.arena_id, config.epoch_index, "", 0),
    )
    runtime.key = secrets.token_hex(4)
    runtime.network = f"mt-replay-{runtime.key}"
    engine.create_network(runtime.network)
    served.attach(runtime.key, session)
    try:
        runtime.start_gateway()
        for row in recorded.get("items") or []:
            if not isinstance(row, Mapping):
                continue
            item_id = str(row.get("item_id") or "")
            condition = str(row.get("condition") or "")
            pool = str(row.get("pool") or "")
            item = known.get(item_id)
            if item is None:
                return outcome.fail(item_id, 0, "unknown item")
            if int(row.get("seed") or 0) != item.seed:
                return outcome.fail(item_id, 0, "seed")
            outcome.items += 1
            if pool == SLM or condition == transcript.BARE:
                reason = _direct_check(row, item, definition)
                if reason:
                    return outcome.fail(item_id, 0, reason)
                outcome.calls += 1
                continue
            route = ROUTES.get(pool)
            tree = tree_for(condition, root, manifest.has, inputs)
            if route is None or tree is None:
                return outcome.fail(item_id, 0, "condition")
            url = runtime.main(tree)
            purpose = (
                REPAIR
                if condition == transcript.REPAIR
                else SYSTEM_PURPOSE
                if transcript.floor_of(condition) is not None
                else ""
            )
            calls = [dict(c) for c in row.get("calls") or [] if isinstance(c, Mapping)]
            request = row.get("request")
            if not isinstance(request, Mapping) or request.get(
                transcript.TASK_SHA256
            ) != transcript.task_sha256(item.task):
                return outcome.fail(item_id, 0, "request task")
            session.begin_item(
                item_id,
                item.seed,
                calls,
                pool=pool,
                condition=condition,
                request=dict(request),
                purpose=purpose,
            )
            call = call_main(
                url,
                route,
                transcript.posted_request(request, item.task),
                timeout=config.request_seconds,
            )
            mismatch = session.finish_item()
            if mismatch is not None:
                return outcome.fail(mismatch.item_id, mismatch.seq, mismatch.what)
            body: Any = call.body if call.ok else {"error": call.error}
            if not transcript.canonical_equal(body, row.get("output")):
                return outcome.fail(item_id, len(calls), "output")
            outcome.calls += len(calls)
    except DockerError as exc:
        return outcome.fail("", 0, f"docker: {exc}")
    finally:
        runtime.close()
        served.stop()
    return outcome
