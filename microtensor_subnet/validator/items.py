from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from microtensor_subnet.artifact.params import (
    ParameterError,
    model_parameters,
)
from microtensor_subnet.core.hashing import canonical_json
from microtensor_subnet.core.items import (
    ItemError,
    ReleaseManifest,
    items_from,
    open_release,
)
from microtensor_subnet.validator.client import ServerClient, ServerError
from microtensor_subnet.validator.collect import download
from microtensor_subnet.validator.context import ADVERSARIAL, EpochItems, Reference, split_items

log = logging.getLogger("microtensor_subnet.validator.items")

REFERENCE_PREFIX = "reference/"
BLOB_PREFIX = "db/"
ITEMS_FILE = "items.jsonl"


class ReleaseError(RuntimeError):
    pass


def verify_document(document: dict[str, Any], public_key_hex: str) -> str:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise ReleaseError("cryptography is required to verify server documents") from exc
    body = dict(document.get("body") or {})
    signature = str(document.get("signature", ""))
    message = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    digest = hashlib.sha256(message).hexdigest()
    if digest != str(document.get("digest", "")):
        raise ReleaseError("the document digest does not match its body")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            bytes.fromhex(signature), message
        )
    except Exception as exc:
        raise ReleaseError("the document is not signed by the server") from exc
    return digest


def server_key(client: ServerClient) -> str:
    keys = client.public_keys()
    for name in ("token_public_key", "public_key", "ed25519"):
        value = keys.get(name)
        if isinstance(value, str) and value:
            return value
    raise ReleaseError("the server publishes no signing key")


def fetch_release(
    client: ServerClient, arena_id: str, epoch_index: int, public_key: str
) -> tuple[ReleaseManifest, bytes, dict[str, str]]:
    document = client.document(arena_id, epoch_index, "release")
    if document is None:
        raise ReleaseError("the epoch has not been released")
    digest = verify_document(document, public_key)
    body = dict(document["body"])
    manifest = ReleaseManifest.from_dict({**body, "signature": document.get("signature", "")})
    bundle = client.bundle(arena_id, epoch_index)
    if not manifest.matches(bundle):
        raise ReleaseError("the bundle does not match the signed release")
    digests = {"release": digest}
    for kind in ("roster", "snapshot"):
        extra = client.document(arena_id, epoch_index, kind)
        digests[kind] = verify_document(extra, public_key) if extra is not None else ""
    return manifest, bundle, digests


def open_bundle(
    manifest: ReleaseManifest, bundle: bytes, hotkey: str, recipient_private: bytes
) -> dict[str, bytes]:
    try:
        return open_release(manifest, bundle, hotkey, recipient_private)
    except ItemError as exc:
        raise ReleaseError(str(exc)) from exc


def _write_tree(files: dict[str, bytes], prefix: str, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        if not name.startswith(prefix):
            continue
        target = into / name[len(prefix) :]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    return into


def unpack(
    files: dict[str, bytes], *, seed: str, work_dir: Path, limit: int = 0
) -> tuple[EpochItems, dict[str, Path]]:
    items = items_from(files)
    by_test = split_items(seed, [i for i in items if ADVERSARIAL not in i.tests], limit)
    adversarial = [
        {"item_id": i.item_id, "family": i.family, **dict(i.label), "task": dict(i.task)}
        for i in items
        if ADVERSARIAL in i.tests
    ]
    databases = {
        name[len(BLOB_PREFIX) :]: body
        for name, body in files.items()
        if name.startswith(BLOB_PREFIX)
    }
    families = {i.item_id: i.family for i in items}
    trees: dict[str, Path] = {}
    names = {
        name[len(REFERENCE_PREFIX) :].split("/", 1)[0]
        for name in files
        if name.startswith(REFERENCE_PREFIX)
    }
    for name in sorted(names):
        trees[name] = _write_tree(
            files, f"{REFERENCE_PREFIX}{name}/", work_dir / "reference" / name
        )
    return EpochItems(
        by_test=by_test, databases=databases, adversarial=adversarial, families=families
    ), trees


TESTNET = "test"
TEMPLATE_REPO = "https://github.com/microtensor-io/microtensor-miner-template"
TEMPLATE_FILES = ("main.py", "config.json", "requirements.txt")
TEMPLATE_DIRS = ("harness", "escalation")


def template_harness(task: str, cache_dir: Path) -> Path:
    clone = cache_dir / "reference-template"
    if not (clone / "main.py").is_file():
        shutil.rmtree(clone, ignore_errors=True)
        try:
            subprocess.run(  # noqa: S603
                ["git", "clone", "--quiet", "--depth", "1", TEMPLATE_REPO, str(clone)],  # noqa: S607
                check=True,
                capture_output=True,
                timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            reason = f"the reference harness could not be built from the template: {exc}"
            raise ReleaseError(reason) from exc
    into = cache_dir / f"reference-harness-{task or 'generic'}"
    shutil.rmtree(into, ignore_errors=True)
    into.mkdir(parents=True)
    for name in TEMPLATE_FILES:
        shutil.copy2(clone / name, into / name)
    for name in TEMPLATE_DIRS:
        shutil.copytree(clone / name, into / name, ignore=shutil.ignore_patterns("__pycache__"))
    if task and (into / "harness" / "prompts" / task).is_dir():
        config = json.loads((into / "config.json").read_text(encoding="utf-8"))
        config["adapter"] = task
        (into / "config.json").write_text(json.dumps(config, indent=2) + chr(10), encoding="utf-8")
    log.warning("testnet: the bundle has no reference harness; built one from the miner template")
    return into


def reference_components(
    trees: dict[str, Path], definition: dict[str, Any], cache_dir: Path, network: str = ""
) -> Reference:
    references = dict(definition.get("references") or {})
    slm_repo = str(references.get("slm", ""))
    slm_revision = str(references.get("slm_revision", "main"))
    if not slm_repo:
        raise ReleaseError("the arena names no reference slm")
    slm_dir = download(slm_repo, slm_revision, cache_dir)
    try:
        parameters = model_parameters(slm_dir)
    except ParameterError as exc:
        raise ReleaseError(f"the reference slm has no readable weights: {exc}") from exc
    harness = trees.get("harness")
    if harness is None and network == TESTNET:
        harness = template_harness(str(definition.get("task") or ""), cache_dir)
    if harness is None:
        raise ReleaseError("the bundle carries no reference harness")
    variants = tuple(path for name, path in sorted(trees.items()) if name.startswith("variant-"))
    router_dir = trees.get("router")
    router_repo = str(references.get("router", ""))
    if router_dir is None and router_repo:
        router_dir = download(
            router_repo, str(references.get("router_revision", "main")), cache_dir
        )
    return Reference(
        slm_dir=slm_dir,
        harness_dir=harness,
        variant_dirs=variants,
        router_dir=router_dir,
        escalation_dir=trees.get("escalation"),
        slm_parameters=parameters,
    )


def snapshot_bests(document: dict[str, Any] | None) -> dict[tuple[str, str, str], float]:
    found: dict[tuple[str, str, str], float] = {}
    for row in (document or {}).get("body", {}).get("standings") or []:
        try:
            found[(str(row["hotkey"]), str(row.get("artifact_sha", "")), str(row["pool"]))] = float(
                row.get("published") or 0.0
            )
        except (KeyError, TypeError, ValueError):
            continue
    return found


def roster_entries(document: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("hotkey")): dict(row)
        for row in (document or {}).get("body", {}).get("submissions") or []
        if row.get("hotkey")
    }


def item_ids_digest(items: EpochItems) -> str:
    ids = sorted({item.item_id for chosen in items.by_test.values() for item in chosen})
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def digest_of(body: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(body)).hexdigest()


def load_recipient_key(keys_dir: Path) -> bytes:
    path = keys_dir / "x25519.key"
    try:
        raw = bytes.fromhex(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise ReleaseError("no registered x25519 key; run `mt validator register-key`") from exc
    if len(raw) != 32:
        raise ReleaseError("the stored x25519 key is malformed")
    return raw


def catch_server(exc: ServerError) -> ReleaseError:
    return ReleaseError(str(exc))


INPUTS_FILE = "inputs.jsonl"
DRAFTS_FILE = "reference/drafts.jsonl"
CANARIES_FILE = "canaries.jsonl"
REDO_FILE = "redo.jsonl"
GZIP_MAGIC = b"\x1f\x8b"


@dataclass(frozen=True, slots=True)
class InputItem:
    item_id: str
    pool: str
    family: str
    task: dict[str, Any]
    twin_of: str
    seed: int


@dataclass(slots=True)
class InputsBundle:
    sha256: str
    seed: str
    items: list[InputItem]
    databases: dict[str, bytes]
    trees: dict[str, Path]
    drafts: dict[str, str]
    open_block: int = 0
    close_block: int = 0

    def by_id(self) -> dict[str, InputItem]:
        return {item.item_id: item for item in self.items}

    def seeds(self) -> dict[str, int]:
        return {item.item_id: item.seed for item in self.items}


@dataclass(frozen=True, slots=True)
class RedoRow:
    item_id: str
    pool: str
    correct: bool
    cost_micros: int
    latency_ms: float
    answer: str = ""
    model: str = ""
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0


@dataclass(slots=True)
class LabelsBundle:
    canaries: list[dict[str, Any]] = field(default_factory=list)
    redo: list[RedoRow] = field(default_factory=list)

    def rows(self, pool: str) -> dict[str, RedoRow]:
        return {row.item_id: row for row in self.redo if row.pool == pool}


def item_seed(epoch_seed: str, item_id: str) -> int:
    digest = hashlib.sha256(f"{epoch_seed}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _lines(raw: bytes | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    found: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            found.append(parsed)
    return found


def fetch_inputs(
    client: ServerClient, arena_id: str, epoch_index: int, public_key: str
) -> tuple[dict[str, Any], bytes]:
    from microtensor_subnet.validator.transcripts import client_method

    document = client_method(client, "inputs")(arena_id, epoch_index)
    if document is None:
        raise ReleaseError("the epoch inputs have not been published")
    verify_document(document, public_key)
    body = dict(document.get("body") or {})
    bundle = client_method(client, "inputs_bundle")(arena_id, epoch_index)
    if len(bundle) != int(body.get("bundle_bytes") or -1):
        raise ReleaseError("the inputs bundle does not match the signed inputs document")
    if hashlib.sha256(bundle).hexdigest() != str(body.get("bundle_sha256", "")).lower():
        raise ReleaseError("the inputs bundle does not match the signed inputs document")
    return body, bundle


def unpack_inputs(
    body: dict[str, Any], bundle: bytes, *, epoch_seed: str, work_dir: Path
) -> InputsBundle:
    from microtensor_subnet.core.items import unpack_bundle

    try:
        files = unpack_bundle(bundle)
    except (ItemError, OSError) as exc:
        raise ReleaseError(f"the inputs bundle cannot be opened: {exc}") from exc
    rows = _lines(files.get(INPUTS_FILE))
    if not rows:
        raise ReleaseError(f"the inputs bundle has no {INPUTS_FILE}")
    items: list[InputItem] = []
    for row in rows:
        item_id = str(row.get("item_id", "")).strip()
        task = row.get("task")
        if not item_id or not isinstance(task, dict):
            raise ReleaseError(f"inputs item {item_id or '?'} is malformed")
        wanted = item_seed(epoch_seed, item_id)
        raw_seed = row.get("seed")
        seed = int(raw_seed) if raw_seed is not None else wanted
        if seed != wanted:
            raise ReleaseError(f"inputs item {item_id} carries a seed the epoch seed does not give")
        items.append(
            InputItem(
                item_id=item_id,
                pool=str(row.get("pool") or ""),
                family=str(row.get("family") or item_id),
                task=dict(task),
                twin_of=str(row.get("twin_of") or ""),
                seed=seed,
            )
        )
    databases = {
        name[len(BLOB_PREFIX) :]: blob
        for name, blob in files.items()
        if name.startswith(BLOB_PREFIX)
    }
    trees: dict[str, Path] = {}
    names = {
        name[len(REFERENCE_PREFIX) :].split("/", 1)[0]
        for name in files
        if name.startswith(REFERENCE_PREFIX) and "/" in name[len(REFERENCE_PREFIX) :]
    }
    for name in sorted(names):
        trees[name] = _write_tree(
            files, f"{REFERENCE_PREFIX}{name}/", work_dir / "inputs" / "reference" / name
        )
    drafts = {
        str(row.get("item_id", "")): str(row.get("draft", ""))
        for row in _lines(files.get(DRAFTS_FILE))
        if row.get("item_id")
    }
    return InputsBundle(
        sha256=str(body.get("bundle_sha256", "")).lower(),
        seed=str(body.get("seed") or epoch_seed),
        items=items,
        databases=databases,
        trees=trees,
        drafts=drafts,
        open_block=int(body.get("open_block") or 0),
        close_block=int(body.get("close_block") or 0),
    )


def unpack_labels(files: Mapping[str, bytes]) -> LabelsBundle:
    canaries = [
        {
            "item_id": str(row.get("item_id", "")),
            "prefix": str(row.get("prefix", "")),
            "fact": str(row.get("fact", "")),
            "epoch": int(row.get("epoch") or 0),
        }
        for row in _lines(files.get(CANARIES_FILE))
        if row.get("item_id") and row.get("fact")
    ]
    redo = [
        RedoRow(
            item_id=str(row.get("item_id", "")),
            pool=str(row.get("pool") or ""),
            correct=bool(row.get("correct")),
            cost_micros=int(row.get("cost_micros") or 0),
            latency_ms=float(row.get("latency_ms") or 0.0),
            answer=str(row.get("answer") if row.get("answer") is not None else ""),
            model=str(row.get("model") or ""),
            ttft_ms=float(row.get("ttft_ms") or 0.0),
            tpot_ms=float(row.get("tpot_ms") or 0.0),
        )
        for row in _lines(files.get(REDO_FILE))
        if row.get("item_id")
    ]
    return LabelsBundle(canaries=canaries, redo=redo)
