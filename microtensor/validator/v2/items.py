from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from microtensor.artifact.params import ParameterError, count_parameters, safetensors_in
from microtensor.core.hashing import canonical_json
from microtensor.core.items import (
    ItemError,
    ReleaseManifest,
    items_from,
    open_release,
)
from microtensor.validator.v2.client import ServerClient, ServerError
from microtensor.validator.v2.collect import download
from microtensor.validator.v2.context import ADVERSARIAL, EpochItems, Reference, split_items

log = logging.getLogger("microtensor.validator.v2.items")

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


def reference_components(
    trees: dict[str, Path], definition: dict[str, Any], cache_dir: Path
) -> Reference:
    references = dict(definition.get("references") or {})
    slm_repo = str(references.get("slm", ""))
    slm_revision = str(references.get("slm_revision", "main"))
    if not slm_repo:
        raise ReleaseError("the arena names no reference slm")
    slm_dir = download(slm_repo, slm_revision, cache_dir)
    try:
        parameters = count_parameters(safetensors_in(slm_dir))
    except ParameterError as exc:
        raise ReleaseError(f"the reference slm has no readable weights: {exc}") from exc
    harness = trees.get("harness")
    if harness is None:
        raise ReleaseError("the bundle carries no reference harness")
    variants = tuple(path for name, path in sorted(trees.items()) if name.startswith("variant-"))
    return Reference(
        slm_dir=slm_dir,
        harness_dir=harness,
        variant_dirs=variants,
        router_dir=trees.get("router"),
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
