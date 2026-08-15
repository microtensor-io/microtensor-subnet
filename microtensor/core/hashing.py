from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

_STREAM_CHUNK_BYTES = 8 * 1024 * 1024

DIGEST_PREFIX = "sha256:"


def _normalise(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        return {_normalise(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    return value


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _normalise(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_digest(payload: Mapping[str, Any]) -> bytes:
    return hashlib.sha256(canonical_json(payload)).digest()


def canonical_hash(payload: Mapping[str, Any]) -> str:
    return DIGEST_PREFIX + canonical_digest(payload).hex()


def digest_bytes(data: bytes) -> str:
    return DIGEST_PREFIX + hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_STREAM_CHUNK_BYTES):
            h.update(chunk)
    return DIGEST_PREFIX + h.hexdigest()


def _walk_sorted(root: Path) -> Iterator[Path]:
    files = (p for p in root.rglob("*") if p.is_file())
    return iter(sorted(files, key=lambda p: p.relative_to(root).as_posix()))


def digest_entries(entries: Sequence[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for rel, digest in sorted(entries, key=lambda e: e[0]):
        h.update(unicodedata.normalize("NFC", rel).encode("utf-8"))
        h.update(b"\x00")
        h.update(digest.encode("ascii"))
        h.update(b"\x00")
    return DIGEST_PREFIX + h.hexdigest()


def digest_tree(root: Path) -> str:
    return digest_entries(
        [(path.relative_to(root).as_posix(), digest_file(path)) for path in _walk_sorted(root)]
    )


def tree_size_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in _walk_sorted(root))


def round_seed(block_hash: str, track: str, hardware_class: str) -> str:
    material = f"{block_hash}:{track}:{hardware_class}".encode()
    return hashlib.sha256(material).hexdigest()


def task_nonce(seed: str, task_ref: str, artifact_digest: str = "") -> str:
    material = f"{seed}:{task_ref}:{artifact_digest}".encode()
    return hashlib.sha256(material).hexdigest()


def select_deterministic(items: Sequence[str], seed: str, count: int) -> list[str]:
    if count >= len(items):
        return sorted(items)
    keyed = sorted(
        (hashlib.sha256(f"{seed}:{ref}".encode()).hexdigest(), ref) for ref in items
    )
    return sorted(ref for _, ref in keyed[:count])
