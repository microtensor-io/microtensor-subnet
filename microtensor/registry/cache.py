from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from microtensor.core.constants import ARTIFACT_CACHE_CAP_BYTES

INDEX_NAME = "index.json"
STORE_NAME = "objects"
DIGEST_PREFIX = "sha256:"
DISK_FRACTION = 0.9


class CacheError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CacheEntry:
    digest: str
    size_bytes: int
    last_used: float
    sequence: int = 0
    released: bool = False


def _key(digest: str) -> str:
    body = digest[len(DIGEST_PREFIX):] if digest.startswith(DIGEST_PREFIX) else digest
    if len(body) < 32 or not all(c in "0123456789abcdef" for c in body.lower()):
        raise CacheError(f"malformed digest {digest!r}")
    return body.lower()


def _tree_size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


class ArtifactCache:
    def __init__(self, root: Path, cap_bytes: int = ARTIFACT_CACHE_CAP_BYTES) -> None:
        if cap_bytes < 1:
            raise CacheError("cache cap must be positive")
        self.root = root
        self.store = root / STORE_NAME
        self.store.mkdir(parents=True, exist_ok=True)
        self.cap_bytes = min(cap_bytes, int(shutil.disk_usage(self.store).total * DISK_FRACTION))
        self._index: dict[str, CacheEntry] = {}
        self._sequence = 0
        self._lock = threading.RLock()
        self._load_index()

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_NAME

    def _load_index(self) -> None:
        if not self.index_path.is_file():
            self._reconcile()
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._index = {
                key: CacheEntry(
                    digest=str(value["digest"]),
                    size_bytes=int(value["size_bytes"]),
                    last_used=float(value["last_used"]),
                    sequence=int(value.get("sequence", 0)),
                    released=bool(value.get("released", False)),
                )
                for key, value in payload.items()
            }
        except (OSError, ValueError, KeyError, TypeError):
            self._index = {}
        self._sequence = max((e.sequence for e in self._index.values()), default=0)
        self._reconcile()

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _reconcile(self) -> None:
        on_disk = {p.name for p in self.store.iterdir() if p.is_dir()}
        for key in list(self._index):
            if key not in on_disk:
                del self._index[key]
        now = time.time()
        for key in sorted(on_disk - set(self._index)):
            path = self.store / key
            self._index[key] = CacheEntry(
                digest=DIGEST_PREFIX + key,
                size_bytes=_tree_size(path),
                last_used=now,
                sequence=self._next_sequence(),
            )
        self._save_index()

    def _save_index(self) -> None:
        payload = {
            key: {
                "digest": entry.digest,
                "size_bytes": entry.size_bytes,
                "last_used": entry.last_used,
                "sequence": entry.sequence,
                "released": entry.released,
            }
            for key, entry in self._index.items()
        }
        staging = self.index_path.with_suffix(".tmp")
        staging.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        staging.replace(self.index_path)

    def path_for(self, digest: str) -> Path:
        return self.store / _key(digest)

    def has(self, digest: str) -> bool:
        return _key(digest) in self._index and self.path_for(digest).is_dir()

    def get(self, digest: str) -> Path | None:
        with self._lock:
            if not self.has(digest):
                return None
            self.touch(digest)
            return self.path_for(digest)

    def touch(self, digest: str) -> None:
        with self._lock:
            key = _key(digest)
            entry = self._index.get(key)
            if entry is None:
                return
            self._index[key] = CacheEntry(
                entry.digest, entry.size_bytes, time.time(), self._next_sequence()
            )
            self._save_index()

    def release(self, digest: str) -> None:
        with self._lock:
            key = _key(digest)
            entry = self._index.get(key)
            if entry is None:
                return
            self._index[key] = CacheEntry(
                entry.digest, entry.size_bytes, entry.last_used, entry.sequence, released=True
            )
            self._save_index()

    def released(self) -> tuple[CacheEntry, ...]:
        return tuple(e for e in self.entries() if e.released)

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self._index.values())

    def entries(self) -> tuple[CacheEntry, ...]:
        return tuple(sorted(self._index.values(), key=lambda e: (e.sequence, e.digest)))

    def reserve(self, needed_bytes: int, *, keep: frozenset[str] = frozenset()) -> list[str]:
        with self._lock:
            if needed_bytes > self.cap_bytes:
                raise CacheError(
                    f"artifact needs {needed_bytes} bytes, over the {self.cap_bytes} byte cache cap"
                )
            protected = {_key(d) for d in keep}
            evicted: list[str] = []
            for entry in self.released() + tuple(e for e in self.entries() if not e.released):
                if self.total_bytes + needed_bytes <= self.cap_bytes:
                    break
                key = _key(entry.digest)
                if key in protected:
                    continue
                self.drop(entry.digest)
                evicted.append(entry.digest)
            if self.total_bytes + needed_bytes > self.cap_bytes:
                raise CacheError(
                    "cache cannot free enough space without evicting a protected artifact"
                )
            return evicted

    def adopt(self, digest: str, staged: Path) -> Path:
        with self._lock:
            if not staged.is_dir():
                raise CacheError(f"{staged} is not a staged artifact directory")
            destination = self.path_for(digest)
            if destination.exists():
                shutil.rmtree(staged, ignore_errors=True)
                self.touch(digest)
                return destination

            size = _tree_size(staged)
            self.reserve(size)
            staged.replace(destination)
            self._index[_key(digest)] = CacheEntry(
                digest=DIGEST_PREFIX + _key(digest),
                size_bytes=size,
                last_used=time.time(),
                sequence=self._next_sequence(),
            )
            self._save_index()
            return destination

    def drop(self, digest: str) -> None:
        with self._lock:
            key = _key(digest)
            shutil.rmtree(self.store / key, ignore_errors=True)
            self._index.pop(key, None)
            self._save_index()

    def clear(self) -> None:
        with self._lock:
            for key in list(self._index):
                shutil.rmtree(self.store / key, ignore_errors=True)
            self._index.clear()
            self._save_index()
