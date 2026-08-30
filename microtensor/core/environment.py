from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from microtensor.core.hashing import digest_file


def _distribution_pins(root: Path) -> list[str]:
    lock = root / "lock.txt"
    if not lock.is_file():
        return []
    lines = (line.strip() for line in lock.read_text(encoding="utf-8").splitlines())
    return sorted(line for line in lines if line and not line.startswith("#"))


def _tree_entries(root: Path, name: str) -> list[str]:
    base = root / name
    if not base.is_dir():
        return []
    files = sorted(
        (p for p in base.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(base).as_posix(),
    )
    return [f"{name}/{p.relative_to(base).as_posix()}:{digest_file(p)}" for p in files]


def environment_digest(root: Path) -> str:
    if not root.is_dir():
        return ""
    parts = [
        f"python:{sys.version_info.major}.{sys.version_info.minor}",
        *_distribution_pins(root),
        *_tree_entries(root, "nltk_data"),
        *_tree_entries(root, "mplconfig"),
    ]
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return f"env:{h.hexdigest()[:16]}"
