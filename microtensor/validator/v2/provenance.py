from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from microtensor.artifact.params import read_header, safetensors_in
from microtensor.artifact.validate import sha256_of

log = logging.getLogger("microtensor.validator.v2.provenance")

ATTENTION_MARKERS = ("q_proj", "k_proj", "v_proj", "o_proj")
SINGULAR_VALUES = 32
MAX_LAYERS = 8
DEFAULT_THRESHOLD = 0.9


def public_checkpoint_hashes(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    found: set[str] = set()
    for row in rows if isinstance(rows, list) else []:
        digest = str(row).lower()
        found.add(digest if digest.startswith("sha256:") else f"sha256:{digest}")
    return found


def matches_public(slm_dir: Path, known: Iterable[str]) -> list[str]:
    wanted = set(known)
    if not wanted:
        return []
    return [path.name for path in safetensors_in(slm_dir) if sha256_of(path) in wanted]


def _spectra(model_dir: Path, *, device: str = "cpu") -> dict[str, list[float]]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("torch and safetensors are required for the weight fingerprint") from exc
    found: dict[str, list[float]] = {}
    for path in safetensors_in(model_dir):
        header = read_header(path)
        names = sorted(
            name
            for name, entry in header.items()
            if isinstance(entry, Mapping)
            and any(marker in name for marker in ATTENTION_MARKERS)
            and name.endswith(".weight")
            and len(entry.get("shape") or []) == 2
        )
        with safe_open(str(path), framework="pt", device=device) as handle:
            for name in names:
                layer = _layer_of(name)
                if layer is None or layer >= MAX_LAYERS:
                    continue
                tensor = handle.get_tensor(name).to(torch.float32)
                values = torch.linalg.svdvals(tensor)[:SINGULAR_VALUES]
                scale = float(values[0]) if values.numel() and float(values[0]) > 0 else 1.0
                found[name] = [float(v) / scale for v in values.tolist()]
    return found


def _layer_of(name: str) -> int | None:
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return None


def fingerprint_similarity(model_dir: Path, base_dir: Path) -> float:
    mine = _spectra(model_dir)
    base = _spectra(base_dir)
    shared = sorted(set(mine) & set(base))
    if not shared:
        return 0.0
    scores: list[float] = []
    for name in shared:
        a, b = mine[name], base[name]
        length = min(len(a), len(b))
        if not length:
            continue
        dot = sum(x * y for x, y in zip(a[:length], b[:length], strict=True))
        norm = (sum(x * x for x in a[:length]) ** 0.5) * (sum(y * y for y in b[:length]) ** 0.5)
        scores.append(dot / norm if norm else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def declared_base(slm_dir: Path) -> str:
    recipe = slm_dir / "training" / "recipe.json"
    try:
        return str(json.loads(recipe.read_text(encoding="utf-8")).get("base_model", ""))
    except (OSError, ValueError):
        return ""


def check(
    slm_dir: Path,
    *,
    base_dirs: Mapping[str, Path],
    known_public: Iterable[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    base = declared_base(slm_dir)
    result: dict[str, Any] = {
        "declared_base": base,
        "public_match": [],
        "similarity": None,
        "passed": False,
    }
    result["public_match"] = matches_public(slm_dir, known_public)
    base_dir = base_dirs.get(base) or base_dirs.get(base.lower())
    if base_dir is None:
        result["reason"] = "declared base is not available to compare against"
        return result
    try:
        similarity = fingerprint_similarity(slm_dir, base_dir)
    except RuntimeError as exc:
        result["reason"] = str(exc)
        return result
    result["similarity"] = round(similarity, 6)
    if result["public_match"]:
        result["reason"] = "weights match a public checkpoint"
        return result
    if similarity < threshold:
        result["reason"] = f"weights do not fingerprint to {base}"
        return result
    result["passed"] = True
    return result


def digest_of_dir(root: Path) -> str:
    h = hashlib.sha256()
    for path in safetensors_in(root):
        h.update(sha256_of(path).encode())
    return h.hexdigest()
