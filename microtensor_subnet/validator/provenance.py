from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from microtensor_subnet.artifact.gguf import gguf_in
from microtensor_subnet.artifact.params import read_header, safetensors_in
from microtensor_subnet.artifact.validate import sha256_of

log = logging.getLogger("microtensor_subnet.validator.provenance")

ATTENTION_MARKERS = ("q_proj", "k_proj", "v_proj", "o_proj")
GGUF_NAMES = {"attn_q": "q_proj", "attn_k": "k_proj", "attn_v": "v_proj", "attn_output": "o_proj"}
AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
NIBBLE_SHIFTS = tuple(range(0, 32, 4))
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
    files = safetensors_in(slm_dir) or gguf_in(slm_dir)
    return [path.name for path in files if sha256_of(path) in wanted]


def _normalised(matrix: Any) -> list[float]:
    import numpy as np

    values = np.linalg.svd(np.asarray(matrix, dtype=np.float32), compute_uv=False)[:SINGULAR_VALUES]
    scale = float(values[0]) if values.size and float(values[0]) > 0 else 1.0
    return [float(v) / scale for v in values]


def _unpack(packed: Any, axis: int) -> Any:
    import numpy as np

    raw = np.asarray(packed).astype(np.uint32)
    nibbles = np.stack([(raw >> shift) & 0xF for shift in NIBBLE_SHIFTS], axis=axis + 1)
    shape = list(raw.shape)
    shape[axis] *= len(NIBBLE_SHIFTS)
    return nibbles.reshape(shape).astype(np.float32)


def dequantise_awq(qweight: Any, qzeros: Any, scales: Any, group: int) -> Any:
    import numpy as np

    order = np.array(AWQ_ORDER)

    def unpacked(packed: Any) -> Any:
        values = _unpack(packed, 1)
        rows, cols = values.shape
        return values.reshape(rows, cols // 8, 8)[:, :, order].reshape(rows, cols)

    weight = unpacked(qweight)
    zeros = np.repeat(unpacked(qzeros), group, axis=0)[: weight.shape[0]]
    factors = np.repeat(np.asarray(scales, dtype=np.float32), group, axis=0)[: weight.shape[0]]
    return (weight - zeros) * factors


def dequantise_gptq(
    qweight: Any, qzeros: Any, scales: Any, group: int, g_idx: Any | None = None
) -> Any:
    import numpy as np

    weight = _unpack(qweight, 0)
    zeros = _unpack(qzeros, 1) + 1
    rows = weight.shape[0]
    index = np.asarray(g_idx) if g_idx is not None else np.arange(rows) // group
    return (weight - zeros[index]) * np.asarray(scales, dtype=np.float32)[index]


def _quantisation(model_dir: Path) -> dict[str, Any]:
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    found = config.get("quantization_config") if isinstance(config, Mapping) else None
    return dict(found) if isinstance(found, Mapping) else {}


def _wanted(name: str, suffix: str) -> bool:
    if not name.endswith(suffix) or not any(m in name for m in ATTENTION_MARKERS):
        return False
    layer = _layer_of(name)
    return layer is not None and layer < MAX_LAYERS


def _safetensors_spectra(model_dir: Path) -> dict[str, list[float]]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("torch and safetensors are required for the weight fingerprint") from exc

    def array(tensor: Any) -> Any:
        return (tensor if not tensor.is_floating_point() else tensor.to(torch.float32)).numpy()

    quant = _quantisation(model_dir)
    method = str(quant.get("quant_method") or "").lower()
    group = int(quant.get("group_size") or 128)
    found: dict[str, list[float]] = {}
    for path in safetensors_in(model_dir):
        header = read_header(path)
        names = sorted(n for n, e in header.items() if isinstance(e, Mapping))
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in names:
                if _wanted(name, ".weight") and len(header[name].get("shape") or []) == 2:
                    found[name] = _normalised(array(handle.get_tensor(name)))
                elif _wanted(name, ".qweight") and method in ("awq", "gptq"):
                    stem = name[: -len(".qweight")]
                    parts = {
                        key: array(handle.get_tensor(f"{stem}.{key}"))
                        for key in ("qzeros", "scales", "g_idx")
                        if f"{stem}.{key}" in header
                    }
                    packed = array(handle.get_tensor(name))
                    matrix = (
                        dequantise_awq(packed, parts["qzeros"], parts["scales"], group)
                        if method == "awq"
                        else dequantise_gptq(
                            packed, parts["qzeros"], parts["scales"], group, parts.get("g_idx")
                        )
                    )
                    found[f"{stem}.weight"] = _normalised(matrix)
    return found


def _gguf_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) != 4 or parts[0] != "blk" or parts[3] != "weight" or not parts[1].isdigit():
        return ""
    projection = GGUF_NAMES.get(parts[2], "")
    return f"model.layers.{parts[1]}.self_attn.{projection}.weight" if projection else ""


def _gguf_spectra(path: Path) -> dict[str, list[float]]:
    try:
        from gguf import GGUFReader
        from gguf.quants import dequantize
    except ImportError as exc:
        raise RuntimeError("the gguf package is required to fingerprint GGUF weights") from exc
    found: dict[str, list[float]] = {}
    for tensor in GGUFReader(str(path)).tensors:
        name = _gguf_name(str(tensor.name))
        layer = _layer_of(name) if name else None
        if layer is None or layer >= MAX_LAYERS:
            continue
        found[name] = _normalised(dequantize(tensor.data, tensor.tensor_type))
    return found


def _spectra(model_dir: Path) -> dict[str, list[float]]:
    if safetensors_in(model_dir):
        return _safetensors_spectra(model_dir)
    ggufs = gguf_in(model_dir)
    if ggufs:
        return _gguf_spectra(ggufs[0])
    raise RuntimeError(f"no weights to fingerprint in {model_dir.name}")


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
