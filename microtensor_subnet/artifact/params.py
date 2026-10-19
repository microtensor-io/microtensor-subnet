from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

HEADER_PREFIX_BYTES = 8
QUANT_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx")
INT4_PACK = 8
INT8_PACK = 4
KV_BYTES = 2
SKIPPED_SUFFIXES = (".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map", ".quant_state")


class ParameterError(ValueError):
    pass


def read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        prefix = handle.read(HEADER_PREFIX_BYTES)
        if len(prefix) != HEADER_PREFIX_BYTES:
            raise ParameterError(f"{path.name} is not a safetensors file")
        (length,) = struct.unpack("<Q", prefix)
        raw = handle.read(length)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ParameterError(f"{path.name} has a malformed safetensors header") from exc
    if not isinstance(header, dict):
        raise ParameterError(f"{path.name} has a malformed safetensors header")
    return header


def _elements(shape: Iterable[Any]) -> int:
    dims = [int(d) for d in shape]
    return math.prod(dims) if dims else 1


def _counted(name: str, entry: Mapping[str, Any]) -> int:
    if name == "__metadata__" or name.endswith(SKIPPED_SUFFIXES):
        return 0
    shape = entry.get("shape") or []
    dtype = str(entry.get("dtype", ""))
    elements = _elements(shape)
    if name.endswith(".qweight"):
        return elements * (INT4_PACK if dtype in ("I32", "U32") else 1)
    if name.endswith((".qzeros", ".scales", ".g_idx")):
        return 0
    return elements


def count_parameters(files: Iterable[Path]) -> int:
    total = 0
    seen = 0
    for path in files:
        header = read_header(path)
        for name, entry in header.items():
            if isinstance(entry, Mapping):
                total += _counted(name, entry)
        seen += 1
    if seen == 0:
        raise ParameterError("no safetensors files to count")
    return total


def safetensors_in(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*.safetensors") if p.is_file())


def _text_config(root: Path) -> Mapping[str, Any]:
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ParameterError(f"config.json in {root.name} cannot be read") from exc
    if not isinstance(config, Mapping):
        raise ParameterError(f"config.json in {root.name} is not an object")
    nested = config.get("text_config")
    return nested if isinstance(nested, Mapping) else config


def kv_cache_bytes(root: Path, max_len: int) -> int:
    config = _text_config(root)
    layers = int(config.get("num_hidden_layers") or 0)
    heads = int(config.get("num_attention_heads") or 0)
    kv_heads = int(config.get("num_key_value_heads") or heads)
    hidden = int(config.get("hidden_size") or 0)
    head_dim = int(config.get("head_dim") or (hidden // heads if heads else 0))
    return 2 * layers * kv_heads * head_dim * max_len * KV_BYTES


def footprint_bytes(root: Path, max_len: int) -> int:
    weights = sum(path.stat().st_size for path in safetensors_in(root))
    return weights + kv_cache_bytes(root, max_len)
