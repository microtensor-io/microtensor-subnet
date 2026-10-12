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
