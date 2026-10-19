from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Final

MAGIC: Final[bytes] = b"GGUF"
KV_BYTES: Final[int] = 2
SCALARS: Final[dict[int, str]] = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
STRING: Final[int] = 8
ARRAY: Final[int] = 9
TYPE_NAMES: Final[dict[int, str]] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
}
BELOW_FOUR_BITS: Final[frozenset[str]] = frozenset(
    {
        "Q2_K",
        "Q3_K",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ3_XXS",
        "IQ1_S",
        "IQ3_S",
        "IQ2_S",
        "IQ1_M",
        "TQ1_0",
        "TQ2_0",
    }
)
FILE_TYPES: Final[dict[int, str]] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    30: "IQ4_XS",
    32: "BF16",
}


class GgufError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    kind: str

    @property
    def elements(self) -> int:
        return math.prod(self.shape) if self.shape else 1


@dataclass(slots=True)
class Gguf:
    version: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: list[Tensor] = field(default_factory=list)

    @property
    def parameters(self) -> int:
        shapes = {t.name: t.shape for t in self.tensors}
        tied = "output.weight" in shapes and shapes["output.weight"] == shapes.get(
            "token_embd.weight"
        )
        return sum(t.elements for t in self.tensors if not (tied and t.name == "output.weight"))

    @property
    def file_type(self) -> str:
        code = self.metadata.get("general.file_type")
        return FILE_TYPES.get(int(code), str(code)) if code is not None else ""

    @property
    def below_four_bits(self) -> list[str]:
        return [t.name for t in self.tensors if t.kind in BELOW_FOUR_BITS]

    def number(self, suffix: str) -> int:
        arch = str(self.metadata.get("general.architecture") or "")
        value = self.metadata.get(f"{arch}.{suffix}")
        return int(value) if isinstance(value, (int, float)) else 0

    def kv_cache_bytes(self, max_len: int) -> int:
        layers = self.number("block_count")
        heads = self.number("attention.head_count")
        kv_heads = self.number("attention.head_count_kv") or heads
        width = self.number("embedding_length")
        head_dim = self.number("attention.key_length") or (width // heads if heads else 0)
        return 2 * layers * kv_heads * head_dim * max_len * KV_BYTES


def _unpack(handle: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    raw = handle.read(size)
    if len(raw) != size:
        raise GgufError("the GGUF header ends early")
    return struct.unpack(fmt, raw)[0]


def _string(handle: BinaryIO) -> str:
    length = _unpack(handle, "<Q")
    raw = handle.read(length)
    if len(raw) != length:
        raise GgufError("the GGUF header ends early")
    return raw.decode("utf-8", "replace")


def _value(handle: BinaryIO, kind: int) -> Any:
    if kind in SCALARS:
        return _unpack(handle, SCALARS[kind])
    if kind == STRING:
        return _string(handle)
    if kind == ARRAY:
        inner = _unpack(handle, "<I")
        count = _unpack(handle, "<Q")
        if inner in SCALARS:
            size = struct.calcsize(SCALARS[inner])
            handle.seek(size * count, 1)
            return count
        for _ in range(count):
            _value(handle, inner)
        return count
    raise GgufError(f"unknown GGUF value type {kind}")


def read(path: Path) -> Gguf:
    with path.open("rb") as handle:
        if handle.read(4) != MAGIC:
            raise GgufError(f"{path.name} is not a GGUF file")
        version = _unpack(handle, "<I")
        if version < 2:
            raise GgufError(f"{path.name} is GGUF version {version}; 2 or later is required")
        tensors = _unpack(handle, "<Q")
        pairs = _unpack(handle, "<Q")
        found = Gguf(version=version)
        for _ in range(pairs):
            key = _string(handle)
            found.metadata[key] = _value(handle, _unpack(handle, "<I"))
        for _ in range(tensors):
            name = _string(handle)
            dims = _unpack(handle, "<I")
            shape = tuple(_unpack(handle, "<Q") for _ in range(dims))
            kind = _unpack(handle, "<I")
            _unpack(handle, "<Q")
            found.tensors.append(Tensor(name, shape, TYPE_NAMES.get(kind, f"type{kind}")))
    return found


def gguf_in(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*.gguf") if p.is_file())
