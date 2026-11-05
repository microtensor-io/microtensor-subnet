from __future__ import annotations

import functools
import hashlib
import logging
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from operator import mul
from pathlib import Path
from typing import Any, Final

from microtensor_subnet.artifact.params import QUANT_SUFFIXES, loaded_weights, read_header
from microtensor_subnet.validator import transcripts
from microtensor_subnet.validator.context import ROUTER, SLM, SYSTEM

log = logging.getLogger("microtensor_subnet.validator.duplicates")

SKETCH_SIZE: Final[int] = 4096
SKETCH_SEED: Final[bytes] = b"microtensor-duplicate-sketch"
SKETCH_DIGITS: Final[int] = 9
CHUNK_ELEMENTS: Final[int] = 1 << 20
PROBE_ELEMENTS: Final[int] = 4096
MIN_COVERAGE: Final[float] = 0.9
MASK_32: Final[int] = 0xFFFFFFFF
MIX_ONE: Final[int] = 0x7FEB352D
MIX_TWO: Final[int] = 0x5BD1E995
SIGN_BIT: Final[int] = 12
FLOAT_TYPES: Final[frozenset[str]] = frozenset({"BF16", "F16", "F32", "F64"})
METADATA: Final[str] = "__metadata__"

SKETCH_SIMILARITY: Final[float] = 0.9
SAME_OUTPUTS: Final[float] = 0.95
MIN_COMMON_ITEMS: Final[int] = 20

MODEL: Final[str] = "model"
ROUTER_PART: Final[str] = "router"
POOLS_OF: Final[dict[str, tuple[str, str]]] = {
    MODEL: (SLM, SYSTEM),
    ROUTER_PART: (ROUTER, SYSTEM),
}


@dataclass(frozen=True, slots=True)
class Stored:
    path: Path
    shape: tuple[int, ...]
    dtype: str

    @property
    def elements(self) -> int:
        return math.prod(self.shape) if self.shape else 1


def stored_tensors(root: Path) -> dict[str, Stored]:
    found: dict[str, Stored] = {}
    for path in loaded_weights(root):
        for name, entry in read_header(path).items():
            if name == METADATA or not isinstance(entry, Mapping):
                continue
            found[name] = Stored(
                path, tuple(int(d) for d in entry.get("shape") or ()), str(entry.get("dtype", ""))
            )
    return found


def _mixed(index: Any) -> Any:
    value = index & MASK_32
    value = ((value ^ (value >> 16)) * MIX_ONE) & MASK_32
    value = ((value ^ (value >> 13)) * MIX_TWO) & MASK_32
    return value ^ (value >> 16)


@functools.lru_cache(maxsize=1)
def _table() -> tuple[Any, Any]:
    import torch

    mixed = _mixed(torch.arange(CHUNK_ELEMENTS, dtype=torch.int64))
    slots = mixed & (SKETCH_SIZE - 1)
    signs = (((mixed >> SIGN_BIT) & 1) * 2 - 1).to(torch.float64)
    return slots, signs


def _placement(name: str, chunk: int) -> tuple[int, float]:
    digest = hashlib.blake2b(
        SKETCH_SEED + name.encode("utf-8") + b"\x00" + str(chunk).encode("ascii"), digest_size=4
    ).digest()
    value = int.from_bytes(digest, "big")
    return value & (SKETCH_SIZE - 1), 1.0 if (value >> SIGN_BIT) & 1 else -1.0


def _chunks(name: str, stored: Stored) -> Iterator[Any]:
    from safetensors import safe_open

    with safe_open(str(stored.path), framework="pt", device="cpu") as handle:
        if not stored.shape:
            yield handle.get_tensor(name).reshape(-1)
            return
        rows = stored.shape[0]
        per_row = math.prod(stored.shape[1:]) if stored.shape[1:] else 1
        step = max(1, CHUNK_ELEMENTS // max(1, per_row))
        tensor = handle.get_slice(name)
        for start in range(0, rows, step):
            yield from tensor[start : min(rows, start + step)].reshape(-1).split(CHUNK_ELEMENTS)


def _probe(name: str, stored: Stored) -> Any:
    import torch

    for chunk in _chunks(name, stored):
        return chunk[:PROBE_ELEMENTS].to(torch.float64)
    return torch.zeros(0, dtype=torch.float64)


def _aligned(model: Mapping[str, Stored], base: Mapping[str, Stored]) -> list[tuple[str, str]]:
    import torch

    mine = {name: entry for name, entry in model.items() if entry.dtype in FLOAT_TYPES}
    theirs = {name: entry for name, entry in base.items() if entry.dtype in FLOAT_TYPES}
    pairs = {
        name: name
        for name, entry in mine.items()
        if name in theirs and theirs[name].shape == entry.shape
    }
    probes: dict[str, Any] = {}
    for name in sorted(set(mine) - set(pairs)):
        shape = mine[name].shape
        free = [b for b in sorted(theirs) if b not in pairs and theirs[b].shape == shape]
        if not free:
            continue
        probe = _probe(name, mine[name])
        for candidate in free:
            if candidate not in probes:
                probes[candidate] = _probe(candidate, theirs[candidate])
        distances = [(float(torch.linalg.vector_norm(probe - probes[b])), b) for b in free]
        pairs[min(distances)[1]] = name
    covered = sum(theirs[name].elements for name in pairs)
    total = sum(entry.elements for entry in theirs.values())
    if not total or covered / total < MIN_COVERAGE:
        return []
    return sorted(pairs.items())


def sketch_weights(model_dir: Path, base_dir: Path) -> list[float] | None:
    import torch

    model = stored_tensors(model_dir)
    base = stored_tensors(base_dir)
    if not model or not base:
        return None
    if any(name.endswith(QUANT_SUFFIXES) for name in (*model, *base)):
        return None
    with torch.no_grad():
        pairs = _aligned(model, base)
        if not pairs:
            return None
        slots, signs = _table()
        total = torch.zeros(SKETCH_SIZE, dtype=torch.float64)
        for base_name, model_name in pairs:
            parts = zip(
                _chunks(model_name, model[model_name]),
                _chunks(base_name, base[base_name]),
                strict=True,
            )
            for chunk, (left, right) in enumerate(parts):
                delta = left.to(torch.float64) - right.to(torch.float64)
                delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                size = int(delta.numel())
                offset, flip = _placement(base_name, chunk)
                total += torch.bincount(
                    (slots[:size] + offset) & (SKETCH_SIZE - 1),
                    weights=delta * signs[:size] * flip,
                    minlength=SKETCH_SIZE,
                )
        norm = float(torch.linalg.vector_norm(total))
        if not norm or not math.isfinite(norm):
            return [0.0] * SKETCH_SIZE
        return [round(float(value), SKETCH_DIGITS) for value in total / norm]


def similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = math.fsum(map(mul, a, b))
    left = math.fsum(map(mul, a, a))
    right = math.fsum(map(mul, b, b))
    return dot / math.sqrt(left * right) if left and right else 0.0


def outputs_of(transcript: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    found: dict[str, tuple[int, ...]] = {}
    for row in transcripts.items_of(transcript):
        if str(row.get("pool", "")) != SLM:
            continue
        if str(row.get("condition", "")) != transcripts.CONDITION_MINER:
            continue
        calls = transcripts.slm_calls(row)
        if calls and calls[0].get("fault"):
            continue
        tokens = tuple(int(t) for t in (calls[0].get("output_tokens") or ())) if calls else ()
        if tokens:
            found[str(row.get("item_id", ""))] = tokens
    return found


def behaviour_overlap(
    a: Mapping[str, Sequence[int]], b: Mapping[str, Sequence[int]]
) -> tuple[float, int]:
    common = set(a) & set(b)
    if not common:
        return 0.0, 0
    same = sum(1 for item_id in common if a[item_id] == b[item_id])
    return same / len(common), len(common)


@dataclass(slots=True)
class Entry:
    hotkey: str
    weights_sha256: str = ""
    sketch: list[float] | None = None
    router_sha256: str = ""
    router_sketch: list[float] | None = None
    outputs: dict[str, tuple[int, ...]] = field(default_factory=dict)
    outputs_checked: bool = False


def _same_weights(
    later_hash: str,
    earlier_hash: str,
    later_sketch: Sequence[float] | None,
    earlier_sketch: Sequence[float] | None,
) -> bool:
    if later_hash and later_hash == earlier_hash:
        return True
    if later_sketch is None or earlier_sketch is None:
        return False
    return similarity(later_sketch, earlier_sketch) >= SKETCH_SIMILARITY


def _same_model(later: Entry, earlier: Entry) -> bool:
    if _same_weights(later.weights_sha256, earlier.weights_sha256, later.sketch, earlier.sketch):
        return True
    if not earlier.outputs_checked:
        return False
    share, common = behaviour_overlap(later.outputs, earlier.outputs)
    return common >= MIN_COMMON_ITEMS and share >= SAME_OUTPUTS


def _matched_parts(later: Entry, earlier: Entry) -> Iterable[str]:
    if _same_model(later, earlier):
        yield MODEL
    if _same_weights(
        later.router_sha256, earlier.router_sha256, later.router_sketch, earlier.router_sketch
    ):
        yield ROUTER_PART


def compare(entries: Sequence[Entry]) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for position, later in enumerate(entries):
        zeroed: dict[str, str] = {}
        for earlier in entries[:position]:
            for part in _matched_parts(later, earlier):
                log.info("%s duplicates %s of the earlier %s", later.hotkey, part, earlier.hotkey)
                for pool in POOLS_OF[part]:
                    zeroed.setdefault(pool, earlier.hotkey)
        if zeroed:
            found[later.hotkey] = zeroed
    return found


def commit_positions(chain: Any, blocks: Mapping[str, int]) -> dict[str, int]:
    by_block: dict[int, list[str]] = {}
    for hotkey, block in blocks.items():
        if block > 0:
            by_block.setdefault(block, []).append(hotkey)
    tied = {block: hotkeys for block, hotkeys in by_block.items() if len(hotkeys) > 1}
    if not tied:
        return {}
    reader = getattr(chain, "commitment_position", None)
    if not callable(reader):
        log.warning("this chain client cannot read commit order; hotkeys decide ties")
        return {}
    found: dict[str, int] = {}
    for block, hotkeys in sorted(tied.items()):
        try:
            found.update({hotkey: int(reader(block, hotkey)) for hotkey in hotkeys})
        except Exception as exc:
            log.warning("the commit order in block %d was not read; hotkeys decide: %s", block, exc)
    return found


def commit_order(chain: Any, blocks: Mapping[str, int]) -> list[str]:
    positions = commit_positions(chain, blocks)

    def key(hotkey: str) -> tuple[bool, int, int, str]:
        block = blocks[hotkey]
        return (block <= 0, block, positions.get(hotkey, 0), hotkey)

    return sorted(blocks, key=key)
