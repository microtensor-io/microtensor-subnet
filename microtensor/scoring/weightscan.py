"""Reading weight statistics from an artifact, best effort.

Derivation detection compares numbers derived from weights, not the raw
tensors: a projected sample of the flattened weights for displacement, and the
per-layer attention standard-deviation profile for the attention signal. Both
are read here, and anything unreadable returns None so the caller records the
signal as unavailable rather than failing a round over a parsing quirk.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

log = logging.getLogger("microtensor.derivation")

_GGUF_MAGIC = b"GGUF"
SAMPLE_DIMS = 256



def _entrypoint(artifact: Path) -> Path | None:
    for name in ("model.gguf", "front.gguf"):
        candidate = artifact / name
        if candidate.is_file():
            return candidate
    ggufs = sorted(artifact.rglob("*.gguf"))
    return ggufs[0] if ggufs else None


def _project(values: list[float], dims: int = SAMPLE_DIMS) -> list[float]:
    """Fold a long weight sample into a fixed-length vector by summed buckets.

    A deterministic projection, so two reads of the same file land in the same
    space and cosine is meaningful. Not a random projection: it needs no shared
    seed between validators, which a random one would.
    """
    if not values:
        return [0.0] * dims
    out = [0.0] * dims
    for i, v in enumerate(values):
        out[i % dims] += v
    return out


def weight_sample(artifact: Path, limit: int = 1_000_000) -> list[float] | None:
    """A projected sample of float32 weights, or None if unreadable.

    Reads raw float32 runs out of the gguf tensor region up to `limit` values.
    This is a fingerprinting sample, not a faithful dequantisation: it only has
    to be stable and comparable across artifacts, which raw float32 runs are.
    """
    path = _entrypoint(artifact)
    if path is None:
        return None
    try:
        raw = path.read_bytes()
    except OSError as exc:
        log.warning("weight sample unreadable at %s: %s", path, exc)
        return None
    if raw[:4] != _GGUF_MAGIC:
        return None

    floats: list[float] = []
    stride = max(4, (len(raw) // (limit * 4)) * 4) if len(raw) > limit * 4 else 4
    for offset in range(0, len(raw) - 4, stride):
        (value,) = struct.unpack_from("<f", raw, offset)
        if value == value and abs(value) < 1e6:  # drop NaN and non-weight noise
            floats.append(value)
        if len(floats) >= limit:
            break
    return _project(floats) if floats else None


def attention_profile(artifact: Path, layers: int = 32) -> list[float] | None:
    """A crude per-region standard-deviation profile, or None.

    A faithful profile reads each attention projection tensor by name. Absent a
    full gguf tensor parser here, this slices the weight region into `layers`
    bands and takes each band's standard deviation, which preserves the shape
    of the curve across depth well enough for a comparison and degrades to None
    when the file cannot be read.
    """
    path = _entrypoint(artifact)
    if path is None:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if raw[:4] != _GGUF_MAGIC or len(raw) < layers * 64:
        return None

    body = raw[len(raw) % 4 :]
    band = (len(body) // 4) // layers
    if band == 0:
        return None
    profile: list[float] = []
    for layer in range(layers):
        start = layer * band * 4
        values = [
            struct.unpack_from("<f", body, start + i * 4)[0]
            for i in range(0, band, max(1, band // 512))
        ]
        values = [v for v in values if v == v and abs(v) < 1e6]
        if not values:
            profile.append(0.0)
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        profile.append(var**0.5)
    return profile
