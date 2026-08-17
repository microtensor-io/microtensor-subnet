from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

SAFETENSORS_SUFFIX: Final[str] = ".safetensors"
HEADER_LENGTH_BYTES: Final[int] = 8
MAX_HEADER_BYTES: Final[int] = 100 * 1024 * 1024

READABLE_DTYPES: Final[dict[str, str]] = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
}

UNSUPPORTED_FORMAT: Final[str] = (
    "divergence needs both sides in safetensors with matching tensor names; "
    "an exported or quantised graph does not align to its source checkpoint"
)


class DivergenceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Divergence:
    value: float | None
    aligned_tensors: int
    compared_elements: int
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.value is not None

    def payload(self) -> dict[str, Any]:
        if not self.available:
            return {"base_divergence": None, "base_divergence_unavailable": self.reason}
        return {
            "base_divergence": round(float(self.value or 0.0), 4),
            "aligned_tensors": self.aligned_tensors,
        }


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as fh:
        raw = fh.read(HEADER_LENGTH_BYTES)
        if len(raw) < HEADER_LENGTH_BYTES:
            raise DivergenceUnavailable(f"{path.name} is not a safetensors file")
        length = struct.unpack("<Q", raw)[0]
        if length <= 0 or length > MAX_HEADER_BYTES:
            raise DivergenceUnavailable(f"{path.name} declares an implausible header")
        try:
            header = json.loads(fh.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise DivergenceUnavailable(f"{path.name} has an unreadable header: {exc}") from exc
    if not isinstance(header, dict):
        raise DivergenceUnavailable(f"{path.name} has a malformed header")
    return header, HEADER_LENGTH_BYTES + length


def tensor_index(root: Path) -> dict[str, tuple[Path, str, int, int]]:
    files = sorted(root.rglob(f"*{SAFETENSORS_SUFFIX}")) if root.is_dir() else [root]
    index: dict[str, tuple[Path, str, int, int]] = {}
    for path in files:
        if not path.is_file():
            continue
        header, offset = _read_header(path)
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            dtype = str(meta.get("dtype", ""))
            spans = meta.get("data_offsets") or [0, 0]
            start, end = int(spans[0]), int(spans[1])
            if end > start:
                index[name] = (path, dtype, offset + start, offset + end)
    return index


def _read_tensor(entry: tuple[Path, str, int, int]) -> Any:
    import numpy

    path, dtype, start, end = entry
    code = READABLE_DTYPES.get(dtype)
    if code is None:
        raise DivergenceUnavailable(f"dtype {dtype} is not readable for divergence")
    with path.open("rb") as fh:
        fh.seek(start)
        buffer = fh.read(end - start)
    return numpy.frombuffer(buffer, dtype=numpy.dtype(code)).astype(numpy.float64, copy=False)


def aligned_names(artifact: dict[str, Any], base: dict[str, Any]) -> list[str]:
    return sorted(set(artifact) & set(base))


def _accumulate(
    artifact: dict[str, tuple[Path, str, int, int]],
    base: dict[str, tuple[Path, str, int, int]],
    names: list[str],
) -> Iterator[tuple[float, float, float, int]]:
    for name in names:
        left = _read_tensor(artifact[name])
        right = _read_tensor(base[name])
        if left.shape != right.shape or left.size == 0:
            continue
        yield (
            float((left * right).sum()),
            float((left * left).sum()),
            float((right * right).sum()),
            int(left.size),
        )


def base_divergence(artifact: Path, base: Path) -> Divergence:
    try:
        artifact_index = tensor_index(artifact)
        base_index = tensor_index(base)
    except DivergenceUnavailable as exc:
        return Divergence(None, 0, 0, str(exc))

    if not artifact_index or not base_index:
        return Divergence(None, 0, 0, UNSUPPORTED_FORMAT)

    names = aligned_names(artifact_index, base_index)
    if not names:
        return Divergence(
            None,
            0,
            0,
            "no tensor name is present in both the artifact and its declared base; "
            + UNSUPPORTED_FORMAT,
        )

    try:
        import numpy  # noqa: F401
    except ImportError:
        return Divergence(None, 0, 0, "numpy is required to compute divergence")

    dot = left_sq = right_sq = 0.0
    elements = 0
    compared = 0
    try:
        for d, ls, rs, size in _accumulate(artifact_index, base_index, names):
            dot += d
            left_sq += ls
            right_sq += rs
            elements += size
            compared += 1
    except DivergenceUnavailable as exc:
        return Divergence(None, 0, 0, str(exc))

    if compared == 0 or left_sq <= 0.0 or right_sq <= 0.0:
        return Divergence(
            None, compared, elements, "no comparable tensor pair had matching shapes"
        )

    cosine = dot / (math.sqrt(left_sq) * math.sqrt(right_sq))
    return Divergence(max(0.0, 1.0 - min(1.0, cosine)), compared, elements)


@dataclass(frozen=True, slots=True)
class Uplift:
    artifact_score: float
    base_score: float | None
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.base_score is not None

    @property
    def value(self) -> float | None:
        if self.base_score is None:
            return None
        return self.artifact_score - self.base_score

    def payload(self, decimals: int) -> dict[str, Any]:
        if not self.available:
            return {"base_score": None, "uplift": None, "uplift_unavailable": self.reason}
        return {
            "base_score": round(float(self.base_score or 0.0), decimals),
            "uplift": round(float(self.value or 0.0), decimals),
        }


class BaseScoreCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._scores: dict[str, float] = {}
        self.loads = 0
        self.misses = 0
        self._load()

    @staticmethod
    def key(base_model: str, corpus_version: str, track: str, hardware_class: str) -> str:
        return f"{base_model}|{corpus_version}|{track}|{hardware_class}"

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._scores = {str(k): float(v) for k, v in payload.items()}
        except (OSError, ValueError, TypeError):
            self._scores = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._scores, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def get(self, key: str) -> float | None:
        value = self._scores.get(key)
        if value is None:
            self.misses += 1
        else:
            self.loads += 1
        return value

    def put(self, key: str, score: float) -> None:
        self._scores[key] = float(score)
        self._save()

    def uplift(self, key: str, artifact_score: float) -> Uplift:
        base = self.get(key)
        if base is None:
            return Uplift(
                artifact_score=artifact_score,
                base_score=None,
                reason=(
                    "no scored baseline for this base model and corpus version; publish a "
                    "runnable base reference artifact and score it once"
                ),
            )
        return Uplift(artifact_score=artifact_score, base_score=base)


def work_evidence(
    base_model: str,
    divergence: Divergence,
    uplift: Uplift,
    *,
    decimals: int = 4,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"base_model": base_model}
    payload.update(divergence.payload())
    payload.update(uplift.payload(decimals))
    return payload
