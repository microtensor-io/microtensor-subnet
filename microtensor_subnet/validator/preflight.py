from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from microtensor_subnet.artifact import gguf
from microtensor_subnet.artifact.params import (
    ParameterError,
    count_parameters,
    footprint_bytes,
    safetensors_in,
)
from microtensor_subnet.artifact.validate import sha256_of, weights_sha256
from microtensor_subnet.core import profiles

log = logging.getLogger("microtensor_subnet.validator.preflight")

OK: Final[str] = "ok"
CONFIG: Final[str] = "config"
TOKENIZER: Final[str] = "tokenizer"
WEIGHTS: Final[str] = "weights"
SIZE: Final[str] = "size"
PROBE_TEXT: Final[str] = "SELECT 1; def main(): return {'ok': True}"
DEFAULT_MAX_LEN: Final[int] = 8192
REASON_LIMIT: Final[int] = 300


@dataclass(slots=True)
class Preflight:
    stage: str
    reason: str = ""
    parameters: int = 0
    weights_sha256: str = ""
    footprint_bytes: int = 0
    files: dict[str, str] = field(default_factory=dict)
    vocab_size: int = 0
    architectures: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.stage == OK

    @property
    def message(self) -> str:
        return OK if self.ok else f"{self.stage}: {self.reason}"


def _reason(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text[:REASON_LIMIT] or exc.__class__.__name__


def _transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError("transformers is not installed; install the validator extra") from exc
    return transformers


def load_config(root: Path) -> Any:
    transformers = _transformers()
    return transformers.AutoConfig.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False
    )


def load_tokenizer(root: Path) -> Any:
    transformers = _transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False
    )
    encoded = tokenizer(PROBE_TEXT)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if not ids:
        raise ValueError("the tokenizer produced no tokens for a plain sentence")
    return tokenizer


def read_headers(files: list[Path]) -> int:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is not installed; install the validator extra") from exc
    tensors = 0
    for path in files:
        with safe_open(str(path), framework="np") as handle:
            handle.metadata()
            names = list(handle.keys())
        if not names:
            raise ValueError(f"{path.name} holds no tensors")
        tensors += len(names)
    return tensors


def hash_weights(root: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name.endswith((".safetensors", ".gguf")):
            found[path.relative_to(root).as_posix()] = sha256_of(path)
    return found


def check(
    root: Path,
    *,
    profile: str = "",
    bounds: tuple[int, int] | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    memory_bytes: int = 0,
) -> Preflight:
    root = Path(root)
    if not root.is_dir():
        return Preflight(CONFIG, f"{root} is not a directory")
    weights = safetensors_in(root)
    ggufs = gguf.gguf_in(root)
    architectures: tuple[str, ...] = ()
    vocab = 0
    if weights or not ggufs:
        try:
            config = load_config(root)
        except Exception as exc:
            return Preflight(CONFIG, _reason(exc))
        architectures = tuple(str(a) for a in getattr(config, "architectures", None) or ())
        vocab = int(getattr(config, "vocab_size", 0) or 0)
        try:
            tokenizer = load_tokenizer(root)
        except Exception as exc:
            return Preflight(TOKENIZER, _reason(exc))
        size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        if vocab and size and size > vocab:
            return Preflight(
                TOKENIZER, f"the tokenizer has {size} tokens but config.json allows {vocab}"
            )
    if not weights and not ggufs:
        return Preflight(WEIGHTS, "no model.safetensors (or shards) or model.gguf in the root")
    try:
        if weights:
            read_headers(weights)
            parameters = count_parameters(weights)
        else:
            if len(ggufs) != 1:
                raise ValueError("the model needs exactly one GGUF file")
            parameters = gguf.read(ggufs[0]).parameters
    except (OSError, ValueError, ParameterError, gguf.GgufError, RuntimeError) as exc:
        return Preflight(WEIGHTS, _reason(exc))
    except Exception as exc:
        return Preflight(WEIGHTS, _reason(exc))
    files = hash_weights(root)
    digest = weights_sha256(files)
    try:
        footprint = footprint_bytes(root, max_len)
    except (OSError, ParameterError) as exc:
        return Preflight(WEIGHTS, _reason(exc))
    if bounds is not None:
        low, high = bounds
        if parameters < low or parameters > high:
            return Preflight(
                SIZE,
                f"{parameters} parameters; the arena allows {low} to {high}",
                parameters=parameters,
                weights_sha256=digest,
                footprint_bytes=footprint,
                files=files,
            )
    cap = memory_bytes or (profiles.memory_bytes(profile) if profile else 0)
    if cap and footprint > cap:
        return Preflight(
            SIZE,
            f"{footprint} bytes with the KV cache at {max_len} tokens, over the "
            f"{cap} byte memory of profile {profile or 'this host'}",
            parameters=parameters,
            weights_sha256=digest,
            footprint_bytes=footprint,
            files=files,
        )
    return Preflight(
        OK,
        parameters=parameters,
        weights_sha256=digest,
        footprint_bytes=footprint,
        files=files,
        vocab_size=vocab,
        architectures=architectures,
    )


def bounds_of(definition: dict[str, Any]) -> tuple[int, int] | None:
    budgets = definition.get("budgets") or {}
    if not isinstance(budgets, dict) or not budgets.get("max_parameters"):
        return None
    return int(budgets.get("min_parameters") or 0), int(budgets["max_parameters"])


def max_len_of(definition: dict[str, Any]) -> int:
    budgets = definition.get("budgets") or {}
    if isinstance(budgets, dict) and budgets.get("max_input_tokens"):
        return int(budgets["max_input_tokens"]) + int(budgets.get("max_output_tokens") or 0)
    return DEFAULT_MAX_LEN
