from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

log = logging.getLogger("microtensor_subnet.validator.sandbox.compat")

GUEST: Final[str] = "/model"
CONFIG: Final[str] = "config.json"
TOKENIZER_CONFIG: Final[str] = "tokenizer_config.json"
MARK: Final[str] = ".compat-"
COPY_BYTES: Final[int] = 64 * 1024**2
DEFAULT_ROPE: Final[str] = "default"
LIFTED_ROPE: Final[tuple[str, ...]] = ("rope_theta", "partial_rotary_factor")
FULL_LAYERS: Final[str] = "full_attention"
LOCAL_LAYERS: Final[str] = "sliding_attention"
LOCAL_THETA: Final[str] = "rope_local_base_freq"
SUB_CONFIG: Final[str] = "_config"
DROPPED_TOKENIZER: Final[tuple[str, ...]] = ("backend", "is_local", "local_files_only")
TOKENIZER_CLASSES: Final[dict[str, str]] = {"TokenizersBackend": "PreTrainedTokenizerFast"}

_live: set[Path] = set()
_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class Mount:
    source: Path
    served: Path
    volumes: tuple[tuple[str, str, str], ...]


def plain(source: Path, guest: str = GUEST) -> Mount:
    return Mount(source, source, ((str(source), guest, "ro"),))


def _read(path: Path) -> dict[str, Any] | None:
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def config_needs(config: Mapping[str, Any]) -> bool:
    if "rope_parameters" in config or ("dtype" in config and "torch_dtype" not in config):
        return True
    return any(
        isinstance(value, Mapping) and config_needs(value)
        for key, value in config.items()
        if key.endswith(SUB_CONFIG)
    )


def tokenizer_needs(config: Mapping[str, Any]) -> bool:
    return isinstance(config.get("extra_special_tokens"), list) or any(
        key in config for key in DROPPED_TOKENIZER
    )


def _rope(config: dict[str, Any]) -> None:
    raw = config.pop("rope_parameters", None)
    if not isinstance(raw, Mapping):
        return
    if raw and all(isinstance(value, Mapping) for value in raw.values()):
        local = raw.get(LOCAL_LAYERS)
        if isinstance(local, Mapping) and "rope_theta" in local:
            config.setdefault(LOCAL_THETA, local["rope_theta"])
        full = raw.get(FULL_LAYERS)
        raw = full if isinstance(full, Mapping) else next(iter(raw.values()))
    params = dict(raw)
    for key in LIFTED_ROPE:
        if key in params:
            config.setdefault(key, params.pop(key))
    kind = str(params.pop("rope_type", "") or params.pop("type", "") or DEFAULT_ROPE)
    params.pop("type", None)
    if (kind != DEFAULT_ROPE or params) and not config.get("rope_scaling"):
        config["rope_scaling"] = {"rope_type": kind, **params}


def config_rewrite(config: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(config)
    _rope(out)
    if "dtype" in out:
        dtype = out.pop("dtype")
        out.setdefault("torch_dtype", dtype)
    for key, value in config.items():
        if key.endswith(SUB_CONFIG) and isinstance(value, Mapping):
            out[key] = config_rewrite(value)
    return out


def tokenizer_rewrite(config: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in config.items() if k not in DROPPED_TOKENIZER}
    extra = out.get("extra_special_tokens")
    if isinstance(extra, list):
        out.pop("extra_special_tokens")
        merged = list(out.get("additional_special_tokens") or [])
        merged += [token for token in extra if token not in merged]
        if merged:
            out["additional_special_tokens"] = merged
    named = str(out.get("tokenizer_class") or "")
    if named in TOKENIZER_CLASSES:
        out["tokenizer_class"] = TOKENIZER_CLASSES[named]
    return out


def rewrites(source: Path) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    config = _read(source / CONFIG)
    if config is not None and config_needs(config):
        found[CONFIG] = config_rewrite(config)
    tokenizer = _read(source / TOKENIZER_CONFIG)
    if tokenizer is not None and tokenizer_needs(tokenizer):
        found[TOKENIZER_CONFIG] = tokenizer_rewrite(tokenizer)
    return found


def _siblings(source: Path) -> list[Path]:
    prefix = source.name + MARK
    try:
        return [p for p in source.parent.iterdir() if p.name.startswith(prefix) and p.is_dir()]
    except OSError:
        return []


def _place(path: Path, destination: Path) -> bool:
    try:
        os.link(path, destination)
    except OSError:
        if path.stat().st_size < COPY_BYTES:
            shutil.copy2(path, destination)
            return False
        destination.symlink_to(path)
        return True
    return False


def _mirror(source: Path, target: Path, skip: set[str]) -> bool:
    outside = False
    target.mkdir()
    for root, folders, files in os.walk(source):
        here = Path(root)
        relative = here.relative_to(source)
        for folder in folders:
            (target / relative / folder).mkdir(exist_ok=True)
        for name in files:
            if (relative / name).as_posix() in skip:
                continue
            outside = _place(here / name, target / relative / name) or outside
    return outside


def mount(source: Path, guest: str = GUEST) -> Mount:
    changed = rewrites(source)
    if not changed:
        return plain(source, guest)
    with _lock:
        served = source.parent / f"{source.name}{MARK}{secrets.token_hex(4)}"
        _live.add(served)
        stale = [p for p in _siblings(source) if p not in _live]
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)
    try:
        outside = _mirror(source, served, set(changed))
        for name, content in changed.items():
            (served / name).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        log.warning("%s: no transformers v4 copy, serving the original: %s", source, exc)
        release(Mount(source, served, ()))
        return plain(source, guest)
    log.info("%s: serving a transformers v4 copy of %s", source, ", ".join(sorted(changed)))
    volumes = [(str(served), guest, "ro")]
    if outside:
        volumes.append((str(source), str(source), "ro"))
    return Mount(source, served, tuple(volumes))


def release(made: Mount) -> None:
    if made.served == made.source:
        return
    shutil.rmtree(made.served, ignore_errors=True)
    with _lock:
        _live.discard(made.served)
