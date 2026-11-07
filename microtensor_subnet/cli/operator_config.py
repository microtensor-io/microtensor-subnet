from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final

KEYS: Final[tuple[str, ...]] = (
    "hotkey_path",
    "gateway",
    "gpus",
    "models",
    "max_concurrency_per_gpu",
    "cache_dir",
    "collateral_tao_per_gpu",
)
AUTO: Final[str] = "auto"


class ConfigError(RuntimeError):
    pass


def _value(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
        return [part.strip().strip("'\"") for part in text.split(",") if part.strip()]
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return text


def parse(text: str) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        body = line.split("#", 1)[0].rstrip()
        if not body.strip():
            continue
        if ":" not in body or body.startswith((" ", "\t")):
            raise ConfigError(f"line {number} is not a key and a value")
        key, raw = body.split(":", 1)
        key = key.strip()
        if key not in KEYS:
            raise ConfigError(
                f"line {number}: {key!r} is not a setting; settings are {', '.join(KEYS)}"
            )
        found[key] = _value(raw)
    return found


def read(path: Path) -> dict[str, Any]:
    try:
        return parse(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"the config file {path} is unreadable: {exc}") from exc


def wallet_of(hotkey_path: str) -> tuple[str, str, str]:
    path = PurePosixPath(str(hotkey_path).strip().replace("\\", "/"))
    if path.parent.name != "hotkeys" or not path.name or not path.parent.parent.name:
        raise ConfigError("hotkey_path must look like <wallets>/<coldkey>/hotkeys/<hotkey>")
    return str(path.parent.parent.parent), path.parent.parent.name, path.name


def apply(
    args: argparse.Namespace, found: Mapping[str, Any], defaults: Mapping[str, Any]
) -> list[str]:
    notes: list[str] = []

    def unset(name: str) -> bool:
        return getattr(args, name, None) == defaults.get(name)

    if found.get("hotkey_path"):
        wallet_path, name, hotkey = wallet_of(str(found["hotkey_path"]))
        if unset("wallet_path"):
            args.wallet_path = wallet_path
        if unset("wallet_name"):
            args.wallet_name = name
        if unset("wallet_hotkey"):
            args.wallet_hotkey = hotkey
    if found.get("gateway") and unset("gateway"):
        args.gateway = str(found["gateway"])
    if found.get("gpus") and unset("gpus"):
        raw = found["gpus"]
        args.gpus = ",".join(str(g) for g in raw) if isinstance(raw, list) else str(raw)
    models = found.get("models")
    if models and unset("models") and unset("auto"):
        if isinstance(models, list):
            args.models = [str(m) for m in models]
            args.auto = True
        elif str(models).strip().lower() == AUTO:
            args.auto = True
        else:
            args.models = [str(models).strip()]
            args.auto = True
    if found.get("max_concurrency_per_gpu") and unset("max_concurrency"):
        try:
            args.max_concurrency = max(0, int(str(found["max_concurrency_per_gpu"])))
        except ValueError as exc:
            raise ConfigError("max_concurrency_per_gpu must be a whole number") from exc
    if found.get("cache_dir") and unset("cache_dir"):
        args.cache_dir = str(found["cache_dir"])
    if found.get("collateral_tao_per_gpu") is not None:
        notes.append(f"collateral per gpu in the config: {found['collateral_tao_per_gpu']} TAO")
    return notes
