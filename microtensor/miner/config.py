from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from microtensor.chain.commitment import SOURCE_SCHEMES
from microtensor.chain.config import ChainConfig
from microtensor.core.constants import GENESIS_BLOCK, ROUND_BLOCKS
from microtensor.core.protocol import ArtifactFormat
from microtensor.core.tracks import CLASSES, enabled_tracks, is_competable

CONFIG_NAME = "miner.json"

HF_PINNED_LOCATOR = re.compile(r"^[\w.-]+/[\w.-]+@[0-9a-f]{7,40}$")


class MinerConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MinerConfig:
    chain: ChainConfig
    home: Path
    artifact_dir: Path
    track: str
    hardware_class: str
    source: str
    entrypoint: str = "model.onnx"
    artifact_format: str = ArtifactFormat.ONNX.value
    quantization: str = ""
    max_input_tokens: int = 4096
    tokenizer: str = "tokenizer.json"
    base_model: str = ""
    round_blocks: int = ROUND_BLOCKS
    genesis_block: int = GENESIS_BLOCK
    republish_margin_blocks: int = 120
    allow_unsandboxed: bool = False

    def __post_init__(self) -> None:
        if not is_competable(self.track, self.hardware_class):
            raise MinerConfigError(
                f"{self.track}/{self.hardware_class} is not an open competition; "
                f"tracks: {[t.id for t in enabled_tracks()]}, classes: {sorted(CLASSES)}"
            )
        scheme, _, locator = self.source.partition(":")
        if scheme not in SOURCE_SCHEMES:
            raise MinerConfigError(
                f"source scheme {scheme!r} is not one validators can fetch: "
                f"{sorted(SOURCE_SCHEMES)}"
            )
        if not locator:
            raise MinerConfigError("source must carry a locator after the scheme")
        if scheme == "hf" and not HF_PINNED_LOCATOR.match(locator):
            raise MinerConfigError(
                f"hf source {locator!r} must pin a commit as <org>/<repo>@<sha>; "
                "a branch or tag can move after you commit the digest, so validators "
                "would fetch a tree that no longer matches your manifest"
            )

    @property
    def manifest_path(self) -> Path:
        return self.artifact_dir / "manifest.json"

    @property
    def competition(self) -> tuple[str, str]:
        return self.track, self.hardware_class

    @property
    def config_path(self) -> Path:
        return self.home / CONFIG_NAME

    @property
    def selfcheck_path(self) -> Path:
        return self.home / "selfcheck.json"

    @property
    def scheme(self) -> str:
        return self.source.partition(":")[0]

    @property
    def locator(self) -> str:
        return self.source.partition(":")[2]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            k: v for k, v in asdict(self).items() if k not in ("chain", "home", "artifact_dir")
        }
        payload["artifact_dir"] = str(self.artifact_dir)
        payload["netuid"] = self.chain.netuid
        payload["network"] = self.chain.network
        payload["wallet_name"] = self.chain.wallet_name
        payload["wallet_hotkey"] = self.chain.wallet_hotkey
        return payload

    def save(self) -> Path:
        self.home.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return self.config_path

    def with_overrides(self, **overrides: Any) -> MinerConfig:
        supplied = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **supplied) if supplied else self

    @classmethod
    def load(cls, home: Path, chain: ChainConfig, **overrides: Any) -> MinerConfig:
        path = home / CONFIG_NAME
        if not path.is_file():
            raise MinerConfigError(
                f"no miner config at {path}; run `mt miner init` once, or pass "
                "--artifact --track --hardware-class --source on every command"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MinerConfigError(f"{path} is unreadable: {exc}") from exc

        for key in ("netuid", "network", "wallet_name", "wallet_hotkey"):
            payload.pop(key, None)
        payload["artifact_dir"] = Path(payload.get("artifact_dir", ""))

        supplied = {k: v for k, v in overrides.items() if v is not None}
        payload.update(supplied)

        known = {f for f in cls.__dataclass_fields__ if f not in ("chain", "home")}
        unknown = set(payload) - known
        for key in unknown:
            payload.pop(key)

        try:
            return cls(chain=chain, home=home, **payload)
        except TypeError as exc:
            raise MinerConfigError(f"{path} is missing a required setting: {exc}") from exc

    @classmethod
    def build(cls, home: Path, chain: ChainConfig, **fields: Any) -> MinerConfig:
        supplied = {k: v for k, v in fields.items() if v is not None}
        missing = [
            name
            for name in ("artifact_dir", "track", "hardware_class", "source")
            if name not in supplied
        ]
        if missing:
            raise MinerConfigError(f"missing required settings: {', '.join(missing)}")
        return cls(chain=chain, home=home, **supplied)
