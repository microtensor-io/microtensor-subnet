from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any, Final

from microtensor.core.constants import DEFAULT_NETUID

NETWORKS: Final[dict[str, str]] = {
    "finney": "wss://entrypoint-finney.opentensor.ai:443",
    "test": "wss://test.finney.opentensor.ai:443",
    "archive": "wss://archive.chain.opentensor.ai:443",
    "local": "ws://127.0.0.1:9944",
}

ENV_PREFIX: Final[str] = "MT_"


@dataclass(frozen=True, slots=True)
class ChainConfig:
    netuid: int = DEFAULT_NETUID
    network: str = "finney"
    endpoint: str = ""
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    wallet_path: str = ""

    def __post_init__(self) -> None:
        if self.netuid < 0:
            raise ValueError("netuid must not be negative")
        if not self.endpoint and self.network not in NETWORKS:
            raise ValueError(f"unknown network {self.network!r}; known: {sorted(NETWORKS)}")
        if not self.wallet_name or not self.wallet_hotkey:
            raise ValueError("wallet name and hotkey must both be set")

    @property
    def resolved_endpoint(self) -> str:
        return self.endpoint or NETWORKS[self.network]

    @property
    def is_local(self) -> bool:
        return self.resolved_endpoint.startswith("ws://")

    def with_overrides(self, **overrides: Any) -> ChainConfig:
        supplied = {k: v for k, v in overrides.items() if v is not None}
        unknown = set(supplied) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"unknown chain settings: {sorted(unknown)}")
        return replace(self, **supplied)

    @classmethod
    def from_env(cls, **overrides: Any) -> ChainConfig:
        env: dict[str, Any] = {}
        for field in fields(cls):
            raw = os.environ.get(f"{ENV_PREFIX}{field.name.upper()}", "").strip()
            if not raw:
                continue
            env[field.name] = int(raw) if field.type in ("int", int) else raw
        return cls(**env).with_overrides(**overrides)
