from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from microtensor.chain.client import SubtensorClient
from microtensor.chain.config import ChainConfig
from microtensor.chain.wallet import WalletError, load_wallet

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-32s %(message)s"


def default_home() -> Path:
    return Path(os.environ.get("MT_HOME", Path.home() / ".microtensor"))


def configure_logging(level: str = "INFO") -> None:
    resolved = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=resolved, format=LOG_FORMAT, datefmt="%H:%M:%S", stream=sys.stderr)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)

    # Own handler, no propagation. bittensor reconfigures the root logger to
    # WARNING when a wallet loads, which silently swallowed every INFO line
    # after startup: a validator waiting hours for a round close looked hung.
    ours = logging.getLogger("microtensor")
    ours.setLevel(resolved)
    ours.propagate = False
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
    ours.addHandler(handler)


def add_chain_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--netuid", type=int, help="subnet id")
    parser.add_argument("--network", help="finney, test, local, or archive")
    parser.add_argument("--endpoint", help="explicit chain websocket endpoint")
    parser.add_argument("--wallet-name", help="coldkey wallet name")
    parser.add_argument("--wallet-hotkey", help="hotkey name")
    parser.add_argument("--wallet-path", help="wallet directory")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--home", type=Path, default=default_home(), help="state directory")
    parser.add_argument("--log-level", default="INFO")


def chain_config(args: argparse.Namespace) -> ChainConfig:
    return ChainConfig.from_env(
        netuid=args.netuid,
        network=args.network,
        endpoint=args.endpoint,
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        wallet_path=args.wallet_path,
    )


def open_wallet(config: ChainConfig, *, required: bool = True):  # type: ignore[no-untyped-def]
    try:
        return load_wallet(config)
    except WalletError as exc:
        if required:
            raise SystemExit(f"wallet unavailable: {exc}") from exc
        return None


def open_client(config: ChainConfig, wallet: object | None = None) -> SubtensorClient:
    return SubtensorClient(config, wallet)


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1
