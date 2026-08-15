from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from microtensor.chain.config import ChainConfig
from microtensor.core.hashing import canonical_json


class WalletError(RuntimeError):
    pass


def _bittensor() -> Any:
    try:
        import bittensor
    except ImportError as exc:
        raise WalletError(
            "bittensor is not installed; install the validator or miner extra"
        ) from exc
    return bittensor


def _keypair_class() -> Any:
    try:
        from bittensor_wallet import Keypair
    except ImportError:
        try:
            from substrateinterface import Keypair
        except ImportError as exc:
            raise WalletError("no keypair implementation is available") from exc
    return Keypair


def load_wallet(config: ChainConfig) -> Any:
    bittensor = _bittensor()
    kwargs: dict[str, Any] = {"name": config.wallet_name, "hotkey": config.wallet_hotkey}
    if config.wallet_path:
        kwargs["path"] = config.wallet_path
    wallet = bittensor.wallet(**kwargs)
    try:
        _ = wallet.hotkey.ss58_address
    except Exception as exc:
        raise WalletError(
            f"wallet {config.wallet_name}/{config.wallet_hotkey} could not be unlocked"
        ) from exc
    return wallet


def hotkey_address(wallet: Any) -> str:
    address = getattr(getattr(wallet, "hotkey", None), "ss58_address", "")
    if not address:
        raise WalletError("wallet exposes no hotkey address")
    return str(address)


def coldkey_address(wallet: Any) -> str:
    return str(getattr(getattr(wallet, "coldkeypub", None), "ss58_address", ""))


def sign_payload(wallet: Any, payload: Mapping[str, Any]) -> str:
    message = canonical_json(payload)
    try:
        signature = wallet.hotkey.sign(message)
    except Exception as exc:
        raise WalletError("hotkey refused to sign the payload") from exc
    return signature.hex()


def verify_payload(hotkey: str, payload: Mapping[str, Any], signature: str) -> bool:
    if not signature or not hotkey:
        return False
    try:
        keypair = _keypair_class()(ss58_address=hotkey)
        return bool(keypair.verify(canonical_json(payload), bytes.fromhex(signature)))
    except (WalletError, ValueError, TypeError):
        return False
    except Exception:
        return False
