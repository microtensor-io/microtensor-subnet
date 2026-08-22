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
    """Whichever keypair implementation is installed.

    Aliased on import: binding the same name twice in one scope is a
    redefinition, and the type checker is right to say so.
    """
    try:
        from bittensor_wallet import Keypair as WalletKeypair

        return WalletKeypair
    except ImportError:
        pass

    try:
        from substrateinterface import Keypair as SubstrateKeypair

        return SubstrateKeypair
    except ImportError as exc:
        raise WalletError("no keypair implementation is available") from exc


def load_wallet(config: ChainConfig) -> Any:
    bittensor = _bittensor()
    kwargs: dict[str, Any] = {"name": config.wallet_name, "hotkey": config.wallet_hotkey}
    if config.wallet_path:
        kwargs["path"] = config.wallet_path
    # bittensor 10 dropped the lowercase module aliases and kept only the
    # classes, so the name that worked on 9 raises AttributeError on 10. The
    # subtensor factory in chain/client.py already picks whichever exists;
    # this path did not, and it is the one every wallet-using command needs.
    factory = getattr(bittensor, "wallet", None) or getattr(bittensor, "Wallet", None)
    if factory is None:
        raise WalletError("this bittensor build exposes no wallet class")
    wallet = factory(**kwargs)
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
    return str(bytes(signature).hex())


def sign_bytes(wallet: Any, message: bytes) -> str:
    """Sign raw bytes rather than a canonicalised mapping.

    Request authentication covers the method and path alongside the body, so
    a signed report cannot be replayed against a different endpoint, and that
    is not a JSON document.
    """
    try:
        signature = wallet.hotkey.sign(message)
    except Exception as exc:
        raise WalletError("hotkey refused to sign the message") from exc
    return str(bytes(signature).hex())


def verify_bytes(hotkey: str, message: bytes, signature: str) -> bool:
    if not signature or not hotkey:
        return False
    try:
        keypair = _keypair_class()(ss58_address=hotkey)
        return bool(keypair.verify(message, bytes.fromhex(signature)))
    except (WalletError, ValueError, TypeError):
        return False
    except Exception:
        return False


def verify_payload(hotkey: str, payload: Mapping[str, Any], signature: str) -> bool:
    if not signature or not hotkey:
        return False
    try:
        return verify_bytes(hotkey, canonical_json(payload), signature)
    except (WalletError, ValueError, TypeError):
        return False
