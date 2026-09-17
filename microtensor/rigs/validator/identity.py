from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from microtensor.rigs.protocol.session import (
    HEADER_HOTKEY,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    signing_bytes,
)
from microtensor.rigs.validator.config import Settings

try:
    from bittensor_wallet import Keypair
except ImportError:  # pragma: no cover
    from substrateinterface import Keypair

ED25519 = 0
SR25519 = 1

SS58_FORMAT = 42


class IdentityError(RuntimeError):
    pass


def canonical_json(body: Any) -> bytes:
    return json.dumps(body, separators=(",", ":"), sort_keys=True).encode()


def seed_bytes(text: str) -> bytes:
    clean = text.strip()
    if clean.startswith("0x"):
        clean = clean[2:]
    try:
        raw = bytes.fromhex(clean)
    except ValueError as exc:
        raise IdentityError("the hotkey seed must be hex") from exc
    if len(raw) != 32:
        raise IdentityError("the hotkey seed must be 32 bytes")
    return raw


def load_keypair(settings: Settings) -> Keypair:
    if settings.hotkey_seed:
        return Keypair.create_from_seed(seed_bytes(settings.hotkey_seed), ss58_format=SS58_FORMAT)
    if settings.hotkey_mnemonic:
        return Keypair.create_from_mnemonic(
            settings.hotkey_mnemonic.strip(), ss58_format=SS58_FORMAT
        )
    if settings.wallet_hotkey_file:
        try:
            data = json.loads(Path(settings.wallet_hotkey_file).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise IdentityError(f"cannot read the wallet hotkey file: {exc}") from exc
        seed = str(data.get("secretSeed", "") or "")
        phrase = str(data.get("secretPhrase", "") or "")
        if seed:
            return Keypair.create_from_seed(seed_bytes(seed), ss58_format=SS58_FORMAT)
        if phrase:
            return Keypair.create_from_mnemonic(phrase.strip(), ss58_format=SS58_FORMAT)
        private = str(data.get("privateKey", "") or "")
        if private:
            raw = bytes.fromhex(private[2:] if private.startswith("0x") else private)
            public = str(data.get("publicKey", "") or "")
            crypto = ED25519 if int(data.get("cryptoType", 1) or 1) == 0 else SR25519
            return Keypair(
                public_key=bytes.fromhex(public[2:] if public.startswith("0x") else public)
                if public
                else None,
                private_key=raw,
                ss58_format=SS58_FORMAT,
                crypto_type=crypto,
            )
        raise IdentityError(
            "the wallet hotkey file has neither secretSeed, secretPhrase nor privateKey"
        )
    raise IdentityError("set CV_HOTKEY_SEED, CV_HOTKEY_MNEMONIC or CV_WALLET_HOTKEY_FILE")


class Identity:
    def __init__(self, keypair: Keypair) -> None:
        self._key = keypair
        self._lock = threading.Lock()
        self._last_stamp = 0.0

    @property
    def keypair(self) -> Keypair:
        return self._key

    @property
    def hotkey(self) -> str:
        return str(self._key.ss58_address)

    def sign(self, message: bytes) -> str:
        return "0x" + bytes(self._key.sign(message)).hex()

    def verify(self, message: bytes, signature: str) -> bool:
        raw = signature[2:] if signature.startswith("0x") else signature
        try:
            return bool(self._key.verify(message, bytes.fromhex(raw)))
        except (ValueError, TypeError):
            return False

    def timestamp(self) -> str:
        with self._lock:
            stamp = max(time.time(), self._last_stamp + 0.001)
            self._last_stamp = stamp
        return f"{stamp:.3f}"

    def headers(self, method: str, path: str, body: bytes = b"") -> dict[str, str]:
        timestamp = self.timestamp()
        return {
            HEADER_HOTKEY: self.hotkey,
            HEADER_TIMESTAMP: timestamp,
            HEADER_SIGNATURE: self.sign(signing_bytes(method, path, timestamp, body)),
        }
