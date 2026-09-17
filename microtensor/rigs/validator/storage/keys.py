from __future__ import annotations

import base64
import binascii
import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MASTER_SECRET_ENV = "CV_VOLUME_MASTER_SECRET"
INFO = b"microtensor-compute volume passphrase v1"
MIN_SECRET_BYTES = 32
PASSPHRASE_BYTES = 32


def decode_secret(text: str) -> bytes:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("volume master secret is empty")
    try:
        secret = binascii.unhexlify(raw)
    except (binascii.Error, ValueError):
        try:
            secret = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            secret = raw.encode("utf-8")
    if len(secret) < MIN_SECRET_BYTES:
        raise ValueError(f"volume master secret must be at least {MIN_SECRET_BYTES} bytes")
    return secret


def master_secret() -> bytes:
    return decode_secret(os.environ.get(MASTER_SECRET_ENV, ""))


def volume_passphrase(secret: bytes, volume_id: str) -> str:
    if not volume_id:
        raise ValueError("volume id is required")
    derived = HKDF(
        algorithm=hashes.SHA256(),
        length=PASSPHRASE_BYTES,
        salt=volume_id.encode("utf-8"),
        info=INFO,
    ).derive(secret)
    return base64.urlsafe_b64encode(derived).decode("ascii").rstrip("=")
