from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from microtensor_subnet.core.hashing import canonical_json

FIELDS: Final[tuple[str, ...]] = (
    "request_id",
    "time",
    "key_hash",
    "model",
    "image_digest",
    "operator",
    "instance",
    "input_tokens",
    "output_tokens",
    "ttft_ms",
    "total_ms",
    "outcome",
    "probe",
    "price_usd",
)
OUTCOMES: Final[tuple[str, ...]] = ("ok", "retried", "fallback", "error", "refused")
EMPTY_ROOT: Final[str] = ""


class ReceiptError(RuntimeError):
    pass


def canonical(receipt: Mapping[str, Any]) -> bytes:
    return canonical_json(dict(receipt))


def leaf(receipt: Mapping[str, Any]) -> bytes:
    return hashlib.sha256(canonical(receipt)).digest()


def root(receipts: Iterable[Mapping[str, Any]]) -> str:
    ordered = sorted(receipts, key=lambda r: str(r.get("request_id", "")))
    level = [leaf(r) for r in ordered]
    if not level:
        return EMPTY_ROOT
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


def verify(receipt: Mapping[str, Any], signature: str, public_key: str) -> bool:
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key))
        key.verify(bytes.fromhex(signature), canonical(receipt))
    except (ValueError, TypeError, InvalidSignature):
        return False
    return True


def shaped(receipt: Mapping[str, Any]) -> str:
    missing = [name for name in FIELDS if name not in receipt]
    if missing:
        return f"missing {', '.join(missing)}"
    if receipt.get("outcome") not in OUTCOMES:
        return f"outcome {receipt.get('outcome')!r} is not one of {', '.join(OUTCOMES)}"
    return ""


def check(
    rows: Sequence[Mapping[str, Any]], *, public_key: str, expected_root: str
) -> tuple[bool, int, int]:
    receipts: list[Mapping[str, Any]] = []
    bad = 0
    for row in rows:
        receipt = row.get("receipt")
        if not isinstance(receipt, Mapping):
            bad += 1
            continue
        receipts.append(receipt)
        if public_key and not verify(receipt, str(row.get("signature") or ""), public_key):
            bad += 1
    found = root(receipts)
    return found == expected_root, bad, len(receipts)
