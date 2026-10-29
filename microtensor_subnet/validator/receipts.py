from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Final

from microtensor_subnet.core.hashing import canonical_json

SIGNED_FIELDS: Final[tuple[str, ...]] = (
    "receipt",
    "ticket",
    "arena_id",
    "epoch_index",
    "validator",
    "miner",
    "item_id",
    "model",
    "tokens_in",
    "tokens_out",
    "cost_micros",
    "latency_ms",
    "status",
    "prompt_sha256",
    "completion_sha256",
    "purpose",
    "subject",
)
VALIDATOR_SUBJECT: Final[str] = "validator"
MINER_SUBJECT: Final[str] = "miner"
OK_STATUS: Final[str] = "ok"


def signed_body(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: receipt[key] for key in SIGNED_FIELDS if key in receipt}


def signed_bytes(receipt: Mapping[str, Any]) -> bytes:
    return canonical_json(signed_body(receipt))


def _nfc(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        return {_nfc(k): _nfc(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nfc(v) for v in value]
    return value


def prompt_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        _nfc(list(messages)), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def completion_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify(receipt: Mapping[str, Any], server_public_key: str) -> bool:
    signature = str(receipt.get("signature", "") or "")
    if not signature or not server_public_key or not receipt.get("receipt"):
        return False
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        key = server_public_key.strip().lower()
        key = key[2:] if key.startswith("0x") else key
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key)).verify(
            bytes.fromhex(signature), signed_bytes(receipt)
        )
    except Exception:
        return False
    return True


def check(
    receipt: Mapping[str, Any],
    *,
    server_public_key: str,
    arena_id: str = "",
    epoch_index: int | None = None,
    miner: str = "",
    item_id: str = "",
    purpose: str = "",
    cap_micros: int = 0,
    subject: str = "",
) -> str:
    if not verify(receipt, server_public_key):
        return "receipt signature does not verify against the server key"
    if arena_id and str(receipt.get("arena_id", "")) != arena_id:
        return "receipt names another arena"
    if epoch_index is not None and int(receipt.get("epoch_index") or -1) != int(epoch_index):
        return "receipt names another epoch"
    if miner and str(receipt.get("miner", "")) != miner:
        return "receipt names another miner"
    if item_id and str(receipt.get("item_id", "")) != item_id:
        return "receipt names another item"
    if purpose and str(receipt.get("purpose", "")) != purpose:
        return f"receipt purpose {receipt.get('purpose')!r} is not {purpose!r}"
    if subject and str(receipt.get("subject", "")) != subject:
        return f"receipt subject {receipt.get('subject')!r} is not {subject!r}"
    if cap_micros and int(receipt.get("cost_micros") or 0) > cap_micros:
        return f"receipt cost {receipt.get('cost_micros')} is over the cap {cap_micros}"
    return ""


def total_cost(receipts: Sequence[Mapping[str, Any]]) -> int:
    return sum(int(r.get("cost_micros") or 0) for r in receipts)


def total_latency(receipts: Sequence[Mapping[str, Any]]) -> int:
    return sum(int(r.get("latency_ms") or 0) for r in receipts)
