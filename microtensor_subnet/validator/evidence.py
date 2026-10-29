from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from microtensor_subnet.chain.wallet import sign_bytes, verify_bytes
from microtensor_subnet.validator.transcripts import canonical_json, client_method

log = logging.getLogger("microtensor_subnet.validator.evidence")

SPOT_CHECK: Final[str] = "spot_check"
CANARY: Final[str] = "canary"
KINDS: Final[tuple[str, ...]] = (SPOT_CHECK, CANARY)
SIGNATURE: Final[str] = "validator_sig"
COMPLETION_LIMIT: Final[int] = 2000


def digest(body: Mapping[str, Any]) -> bytes:
    unsigned = {k: v for k, v in body.items() if k != SIGNATURE}
    return hashlib.sha256(canonical_json(unsigned)).digest()


def sign(body: Mapping[str, Any], wallet: Any, hotkey: str) -> dict[str, Any]:
    unsigned = {k: v for k, v in body.items() if k != SIGNATURE}
    unsigned["validator_hotkey"] = hotkey
    return {**unsigned, SIGNATURE: sign_bytes(wallet, digest(unsigned))}


def verify(body: Mapping[str, Any]) -> bool:
    return verify_bytes(
        str(body.get("validator_hotkey", "")), digest(body), str(body.get(SIGNATURE, ""))
    )


def first_divergence(claimed: Sequence[int], recomputed: Sequence[int]) -> int:
    for index, (left, right) in enumerate(zip(claimed, recomputed, strict=False)):
        if left != right:
            return index
    return min(len(claimed), len(recomputed))


def spot_check_body(
    *,
    arena_id: str,
    epoch: int,
    miner_hotkey: str,
    profile: str,
    weights_sha256: str,
    item_id: str,
    call_seq: int,
    seed: int,
    prompt_sha256: str,
    claimed_tokens: Sequence[int],
    recomputed_tokens: Sequence[int],
) -> dict[str, Any]:
    return {
        "kind": SPOT_CHECK,
        "arena_id": arena_id,
        "epoch": int(epoch),
        "miner_hotkey": miner_hotkey,
        "profile": profile,
        "weights_sha256": weights_sha256,
        "item_id": item_id,
        "call_seq": int(call_seq),
        "seed": int(seed),
        "prompt_sha256": prompt_sha256,
        "claimed_tokens": [int(t) for t in claimed_tokens],
        "recomputed_tokens": [int(t) for t in recomputed_tokens],
        "first_divergence": first_divergence(claimed_tokens, recomputed_tokens),
    }


def canary_body(
    *,
    arena_id: str,
    epoch: int,
    miner_hotkey: str,
    profile: str,
    weights_sha256: str,
    item_id: str,
    seed: int,
    prefix_sha256: str,
    recomputed_tokens: Sequence[int],
    completion_text: str,
) -> dict[str, Any]:
    return {
        "kind": CANARY,
        "arena_id": arena_id,
        "epoch": int(epoch),
        "miner_hotkey": miner_hotkey,
        "profile": profile,
        "weights_sha256": weights_sha256,
        "item_id": item_id,
        "call_seq": 0,
        "seed": int(seed),
        "prompt_sha256": prefix_sha256,
        "prefix_sha256": prefix_sha256,
        "claimed_tokens": [],
        "recomputed_tokens": [int(t) for t in recomputed_tokens],
        "completion_text": completion_text[:COMPLETION_LIMIT],
        "first_divergence": 0,
    }


def post(client: Any, body: Mapping[str, Any]) -> str:
    found = client_method(client, "post_evidence")(dict(body))
    return str(found.get("id") or found.get("evidence_id") or "")


def fetch(client: Any, arena_id: str, epoch: int) -> list[dict[str, Any]]:
    try:
        rows = client_method(client, "evidence")(arena_id, epoch)
    except Exception as exc:
        log.warning("evidence for epoch %d not read: %s", epoch, exc)
        return []
    return [dict(r) for r in rows if isinstance(r, Mapping)]


def confirmation(
    *, confirmed: bool, recomputed_tokens: Sequence[int], wallet: Any, hotkey: str
) -> dict[str, Any]:
    return sign(
        {"confirmed": bool(confirmed), "recomputed_tokens": [int(t) for t in recomputed_tokens]},
        wallet,
        hotkey,
    )


def confirm(client: Any, evidence_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
    return dict(client_method(client, "confirm_evidence")(evidence_id, dict(body)))


def evidence_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id") or row.get("evidence_id") or "")


def confirmers(row: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()
    listed = row.get("confirmations")
    if isinstance(listed, list):
        for entry in listed:
            if isinstance(entry, Mapping) and entry.get("confirmed", True):
                found.add(str(entry.get("validator_hotkey") or entry.get("validator") or ""))
            elif isinstance(entry, str):
                found.add(entry)
    found.discard("")
    return found


def confirmation_count(row: Mapping[str, Any]) -> int:
    listed = row.get("confirmations")
    if isinstance(listed, int) and not isinstance(listed, bool):
        return listed
    poster = str(row.get("validator_hotkey", ""))
    return len(confirmers(row) | ({poster} if poster else set()))


def is_confirmed(row: Mapping[str, Any], required: int) -> bool:
    if row.get("confirmed") is True:
        return True
    return confirmation_count(row) >= max(1, required)


def confirmed_by_miner(rows: Iterable[Mapping[str, Any]], required: int) -> dict[str, str]:
    found: dict[str, str] = {}
    for row in rows:
        if str(row.get("kind", "")) not in KINDS or not verify(row):
            continue
        if is_confirmed(row, required):
            found.setdefault(str(row.get("miner_hotkey", "")), evidence_id(row))
    found.pop("", None)
    return found


def already_confirmed(row: Mapping[str, Any], hotkey: str) -> bool:
    return hotkey in confirmers(row) or str(row.get("validator_hotkey", "")) == hotkey
