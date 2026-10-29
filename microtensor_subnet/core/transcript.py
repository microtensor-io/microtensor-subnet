from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from microtensor_subnet.chain.wallet import sign_bytes, verify_bytes

VERSION: Final[str] = "t1"
FLOAT_DECIMALS: Final[int] = 6
SIGNATURE_KEY: Final[str] = "miner_sig"
SLM: Final[str] = "slm"
TOOL: Final[str] = "tool"
RELAY: Final[str] = "relay"
ROUTER: Final[str] = "router"
KINDS: Final[tuple[str, ...]] = (SLM, TOOL, RELAY, ROUTER)
MINER: Final[str] = "miner"
REPAIR: Final[str] = "repair"
BARE: Final[str] = "B0"
SYSTEM_PREFIX: Final[str] = "system@"
IMAGE_KINDS: Final[tuple[str, ...]] = ("engine", "runner", "sandbox")
REQUIRED: Final[tuple[str, ...]] = (
    "version",
    "arena_id",
    "epoch",
    "miner_hotkey",
    "profile",
    "commitment_sha256",
    "manifest_sha256",
    "weights_sha256",
    "images",
    "items",
)
ITEM_REQUIRED: Final[tuple[str, ...]] = ("item_id", "pool", "condition", "seed", "calls", "output")
SLM_FIELDS: Final[tuple[str, ...]] = (
    "prompt_sha256",
    "prompt_tokens",
    "max_tokens",
    "stop",
    "seed",
    "output_tokens",
    "output_text",
    "logprobs",
    "finish_reason",
)


class TranscriptError(ValueError):
    pass


def round_floats(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, FLOAT_DECIMALS)
    if isinstance(value, Mapping):
        return {str(k): round_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(v) for v in value]
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        round_floats(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_hex(canonical_json(value))


def canonical_equal(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def prompt_sha256(prompt: str) -> str:
    return sha256_hex(prompt.encode("utf-8"))


def commitment_sha256(raw: str) -> str:
    return sha256_hex(raw.encode("utf-8"))


def item_seed(epoch_seed: str, item_id: str) -> int:
    digest = hashlib.sha256(f"{epoch_seed}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def reference_condition(index: int) -> str:
    return f"B{index}"


def system_condition(floor: float) -> str:
    return f"{SYSTEM_PREFIX}{floor}"


def floor_of(condition: str) -> float | None:
    if not condition.startswith(SYSTEM_PREFIX):
        return None
    try:
        return float(condition[len(SYSTEM_PREFIX) :])
    except ValueError:
        return None


def slm_call(
    seq: int,
    *,
    prompt: str,
    prompt_tokens: int,
    max_tokens: int,
    stop: Sequence[str],
    seed: int,
    output_tokens: Sequence[int],
    output_text: str,
    logprobs: Sequence[float] = (),
    finish_reason: str = "",
    ttft_ms: float = 0.0,
    total_ms: float = 0.0,
    fault: str = "",
) -> dict[str, Any]:
    return {
        "seq": seq,
        "kind": SLM,
        "prompt_sha256": prompt_sha256(prompt),
        "prompt_tokens": int(prompt_tokens),
        "max_tokens": int(max_tokens),
        "stop": [str(s) for s in stop],
        "seed": int(seed),
        "output_tokens": [int(t) for t in output_tokens],
        "output_text": output_text,
        "logprobs": [float(p) for p in logprobs],
        "finish_reason": finish_reason,
        "ttft_ms": float(ttft_ms),
        "total_ms": float(total_ms),
        "fault": fault,
    }


def tool_call(
    seq: int, *, route: str, request: Mapping[str, Any], response: Mapping[str, Any], fault: str
) -> dict[str, Any]:
    return {
        "seq": seq,
        "kind": TOOL,
        "route": route,
        "request": dict(request),
        "response": dict(response),
        "fault": fault,
    }


def router_call(
    seq: int, *, route: str, request: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "seq": seq,
        "kind": ROUTER,
        "route": route,
        "request": dict(request),
        "response": dict(response),
    }


def relay_call(
    seq: int,
    *,
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    receipt: Mapping[str, Any],
    status: int = 200,
) -> dict[str, Any]:
    return {
        "seq": seq,
        "kind": RELAY,
        "request_sha256": canonical_sha256(request),
        "response": dict(response),
        "receipt": dict(receipt),
        "status": int(status),
    }


def item_entry(
    *,
    item_id: str,
    pool: str,
    condition: str,
    seed: int,
    calls: Sequence[Mapping[str, Any]],
    output: Any,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "item_id": item_id,
        "pool": pool,
        "condition": condition,
        "seed": int(seed),
        "calls": [dict(c) for c in calls],
        "output": output,
    }
    if request is not None:
        entry["request"] = dict(request)
    return entry


@dataclass(slots=True)
class Transcript:
    arena_id: str
    epoch: int
    miner_hotkey: str
    profile: str
    commitment_sha256: str
    manifest_sha256: str
    weights_sha256: str
    images: dict[str, str]
    items: list[dict[str, Any]] = field(default_factory=list)
    miner_sig: str = ""

    def body(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "arena_id": self.arena_id,
            "epoch": int(self.epoch),
            "miner_hotkey": self.miner_hotkey,
            "profile": self.profile,
            "commitment_sha256": self.commitment_sha256,
            "manifest_sha256": self.manifest_sha256,
            "weights_sha256": self.weights_sha256,
            "images": dict(self.images),
            "items": [dict(i) for i in self.items],
        }

    def to_dict(self) -> dict[str, Any]:
        found = self.body()
        found[SIGNATURE_KEY] = self.miner_sig
        return found

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Transcript:
        problems = validate(raw)
        if problems:
            raise TranscriptError(problems[0])
        return cls(
            arena_id=str(raw["arena_id"]),
            epoch=int(raw["epoch"]),
            miner_hotkey=str(raw["miner_hotkey"]),
            profile=str(raw["profile"]),
            commitment_sha256=str(raw["commitment_sha256"]),
            manifest_sha256=str(raw["manifest_sha256"]),
            weights_sha256=str(raw["weights_sha256"]),
            images={str(k): str(v) for k, v in dict(raw["images"]).items()},
            items=[dict(i) for i in raw["items"]],
            miner_sig=str(raw.get(SIGNATURE_KEY, "")),
        )


def validate(raw: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    for key in REQUIRED:
        if key not in raw:
            problems.append(f"transcript lacks {key}")
    if problems:
        return problems
    if raw.get("version") != VERSION:
        problems.append(f"transcript version {raw.get('version')!r} is not {VERSION}")
    if not isinstance(raw.get("images"), Mapping):
        problems.append("images must map image kinds to digests")
    items = raw.get("items")
    if not isinstance(items, list):
        return [*problems, "items must be a list"]
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            problems.append(f"item {index} is not an object")
            continue
        missing = [key for key in ITEM_REQUIRED if key not in item]
        if missing:
            problems.append(f"item {item.get('item_id', index)} lacks {', '.join(missing)}")
            continue
        calls = item.get("calls")
        if not isinstance(calls, list):
            problems.append(f"item {item['item_id']} calls must be a list")
            continue
        for seq, call in enumerate(calls):
            if not isinstance(call, Mapping):
                problems.append(f"item {item['item_id']} call {seq} is not an object")
            elif call.get("seq") != seq:
                problems.append(f"item {item['item_id']} call {seq} is out of sequence")
            elif call.get("kind") not in KINDS:
                problems.append(f"item {item['item_id']} call {seq} has kind {call.get('kind')!r}")
            elif call["kind"] == SLM and any(key not in call for key in SLM_FIELDS):
                problems.append(f"item {item['item_id']} call {seq} is an incomplete slm call")
    return problems


def unsigned(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in raw.items() if k != SIGNATURE_KEY}


def signing_digest(raw: Mapping[str, Any]) -> bytes:
    return hashlib.sha256(canonical_json(unsigned(raw))).digest()


def sign(raw: Mapping[str, Any], wallet: Any) -> dict[str, Any]:
    found = dict(unsigned(raw))
    found[SIGNATURE_KEY] = sign_bytes(wallet, signing_digest(found))
    return found


def verify_signature(raw: Mapping[str, Any]) -> bool:
    signature = str(raw.get(SIGNATURE_KEY, "") or "")
    hotkey = str(raw.get("miner_hotkey", "") or "")
    if not signature or not hotkey:
        return False
    return verify_bytes(hotkey, signing_digest(raw), signature)


def transcript_sha256(raw: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(raw))


def encode(raw: Mapping[str, Any]) -> bytes:
    return canonical_json(raw)


def decode(data: bytes) -> dict[str, Any]:
    try:
        found = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TranscriptError(f"the transcript is not valid json: {exc}") from exc
    if not isinstance(found, dict):
        raise TranscriptError("the transcript must be a json object")
    return found


def leaf(hotkey: str, transcript_sha: str) -> bytes:
    return hashlib.sha256((hotkey + transcript_sha).encode("utf-8")).digest()


def leaf_hex(hotkey: str, transcript_sha: str) -> str:
    return leaf(hotkey, transcript_sha).hex()


def _parent(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(left + right).digest()


def _levels(leaves: Sequence[bytes]) -> list[list[bytes]]:
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        current = levels[-1]
        above: list[bytes] = []
        for index in range(0, len(current), 2):
            left = current[index]
            right = current[index + 1] if index + 1 < len(current) else left
            above.append(_parent(left, right))
        levels.append(above)
    return levels


def ordered_leaves(leaves: Mapping[str, str]) -> list[tuple[str, bytes]]:
    return [(hotkey, leaf(hotkey, leaves[hotkey])) for hotkey in sorted(leaves)]


def merkle_root(leaves: Mapping[str, str]) -> str:
    rows = ordered_leaves(leaves)
    if not rows:
        return sha256_hex(b"")
    return _levels([digest for _, digest in rows])[-1][0].hex()


def merkle_proof(leaves: Mapping[str, str], hotkey: str) -> list[dict[str, str]]:
    rows = ordered_leaves(leaves)
    names = [name for name, _ in rows]
    if hotkey not in names:
        raise TranscriptError(f"{hotkey} has no leaf")
    position = names.index(hotkey)
    proof: list[dict[str, str]] = []
    for level in _levels([digest for _, digest in rows])[:-1]:
        sibling = position ^ 1
        partner = level[sibling] if sibling < len(level) else level[position]
        proof.append({"side": "left" if sibling < position else "right", "hash": partner.hex()})
        position //= 2
    return proof


def check_proof(leaf_value: str, proof: Sequence[Mapping[str, str]], root: str) -> bool:
    try:
        current = bytes.fromhex(leaf_value)
        for step in proof:
            partner = bytes.fromhex(str(step["hash"]))
            if step.get("side") == "left":
                current = _parent(partner, current)
            else:
                current = _parent(current, partner)
    except (ValueError, KeyError, TypeError):
        return False
    return current.hex() == root.lower()


def verify_leaf(hotkey: str, transcript_sha: str, leaves: Mapping[str, str], root: str) -> bool:
    if leaves.get(hotkey) != transcript_sha:
        return False
    return merkle_root(leaves) == root.lower()


def usage_by_condition(raw: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    found: dict[str, dict[str, int]] = {}
    for item in raw.get("items") or []:
        bucket = found.setdefault(str(item.get("condition", "")), {"in": 0, "out": 0, "calls": 0})
        for call in item.get("calls") or []:
            if call.get("kind") != SLM:
                continue
            bucket["calls"] += 1
            bucket["in"] += int(call.get("prompt_tokens") or 0)
            bucket["out"] += len(call.get("output_tokens") or [])
    return found
