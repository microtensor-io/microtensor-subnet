from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from microtensor_subnet.chain.wallet import verify_bytes
from microtensor_subnet.validator import items, transcripts

log = logging.getLogger("microtensor_subnet.validator.spotcheck")

SPOTCHECK_KIND: Final[str] = "spotcheck"
APPEALS_KIND: Final[str] = "appeals"
SIGNATURE: Final[str] = "signature"
SPOT_CHECK: Final[str] = "spot_check"
CANARY: Final[str] = "canary"
FLAG_KINDS: Final[tuple[str, ...]] = (SPOT_CHECK, CANARY)
DEFAULT_MARGIN_BLOCKS: Final[int] = 150
DEFAULT_POLL_SECONDS: Final[int] = 60
DEFAULT_BAN_EPOCHS: Final[int] = 3
FAILURE_CHARS: Final[int] = 300
TIMING_KEYS: Final[frozenset[str]] = frozenset(
    {"ttft_p95_ms", "tpot_p95_ms", "vram_peak_bytes", "drafts_checked", "drafts_mismatched"}
)
DRAFTS_KEY: Final[str] = "drafts"
DRAFTS_DONE_KEY: Final[str] = "drafts_complete"
ROWS_KEY: Final[str] = "rows"
ROWS_DONE_KEY: Final[str] = "rows_complete"
NO_TOKENS: Final[str] = hashlib.sha256(b"[]").hexdigest()


class SpotCheckError(RuntimeError):
    pass


def _float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int(row: Mapping[str, Any], key: str) -> int:
    try:
        return int(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0


@dataclass(slots=True)
class SpotCheck:
    calls: int = 0
    matched: int = 0
    mismatched: int = 0
    ttft_p95_ms: float = 0.0
    tpot_p95_ms: float = 0.0
    vram_peak_bytes: int = 0
    reference_ttft_p95_ms: float = 0.0
    reference_tpot_p95_ms: float = 0.0
    reference_vram_peak_bytes: int = 0
    uniform_calls: int = 0
    over_budget: int = 0
    envelope_lb: float = 0.0
    envelope_ok: bool = True
    timing: dict[tuple[str, int], tuple[float, float]] = field(default_factory=dict)
    unsettled: int = 0
    errors: int = 0
    skipped_prompt: int = 0
    e2e_p95_ms: float = 0.0
    memory_ok: bool = True
    router_latency_p95_ms: float | None = None
    router_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "matched": self.matched,
            "mismatched": self.mismatched,
            "ttft_p95_ms": round(self.ttft_p95_ms, 1),
            "tpot_p95_ms": round(self.tpot_p95_ms, 2),
            "vram_peak_bytes": self.vram_peak_bytes,
            "reference_ttft_p95_ms": round(self.reference_ttft_p95_ms, 1),
            "reference_tpot_p95_ms": round(self.reference_tpot_p95_ms, 2),
            "reference_vram_peak_bytes": self.reference_vram_peak_bytes,
            "uniform_calls": self.uniform_calls,
            "over_budget": self.over_budget,
            "envelope_lb": round(self.envelope_lb, 6),
            "envelope_ok": self.envelope_ok,
            "unsettled": self.unsettled,
            "errors": self.errors,
            "skipped_prompt": self.skipped_prompt,
            "e2e_p95_ms": round(self.e2e_p95_ms, 1),
            "memory_ok": self.memory_ok,
            "router_latency_p95_ms": self.router_latency_p95_ms,
            "router_reason": self.router_reason,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> SpotCheck:
        timing: dict[tuple[str, int], tuple[float, float]] = {}
        for entry in row.get("timing") or []:
            if isinstance(entry, (list, tuple)) and len(entry) == 4:
                try:
                    timing[(str(entry[0]), int(entry[1]))] = (float(entry[2]), float(entry[3]))
                except (TypeError, ValueError):
                    continue
        latency = row.get("router_latency_p95_ms")
        return cls(
            calls=_int(row, "calls"),
            matched=_int(row, "matched"),
            mismatched=_int(row, "mismatched"),
            ttft_p95_ms=_float(row, "ttft_p95_ms"),
            tpot_p95_ms=_float(row, "tpot_p95_ms"),
            vram_peak_bytes=_int(row, "vram_peak_bytes"),
            uniform_calls=_int(row, "uniform_calls"),
            over_budget=_int(row, "over_budget"),
            envelope_lb=_float(row, "envelope_lb"),
            envelope_ok=bool(row.get("envelope_ok", True)),
            timing=timing,
            unsettled=_int(row, "unsettled"),
            errors=_int(row, "errors"),
            skipped_prompt=_int(row, "skipped_prompt"),
            e2e_p95_ms=_float(row, "e2e_p95_ms"),
            memory_ok=bool(row.get("memory_ok", True)),
            router_latency_p95_ms=_float(row, "router_latency_p95_ms")
            if latency is not None
            else None,
            router_reason=str(row.get("router_reason") or "")[:FAILURE_CHARS],
        )


@dataclass(frozen=True, slots=True)
class ReferenceTiming:
    ttft_p95_ms: float
    tpot_p95_ms: float
    vram_peak_bytes: int
    drafts_checked: int
    drafts_mismatched: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ttft_p95_ms": round(self.ttft_p95_ms, 1),
            "tpot_p95_ms": round(self.tpot_p95_ms, 2),
            "vram_peak_bytes": self.vram_peak_bytes,
            "drafts_checked": self.drafts_checked,
            "drafts_mismatched": self.drafts_mismatched,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ReferenceTiming:
        return cls(
            ttft_p95_ms=_float(row, "ttft_p95_ms"),
            tpot_p95_ms=_float(row, "tpot_p95_ms"),
            vram_peak_bytes=_int(row, "vram_peak_bytes"),
            drafts_checked=_int(row, "drafts_checked"),
            drafts_mismatched=_int(row, "drafts_mismatched"),
        )


@dataclass(frozen=True, slots=True)
class Flag:
    epoch_index: int
    miner_hotkey: str
    kind: str
    item_id: str
    call_seq: int

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.miner_hotkey, self.kind, self.item_id, self.call_seq)

    def label(self) -> str:
        return f"epoch {self.epoch_index} {self.kind} {self.item_id}#{self.call_seq}"


@dataclass(slots=True)
class Coordinated:
    epoch_index: int
    complete: bool = False
    reference: ReferenceTiming | None = None
    spot_checks: dict[str, SpotCheck] = field(default_factory=dict)
    canaries: dict[str, dict[str, int]] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    flags: dict[str, list[Flag]] = field(default_factory=dict)
    upheld: set[tuple[str, str, str, int]] = field(default_factory=set)
    coordinators: set[str] = field(default_factory=set)
    drafts: dict[str, str] = field(default_factory=dict)
    drafts_complete: bool = False
    rows: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    rows_complete: bool = False

    def standing(self, hotkey: str) -> list[Flag]:
        return [f for f in self.flags.get(hotkey, []) if f.key not in self.upheld]

    def covers(self, hotkey: str) -> bool:
        return hotkey in self.spot_checks or hotkey in self.failures


def report_message(body: Mapping[str, Any]) -> bytes:
    unsigned = {k: v for k, v in body.items() if k != SIGNATURE}
    return hashlib.sha256(transcripts.canonical_json(unsigned)).hexdigest().encode("ascii")


def verify_report(body: Mapping[str, Any]) -> bool:
    hotkey = str(body.get("coordinator_hotkey", ""))
    signature = str(body.get(SIGNATURE, ""))
    message = report_message(body)
    return verify_bytes(hotkey, message, signature) or verify_bytes(
        hotkey, b"<Bytes>" + message + b"</Bytes>", signature
    )


def stands(entry: Mapping[str, Any]) -> bool:
    if str(entry.get("kind", "")) == CANARY:
        return bool(entry.get("first_hit")) and bool(entry.get("second_hit"))
    first = str(entry.get("first_sha256") or "").lower()
    second = str(entry.get("second_sha256") or "").lower()
    claimed = str(entry.get("claimed_sha256") or "").lower()
    if not first or not claimed or first == NO_TOKENS:
        return False
    return first == second and first != claimed


def _take_reference(found: Coordinated, block: Mapping[str, Any]) -> None:
    if TIMING_KEYS & set(block):
        found.reference = ReferenceTiming.from_dict(block)
    for row in block.get(DRAFTS_KEY) or []:
        if isinstance(row, Mapping) and row.get("item_id"):
            found.drafts[str(row["item_id"])] = str(row.get("draft") or "")
    found.drafts_complete = found.drafts_complete or bool(block.get(DRAFTS_DONE_KEY))
    for row in block.get(ROWS_KEY) or []:
        if isinstance(row, Mapping) and row.get("item_id"):
            found.rows[(str(row.get("pool") or ""), str(row["item_id"]))] = dict(row)
    found.rows_complete = found.rows_complete or bool(block.get(ROWS_DONE_KEY))


def merge(
    epoch_index: int,
    reports: Iterable[Mapping[str, Any]],
    leaves: Mapping[str, str] | None = None,
) -> Coordinated:
    found = Coordinated(epoch_index=epoch_index)
    for body in reports:
        if not isinstance(body, Mapping) or not verify_report(body):
            log.warning("a coordinator report for epoch %d does not verify; skipped", epoch_index)
            continue
        if body.get("epoch_index") != epoch_index:
            continue
        found.coordinators.add(str(body.get("coordinator_hotkey", "")))
        found.complete = found.complete or bool(body.get("complete"))
        reference = body.get("reference")
        if isinstance(reference, Mapping):
            _take_reference(found, reference)
        for hotkey, row in (body.get("miners") or {}).items():
            if not isinstance(row, Mapping):
                continue
            hotkey = str(hotkey)
            if leaves is not None:
                leaf = str(leaves.get(hotkey) or "").lower()
                if not leaf or str(row.get("transcript_sha256") or "").lower() != leaf:
                    continue
            spot = row.get("spot_check")
            if isinstance(spot, Mapping):
                found.spot_checks[hotkey] = SpotCheck.from_dict(spot)
                found.failures.pop(hotkey, None)
            elif row.get("failure") and hotkey not in found.spot_checks:
                found.failures[hotkey] = str(row.get("failure"))[:FAILURE_CHARS]
            canary = row.get("canary")
            if isinstance(canary, Mapping):
                found.canaries[hotkey] = {
                    "checked": _int(canary, "checked"),
                    "hits": _int(canary, "hits"),
                }
            flags = found.flags.setdefault(hotkey, [])
            seen = {flag.key for flag in flags}
            for entry in row.get("evidence") or []:
                if not isinstance(entry, Mapping):
                    continue
                if str(entry.get("kind", "")) not in FLAG_KINDS or not stands(entry):
                    continue
                flag = Flag(
                    epoch_index,
                    hotkey,
                    str(entry.get("kind", "")),
                    str(entry.get("item_id", "")),
                    _int(entry, "call_seq"),
                )
                if flag.key not in seen:
                    seen.add(flag.key)
                    flags.append(flag)
        for outcome in body.get("appeals") or []:
            if isinstance(outcome, Mapping) and bool(outcome.get("upheld")):
                found.upheld.add(
                    (
                        str(outcome.get("miner_hotkey", "")),
                        str(outcome.get("kind", "")),
                        str(outcome.get("item_id", "")),
                        _int(outcome, "call_seq"),
                    )
                )
    return found


def receipt_leaves(client: Any, arena_id: str, epoch_index: int, key: str) -> dict[str, str]:
    try:
        document = client.receipts(arena_id, epoch_index)
        if not document:
            return {}
        items.verify_document(document, key)
    except Exception as exc:
        log.warning("the receipts of epoch %d were not read: %s", epoch_index, exc)
        return {}
    leaves = dict(document.get("body") or {}).get("leaves") or {}
    return {str(h): str(s).lower() for h, s in dict(leaves).items() if h and s}


def fetch(
    client: Any,
    arena_id: str,
    epoch_index: int,
    key: str,
    leaves: Mapping[str, str] | None = None,
) -> Coordinated | None:
    try:
        document = client.document(arena_id, epoch_index, SPOTCHECK_KIND)
    except Exception as exc:
        log.debug("coordinator document for epoch %d not read: %s", epoch_index, exc)
        return None
    if not document:
        return None
    try:
        items.verify_document(document, key)
    except Exception as exc:
        log.warning("the coordinator document for epoch %d does not verify: %s", epoch_index, exc)
        return None
    body = dict(document.get("body") or {})
    held = receipt_leaves(client, arena_id, epoch_index, key) if leaves is None else leaves
    return merge(epoch_index, body.get("reports") or [], held)


def ban_epochs(definition: Mapping[str, Any]) -> int:
    block = definition.get("evaluation") or {}
    ban = block.get("ban") if isinstance(block, Mapping) else None
    found = ban.get("epochs") if isinstance(ban, Mapping) else None
    return max(1, int(found or DEFAULT_BAN_EPOCHS))
