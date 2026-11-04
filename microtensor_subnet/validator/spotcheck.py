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
    first = [int(t) for t in entry.get("first_tokens") or []]
    second = [int(t) for t in entry.get("second_tokens") or []]
    if str(entry.get("kind", "")) == CANARY:
        return bool(entry.get("first_hit")) and bool(entry.get("second_hit"))
    claimed = [int(t) for t in entry.get("claimed_tokens") or []]
    return bool(first) and first == second and first != claimed


def merge(epoch_index: int, reports: Iterable[Mapping[str, Any]]) -> Coordinated:
    found = Coordinated(epoch_index=epoch_index)
    for body in reports:
        if not isinstance(body, Mapping) or not verify_report(body):
            log.warning("a coordinator report for epoch %d does not verify; skipped", epoch_index)
            continue
        if _int(body, "epoch_index") != epoch_index:
            continue
        found.coordinators.add(str(body.get("coordinator_hotkey", "")))
        found.complete = found.complete or bool(body.get("complete"))
        reference = body.get("reference")
        if isinstance(reference, Mapping):
            found.reference = ReferenceTiming.from_dict(reference)
        for hotkey, row in (body.get("miners") or {}).items():
            if not isinstance(row, Mapping):
                continue
            hotkey = str(hotkey)
            spot = row.get("spot_check")
            if isinstance(spot, Mapping):
                found.spot_checks[hotkey] = SpotCheck.from_dict(spot)
                found.failures.pop(hotkey, None)
            elif row.get("failure"):
                found.failures[hotkey] = str(row.get("failure"))[:300]
            canary = row.get("canary")
            if isinstance(canary, Mapping):
                found.canaries[hotkey] = {
                    "checked": _int(canary, "checked"),
                    "hits": _int(canary, "hits"),
                }
            flags = [
                Flag(
                    epoch_index,
                    hotkey,
                    str(entry.get("kind", "")),
                    str(entry.get("item_id", "")),
                    _int(entry, "call_seq"),
                )
                for entry in row.get("evidence") or []
                if isinstance(entry, Mapping)
                and str(entry.get("kind", "")) in FLAG_KINDS
                and stands(entry)
            ]
            found.flags[hotkey] = flags
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


def fetch(client: Any, arena_id: str, epoch_index: int, key: str) -> Coordinated | None:
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
    return merge(epoch_index, body.get("reports") or [])


def ban_epochs(definition: Mapping[str, Any]) -> int:
    block = definition.get("evaluation") or {}
    ban = block.get("ban") if isinstance(block, Mapping) else None
    found = ban.get("epochs") if isinstance(ban, Mapping) else None
    return max(1, int(found or DEFAULT_BAN_EPOCHS))
