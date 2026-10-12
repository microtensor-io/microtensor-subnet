from __future__ import annotations

import secrets
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import httpx

TIMEOUT_SECONDS: Final[float] = 180.0
REPAIR: Final[str] = "repair"
SYSTEM: Final[str] = "system"
REFERENCE: Final[str] = "reference"


class RelayError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code


@dataclass(frozen=True, slots=True)
class Receipt:
    receipt: str
    ticket: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_micros: int
    latency_ms: int
    status: str
    signature: str
    body: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Receipt:
        return cls(
            receipt=str(raw.get("receipt", "")),
            ticket=str(raw.get("ticket", "")),
            model=str(raw.get("model", "")),
            tokens_in=int(raw.get("tokens_in") or 0),
            tokens_out=int(raw.get("tokens_out") or 0),
            cost_micros=int(raw.get("cost_micros") or 0),
            latency_ms=int(raw.get("latency_ms") or 0),
            status=str(raw.get("status", "")),
            signature=str(raw.get("signature", "")),
            body=dict(raw),
        )


@dataclass(frozen=True, slots=True)
class RelayAnswer:
    text: str
    receipt: Receipt
    finish_reason: str


@dataclass(slots=True)
class RelayClient:
    base_url: str
    token: str
    timeout: float = TIMEOUT_SECONDS

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            with httpx.Client(base_url=self.base_url.rstrip("/"), timeout=self.timeout) as http:
                response = http.post(
                    path, json=body, headers={"authorization": f"Bearer {self.token}"}
                )
        except httpx.HTTPError as exc:
            raise RelayError("unreachable", f"the relay could not be reached: {exc}") from exc
        if response.status_code >= 400:
            detail: Any = {}
            try:
                detail = response.json().get("detail", {})
            except ValueError:
                detail = {}
            code = str(detail.get("code", "refused")) if isinstance(detail, dict) else "refused"
            message = (
                str(detail.get("message", "")) if isinstance(detail, dict) else response.text[:200]
            )
            raise RelayError(code, message or f"the relay answered {response.status_code}")
        return dict(response.json())

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            with httpx.Client(base_url=self.base_url.rstrip("/"), timeout=self.timeout) as http:
                response = http.get(
                    path, params=params, headers={"authorization": f"Bearer {self.token}"}
                )
        except httpx.HTTPError as exc:
            raise RelayError("unreachable", f"the relay could not be reached: {exc}") from exc
        if response.status_code >= 400:
            raise RelayError("refused", f"the relay answered {response.status_code}")
        return dict(response.json())

    def catalog(self, arena_id: str) -> list[dict[str, Any]]:
        return list(self._get(f"/v1/relay/catalog/{arena_id}", {}).get("models") or [])

    def ticket(
        self,
        *,
        arena_id: str,
        epoch_index: int,
        miner: str,
        item_id: str,
        purpose: str,
        cap_micros: int,
    ) -> str:
        found = self._post(
            "/v1/relay/tickets",
            {
                "arena_id": arena_id,
                "epoch_index": epoch_index,
                "miner": miner,
                "item_id": item_id,
                "purpose": purpose,
                "cap_micros": cap_micros,
            },
        )
        return str(found["ticket"])

    def complete(
        self,
        *,
        ticket: str,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int | None = None,
    ) -> RelayAnswer:
        body: dict[str, Any] = {"ticket": ticket, "model": model, "messages": list(messages)}
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        found = self._post("/v1/relay/chat/completions", body)
        choices = found.get("choices") or [{}]
        return RelayAnswer(
            text=str((choices[0].get("message") or {}).get("content") or ""),
            receipt=Receipt.from_dict(found.get("receipt") or {}),
            finish_reason=str(choices[0].get("finish_reason") or ""),
        )

    def receipts(self, arena_id: str, epoch_index: int) -> list[Receipt]:
        found = self._get("/v1/relay/receipts", {"arena_id": arena_id, "epoch_index": epoch_index})
        return [Receipt.from_dict(r) for r in found.get("receipts") or []]


@dataclass(slots=True)
class TicketDesk:
    relay: RelayClient
    arena_id: str
    epoch_index: int
    _opaque: dict[str, str] = field(default_factory=dict)
    _spent: dict[str, list[Receipt]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def issue(self, *, miner: str, item_id: str, purpose: str, cap_micros: int) -> str:
        real = self.relay.ticket(
            arena_id=self.arena_id,
            epoch_index=self.epoch_index,
            miner=miner,
            item_id=item_id,
            purpose=purpose,
            cap_micros=cap_micros,
        )
        opaque = "mtk_" + secrets.token_urlsafe(18)
        with self._lock:
            self._opaque[opaque] = real
            self._spent[opaque] = []
        return opaque

    def complete(
        self,
        opaque: str,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int | None,
    ) -> RelayAnswer:
        with self._lock:
            real = self._opaque.get(opaque)
        if real is None:
            raise RelayError("refused", "unknown ticket")
        answer = self.relay.complete(
            ticket=real, model=model, messages=messages, max_tokens=max_tokens
        )
        with self._lock:
            self._spent.setdefault(opaque, []).append(answer.receipt)
        return answer

    def receipts_for(self, opaque: str) -> list[Receipt]:
        with self._lock:
            return list(self._spent.get(opaque, []))

    def cost_for(self, opaque: str) -> int:
        return sum(r.cost_micros for r in self.receipts_for(opaque))

    def forget(self, opaque: str) -> None:
        with self._lock:
            self._opaque.pop(opaque, None)
