from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

REVISION: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")


class EscalationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EscalationModel:
    model: str
    revision: str
    usd_per_mtok_in: float
    usd_per_mtok_out: float

    def __post_init__(self) -> None:
        if "/" not in self.model:
            raise EscalationError(f"escalation model {self.model!r} is not an org/name repository")
        if not REVISION.match(self.revision):
            raise EscalationError(
                f"escalation model {self.model!r} must be pinned to a full commit revision"
            )
        if self.usd_per_mtok_in < 0 or self.usd_per_mtok_out < 0:
            raise EscalationError(f"escalation model {self.model!r} has a negative price")

    @property
    def key(self) -> str:
        return f"{self.model}@{self.revision}"

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.usd_per_mtok_in + completion_tokens * self.usd_per_mtok_out
        ) / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "revision": self.revision,
            "usd_per_mtok_in": self.usd_per_mtok_in,
            "usd_per_mtok_out": self.usd_per_mtok_out,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EscalationModel:
        try:
            return cls(
                model=str(raw["model"]),
                revision=str(raw["revision"]),
                usd_per_mtok_in=float(raw["usd_per_mtok_in"]),
                usd_per_mtok_out=float(raw["usd_per_mtok_out"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EscalationError(f"escalation allowlist entry is malformed: {exc}") from exc


def allowlist(entries: Iterable[Any]) -> dict[str, EscalationModel]:
    found: dict[str, EscalationModel] = {}
    for raw in entries:
        if isinstance(raw, Mapping):
            model = EscalationModel.from_dict(raw)
            found[model.key] = model
    return found
