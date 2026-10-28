from __future__ import annotations

import functools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from microtensor_subnet.artifact import gguf
from microtensor_subnet.artifact.params import safetensors_in
from microtensor_subnet.validator import collect, provenance, score
from microtensor_subnet.validator.context import Artifact, EpochContext
from microtensor_subnet.validator.results import SlmRun

PROVENANCE: Final[str] = "provenance"
NOT_A_COPY: Final[str] = "not_a_copy"
ENVELOPE: Final[str] = "envelope"
SPEED: Final[str] = "speed"
MEMORISATION: Final[str] = "memorisation"
LICENCE: Final[str] = "licence"
FORMAT: Final[str] = "format"
VALUES: Final[dict[str, frozenset[str]]] = {
    PROVENANCE: frozenset({"similarity", "threshold", "declared_base"}),
    NOT_A_COPY: frozenset(),
    ENVELOPE: frozenset(
        {
            "group",
            "vram_peak_bytes",
            "memory_cap_bytes",
            "parameters",
            "parameters_min",
            "parameters_max",
        }
    ),
    SPEED: frozenset(
        {
            "ttft_p95_ms",
            "tpot_p95_ms",
            "ttft_budget_ms",
            "tpot_budget_ms",
            "violation_share",
            "violation_limit",
        }
    ),
    MEMORISATION: frozenset({"twin_rate", "main_rate", "twin_gap"}),
    LICENCE: frozenset({"licence", "licence_name"}),
    FORMAT: frozenset({"format"}),
}
ORDER: Final[tuple[str, ...]] = tuple(VALUES)
TEXT_LIMIT: Final[int] = 200
DIGITS: Final[int] = 6
DEFAULT_MEMORY_BYTES: Final[int] = 24 * 1024**3
DEFAULT_MAX_PARAMETERS: Final[int] = 8_000_000_000


@dataclass(frozen=True, slots=True)
class Envelope:
    group: str
    low: int
    high: int
    memory_cap: int
    ttft_ms: float
    tpot_ms: float

    @classmethod
    def of(cls, context: EpochContext) -> Envelope:
        budgets = context.budgets
        block = context.definition.get("envelope")
        return cls(
            group=str(block.get("group") or "") if isinstance(block, Mapping) else "",
            low=int(budgets.get("min_parameters") or 0),
            high=int(budgets.get("max_parameters") or DEFAULT_MAX_PARAMETERS),
            memory_cap=int(
                budgets.get("memory_bytes") or budgets.get("vram_bytes") or DEFAULT_MEMORY_BYTES
            ),
            ttft_ms=float(budgets.get("ttft_ms") or 0),
            tpot_ms=float(budgets.get("tpot_ms") or 0),
        )

    def sized(self, parameters: int) -> bool:
        return parameters > 0 and self.low <= parameters <= self.high

    def fits(self, run: SlmRun) -> bool:
        return run.vram_peak_bytes <= self.memory_cap

    def fast(self, run: SlmRun) -> bool:
        return (not self.ttft_ms or run.ttft_p95_ms <= self.ttft_ms) and (
            not self.tpot_ms or run.tpot_p95_ms <= self.tpot_ms
        )


def gates(
    context: EpochContext, artifact: Artifact, run: SlmRun, rules: score.Rules
) -> tuple[bool, bool]:
    envelope = Envelope.of(context)
    sized = envelope.sized(artifact.slm_parameters)
    if rules.per_item_envelope:
        over = score.envelope_share(run, context.budgets)
        return sized, envelope.fits(run) and over <= rules.violation_share
    return sized and envelope.fits(run) and envelope.fast(run), True


def clean(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:TEXT_LIMIT]
    return None


def row(name: str, passed: bool | None, **values: Any) -> dict[str, Any]:
    allowed = VALUES.get(name, frozenset())
    kept = {k: clean(v) for k, v in values.items() if k in allowed}
    return {
        "name": name,
        "passed": passed,
        "values": {k: v for k, v in kept.items() if v is not None},
    }


def tidy(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for item in rows:
        name = str(item.get("name") or "")
        if name not in VALUES:
            continue
        passed = item.get("passed")
        values = item.get("values")
        found[name] = row(
            name,
            passed if isinstance(passed, bool) else None,
            **(dict(values) if isinstance(values, Mapping) else {}),
        )
    return [found[name] for name in ORDER if name in found]


def _quant_method(slm_dir: Path) -> str:
    try:
        config = json.loads((slm_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    block = config.get("quantization_config") if isinstance(config, Mapping) else None
    return str(block.get("quant_method") or "").lower() if isinstance(block, Mapping) else ""


def weight_format(slm_dir: Path) -> str:
    if safetensors_in(slm_dir):
        method = _quant_method(slm_dir)
        return f"safetensors {method}" if method else "safetensors"
    files = gguf.gguf_in(slm_dir)
    if not files:
        return ""
    try:
        kind = gguf.read(files[0]).file_type
    except Exception:
        kind = ""
    return f"gguf {kind}".strip()


@functools.lru_cache(maxsize=64)
def labels(slm_dir: Path) -> tuple[str, str, str]:
    return (
        collect.card_value(slm_dir, "license"),
        collect.card_value(slm_dir, "license_name"),
        weight_format(slm_dir),
    )


def _slm_rows(context: EpochContext, artifact: Artifact, run: SlmRun) -> list[dict[str, Any]]:
    rules = score.Rules.from_definition(context.definition)
    envelope = Envelope.of(context)
    found = [
        row(
            ENVELOPE,
            envelope.sized(artifact.slm_parameters) and envelope.fits(run),
            group=envelope.group,
            vram_peak_bytes=run.vram_peak_bytes,
            memory_cap_bytes=envelope.memory_cap,
            parameters=artifact.slm_parameters,
            parameters_min=envelope.low,
            parameters_max=envelope.high,
        )
    ]
    speed: dict[str, Any] = {
        "ttft_p95_ms": round(run.ttft_p95_ms, 1),
        "tpot_p95_ms": round(run.tpot_p95_ms, 2),
        "ttft_budget_ms": envelope.ttft_ms or None,
        "tpot_budget_ms": envelope.tpot_ms or None,
    }
    if rules.per_item_envelope:
        share = score.envelope_share(run, context.budgets)
        speed["violation_share"] = round(share, DIGITS)
        speed["violation_limit"] = rules.violation_share
        fast = share <= rules.violation_share
    else:
        fast = envelope.fast(run)
    found.append(row(SPEED, fast, **speed))
    gap, _ = score.twin_factor(run, rules)
    found.append(
        row(
            MEMORISATION,
            gap <= 0 if run.twins else None,
            twin_rate=round(run.twin_rate, DIGITS) if run.twins else None,
            main_rate=round(run.pass_rate, DIGITS),
            twin_gap=round(gap, DIGITS) if run.twins else None,
        )
    )
    return found


def build(
    context: EpochContext,
    artifact: Artifact,
    *,
    claim: str | None = None,
    found: Mapping[str, Any] | None = None,
    run: SlmRun | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if found:
        rows.append(
            row(
                PROVENANCE,
                bool(found.get("passed")),
                similarity=found.get("similarity"),
                threshold=provenance.DEFAULT_THRESHOLD,
                declared_base=found.get("declared_base"),
            )
        )
    if claim is not None:
        rows.append(row(NOT_A_COPY, not claim))
    if artifact.slm_dir is None:
        return tidy(rows)
    if run is not None:
        rows += _slm_rows(context, artifact, run)
    licence, licence_name, kind = labels(artifact.slm_dir)
    rows.append(row(LICENCE, True, licence=licence, licence_name=licence_name))
    rows.append(row(FORMAT, True, format=kind))
    return tidy(rows)
