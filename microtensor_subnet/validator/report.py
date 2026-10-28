from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from microtensor_subnet.chain.wallet import sign_bytes, sign_payload
from microtensor_subnet.core.constants import MECHANISM_VERSION
from microtensor_subnet.core.reports import EpochReport, MinerReport, PoolScore
from microtensor_subnet.scoring import pools
from microtensor_subnet.validator import checks as checklist
from microtensor_subnet.validator.context import Artifact, EpochContext
from microtensor_subnet.validator.items import item_ids_digest

ARTIFACT_TAG: Final[str] = "mt-artifact-result"
ARTIFACT_FAILURE_LIMIT: Final[int] = 400
LIVE_LIMIT_BYTES: Final[int] = 15_000
DETAIL_KEYS: Final[dict[str, frozenset[str]]] = {
    pools.SLM: frozenset(
        {
            "twin_rate",
            "twin_gap",
            "twin_factor",
            "efficiency",
            "edge_bonus",
            "ttft_p95_ms",
            "tpot_p95_ms",
            "vram_peak_bytes",
            "parameters",
            "versus_reference_lb",
            "beats_base_lb",
            "gate",
            "provenance",
            "envelope",
            "families",
            "passed",
            "reason",
        }
    ),
    pools.HARNESS: frozenset(
        {
            "baseline",
            "baseline_pass_rate",
            "miner_pass_rate",
            "baselines",
            "tokens_miner",
            "tokens_baseline",
            "allowance",
            "standard_error",
            "reason",
            "limits",
        }
    ),
    pools.ESCALATION: frozenset(
        {
            "redo_pass_rate",
            "repair_cost_micros",
            "redo_cost_micros",
            "margin_over_reference_lb",
            "gate",
            "reason",
        }
    ),
    pools.SYSTEM: frozenset(
        {
            "floors",
            "reference_cost_micros",
            "reference_latency_p95_ms",
            "reference_items",
            "reason",
        }
    ),
    pools.ROUTER: frozenset(
        {"sources", "latency_p95_ms", "gate", "reason", "margin_over_reference_lb"}
    ),
}
HARNESS_LIMIT_KEYS: Final[frozenset[str]] = frozenset(
    {"max_model_calls", "added_wall_p95_ms", "added_wall_limit_ms", "within"}
)
ROUTER_SOURCE_KEYS: Final[frozenset[str]] = frozenset({"own", "reference", "peer", "adversarial"})
NESTED_KEYS: Final[dict[tuple[str, str], tuple[frozenset[str] | None, bool]]] = {
    (pools.HARNESS, "baselines"): (None, True),
    (pools.HARNESS, "limits"): (HARNESS_LIMIT_KEYS, False),
    (pools.ROUTER, "sources"): (ROUTER_SOURCE_KEYS, True),
}
FLOOR_KEYS: Final[frozenset[str]] = frozenset(
    {"pass_rate", "lower_bound", "cost_micros", "latency_p95_ms", "escalated", "value", "n", "met"}
)
REFERENCE_KEYS: Final[dict[str, frozenset[str]]] = {
    "slm": frozenset({"pass_rate", "ttft_p95_ms", "tpot_p95_ms", "vram_peak_bytes", "parameters"}),
    "base": frozenset({"name", "pass_rate"}),
    "system": frozenset({"cost_micros", "latency_p95_ms", "items"}),
}


def receipts_digest(receipts: Sequence[Mapping[str, Any]]) -> str:
    body = json.dumps(sorted(r.get("receipt", "") for r in receipts), separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def build(
    context: EpochContext,
    *,
    validator_hotkey: str,
    artifacts: Mapping[str, Artifact],
    scores: Mapping[str, Sequence[PoolScore]],
    failures: Mapping[str, str],
    receipts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> EpochReport:
    miners: list[MinerReport] = []
    for hotkey in sorted(set(artifacts) | set(failures)):
        artifact = artifacts.get(hotkey)
        miners.append(
            MinerReport(
                hotkey=hotkey,
                artifact_sha=artifact.artifact_sha if artifact else "",
                manifest_sha=artifact.manifest_sha if artifact else "",
                pools=tuple(scores.get(hotkey, ())),
                failure=failures.get(hotkey, ""),
                receipts_digest=receipts_digest(receipts.get(hotkey, ())),
            )
        )
    return EpochReport(
        arena_id=context.arena_id,
        epoch_index=context.epoch_index,
        validator_hotkey=validator_hotkey,
        bundle_sha256=context.bundle_sha256,
        open_block_hash=context.open_block_hash,
        seed=context.seed,
        item_ids_digest=item_ids_digest(context.items),
        ledger_snapshot_digest=context.snapshot_digest,
        roster_digest=context.roster_digest,
        miners=tuple(miners),
        mechanism_version=MECHANISM_VERSION,
    )


def sign(report: EpochReport, wallet: Any) -> EpochReport:
    return report.signed_with(sign_payload(wallet, report.body()))


def _number(value: Any) -> float:
    try:
        found = float(value)
    except (TypeError, ValueError):
        return 0.0
    return found if math.isfinite(found) else 0.0


def _flat(raw: Mapping[str, Any], allowed: frozenset[str] | None, numbers: bool) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key)[: checklist.TEXT_LIMIT]
        if allowed is not None and name not in allowed:
            continue
        kept = checklist.clean(value)
        if kept is None or (numbers and isinstance(kept, (bool, str))):
            continue
        found[name] = kept
    return found


def pool_detail(pool: str, detail: Mapping[str, Any]) -> dict[str, Any]:
    allowed = DETAIL_KEYS.get(pool, frozenset())
    found: dict[str, Any] = {}
    for key, value in detail.items():
        if key not in allowed:
            continue
        if isinstance(value, Mapping):
            if pool == pools.SYSTEM and key == "floors":
                found[key] = {
                    str(floor)[: checklist.TEXT_LIMIT]: _flat(row, FLOOR_KEYS, False)
                    for floor, row in value.items()
                    if isinstance(row, Mapping)
                }
            elif (pool, key) in NESTED_KEYS:
                keys, numbers = NESTED_KEYS[(pool, key)]
                found[key] = _flat(value, keys, numbers)
            continue
        kept = checklist.clean(value)
        if kept is not None:
            found[key] = kept
    return found


def tidy_details(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for pool, entry in raw.items():
        if pool not in DETAIL_KEYS or not isinstance(entry, Mapping):
            continue
        detail = entry.get("detail")
        found[pool] = {
            "score": min(1.0, max(0.0, _number(entry.get("score")))),
            "lower_bound": _number(entry.get("lower_bound")),
            "point": _number(entry.get("point")),
            "n": max(0, int(_number(entry.get("n")))),
            "eligible": bool(entry.get("eligible")),
            "detail": pool_detail(pool, detail if isinstance(detail, Mapping) else {}),
        }
    return found


def pool_details(scores: Sequence[PoolScore]) -> dict[str, dict[str, Any]]:
    return tidy_details({s.pool: s.to_dict() for s in scores})


def tidy_reference(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: _flat(value, REFERENCE_KEYS[key], False)
        for key, value in raw.items()
        if key in REFERENCE_KEYS and isinstance(value, Mapping)
    }


def _size(live: Mapping[str, Any]) -> int:
    return len(json.dumps(live).encode("utf-8"))


def _fit(live: dict[str, Any]) -> dict[str, Any]:
    if _size(live) > LIVE_LIMIT_BYTES:
        live.pop("reference", None)
    if _size(live) > LIVE_LIMIT_BYTES:
        for entry in dict(live.get("pool_details") or {}).values():
            entry["detail"] = {}
    if _size(live) > LIVE_LIMIT_BYTES:
        live.pop("checks", None)
    return live


def artifact_body(
    context: EpochContext,
    *,
    validator: str,
    miner: str,
    artifact_sha: str,
    scores: Sequence[PoolScore],
    provenance: Mapping[str, Any],
    failure: str,
    pool_details: Mapping[str, Any] | None = None,
    checks: Sequence[Mapping[str, Any]] | None = None,
    reference: Mapping[str, Any] | None = None,
    progress_final: bool | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "arena_id": context.arena_id,
        "epoch_index": int(context.epoch_index),
        "validator": validator,
        "miner": miner,
        "artifact_sha": artifact_sha,
        "pools": {s.pool: float(s.score) for s in scores},
        "provenance": {
            "passed": bool(provenance.get("passed")),
            "reason": str(provenance.get("reason") or ""),
        },
        "failure": failure[:ARTIFACT_FAILURE_LIMIT],
        "items": {s.pool: int(s.n) for s in scores},
    }
    live: dict[str, Any] = {}
    if pool_details is not None:
        live["pool_details"] = tidy_details(pool_details)
    if checks is not None:
        live["checks"] = checklist.tidy(checks)
    if reference is not None:
        live["reference"] = tidy_reference(reference)
    body.update(_fit(live))
    if progress_final is not None:
        body["progress_final"] = bool(progress_final)
    return body


def artifact_digest(body: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in body.items() if key != "signature"}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_message(body: Mapping[str, Any]) -> bytes:
    return (
        f"{ARTIFACT_TAG}|{body['arena_id']}|{body['epoch_index']}|{body['validator']}"
        f"|{body['miner']}|{body['artifact_sha']}|{artifact_digest(body)}"
    ).encode()


def sign_artifact(body: Mapping[str, Any], wallet: Any) -> dict[str, Any]:
    unsigned = {key: value for key, value in body.items() if key != "signature"}
    return {**unsigned, "signature": sign_bytes(wallet, artifact_message(unsigned))}
