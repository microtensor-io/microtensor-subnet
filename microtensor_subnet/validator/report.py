from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from microtensor_subnet.chain.wallet import sign_bytes, sign_payload
from microtensor_subnet.core.constants import MECHANISM_VERSION
from microtensor_subnet.core.reports import EpochReport, MinerReport, PoolScore
from microtensor_subnet.validator.context import Artifact, EpochContext
from microtensor_subnet.validator.items import item_ids_digest

ARTIFACT_TAG: Final[str] = "mt-artifact-result"
ARTIFACT_FAILURE_LIMIT: Final[int] = 400


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


def artifact_body(
    context: EpochContext,
    *,
    validator: str,
    miner: str,
    artifact_sha: str,
    scores: Sequence[PoolScore],
    provenance: Mapping[str, Any],
    failure: str,
) -> dict[str, Any]:
    return {
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
