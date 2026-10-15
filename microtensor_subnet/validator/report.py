from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from microtensor_subnet.chain.wallet import sign_payload
from microtensor_subnet.core.constants import MECHANISM_VERSION
from microtensor_subnet.core.reports import EpochReport, MinerReport, PoolScore
from microtensor_subnet.validator.context import Artifact, EpochContext
from microtensor_subnet.validator.items import item_ids_digest


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
