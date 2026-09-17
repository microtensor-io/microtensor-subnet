from __future__ import annotations

from typing import Any

from microtensor.rigs.protocol.failures import Failure, FailureClass, Verdict


def enrolment_index(roster: list[dict[str, Any]], exclude_rig: str) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for rig in roster:
        rig_id = str(rig.get("id", "") or "")
        if not rig_id or rig_id == exclude_rig:
            continue
        if str(rig.get("state", "") or "") == "removed":
            continue
        for card in rig.get("gpus") or []:
            uuid = str(card.get("uuid", "") or "") if isinstance(card, dict) else ""
            if uuid:
                index.setdefault(uuid, []).append(rig_id)
    return index


def check(
    gpus: list[dict[str, Any]], rig_id: str, roster: list[dict[str, Any]], seed: str = ""
) -> Verdict:
    index = enrolment_index(roster, rig_id)
    duplicates: dict[str, list[str]] = {}
    for card in gpus:
        uuid = str(card.get("uuid", "") or "")
        if uuid and uuid in index:
            duplicates[uuid] = index[uuid]
    evidence: dict[str, Any] = {
        "scanned": len(gpus),
        "other_rigs": len({r for rigs in index.values() for r in rigs}),
        "duplicates": duplicates,
        "summary": "no card enrolled elsewhere"
        if not duplicates
        else f"{len(duplicates)} cards enrolled elsewhere",
    }
    if duplicates:
        uuid, rigs = next(iter(duplicates.items()))
        return Verdict(
            "duplicates",
            failure=Failure(
                FailureClass.DUPLICATE_UUID,
                f"{uuid} is enrolled on rig {rigs[0]}",
                seed=seed,
                evidence={"duplicates": duplicates},
            ),
            evidence=evidence,
        )
    return Verdict("duplicates", evidence=evidence)
