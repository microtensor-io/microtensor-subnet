from __future__ import annotations

from typing import Any

from microtensor.rigs.protocol.failures import Failure, FailureClass, Verdict
from microtensor.rigs.validator.data import gpu_table
from microtensor.rigs.validator.data.tiers import Tier

FATAL_NVML = {
    "DRIVER_NOT_LOADED",
    "GPU_IS_LOST",
    "GPU_NOT_FOUND",
    "MEMORY",
    "LIBRARY_NOT_FOUND",
    "UNINITIALIZED",
    "LIB_RM_VERSION_MISMATCH",
    "RESET_REQUIRED",
    "IRQ_ISSUE",
    "FUNCTION_NOT_FOUND",
    "UNKNOWN",
}
CORE_FIELDS = ("handle", "name", "uuid", "memory")
MIG_ENABLED = 1
SRIOV = 1


def enrolled_cards(rig: dict[str, Any]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for card in rig.get("gpus") or []:
        if isinstance(card, dict) and card.get("uuid"):
            found[str(card["uuid"])] = card
    return found


def seen_cards(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for card in data.get("gpus") or []:
        if isinstance(card, dict) and card.get("uuid"):
            found[str(card["uuid"])] = card
    return found


def enrolled_tier(card: dict[str, Any]) -> Tier | None:
    try:
        return Tier(str(card.get("tier", "") or ""))
    except ValueError:
        return None


def card_problems(uuid: str, enrolled: dict[str, Any], seen: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    errors = seen.get("errors") or {}
    for name in CORE_FIELDS:
        if name in errors:
            problems.append(f"{uuid}: NVML_ERROR_{errors[name]} reading {name}")
    if problems:
        return problems
    declared = gpu_table.normalize(str(enrolled.get("model", "") or ""))
    observed = gpu_table.normalize(str(seen.get("name", "") or ""))
    if declared != observed:
        problems.append(
            f"{uuid}: enrolled as {declared or 'unknown'}, reads {observed or 'unnamed'}"
        )
    try:
        total_mb = float(seen.get("memory_total_mb") or 0)
    except (TypeError, ValueError):
        total_mb = 0.0
    matched = gpu_table.match(observed, total_mb)
    if matched is None:
        problems.append(f"{uuid}: {gpu_table.mismatch_reason(observed, total_mb)}")
    else:
        declared_tier = enrolled_tier(enrolled)
        if (
            declared_tier is not None
            and matched.tier is not None
            and matched.tier is not declared_tier
        ):
            problems.append(
                f"{uuid}: enrolled in the {declared_tier.value} tier, reads as {matched.tier.value}"
            )
    mig = seen.get("mig_mode")
    if isinstance(mig, dict) and int(mig.get("current") or 0) == MIG_ENABLED:
        problems.append(f"{uuid}: MIG mode is enabled")
    virtualization = seen.get("virtualization")
    if virtualization not in (None, "", "NONE"):
        problems.append(f"{uuid}: virtualization mode {virtualization}")
    host_vgpu = seen.get("host_vgpu_mode")
    if host_vgpu == SRIOV:
        problems.append(f"{uuid}: SR-IOV host vGPU mode")
    return problems


def check(data: dict[str, Any], rig: dict[str, Any], seed: str = "") -> Verdict:
    nvml = data.get("nvml") or {}
    errors = [str(name) for name in (nvml.get("errors") or [])]
    enrolled = enrolled_cards(rig)
    seen = seen_cards(data)
    evidence: dict[str, Any] = {
        "enrolled": len(enrolled),
        "seen": len(seen),
        "nvml_errors": errors,
        "driver": nvml.get("driver", ""),
        "cuda": nvml.get("cuda", ""),
        "cards": [
            {
                "uuid": uuid,
                "model": card.get("name"),
                "memory_mb": card.get("memory_total_mb"),
                "mig": (card.get("mig_mode") or {}).get("current")
                if isinstance(card.get("mig_mode"), dict)
                else None,
                "virtualization": card.get("virtualization"),
                "errors": card.get("errors") or {},
            }
            for uuid, card in seen.items()
        ],
    }
    fatal = [name for name in errors if name in FATAL_NVML]
    if fatal:
        return Verdict(
            "specs",
            failure=Failure(
                FailureClass.SPEC_MISMATCH,
                f"NVML_ERROR_{fatal[0]}",
                seed=seed,
                evidence={"errors": errors},
            ),
            evidence=evidence,
        )
    problems: list[str] = []
    missing = sorted(set(enrolled) - set(seen))
    if missing:
        problems.append(f"enrolled cards absent: {', '.join(missing)}")
    for uuid in sorted(set(enrolled) & set(seen)):
        problems.extend(card_problems(uuid, enrolled[uuid], seen[uuid]))
    extra = sorted(set(seen) - set(enrolled))
    evidence["unenrolled"] = extra
    if problems:
        evidence["problems"] = problems
        return Verdict(
            "specs",
            failure=Failure(
                FailureClass.SPEC_MISMATCH,
                problems[0][:300],
                seed=seed,
                evidence={"problems": problems[:20]},
            ),
            evidence=evidence,
        )
    evidence["summary"] = f"{len(enrolled)} cards as enrolled" + (
        f", {len(extra)} unenrolled" if extra else ""
    )
    return Verdict("specs", evidence=evidence)
