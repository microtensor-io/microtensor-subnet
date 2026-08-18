from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from microtensor.core.constants import (
    ALLOWED_BASE_MODELS,
    DEFAULT_NETUID,
    RELEASE_SIGNING_KEY,
)
from microtensor.core.tracks import CLASSES, enabled_tracks

OPEN: Final[str] = "open"
CLOSED: Final[str] = "closed"
SET: Final[str] = "set"


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    ready: bool
    posture: str
    detail: str
    fix: str = ""

    @property
    def unenforced(self) -> bool:
        return not self.ready and self.posture == OPEN


def launch_classes() -> tuple[str, ...]:
    seen: list[str] = []
    for track in enabled_tracks():
        for class_id in track.live_classes:
            if class_id not in seen:
                seen.append(class_id)
    return tuple(seen)


def _base_model_gate() -> Gate:
    if ALLOWED_BASE_MODELS:
        return Gate(
            name="base-model allowlist",
            ready=True,
            posture=CLOSED,
            detail=f"{len(ALLOWED_BASE_MODELS)} pinned revisions",
        )
    return Gate(
        name="base-model allowlist",
        ready=False,
        posture=OPEN,
        detail="empty, so every declared base model is accepted",
        fix="populate ALLOWED_BASE_MODELS with <repo>@<revision-sha> entries",
    )


def _conformance_gates() -> list[Gate]:
    gates: list[Gate] = []
    for class_id in launch_classes():
        hardware = CLASSES[class_id]
        if hardware.device_profile:
            gates.append(
                Gate(
                    name=f"conformance / {class_id}",
                    ready=True,
                    posture=CLOSED,
                    detail=hardware.device_profile,
                )
            )
            continue
        gates.append(
            Gate(
                name=f"conformance / {class_id}",
                ready=False,
                posture=OPEN,
                detail="no reference profile, so every host counts as conforming",
                fix=f"set device_profile on the {class_id} class from a certified run",
            )
        )
    return gates


def _band_gates() -> list[Gate]:
    from microtensor.envelope.certify import CERT_BANDS

    gates: list[Gate] = []
    for class_id in launch_classes():
        if class_id in CERT_BANDS:
            gates.append(
                Gate(
                    name=f"certification band / {class_id}",
                    ready=True,
                    posture=CLOSED,
                    detail="published",
                )
            )
            continue
        gates.append(
            Gate(
                name=f"certification band / {class_id}",
                ready=False,
                posture=OPEN,
                detail="no band, so certification records but never fails",
                fix=f"mt validator certify {class_id} --fit-band 10",
            )
        )
    return gates


def _signing_gate() -> Gate:
    if RELEASE_SIGNING_KEY:
        return Gate(
            name="release signing key",
            ready=True,
            posture=SET,
            detail="pinned, signed releases will verify",
        )
    return Gate(
        name="release signing key",
        ready=False,
        posture=CLOSED,
        detail="empty, so auto-update refuses to arm rather than trusting anything",
        fix="publish an ed25519 key and set RELEASE_SIGNING_KEY",
    )


def _provenance_gate() -> Gate:
    from microtensor.core.constants import (
        PROVENANCE_ENTITY,
        PROVENANCE_PROJECT,
        PROVENANCE_REQUIRED,
    )
    from microtensor.provenance.wandb_store import credentials_present

    where = f"{PROVENANCE_ENTITY}/{PROVENANCE_PROJECT}"
    if not PROVENANCE_REQUIRED:
        return Gate(
            name="training provenance",
            ready=False,
            posture=OPEN,
            detail="PROVENANCE_REQUIRED is off, so submissions need no training run",
            fix="set PROVENANCE_REQUIRED once the run store is live",
        )
    if not credentials_present():
        return Gate(
            name="training provenance",
            ready=False,
            posture=CLOSED,
            detail=f"required against {where}, but WANDB_API_KEY is unset so rounds abstain",
            fix="export WANDB_API_KEY with read access to the project",
        )
    return Gate(
        name="training provenance",
        ready=True,
        posture=CLOSED,
        detail=f"required against {where}, credentials present",
    )


def _baseline_gates() -> list[Gate]:
    from microtensor.core.constants import ROLE_BASELINES

    gates: list[Gate] = []
    for role, digest in sorted(ROLE_BASELINES.items()):
        if digest:
            gates.append(
                Gate(
                    name=f"role baseline / {role}",
                    ready=True,
                    posture=CLOSED,
                    detail=digest,
                )
            )
            continue
        gates.append(
            Gate(
                name=f"role baseline / {role}",
                ready=False,
                posture=CLOSED,
                detail="unpublished, so this role's contribution reports null rather than zero",
                fix=f"publish a {role} baseline with the corpus and pin its digest",
            )
        )
    return gates


def audit() -> list[Gate]:
    return [
        _base_model_gate(),
        *_conformance_gates(),
        *_band_gates(),
        _signing_gate(),
        _provenance_gate(),
        *_baseline_gates(),
    ]


def unenforced() -> list[Gate]:
    return [gate for gate in audit() if gate.unenforced]


def summary() -> str:
    gates = audit()
    ready = sum(1 for gate in gates if gate.ready)
    return f"netuid {DEFAULT_NETUID}: {ready}/{len(gates)} launch values set"
