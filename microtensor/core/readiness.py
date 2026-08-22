from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from microtensor.core.constants import (
    DEFAULT_NETUID,
    RELEASE_SIGNING_KEY,
)
from microtensor.core.tracks import CLASSES, enabled_tracks

OPEN: Final[str] = "open"
CLOSED: Final[str] = "closed"
SET: Final[str] = "set"
# A gate whose real state could not be read. Distinct from OPEN on purpose:
# open means nothing is configured and everything is admitted, unknown means
# something may well be configured and this host could not find out. Reporting
# the second as the first is the same fail-open mistake as an empty allowlist
# reading as "permit everything".
UNKNOWN: Final[str] = "unknown"


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

    @property
    def unreadable(self) -> bool:
        return self.posture == UNKNOWN


@dataclass(frozen=True, slots=True)
class Served:
    """What a coordinator says the rules are, or why this host cannot tell.

    Three states, and the third is why this exists. Without a coordinator the
    local constants are the whole truth and an unset value really is an open
    gate. With one reachable, the served config is the truth and a value this
    build does not carry is still closed. With one configured but unreachable,
    this host knows nothing either way, and saying "open" would report a
    configured arena as an unguarded one.
    """

    configured: bool = False
    reachable: bool = False
    # Whether the coordinator responded at all. Separate from `reachable`,
    # which means it responded *with a config*. A coordinator with no round
    # open answers 200 with an empty body, and reading that as silence sends
    # an operator to debug a healthy service.
    answered: bool = False
    arenas: Mapping[str, Sequence[str]] = field(default_factory=dict)
    role_baselines: Mapping[str, str] = field(default_factory=dict)

    @property
    def unknown(self) -> bool:
        return self.configured and not self.reachable

    @property
    def idle(self) -> bool:
        """Answered, but publishing nothing to read yet."""
        return self.configured and self.answered and not self.reachable

    @classmethod
    def local(cls) -> Served:
        return cls()

    @classmethod
    def unreachable(cls) -> Served:
        return cls(configured=True, reachable=False)

    @classmethod
    def silent(cls) -> Served:
        """Reachable, with no round open, so no config is served."""
        return cls(configured=True, answered=True, reachable=False)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Served:
        arenas = {
            str(key): [str(m) for m in dict(value).get("allowed_base_models", [])]
            for key, value in dict(config.get("arenas", {})).items()
        }
        baselines = {
            str(role): str(digest)
            for role, digest in dict(config.get("role_baselines", {})).items()
        }
        return cls(
            configured=True,
            answered=True,
            reachable=True,
            arenas=arenas,
            role_baselines=baselines,
        )


def launch_classes() -> tuple[str, ...]:
    seen: list[str] = []
    for track in enabled_tracks():
        for class_id in track.live_classes:
            if class_id not in seen:
                seen.append(class_id)
    return tuple(seen)


def _base_model_gates(
    arenas: Mapping[str, Sequence[str]] | None = None,
    served: Served | None = None,
) -> list[Gate]:
    """One gate per arena, because the allowlist is per arena.

    Sourced from the arena records the control plane holds rather than from a
    constant. With no arena known there is nothing to report on: an empty
    audit is the honest answer, not a passing global gate.
    """
    served = served or Served.local()
    arenas = arenas or served.arenas

    if served.unknown and not arenas:
        return [
            Gate(
                name="base-model allowlist",
                ready=False,
                posture=UNKNOWN,
                detail=(
                    "the coordinator answered but has no round open, so it is not "
                    "serving a config to read"
                    if served.idle
                    else "a coordinator is configured but did not answer, so this "
                    "host cannot tell what is admissible"
                ),
                fix=(
                    "open a round, or check readiness once one is open"
                    if served.idle
                    else "reach the coordinator, or check readiness where it is reachable"
                ),
            )
        ]

    if not arenas:
        return [
            Gate(
                name="base-model allowlist",
                ready=False,
                posture=CLOSED,
                detail="no arena is configured, so nothing is admissible",
                fix="create an arena and attach its allowlist through the operator API",
            )
        ]

    gates: list[Gate] = []
    for arena, entries in sorted(arenas.items()):
        gates.append(
            Gate(
                name=f"base-model allowlist ({arena})",
                ready=bool(entries),
                posture=CLOSED,
                detail=(
                    f"{len(entries)} pinned revisions"
                    if entries
                    else "empty, so no submission is admissible"
                ),
                fix=("" if entries else f"POST /v1/operator/arenas/<id>/allowlist for {arena}"),
            )
        )
    return gates


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


def _provenance_gate(probe: bool = False) -> Gate:
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
            detail=f"required against {where}, but no wandb credentials so rounds abstain",
            fix="export WANDB_API_KEY with read access to the project, or run wandb login",
        )

    # Credentials present is not the same as a store that answers. Reporting
    # on the key alone leaves this gate green in precisely the case that
    # matters: a key that works against a project which does not exist.
    if not probe:
        return Gate(
            name="training provenance",
            ready=False,
            posture=CLOSED,
            detail=f"required against {where}, credentials present but the store was not probed",
            fix="run `mt inspect readiness --probe` to check the store answers",
        )

    from microtensor.provenance.wandb_store import WandbStore

    reachable, why = WandbStore().reachable()
    if not reachable:
        return Gate(
            name="training provenance",
            ready=False,
            posture=CLOSED,
            detail=f"{where} did not answer: {why}",
            fix=f"create the {where} project and confirm miners can write runs to it",
        )

    return Gate(
        name="training provenance",
        ready=True,
        posture=CLOSED,
        detail=f"required against {where}, and it answers",
    )


def _baseline_gates(served: Served | None = None) -> list[Gate]:
    from microtensor.core.constants import ROLE_BASELINES

    served = served or Served.local()
    if served.unknown:
        return [
            Gate(
                name=f"role baseline / {role}",
                ready=False,
                posture=UNKNOWN,
                detail=(
                    "the coordinator answered but has no round open, so it is not "
                    "serving a config to read"
                    if served.idle
                    else "a coordinator is configured but did not answer, so this "
                    "host cannot tell whether a baseline is published"
                ),
                fix=(
                    "open a round, or check readiness once one is open"
                    if served.idle
                    else "reach the coordinator, or check readiness where it is reachable"
                ),
            )
            for role in sorted(ROLE_BASELINES)
        ]

    published = dict(served.role_baselines) if served.reachable else dict(ROLE_BASELINES)

    gates: list[Gate] = []
    for role, digest in sorted(published.items()):
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


def audit(
    arenas: Mapping[str, Sequence[str]] | None = None,
    *,
    probe: bool = False,
    served: Served | None = None,
) -> list[Gate]:
    """Every launch gate and its posture.

    `served` decides where the arena and baseline gates get their answer:
    local constants when no coordinator is configured, the served config when
    one answered, and neither when one is configured but silent.
    """
    return [
        *_base_model_gates(arenas, served),
        *_conformance_gates(),
        *_band_gates(),
        _signing_gate(),
        _provenance_gate(probe),
        *_baseline_gates(served),
    ]


def unenforced(
    arenas: Mapping[str, Sequence[str]] | None = None,
    *,
    probe: bool = False,
    served: Served | None = None,
) -> list[Gate]:
    return [gate for gate in audit(arenas, probe=probe, served=served) if gate.unenforced]


def unreadable(
    arenas: Mapping[str, Sequence[str]] | None = None,
    *,
    probe: bool = False,
    served: Served | None = None,
) -> list[Gate]:
    """Gates whose real state this host could not determine.

    Kept apart from `unenforced` so an outage never reads as a permissive
    gate, and never reads as a satisfied one either.
    """
    return [gate for gate in audit(arenas, probe=probe, served=served) if gate.unreadable]


def summary() -> str:
    gates = audit()
    ready = sum(1 for gate in gates if gate.ready)
    return f"netuid {DEFAULT_NETUID}: {ready}/{len(gates)} launch values set"
