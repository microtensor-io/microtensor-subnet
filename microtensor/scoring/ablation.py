from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from microtensor.core.constants import ROLE_BASELINES
from microtensor.core.protocol import Role
from microtensor.scoring.frontier import Point, exclusive_hypervolume

UNSET_BASELINE = "no published baseline for this role"
NOT_ON_FRONTIER = "system is not on the frontier, so it has no value to divide"


@dataclass(frozen=True, slots=True)
class Contribution:
    role: Role
    artifact_digest: str
    value: int | None
    share: float | None
    reason: str = ""

    @property
    def measurable(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "artifact_digest": self.artifact_digest,
            "value": self.value,
            "share": self.share,
            "reason": self.reason,
        }


def baseline_for(role: Role, baselines: Mapping[str, str] | None = None) -> str:
    table = ROLE_BASELINES if baselines is None else baselines
    return table.get(role.value, "")


def substitute(members: Sequence[Point], key: str, ablated: Point) -> list[Point]:
    """Swap the ablated point in for the system, in place rather than in addition.

    Inserting alongside would measure what adding a point does to the frontier,
    which is a different question from what this component is worth.
    """
    replaced = [p for p in members if p.key != key]
    replaced.append(ablated)
    return replaced


def contribution(
    members: Sequence[Point],
    key: str,
    ablated: Point,
) -> int:
    """phi(c) = V(S) - V(S with c replaced by its role baseline), clamped at zero."""
    present = exclusive_hypervolume(members).get(key, 0)
    without = exclusive_hypervolume(substitute(members, key, ablated)).get(ablated.key, 0)
    return max(present - without, 0)


def settle(
    members: Sequence[Point],
    key: str,
    ablations: Mapping[Role, Point | None],
    baselines: Mapping[str, str] | None = None,
) -> tuple[Contribution, ...]:
    """Price each component of one system by role-baseline ablation.

    A component whose baseline is unpublished reports None, never zero. Zero is
    a measurement meaning the system does not need this component; None means
    the network could not measure it, and the two must not be confused.
    """
    if not any(p.key == key for p in members):
        return tuple(
            Contribution(
                role=role,
                artifact_digest="",
                value=None,
                share=None,
                reason=NOT_ON_FRONTIER,
            )
            for role in ablations
        )

    raw: dict[Role, int] = {}
    results: list[Contribution] = []

    for role, ablated in ablations.items():
        digest = baseline_for(role, baselines)
        if not digest or ablated is None:
            results.append(
                Contribution(
                    role=role,
                    artifact_digest=digest,
                    value=None,
                    share=None,
                    reason=UNSET_BASELINE,
                )
            )
            continue
        value = contribution(members, key, ablated)
        raw[role] = value
        results.append(
            Contribution(role=role, artifact_digest=digest, value=value, share=None)
        )

    total = sum(raw.values())
    if total <= 0:
        return tuple(results)

    return tuple(
        Contribution(
            role=c.role,
            artifact_digest=c.artifact_digest,
            value=c.value,
            share=(raw[c.role] / total) if c.role in raw else None,
            reason=c.reason,
        )
        for c in results
    )


def unset_roles(baselines: Mapping[str, str] | None = None) -> tuple[str, ...]:
    table = ROLE_BASELINES if baselines is None else baselines
    return tuple(sorted(role for role, digest in table.items() if not digest))
