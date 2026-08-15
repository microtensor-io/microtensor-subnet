from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from microtensor.chain.rounds import Round
from microtensor.core.constants import MECHANISM_VERSION
from microtensor.update.release import Release, Version


class Action(str, Enum):
    NONE = "none"
    APPLY = "apply"
    DEFER = "defer"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    reason: str
    release: Release | None = None
    ready_at_block: int | None = None

    @property
    def should_apply(self) -> bool:
        return self.action is Action.APPLY

    @property
    def needs_operator(self) -> bool:
        return self.action is Action.HOLD


def is_safe_point(round_: Round, block: int) -> bool:
    return round_.accepts_submissions(block)


def decide(
    release: Release | None,
    round_: Round,
    block: int,
    *,
    current: str = MECHANISM_VERSION,
    allow_mechanism_change: bool = False,
    allow_major: bool = False,
) -> Decision:
    if release is None:
        return Decision(Action.NONE, "no published release on this channel")

    running = Version.parse(current)
    if release.version <= running:
        return Decision(Action.NONE, f"already running {running}", release)

    if release.version.major != running.major and not allow_major:
        return Decision(
            Action.HOLD,
            f"{release.version} is a major release; upgrade deliberately, not on a timer",
            release,
        )

    if release.changes_mechanism:
        if release.activation_block is None:
            return Decision(
                Action.HOLD,
                (
                    f"{release.tag} changes the mechanism to {release.mechanism_version} "
                    "but declares no activation block; applying it at a different moment "
                    "than other validators would split consensus"
                ),
                release,
            )
        if block < release.activation_block:
            return Decision(
                Action.DEFER,
                (
                    f"{release.tag} activates at block {release.activation_block}, "
                    f"{release.activation_block - block} blocks away"
                ),
                release,
                release.activation_block,
            )
        if not allow_mechanism_change:
            return Decision(
                Action.HOLD,
                (
                    f"{release.tag} moves the mechanism from {current} to "
                    f"{release.mechanism_version}; pass --allow-mechanism-change to accept it"
                ),
                release,
            )

    if not is_safe_point(round_, block):
        return Decision(
            Action.DEFER,
            (
                f"round {round_.index} is sealed; waiting for the next submission window "
                f"at block {round_.next.start_block} so the restart cannot interrupt scoring"
            ),
            release,
            round_.next.start_block,
        )

    return Decision(Action.APPLY, f"{running} → {release.version}", release)
