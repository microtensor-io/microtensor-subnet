from __future__ import annotations

from dataclasses import dataclass

from microtensor.core.constants import (
    REPUTATION_FLOOR,
    REPUTATION_MIN_ROUNDS,
    REPUTATION_RECOVERY_ROUNDS,
)


@dataclass(frozen=True, slots=True)
class Standing:
    """One worker's agreement record.

    This is what replaces "trust the coordinator". The coordinator is not
    judging whether a measurement is correct, it is counting how often each
    worker agrees with the independent majority, which is a fact anyone holding
    the published reports can recompute.
    """

    hotkey: str
    agreed: int = 0
    diverged: int = 0
    streak: int = 0
    advisory: bool = False
    last_round: int = 0

    @property
    def total(self) -> int:
        return self.agreed + self.diverged

    @property
    def rate(self) -> float:
        """Agreement rate. A worker with no history is credited, not doubted."""
        return 1.0 if self.total == 0 else self.agreed / self.total

    @property
    def established(self) -> bool:
        return self.total >= REPUTATION_MIN_ROUNDS

    def to_dict(self) -> dict[str, object]:
        return {
            "hotkey": self.hotkey,
            "agreed": self.agreed,
            "diverged": self.diverged,
            "rate": round(self.rate, 4),
            "streak": self.streak,
            "advisory": self.advisory,
            "established": self.established,
            "last_round": self.last_round,
        }


def update(standing: Standing, *, agreed: bool, round_index: int) -> Standing:
    """Fold one round's outcome into a worker's record.

    Demotion needs an established history, so a worker is never made advisory
    on the strength of its first few rounds. Promotion back needs a run of
    agreeing rounds rather than one, because a single agreement is exactly what
    an intermittently broken worker produces.
    """
    agreed_n = standing.agreed + (1 if agreed else 0)
    diverged_n = standing.diverged + (0 if agreed else 1)
    streak = standing.streak + 1 if agreed else 0

    total = agreed_n + diverged_n
    rate = 1.0 if total == 0 else agreed_n / total

    advisory = standing.advisory
    if advisory:
        if streak >= REPUTATION_RECOVERY_ROUNDS and rate >= REPUTATION_FLOOR:
            advisory = False
    elif total >= REPUTATION_MIN_ROUNDS and rate < REPUTATION_FLOOR:
        advisory = True

    return Standing(
        hotkey=standing.hotkey,
        agreed=agreed_n,
        diverged=diverged_n,
        streak=streak,
        advisory=advisory,
        last_round=round_index,
    )


def advisory_set(standings: list[Standing]) -> tuple[str, ...]:
    """Workers whose reports are stored and served but do not decide a value."""
    return tuple(sorted(s.hotkey for s in standings if s.advisory))
