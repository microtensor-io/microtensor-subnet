from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from microtensor.core.constants import (
    HYSTERESIS_EPSILON,
    INCUMBENT_DECAY,
    MIN_ROUNDS_OBSERVED,
    PAID_RANKS,
    RANK_DECAY,
    TRACK_THRESHOLD,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    hotkey: str
    score: float
    rounds_observed: int = MIN_ROUNDS_OBSERVED
    stale_rounds: int = 0


def geometric_shares(paid_ranks: int = PAID_RANKS, decay: float = RANK_DECAY) -> tuple[float, ...]:
    if paid_ranks < 1:
        raise ValueError("paid_ranks must be at least 1")
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must lie strictly between 0 and 1")
    denominator = 1.0 - decay**paid_ranks
    return tuple(
        (1.0 - decay) * decay**j / denominator for j in range(paid_ranks)
    )


def eligible(
    candidates: Sequence[Candidate],
    threshold: float = TRACK_THRESHOLD,
    min_rounds: int = MIN_ROUNDS_OBSERVED,
) -> list[Candidate]:
    return [
        c
        for c in candidates
        if c.score >= threshold and c.score > 0.0 and c.rounds_observed >= min_rounds
    ]


def rank_with_hysteresis(
    candidates: Sequence[Candidate],
    previous: Sequence[str] = (),
    epsilon: float = HYSTERESIS_EPSILON,
    paid_ranks: int = PAID_RANKS,
) -> list[Candidate]:
    by_score = sorted(candidates, key=lambda c: (-c.score, c.hotkey))
    present = {c.hotkey: c for c in by_score}
    remaining = list(by_score)
    ordered: list[Candidate] = []

    for position in range(min(paid_ranks, len(by_score))):
        if not remaining:
            break

        challenger = remaining[0]
        holder_key = previous[position] if position < len(previous) else None
        holder = present.get(holder_key) if holder_key else None

        chosen = challenger
        if (
            holder is not None
            and holder is not challenger
            and holder in remaining
            and challenger.score <= holder.score + epsilon
        ):
            chosen = holder

        ordered.append(chosen)
        remaining.remove(chosen)

    ordered.extend(remaining)
    return ordered


def apply_incumbent_decay(
    shares: Sequence[float],
    ranked: Sequence[Candidate],
    decay: float = INCUMBENT_DECAY,
) -> tuple[float, ...]:
    if not shares or not ranked:
        return tuple(shares)

    paid = min(len(shares), len(ranked))
    adjusted = list(shares[:paid])
    freed = 0.0

    for index in range(paid):
        stale = ranked[index].stale_rounds
        if stale <= 0:
            continue
        retained = adjusted[index] * (1.0 - decay) ** stale
        freed += adjusted[index] - retained
        adjusted[index] = retained

    if freed <= 0.0:
        return tuple(adjusted)

    receivers = [i for i in range(paid) if ranked[i].stale_rounds <= 0]
    if not receivers:
        return tuple(adjusted)

    basis = sum(adjusted[i] for i in receivers)
    if basis <= 0.0:
        share = freed / len(receivers)
        for i in receivers:
            adjusted[i] += share
        return tuple(adjusted)

    for i in receivers:
        adjusted[i] += freed * adjusted[i] / basis
    return tuple(adjusted)


def allocate(
    candidates: Sequence[Candidate],
    previous: Sequence[str] = (),
    *,
    threshold: float = TRACK_THRESHOLD,
    min_rounds: int = MIN_ROUNDS_OBSERVED,
    epsilon: float = HYSTERESIS_EPSILON,
    paid_ranks: int = PAID_RANKS,
    rank_decay: float = RANK_DECAY,
    incumbent_decay: float = INCUMBENT_DECAY,
) -> dict[str, float]:
    survivors = eligible(candidates, threshold, min_rounds)
    if not survivors:
        return {}

    ranked = rank_with_hysteresis(survivors, previous, epsilon, paid_ranks)[:paid_ranks]
    shares = geometric_shares(paid_ranks, rank_decay)[: len(ranked)]
    shares = apply_incumbent_decay(shares, ranked, incumbent_decay)

    total = sum(shares)
    if total <= 0.0:
        return {}

    return {c.hotkey: s / total for c, s in zip(ranked, shares, strict=True)}


def held_ranks(allocation: Mapping[str, float]) -> list[str]:
    return [hotkey for hotkey, _ in sorted(allocation.items(), key=lambda kv: -kv[1])]
