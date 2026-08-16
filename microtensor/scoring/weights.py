from __future__ import annotations

from collections.abc import Mapping, Sequence

from microtensor.core.constants import (
    CLASS_WEIGHTS,
    CONCENTRATION_CAP_FRACTION,
    DECAY_RATE,
    RECOVERY_RATE,
)
from microtensor.core.tracks import get_track
from microtensor.scoring.schedule import Candidate, allocate


def normalise(weights: Mapping[str, float]) -> dict[str, float]:
    positive = {k: float(v) for k, v in weights.items() if v > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in positive.items()}


def origin_group(coldkey: str) -> str:
    return coldkey.strip()


def apply_concentration_cap(
    weights: Mapping[str, float],
    origins: Mapping[str, str],
    fraction: float = CONCENTRATION_CAP_FRACTION,
) -> dict[str, float]:
    paid = {k: v for k, v in weights.items() if v > 0.0}
    if not paid:
        return {}

    max_per_group = int(len(paid) * fraction)
    if max_per_group < 1:
        return dict(paid)

    grouped: dict[str, list[str]] = {}
    for hotkey in paid:
        group = origins.get(hotkey, "")
        if group:
            grouped.setdefault(group, []).append(hotkey)

    capped = dict(paid)
    for members in grouped.values():
        if len(members) <= max_per_group:
            continue
        ranked = sorted(members, key=lambda h: (-capped[h], h))
        for hotkey in ranked[max_per_group:]:
            capped[hotkey] = 0.0

    return normalise(capped)


def blend(
    new: Mapping[str, float],
    prior: Mapping[str, float],
    decay_rate: float = DECAY_RATE,
    recovery_rate: float = RECOVERY_RATE,
) -> dict[str, float]:
    keys = set(new) | set(prior)
    blended: dict[str, float] = {}

    for key in keys:
        target = float(new.get(key, 0.0))
        held = float(prior.get(key, 0.0))
        rate = recovery_rate if target > held else decay_rate
        value = held + rate * (target - held)
        if value > 1e-9:
            blended[key] = value

    return normalise(blended)


def competition_weights(
    candidates: Sequence[Candidate],
    previous: Sequence[str] = (),
    **kwargs: float | int,
) -> dict[str, float]:
    return allocate(candidates, previous, **kwargs)  # type: ignore[arg-type]


def class_weight(class_id: str, siblings: Sequence[str]) -> float:
    if not siblings:
        return 0.0
    if all(s in CLASS_WEIGHTS for s in siblings):
        total = sum(CLASS_WEIGHTS[s] for s in siblings)
        if total > 0.0:
            return CLASS_WEIGHTS[class_id] / total
    return 1.0 / len(siblings)


def combine_competitions(
    per_competition: Mapping[tuple[str, str], Mapping[str, float]],
) -> dict[str, float]:
    combined: dict[str, float] = {}
    siblings: dict[str, list[str]] = {}

    for track, cls in per_competition:
        siblings.setdefault(track, []).append(cls)

    for (track, cls), allocation in per_competition.items():
        if not allocation:
            continue
        track_share = get_track(track).emission_share
        class_share = track_share * class_weight(cls, siblings[track])
        for hotkey, fraction in allocation.items():
            combined[hotkey] = combined.get(hotkey, 0.0) + class_share * fraction

    return normalise(combined)


def to_uid_weights(
    weights: Mapping[str, float],
    uid_by_hotkey: Mapping[str, int],
) -> tuple[dict[int, float], list[str]]:
    uid_weights: dict[int, float] = {}
    dropped: list[str] = []

    for hotkey, weight in weights.items():
        uid = uid_by_hotkey.get(hotkey)
        if uid is None:
            dropped.append(hotkey)
            continue
        uid_weights[uid] = uid_weights.get(uid, 0.0) + weight

    total = sum(uid_weights.values())
    if total > 0.0:
        uid_weights = {uid: w / total for uid, w in uid_weights.items()}

    return uid_weights, dropped
