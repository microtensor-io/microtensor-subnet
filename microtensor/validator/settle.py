from __future__ import annotations

import logging
from dataclasses import dataclass, field

from microtensor.chain.metagraph import MetagraphSnapshot
from microtensor.chain.weights import WeightVector, quantise_weights
from microtensor.scoring.schedule import allocate, held_ranks
from microtensor.scoring.weights import (
    apply_concentration_cap,
    blend,
    combine_competitions,
    origin_group,
    to_uid_weights,
)
from microtensor.validator.context import ValidatorContext

log = logging.getLogger("microtensor.validator.settle")


@dataclass(frozen=True, slots=True)
class Settlement:
    round_index: int
    per_competition: dict[tuple[str, str], dict[str, float]]
    combined: dict[str, float]
    capped: dict[str, float]
    blended: dict[str, float]
    vector: WeightVector
    dropped: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return self.vector.is_empty

    @property
    def paid(self) -> int:
        return len(self.blended)


def allocations(
    context: ValidatorContext, round_index: int
) -> dict[tuple[str, str], dict[str, float]]:
    per_competition: dict[tuple[str, str], dict[str, float]] = {}

    for track, hardware_class in context.competitions:
        candidates = context.state.candidates(round_index, track, hardware_class)
        if not candidates:
            log.info("%s/%s: no admitted candidate this round", track, hardware_class)
            continue

        previous = context.state.holders(track, hardware_class)
        allocation = allocate(candidates, previous)
        if not allocation:
            log.info("%s/%s: nobody cleared the track threshold", track, hardware_class)
            continue

        per_competition[(track, hardware_class)] = allocation
        context.state.set_holders(track, hardware_class, held_ranks(allocation), round_index)

    return per_competition


def settle(
    context: ValidatorContext, round_index: int, snapshot: MetagraphSnapshot
) -> Settlement:
    per_competition = allocations(context, round_index)
    combined = combine_competitions(per_competition)

    origins = {
        hotkey: origin_group(address) for hotkey, address in snapshot.addresses().items()
    }
    capped = apply_concentration_cap(combined, origins)
    if len(capped) < len(combined):
        log.info(
            "concentration cap removed %d of %d positions",
            len(combined) - len(capped),
            len(combined),
        )

    blended = blend(capped, context.state.last_weights())
    uid_weights, dropped = to_uid_weights(blended, snapshot.uid_by_hotkey)
    if dropped:
        log.warning("dropping %d weights for hotkeys absent from the metagraph", len(dropped))

    return Settlement(
        round_index=round_index,
        per_competition=per_competition,
        combined=combined,
        capped=capped,
        blended=blended,
        vector=quantise_weights(uid_weights),
        dropped=tuple(dropped),
    )


def publish(context: ValidatorContext, settlement: Settlement) -> tuple[bool, str]:
    if settlement.is_empty:
        return False, "no weight survived settlement; nothing to publish"

    if context.config.dry_run:
        log.info(
            "dry run: would set %d weights totalling %d",
            len(settlement.vector),
            settlement.vector.total,
        )
        context.state.record_weights(settlement.round_index, settlement.blended)
        return True, "dry run"

    ok, reason = context.client.set_weights(settlement.vector)
    if ok:
        context.state.record_weights(settlement.round_index, settlement.blended)
        log.info("round %d: %d weights published", settlement.round_index, len(settlement.vector))
    else:
        log.error("round %d: weights rejected — %s", settlement.round_index, reason)
    return ok, reason
