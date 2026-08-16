from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from microtensor.chain.rounds import Round, round_for_block
from microtensor.core.constants import MIN_SCORED_FRACTION
from microtensor.store.state import ABSTAINED, SETTLED
from microtensor.tasks.selection import competition_seed, select
from microtensor.validator.context import ValidatorContext
from microtensor.validator.discover import Roster, discover
from microtensor.validator.evaluate import Abstain, evaluate_competition, require_engines
from microtensor.validator.settle import Settlement, settle
from microtensor.validator.settle import publish as publish_weights

log = logging.getLogger("microtensor.validator.round")


@dataclass(frozen=True, slots=True)
class RoundOutcome:
    round_index: int
    status: str
    reason: str = ""
    participants: int = 0
    scored: int = 0
    settlement: Settlement | None = None
    elapsed_seconds: float = 0.0

    @property
    def settled(self) -> bool:
        return self.status == SETTLED


def current_round(context: ValidatorContext) -> Round:
    return round_for_block(
        context.client.block(),
        length=context.config.round_blocks,
        genesis=context.config.genesis_block,
    )


def run_round(context: ValidatorContext, round_: Round) -> RoundOutcome:
    started = time.monotonic()
    snapshot = context.client.snapshot(refresh=True)
    block_hash = context.client.block_hash(round_.seed_block)
    context.state.open_round(round_.index, round_.seed_block, block_hash)

    def abstain(reason: str, roster: Roster | None = None) -> RoundOutcome:
        log.warning("round %d abstaining: %s", round_.index, reason)
        context.state.finish_round(round_.index, ABSTAINED, reason)
        return RoundOutcome(
            round_index=round_.index,
            status=ABSTAINED,
            reason=reason,
            participants=len(roster) if roster else 0,
            elapsed_seconds=time.monotonic() - started,
        )

    try:
        require_engines()
        roster = discover(context, snapshot, round_)
    except Abstain as exc:
        return abstain(str(exc))

    if context.config.degraded:
        return abstain(
            "degraded host: the cpu limit does not bind, so budgets cannot be "
            "enforced; abstain-only until the host is fixed",
            roster,
        )

    if not roster.participants:
        return abstain("no admissible submission this round", roster)

    scored = 0
    expected = 0
    for track, hardware_class in context.competitions:
        participants = roster.for_competition(track, hardware_class)
        if not participants:
            continue
        expected += len(participants)

        tasks = select(
            context.corpus(track),
            competition_seed(block_hash, track, hardware_class),
            hardware_class,
            budget=context.config.tasks_per_round,
            round_index=round_.index,
        )
        try:
            result = evaluate_competition(context, participants, tasks)
        except Abstain as exc:
            return abstain(str(exc), roster)
        scored += len(result)

    if expected and scored / expected < MIN_SCORED_FRACTION:
        return abstain(
            f"only {scored} of {expected} submissions were scored, below the "
            f"{MIN_SCORED_FRACTION:.0%} floor",
            roster,
        )

    settlement = settle(context, round_.index, snapshot)

    if settlement.is_empty:
        reason = "evaluated cleanly, but no artifact is eligible for emission yet"
        log.info("round %d: %s", round_.index, reason)
        context.state.finish_round(round_.index, SETTLED, reason)
        return RoundOutcome(
            round_index=round_.index,
            status=SETTLED,
            reason=reason,
            participants=len(roster),
            scored=scored,
            settlement=settlement,
            elapsed_seconds=time.monotonic() - started,
        )

    ok, reason = publish_weights(context, settlement)
    if not ok:
        return abstain(f"weights were not published: {reason}", roster)

    context.state.finish_round(round_.index, SETTLED)
    elapsed = time.monotonic() - started
    log.info(
        "round %d settled in %.1fs: %d participants, %d scored, %d weights",
        round_.index,
        elapsed,
        len(roster),
        scored,
        len(settlement.vector),
    )
    return RoundOutcome(
        round_index=round_.index,
        status=SETTLED,
        participants=len(roster),
        scored=scored,
        settlement=settlement,
        elapsed_seconds=elapsed,
    )
