from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from microtensor.chain.rounds import Round, round_for_block
from microtensor.core.constants import MIN_SCORED_FRACTION, REFERENCE_COST_MS, SYSTEMS_ENABLED
from microtensor.core.protocol import Role
from microtensor.provenance.record import ProvenanceUnavailable
from microtensor.scoring import frontier
from microtensor.store.state import ABSTAINED, SETTLED
from microtensor.tasks.selection import competition_seed, select
from microtensor.validator.ablate import (
    BaselineStore,
    Measured,
    OutputCache,
    contributions,
)
from microtensor.validator.context import ValidatorContext
from microtensor.validator.discover import Participant, Roster, discover
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


def price_components(
    context: ValidatorContext,
    round_index: int,
    track: str,
    hardware_class: str,
    participants: Sequence[Participant],
) -> None:
    """Divide each frontier system's value between the components that made it.

    Deliberately off the critical path. A system's emission comes from its own
    exclusive hypervolume, and this only splits that value, so a failure here
    must cost the round nothing.

    The frontier is rebuilt from the same rows the allocation reads, not from
    a fresh set, because a split computed against a different frontier would
    describe a payout nobody received.
    """
    if not SYSTEMS_ENABLED:
        return

    entrants = [
        frontier.Entrant(
            key=row.hotkey,
            quality=row.score,
            cost=row.expected_ms or REFERENCE_COST_MS,
            committed_at=row.committed_at,
            rounds_observed=row.rounds_observed,
            stale_rounds=row.stale_rounds,
        )
        for row in context.state.frontier_candidates(round_index, track, hardware_class)
    ]
    if not entrants:
        return

    members = frontier.frontier(
        frontier.to_points(frontier.eligible(entrants), REFERENCE_COST_MS)
    )
    if not members:
        return

    roles = {
        p.hotkey: tuple(c.role for c in p.system.components) for p in participants
    }

    try:
        priced = contributions(
            [p.hotkey for p in participants],
            roles,
            members,
            BaselineStore(root=context.config.corpus_dir),
            rerun=_no_executor,
            cost_ceiling=REFERENCE_COST_MS,
        )
        context.state.record_contributions(round_index, track, hardware_class, priced)
    except Exception as exc:
        log.warning("%s/%s component settlement failed: %s", track, hardware_class, exc)


def _no_executor(
    key: str, role: Role, digest: str, path: Path, cache: OutputCache
) -> Measured | None:
    """No baseline is published, so no substitution can be executed yet.

    Reached only if ROLE_BASELINES is populated before the substitution
    executor lands, in which case reporting null is correct: the network cannot
    measure the ablation, and null is not zero.
    """
    log.warning(
        "baseline %s is published but no substitution executor exists; "
        "contribution for %s reports null",
        role.value,
        key,
    )
    return None


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
    except ProvenanceUnavailable as exc:
        return abstain(
            f"the training run store is unreachable, so the participant set cannot be "
            f"agreed with other validators: {exc}"
        )

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

        price_components(context, round_.index, track, hardware_class, participants)

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
