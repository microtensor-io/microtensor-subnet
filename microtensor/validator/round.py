from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from microtensor.chain.anchor import AnchorError, read_anchor
from microtensor.chain.metagraph import MetagraphSnapshot
from microtensor.chain.rounds import Round, round_for_block
from microtensor.chain.weights import quantise_weights
from microtensor.coordinator.report import Report
from microtensor.core.constants import (
    COORDINATOR_HOTKEY,
    MECHANISM_VERSION,
    MIN_SCORED_FRACTION,
    REFERENCE_COST_MS,
)
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
from microtensor.validator.client import CoordinatorRefused, SettlementRejected
from microtensor.validator.context import ValidatorContext
from microtensor.validator.coordinated import (
    ROUND_DRIFT,
    CoordinatorDisagrees,
    Mode,
    adopt_settlement,
    emit_reports,
    plan_round,
    to_report,
)
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
    entrants = [
        frontier.Entrant(
            key=row.hotkey,
            quality=row.score,
            cost=row.expected_ms,
            committed_at=row.committed_at,
            rounds_observed=row.rounds_observed,
            stale_rounds=row.stale_rounds,
        )
        for row in context.state.frontier_candidates(round_index, track, hardware_class)
    ]
    if not entrants:
        return

    members = frontier.frontier(frontier.to_points(frontier.eligible(entrants), REFERENCE_COST_MS))
    if not members:
        return

    roles = {p.hotkey: tuple(c.role for c in p.system.components) for p in participants}

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


def adopted(round_index: int, weights: dict[int, float]) -> Settlement:
    """Wrap the coordinator's canonical weights as this round's settlement.

    The vector is taken whole. A worker never merges the canonical weights with
    its own allocation: it measured a subset, so a merge would produce a vector
    that is neither the network's answer nor its own, and no other validator
    would submit the same thing.
    """
    return Settlement(
        round_index=round_index,
        per_competition={},
        combined={},
        capped={},
        blended={},
        vector=quantise_weights(weights),
    )


def run_round(context: ValidatorContext, round_: Round) -> RoundOutcome:
    """Run one round, and leave it in a terminal state whatever happens.

    Every deliberate cause routes through the inner abstain(). This wrapper is
    for the causes nobody enumerated: a sqlite error, a chain fault, a missing
    corpus key. Without it an unforeseen exception escapes to the loop, the
    round row stays open forever, and the loop re-enters the same round on every
    poll. An invariant that holds for the failures you thought of and breaks on
    the one you did not is not an invariant.
    """
    started = time.monotonic()

    try:
        snapshot = context.client.snapshot(refresh=True)
        block_hash = context.client.block_hash(round_.seed_block)
    except Exception as exc:
        reason = f"the chain could not be read, so the round never opened: {exc!r}"
        log.exception("round %d: %s", round_.index, reason)
        return RoundOutcome(
            round_index=round_.index,
            status=ABSTAINED,
            reason=reason,
            elapsed_seconds=time.monotonic() - started,
        )

    context.state.open_round(round_.index, round_.seed_block, block_hash)

    try:
        return _run_round(context, round_, snapshot, block_hash, started)
    except Exception as exc:
        reason = f"the round failed unexpectedly and was abandoned: {exc!r}"
        log.exception("round %d: %s", round_.index, reason)
        context.state.finish_round(round_.index, ABSTAINED, reason)
        return RoundOutcome(
            round_index=round_.index,
            status=ABSTAINED,
            reason=reason,
            elapsed_seconds=time.monotonic() - started,
        )


def _run_round(
    context: ValidatorContext,
    round_: Round,
    snapshot: MetagraphSnapshot,
    block_hash: str,
    started: float,
) -> RoundOutcome:
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

    if context.config.loopback:
        return _run_loopback_round(context, round_, snapshot, block_hash, started, abstain)

    # Looked for again each round rather than only at startup: a worker
    # brought up before an arena is live has none, and the coordinator serves
    # one the moment there is something to measure.
    if not context.refresh_corpora():
        return abstain(
            "no corpus is being served yet, so there is nothing to measure; "
            "this resolves when an arena is live and the coordinator serves its "
            "corpus, and is rechecked every round"
        )

    try:
        plan = plan_round(
            context.coordinator,
            anchor_source=lambda: read_anchor(context.client, COORDINATOR_HOTKEY),
        )
    except AnchorError as exc:
        return abstain(
            f"this validator cannot check the config against the chain, so it did not "
            f"measure: {exc}"
        )
    except SettlementRejected as exc:
        return abstain(
            f"the coordinator's competition config is not the one anchored on chain, "
            f"so this round was not measured: {exc}"
        )
    except CoordinatorRefused as exc:
        return abstain(f"the coordinator refused this validator: {exc}")
    except CoordinatorDisagrees as exc:
        return abstain(f"{ROUND_DRIFT}: {exc}")

    if plan.mode is Mode.HOLD:
        return abstain(f"holding the last settled vector: {plan.reason}")

    if plan.mode is Mode.IDLE:
        log.info("round %d: %s", round_.index, plan.reason)
        # No round open still has a settlement to adopt: the coordinator
        # settles a round with nothing to measure on the reserved hold alone,
        # minted on request. Asking for the chain-derived round keeps this
        # worker setting weights before an arena is live, through the same
        # verified adoption path.
        if not plan.round_index:
            plan = dataclasses.replace(plan, round_index=round_.index)
        if plan.round_index:
            weights, why = adopt_settlement(
                context.coordinator, plan.round_index, uid_by_hotkey=snapshot.uid_by_hotkey
            )
            if weights:
                settlement = adopted(round_.index, weights)
                ok, reason = publish_weights(context, settlement)
                if ok:
                    context.state.finish_round(round_.index, SETTLED, plan.reason)
                    return RoundOutcome(
                        round_index=round_.index,
                        status=SETTLED,
                        reason=plan.reason,
                        settlement=settlement,
                        elapsed_seconds=time.monotonic() - started,
                    )
        return abstain(f"idle without measurement: {plan.reason}")

    if plan.round_index != round_.index:
        return abstain(
            f"the coordinator is on round {plan.round_index} but this validator derived "
            f"{round_.index} from the chain; measuring across that gap would file reports "
            f"against one round and weights against another"
        )

    if not context.refresh_corpora():
        return abstain(
            "no corpus served yet; waiting for an arena to go live, rechecked every round"
        )

    try:
        require_engines()
        roster = discover(context, snapshot, round_, plan.allowlists)
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
    reports: list[Report] = []
    assigned = set(plan.systems)

    for track, hardware_class in context.competitions:
        participants = roster.for_competition(track, hardware_class)
        if assigned is not None:
            participants = tuple(
                p for p in participants if p.commitment.manifest_digest in assigned
            )
        if not participants:
            continue
        expected += len(participants)

        # The arena's numbers, not this build's. They describe the same model
        # class the latency ceilings do, so they travel with the anchored
        # config and a worker on an older build still runs the round its
        # peers are running.
        arena = plan.budgets.get((track, hardware_class))
        tasks = select(
            context.corpus(track),
            competition_seed(block_hash, track, hardware_class),
            hardware_class,
            budget=arena.tasks_per_round if arena else context.config.tasks_per_round,
            round_index=round_.index,
        )
        def publish(evaluation, participant, _plan=plan) -> None:
            if _plan.mode is not Mode.COORDINATED or context.coordinator is None:
                return
            report = to_report(
                evaluation,
                round_index=_plan.round_index,
                worker_hotkey=context.hotkey,
                system_digest=participant.commitment.manifest_digest,
                engine_version=MECHANISM_VERSION,
                corpus_version=context.config.corpus_version,
                corpus_digest=context.corpus_digest,
                components={
                    ref.role.value: ref.artifact_digest for ref in participant.system.components
                },
            )
            emit_reports(context.coordinator, [report], wallet=context.wallet)

        try:
            result = evaluate_competition(
                context,
                participants,
                tasks,
                cpu_seconds=arena.cpu_seconds_per_artifact if arena else 0,
                on_evaluated=publish,
            )
        except Abstain as exc:
            return abstain(str(exc), roster)
        scored += sum(1 for e in result.evaluations if e.measured is not None)

        price_components(context, round_.index, track, hardware_class, participants)

        if plan.mode is Mode.COORDINATED:
            reports.extend(
                to_report(
                    evaluation,
                    round_index=plan.round_index,
                    worker_hotkey=context.hotkey,
                    system_digest=participant.commitment.manifest_digest,
                    engine_version=MECHANISM_VERSION,
                    corpus_version=context.config.corpus_version,
                    corpus_digest=context.corpus_digest,
                    components={
                        ref.role.value: ref.artifact_digest for ref in participant.system.components
                    },
                )
                for evaluation, participant in zip(result.evaluations, participants, strict=True)
            )

    if expected and scored / expected < MIN_SCORED_FRACTION:
        return abstain(
            f"only {scored} of {expected} submissions were scored, below the "
            f"{MIN_SCORED_FRACTION:.0%} floor",
            roster,
        )

    if plan.mode is Mode.COORDINATED:
        sent, failure = emit_reports(context.coordinator, reports, wallet=context.wallet)
        if failure:
            return abstain(
                f"this validator measured {len(reports)} systems but only {sent} reports "
                f"reached the coordinator, so it did not help produce the settlement it "
                f"would be endorsing: {failure}",
                roster,
            )
        log.info("round %d: %d reports sent to the coordinator", plan.round_index, sent)

    weights, why = adopt_settlement(
        context.coordinator, plan.round_index, uid_by_hotkey=snapshot.uid_by_hotkey
    )
    if weights is None:
        return abstain(f"the canonical settlement was not adopted: {why}", roster)
    if not weights:
        return abstain("the coordinator published a settlement with no weights in it", roster)
    settlement = adopted(round_.index, weights)

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


def _run_loopback_round(
    context: ValidatorContext,
    round_: Round,
    snapshot: MetagraphSnapshot,
    block_hash: str,
    started: float,
    abstain: Callable[..., RoundOutcome],
) -> RoundOutcome:
    """The synthetic-chain harness measures and settles in one process.

    The only place a validator still measures without a coordinator. Loopback
    stands up a whole subnet locally, so there is no server to own the round;
    every real deployment holds instead.
    """
    try:
        require_engines()
        roster = discover(context, snapshot, round_, context.config.allowlists)
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
        scored += sum(1 for e in result.evaluations if e.measured is not None)

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
    return RoundOutcome(
        round_index=round_.index,
        status=SETTLED,
        participants=len(roster),
        scored=scored,
        settlement=settlement,
        elapsed_seconds=time.monotonic() - started,
    )
