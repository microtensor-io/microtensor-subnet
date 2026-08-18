from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from microtensor.coordinator.report import CostBlock, QualityBlock, Report
from microtensor.core.protocol import Evaluation
from microtensor.validator.client import (
    CoordinatorClient,
    CoordinatorUnreachable,
    SettlementRejected,
    cross_check,
    verify_config,
    verify_settlement,
)

log = logging.getLogger("microtensor.validator")

FELL_BACK = (
    "the coordinator is unreachable, so this round was measured and settled "
    "standalone; weights are this validator's own"
)
REFUSED = "the published settlement did not recompute, so it was not submitted"
NO_ASSIGNMENT_DOC = "the coordinator returned no assignment for this worker and round"
ROUND_DRIFT = "the coordinator disagrees with itself about which round is open"


class CoordinatorDisagrees(RuntimeError):
    """The coordinator gave two different answers about the open round."""


class Mode(str, Enum):
    """What this worker is doing this round.

    Three states, not two, and the distinction between the last two is the
    point. IDLE means the coordinator answered and had nothing for us, which is
    an ordinary round. STANDALONE means we could not reach it. Collapsing them
    would let an outage read as an empty assignment, so the worker would sit out
    the round believing everything was fine.
    """

    COORDINATED = "coordinated"
    IDLE = "idle"
    STANDALONE = "standalone"


@dataclass(frozen=True, slots=True)
class Plan:
    """How this worker will spend the round."""

    mode: Mode
    round_index: int = 0
    systems: tuple[str, ...] = ()
    config_hash: str = ""
    reason: str = ""

    @property
    def standalone(self) -> bool:
        return self.mode is Mode.STANDALONE

    @property
    def coordinated(self) -> bool:
        """Whether the coordinator is answering, with or without work for us."""
        return self.mode in (Mode.COORDINATED, Mode.IDLE)

    @property
    def measures(self) -> bool:
        return self.mode in (Mode.COORDINATED, Mode.STANDALONE)


def plan_round(
    client: CoordinatorClient | None,
    *,
    require_anchor: bool = True,
) -> Plan:
    """Ask the coordinator for work, or decide to go it alone.

    Two failures, deliberately handled differently. An unreachable coordinator
    falls back, because the subnet must keep setting weights. A config that does
    not match its on-chain anchor does not fall back: it aborts the round on this
    worker, because measuring against ceilings nobody committed to produces
    numbers that look valid and are not.
    """
    if client is None:
        return Plan(mode=Mode.STANDALONE, reason="no coordinator is configured")

    try:
        current = client.current_round()
    except CoordinatorUnreachable as exc:
        log.warning("%s (%s)", FELL_BACK, exc)
        return Plan(mode=Mode.STANDALONE, reason=FELL_BACK)

    if not current:
        return Plan(mode=Mode.STANDALONE, reason="the coordinator has opened no round")

    if require_anchor:
        verify_config(current.get("config", {}), str(current.get("config_hash", "")))

    try:
        assigned_round, systems = client.assignment()
    except CoordinatorUnreachable as exc:
        log.warning("%s (%s)", FELL_BACK, exc)
        return Plan(mode=Mode.STANDALONE, reason=FELL_BACK)

    round_index = int(current["round"])
    config_hash = str(current.get("config_hash", ""))

    if systems is not None and assigned_round is not None and assigned_round != round_index:
        # The two endpoints disagree about which round is open. Measuring under
        # that disagreement puts reports on one round and the settlement on
        # another, so neither is what this worker actually did.
        raise CoordinatorDisagrees(
            f"the coordinator's open round is {round_index} but its assignment is for "
            f"{assigned_round}"
        )

    if systems is None:
        # The endpoint had no assignment document for us at all. That is not the
        # same as being told we have nothing to do, and it is not a transport
        # failure either, so it is neither IDLE nor a silent fall-back.
        log.warning("%s (round %d)", NO_ASSIGNMENT_DOC, round_index)
        return Plan(
            mode=Mode.STANDALONE,
            round_index=round_index,
            config_hash=config_hash,
            reason=NO_ASSIGNMENT_DOC,
        )

    if not systems:
        log.info("round %d: the coordinator assigned this worker nothing", round_index)
        return Plan(
            mode=Mode.IDLE,
            round_index=round_index,
            config_hash=config_hash,
            reason="assigned no systems this round",
        )

    return Plan(
        mode=Mode.COORDINATED,
        round_index=round_index,
        systems=systems,
        config_hash=config_hash,
    )


def to_report(
    evaluation: Evaluation,
    *,
    round_index: int,
    worker_hotkey: str,
    system_digest: str,
    engine_version: str,
    corpus_version: str,
    ablation: dict[str, float] | None = None,
) -> Report:
    """One measurement, in the shape the coordinator reconciles."""
    measured = evaluation.measured
    envelope: dict[str, dict[str, Any]] = {}
    if measured is not None:
        envelope["front"] = {
            "size_bytes": float(measured.size_bytes),
            "peak_rss_bytes": float(measured.peak_rss_bytes),
            "ttft_p50_ms": float(measured.ttft_p50_ms),
            "ttft_p95_ms": float(measured.ttft_p95_ms),
            "tokens_per_second": float(measured.tokens_per_second),
        }

    return Report(
        round_index=round_index,
        worker_hotkey=worker_hotkey,
        system_digest=system_digest,
        quality=QualityBlock(
            rotating=evaluation.score_rotating,
            fixed=evaluation.score_fixed,
            combined=evaluation.score_combined,
        ),
        resolve_rate=evaluation.resolve_rate,
        cost=CostBlock(
            expected_ms=evaluation.expected_ms,
            expected_j=evaluation.expected_j or None,
        ),
        envelope=envelope,
        ablation=ablation,
        device_profile=measured.device_profile if measured else "",
        conforming=bool(measured.conforming) if measured else False,
        engine_version=engine_version,
        corpus_version=corpus_version,
    )


def emit_reports(
    client: CoordinatorClient,
    reports: Sequence[Report],
    wallet: Any = None,
) -> tuple[int, str]:
    """Send every measurement, and say plainly if any did not land.

    Returns the number sent and a failure reason. A worker whose reports did not
    all arrive has not contributed to the settlement it would otherwise endorse,
    so the caller abstains rather than adopting: voting for a number your own
    measurements are missing from is worse than not voting.

    Emission is deliberately all-or-nothing at the round level. A partial send is
    a failure, not a partial success, because the settlement will be computed
    without the missing reports and this worker knows it is incomplete.
    """
    sent = 0
    for report in reports:
        signed = report
        if wallet is not None:
            from microtensor.chain.wallet import sign_payload

            signed = report.signed_with(sign_payload(wallet, report.body()))
        try:
            client.submit(signed)
        except CoordinatorUnreachable as exc:
            return sent, (
                f"only {sent} of {len(reports)} reports reached the coordinator "
                f"before it became unreachable: {exc}"
            )
        except Exception as exc:
            return sent, (
                f"the coordinator refused the report for {report.system_digest}: {exc}"
            )
        sent += 1

    return sent, ""


def adopt_settlement(
    client: CoordinatorClient,
    round_index: int,
    catalogue: dict[str, Any] | None = None,
    uid_by_hotkey: Mapping[str, int] | None = None,
) -> tuple[dict[int, float] | None, str]:
    """Fetch the canonical settlement and recompute it before adopting.

    Returns the weights only if they reproduce from the published reports. A
    worker that skips this is a relay, and a network of relays has no consensus:
    a compromised coordinator would publish whatever it liked and every worker
    would sign it.
    """
    try:
        published = client.settlement(round_index)
    except CoordinatorUnreachable as exc:
        return None, f"the settlement could not be fetched: {exc}"

    if published is None:
        return None, "the coordinator has published no settlement for this round yet"

    try:
        reports = client.reports(round_index)
    except CoordinatorUnreachable as exc:
        return None, f"the published reports could not be fetched: {exc}"

    try:
        verify_settlement(published, reports, catalogue)
        if uid_by_hotkey is not None:
            # Recomputation proves the weights follow from the published inputs.
            # This is the part that checks the inputs themselves: every miner
            # named must hold the uid the metagraph gives it, so emission cannot
            # be redirected to a uid of the coordinator's choosing.
            cross_check(published, uid_by_hotkey)
    except SettlementRejected as exc:
        log.error("%s: %s", REFUSED, exc)
        return None, f"{REFUSED}: {exc}"

    weights = {
        int(uid): float(value) for uid, value in published.get("weights", {}).items()
    }
    return weights, ""
