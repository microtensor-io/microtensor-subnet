from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from microtensor.chain.anchor import ConfigAnchor
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

HELD = (
    "the coordinator is unreachable, so nothing was measured; the last settled "
    "weight vector keeps flowing until contact is restored"
)
REFUSED = "the published settlement did not recompute, so it was not submitted"
NO_ASSIGNMENT_DOC = "the coordinator returned no assignment for this worker and round"
ROUND_DRIFT = "the coordinator disagrees with itself about which round is open"
STALE_ANCHOR = (
    "the coordinator's anchor on chain is for a different round than the one it is serving"
)


class CoordinatorDisagrees(RuntimeError):
    """The coordinator gave two different answers about the open round."""


class Mode(str, Enum):
    """What this worker is doing this round.

    Three states, not two, and the distinction between the last two is the
    point. IDLE means the coordinator answered and had nothing to measure,
    which is the intended state between rounds. HOLD means we could not reach
    it: an outage pauses the network rather than breaking it, and the last
    settled vector keeps flowing throughout. Collapsing them would let an
    outage read as an ordinary quiet round.
    """

    COORDINATED = "coordinated"
    IDLE = "idle"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class Plan:
    """How this worker will spend the round."""

    mode: Mode
    round_index: int = 0
    systems: tuple[str, ...] = ()
    config_hash: str = ""
    reason: str = ""
    allowlists: dict[tuple[str, str], frozenset[str]] = field(default_factory=dict)

    @property
    def coordinated(self) -> bool:
        """Whether the coordinator is answering, with or without work for us."""
        return self.mode in (Mode.COORDINATED, Mode.IDLE)

    @property
    def measures(self) -> bool:
        return self.mode is Mode.COORDINATED


def allowlists_from(config: Mapping[str, Any]) -> dict[tuple[str, str], frozenset[str]]:
    """Per-arena base-model allowlists, from the config the chain anchored.

    Read from the verified config rather than a constant, so adding a
    permitted base model is an operator action rather than a release. An
    arena absent here has no allowlist and therefore admits nothing; the gate
    treats that as "nothing permitted yet", never as "anything goes".
    """
    out: dict[tuple[str, str], frozenset[str]] = {}
    for key, value in dict(config.get("arenas", {})).items():
        track, _, hardware_class = str(key).partition("/")
        if not track or not hardware_class:
            continue
        out[(track, hardware_class)] = frozenset(
            str(entry) for entry in dict(value).get("allowed_base_models", [])
        )
    return out


def plan_round(
    client: CoordinatorClient | None,
    *,
    anchor_source: Callable[[], ConfigAnchor | None] | None = None,
    require_anchor: bool = True,
) -> Plan:
    """Ask the coordinator for work, or hold what already settled.

    Two failures, deliberately handled differently. An unreachable coordinator
    holds: no measurement, the last settled vector keeps flowing, because a
    round now exists only on the server and a validator with no coordinator
    has no round to run. A config that does not match its on-chain anchor does
    not hold: it aborts the round on this worker, because measuring against
    ceilings nobody committed to produces numbers that look valid and are not.

    The anchor comes from `anchor_source`, which reads the chain. It must not
    come from the coordinator: checking a served config against a served hash
    is self-consistent by construction and proves nothing about what anyone
    committed to. Without a source there is nothing to check against, and the
    round is refused rather than admitted.
    """
    if client is None:
        return Plan(mode=Mode.HOLD, reason="no coordinator is configured; holding weights")

    try:
        current = client.current_round()
    except CoordinatorUnreachable as exc:
        log.warning("%s (%s)", HELD, exc)
        return Plan(mode=Mode.HOLD, reason=HELD)

    if not current:
        return Plan(mode=Mode.IDLE, reason="no round is open; the network is idle")

    round_index = int(current["round"])

    if require_anchor:
        anchor = anchor_source() if anchor_source is not None else None
        if anchor is not None and anchor.round_index != round_index:
            raise SettlementRejected(
                f"{STALE_ANCHOR}: anchored round {anchor.round_index} against served "
                f"round {round_index}"
            )
        verify_config(current.get("config", {}), anchor.config_hash if anchor else "")

    try:
        assigned_round, systems = client.assignment()
    except CoordinatorUnreachable as exc:
        log.warning("%s (%s)", HELD, exc)
        return Plan(mode=Mode.HOLD, reason=HELD)

    config_hash = str(current.get("config_hash", ""))
    allowlists = allowlists_from(dict(current.get("config", {})))

    if systems is not None and assigned_round is not None and assigned_round != round_index:
        raise CoordinatorDisagrees(
            f"the coordinator's open round is {round_index} but its assignment is for "
            f"{assigned_round}"
        )

    if systems is None:
        log.warning("%s (round %d)", NO_ASSIGNMENT_DOC, round_index)
        return Plan(
            mode=Mode.HOLD,
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
            allowlists=allowlists,
        )

    return Plan(
        mode=Mode.COORDINATED,
        round_index=round_index,
        systems=systems,
        config_hash=config_hash,
        allowlists=allowlists,
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
    corpus_digest: str = "",
    components: dict[str, str] | None = None,
) -> Report:
    """One measurement, in the shape the coordinator reconciles.

    `components` names each role's artifact digest from the manifest the
    worker fetched. Deterministic across honest workers, and it is what lets
    a settled winner's parts be promoted into a baseline later: the frontier
    row that carries no component digests cannot ratchet.
    """
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
        components=dict(components or {}),
        ablation=ablation,
        device_profile=measured.device_profile if measured else "",
        conforming=bool(measured.conforming) if measured else False,
        engine_version=engine_version,
        corpus_version=corpus_version,
        corpus_digest=corpus_digest,
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
            return sent, (f"the coordinator refused the report for {report.system_digest}: {exc}")
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
            cross_check(published, uid_by_hotkey)
    except SettlementRejected as exc:
        log.error("%s: %s", REFUSED, exc)
        return None, f"{REFUSED}: {exc}"

    weights = {int(uid): float(value) for uid, value in published.get("weights", {}).items()}
    return weights, ""
