from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from microtensor.validator.client import (
    CoordinatorClient,
    CoordinatorUnreachable,
    SettlementRejected,
    verify_config,
    verify_settlement,
)

log = logging.getLogger("microtensor.validator")

FELL_BACK = (
    "the coordinator is unreachable, so this round was measured and settled "
    "standalone; weights are this validator's own"
)
REFUSED = "the published settlement did not recompute, so it was not submitted"


@dataclass(frozen=True, slots=True)
class Plan:
    """How this worker will spend the round.

    `standalone` is not a failure state on its own. It is the mode the network
    runs in when there is no coordinator at all, and the mode every worker
    falls back to when the coordinator is down, which is what keeps a
    coordinator outage from halting the subnet.
    """

    standalone: bool
    round_index: int = 0
    systems: tuple[str, ...] = ()
    config_hash: str = ""
    reason: str = ""

    @property
    def coordinated(self) -> bool:
        return not self.standalone


def plan_round(
    client: CoordinatorClient | None,
    *,
    require_anchor: bool = True,
) -> Plan:
    """Ask the coordinator for work, or decide to go it alone.

    Two distinct failures, deliberately handled differently. An unreachable
    coordinator falls back, because the subnet must keep setting weights. A
    config that does not match its on-chain anchor does not fall back: it
    aborts the round on this worker, because measuring against ceilings nobody
    committed to produces numbers that look valid and are not.
    """
    if client is None:
        return Plan(standalone=True, reason="no coordinator is configured")

    try:
        current = client.current_round()
    except CoordinatorUnreachable as exc:
        log.warning("%s (%s)", FELL_BACK, exc)
        return Plan(standalone=True, reason=FELL_BACK)

    if not current:
        return Plan(standalone=True, reason="the coordinator has opened no round")

    if require_anchor:
        verify_config(current.get("config", {}), str(current.get("config_hash", "")))

    try:
        systems = client.assignment()
    except CoordinatorUnreachable as exc:
        log.warning("%s (%s)", FELL_BACK, exc)
        return Plan(standalone=True, reason=FELL_BACK)

    return Plan(
        standalone=False,
        round_index=int(current["round"]),
        systems=systems,
        config_hash=str(current.get("config_hash", "")),
    )


def adopt_settlement(
    client: CoordinatorClient,
    round_index: int,
    catalogue: dict[str, Any],
) -> dict[int, float] | None:
    """Fetch the canonical settlement and recompute it before adopting.

    Returns the weights only if they reproduce from the published reports. A
    worker that skips this is a relay, and a network of relays has no
    consensus: a compromised coordinator would publish whatever it liked and
    every worker would sign it.
    """
    published = client.settlement(round_index)
    if published is None:
        return None

    reports = client.reports(round_index)
    try:
        verify_settlement(published, reports, catalogue)
    except SettlementRejected as exc:
        log.error("%s: %s", REFUSED, exc)
        return None

    return {int(uid): float(value) for uid, value in published.get("weights", {}).items()}
