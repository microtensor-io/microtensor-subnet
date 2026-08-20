from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("microtensor.miner.axon")

DEFAULT_PORT = 8091


class AxonUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class StatusSource:
    """What the axon answers with, read from the daemon in place.

    Holds a reference rather than a copy so an answer describes where the run is
    when the question arrives, not where it was when the axon started.
    """

    daemon: Any

    def snapshot(self) -> dict[str, Any]:
        reporter = getattr(self.daemon, "reporter", None)
        role = getattr(reporter, "role", None)

        return {
            "phase": self.daemon.phase.value,
            "role": role.value if role else None,
            "epoch": None,
            "loss": None,
            "elapsed_s": getattr(reporter, "elapsed_s", 0) if reporter else 0,
            "eta_s": None,
        }


def serve(daemon: Any, port: int = DEFAULT_PORT, wallet: Any = None) -> Any:
    """Answer on-demand questions about this run.

    A rare call, not a stream: someone clicking into one miner on a dashboard,
    or an operator looking at a run that appears stuck. The steady flow is the
    push to the coordinator, because a miner reporting unprompted is acting as a
    client and an axon is a server. Pushing through an axon would be a call in
    the wrong direction.
    """
    try:
        import bittensor as bt
    except ImportError as exc:
        raise AxonUnavailable(
            "the miner axon needs bittensor: pip install \".[miner]\""
        ) from exc

    source = StatusSource(daemon)

    class TrainingStatus(bt.Synapse):  # type: ignore[misc]
        phase: str = ""
        role: str | None = None
        epoch: int | None = None
        loss: float | None = None
        elapsed_s: int = 0
        eta_s: int | None = None

    def answer(synapse: TrainingStatus) -> TrainingStatus:
        for key, value in source.snapshot().items():
            setattr(synapse, key, value)
        return synapse

    axon = bt.axon(wallet=wallet, port=port)
    axon.attach(forward_fn=answer)
    axon.start()
    log.info("serving training status on port %d", port)
    return axon
