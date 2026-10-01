from __future__ import annotations

import logging
import typing
from collections.abc import Callable, Mapping
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

    serve_axon = getattr(bt, "axon", None) or getattr(bt, "Axon", None)
    if serve_axon is None:
        raise AxonUnavailable("this bittensor build exposes no axon class")
    axon = serve_axon(wallet=wallet, port=port)
    axon.attach(forward_fn=answer)
    axon.start()
    log.info("serving training status on port %d", port)
    return axon


DEFAULT_SYSTEM_PORT = 8091


def serve_system(
    wallet: Any,
    subtensor: Any,
    netuid: int,
    handlers: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
    validators: Callable[[], Mapping[str, float]],
    *,
    port: int = DEFAULT_SYSTEM_PORT,
    external_ip: str = "",
    external_port: int = 0,
) -> Any:
    try:
        import bittensor as bt
    except ImportError as exc:
        raise AxonUnavailable("the system axon needs bittensor: pip install \".[miner]\"") from exc
    from microtensor.chain.synapse import system_task

    task = system_task()

    def forward(synapse: Any) -> Any:
        handler = handlers.get(synapse.system)
        if handler is None:
            synapse.failure = f"this miner hosts no system named {synapse.system!r}"
            return synapse
        try:
            synapse.trace = handler(
                {
                    "round_index": synapse.round_index,
                    "task_ref": synapse.task_ref,
                    "prompt": synapse.prompt,
                    "inputs": dict(synapse.inputs or {}),
                }
            )
        except Exception as exc:
            synapse.failure = f"{type(exc).__name__}: {exc}"
        return synapse

    def caller(synapse: Any) -> str:
        return str(getattr(getattr(synapse, "dendrite", None), "hotkey", "") or "")

    def blacklist(synapse: Any) -> tuple[bool, str]:
        hotkey = caller(synapse)
        if not hotkey:
            return True, "the request carries no hotkey"
        if hotkey not in validators():
            return True, "only validators with a permit test this system"
        return False, "validator"

    def priority(synapse: Any) -> float:
        return float(validators().get(caller(synapse), 0.0))

    forward.__annotations__ = {"synapse": task, "return": task}
    blacklist.__annotations__ = {"synapse": task, "return": typing.Tuple[bool, str]}  # noqa: UP006
    priority.__annotations__ = {"synapse": task, "return": float}

    options: dict[str, Any] = {"wallet": wallet, "port": port}
    if external_ip:
        options["external_ip"] = external_ip
    if external_port:
        options["external_port"] = external_port
    axon = (getattr(bt, "Axon", None) or bt.axon)(**options)
    axon.attach(forward_fn=forward, blacklist_fn=blacklist, priority_fn=priority)
    axon.serve(netuid=netuid, subtensor=subtensor)
    axon.start()
    log.info("serving %s on axon port %d", ", ".join(sorted(handlers)), port)
    return axon
