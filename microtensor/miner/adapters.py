from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

log = logging.getLogger("microtensor.miner.adapters")


class LoopAdapter:
    """For a plain training loop.

    A participant calls `epoch` once per pass and the adapter works out
    throughput and hands the rest to the hook. Nothing here touches the model,
    the optimiser or the data, because none of that is this code's business.

        adapter = LoopAdapter(hook, samples_per_epoch=len(dataset))
        for epoch in range(epochs):
            loss = train_one_epoch(...)
            adapter.epoch(epoch, loss)
    """

    def __init__(self, hook: Any, samples_per_epoch: int = 0) -> None:
        self.hook = hook
        self.samples_per_epoch = samples_per_epoch
        self._last = time.monotonic()

    def epoch(self, epoch: int, loss: float | None = None, **metrics: float) -> None:
        now = time.monotonic()
        seconds = max(1e-6, now - self._last)
        self._last = now

        payload: dict[str, float] = {}
        if loss is not None:
            payload["loss"] = float(loss)
        if self.samples_per_epoch:
            payload["throughput"] = self.samples_per_epoch / seconds
        payload.update({k: float(v) for k, v in metrics.items() if v is not None})

        self.hook.on_epoch_end(epoch, payload)

    def failed(self, reason: str) -> None:
        self.hook.on_failure(reason)


class TrainerCallback:
    """For HuggingFace `Trainer`.

    Duck typed against the callback interface rather than subclassing it, so
    importing this module does not require transformers to be installed. A
    miner that has it gets a callback; a miner that does not still imports the
    rest of the package.

        trainer = Trainer(..., callbacks=[TrainerCallback(hook)])
    """

    def __init__(self, hook: Any) -> None:
        self.hook = hook
        self._epoch = 0

    def on_epoch_end(
        self, args: Any = None, state: Any = None, control: Any = None, **kw: Any
    ) -> Any:
        self._epoch = int(getattr(state, "epoch", self._epoch + 1) or self._epoch + 1)
        self.hook.on_epoch_end(self._epoch, self._metrics(state))
        return control

    def on_log(
        self, args: Any = None, state: Any = None, control: Any = None, **kw: Any
    ) -> Any:
        logs: Mapping[str, Any] = kw.get("logs") or {}
        if "loss" not in logs:
            return control

        metrics = {"loss": float(logs["loss"])}
        for key in ("learning_rate", "grad_norm", "train_samples_per_second"):
            if key in logs:
                metrics["throughput" if key.endswith("per_second") else key] = float(logs[key])

        self.hook.on_epoch_end(self._epoch, metrics)
        return control

    def _metrics(self, state: Any) -> dict[str, float]:
        history = list(getattr(state, "log_history", ()) or ())
        if not history:
            return {}

        latest = history[-1]
        metrics: dict[str, float] = {}
        for source, target in (
            ("loss", "loss"),
            ("train_samples_per_second", "throughput"),
            ("step", "step"),
        ):
            value = latest.get(source)
            if value is not None:
                try:
                    metrics[target] = float(value)
                except (TypeError, ValueError):
                    continue
        return metrics
