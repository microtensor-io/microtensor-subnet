from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable
from types import FrameType

from microtensor.chain.rounds import Round, round_for_block
from microtensor.chain.weights import quantise_weights
from microtensor.core.constants import (
    BLOCK_TIME_SECONDS,
    POLL_INTERVAL_SECONDS,
    WEIGHT_REFRESH_BLOCKS,
)
from microtensor.scoring.weights import to_uid_weights
from microtensor.update.loop import UpdateChecker
from microtensor.validator.context import ValidatorContext
from microtensor.validator.round import RoundOutcome, run_round

log = logging.getLogger("microtensor.validator.loop")


class Stopped(Exception):
    pass


class Restart(Exception):
    pass


class RoundLoop:
    def __init__(
        self,
        context: ValidatorContext,
        *,
        poll_seconds: int = POLL_INTERVAL_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        updater: UpdateChecker | None = None,
    ) -> None:
        self.context = context
        self.poll_seconds = poll_seconds
        self._sleep = sleep
        self._running = False
        self.rounds_run = 0
        self._last_weight_block = 0
        self.updater = updater
        self.restarting = False

    def stop(self, *_: object) -> None:
        if self._running:
            log.info("stop requested; finishing the current round then exiting")
        self._running = False

    def install_signal_handlers(self) -> None:
        def handler(signum: int, frame: FrameType | None) -> None:
            self.stop()

        for name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is not None:
                signal.signal(sig, handler)

    def round_at_head(self) -> Round:
        return round_for_block(
            self.context.client.block(),
            length=self.context.config.round_blocks,
            genesis=self.context.config.genesis_block,
        )

    def consider_update(self, round_: Round, block: int) -> None:
        if self.updater is None:
            return
        applied = self.updater.step(round_, block)
        if applied is not None and applied.restart_required:
            self.restarting = True
            raise Restart(applied.release.tag)

    def refresh_weights(self, block: int) -> None:
        """Re-submit the standing vector so this validator is never silent.

        A round takes a day; an epoch is 360 blocks. A validator that only
        submits when a round completes stops setting weights for twenty epochs
        at a time, and a copier submitting the same stale vector every epoch
        collects the dividends that silence gives up. Nothing here recomputes
        anything: it republishes what the last round already settled.
        """
        if block - self._last_weight_block < WEIGHT_REFRESH_BLOCKS:
            return

        standing = self.context.state.last_weights()
        if not standing:
            standing = self._coordinator_weights()
        if not standing:
            return

        snapshot = self.context.client.snapshot()
        uid_weights, _ = to_uid_weights(standing, snapshot.uid_by_hotkey)
        vector = quantise_weights(uid_weights)
        if vector.is_empty:
            return

        if self.context.config.dry_run:
            self._last_weight_block = block
            return

        ok, reason = self.context.client.set_weights(vector)
        self._last_weight_block = block
        if ok:
            log.info("refreshed %d weights at block %d", len(vector.uids), block)
        else:
            log.warning("weight refresh rejected at block %d: %s", block, reason)

    def _coordinator_weights(self) -> dict[str, float]:
        client = getattr(self.context, "coordinator", None)
        if client is None:
            return {}
        try:
            found = client.weights()
        except Exception as exc:
            log.warning("the coordinator's vector could not be read: %s", exc)
            return {}
        if not found:
            return {}
        uids = self.context.client.snapshot().uid_by_hotkey
        by_uid = {uid: hotkey for hotkey, uid in uids.items()}
        adopted = {by_uid[uid]: value for uid, value in found.items() if uid in by_uid}
        if adopted:
            log.info("adopting the coordinator's vector of %d weights", len(adopted))
        return adopted

    def wait_for_close(self, round_: Round) -> None:
        from microtensor.cli.common import reclaim_logging

        while self._running:
            block = self.context.client.block()
            reclaim_logging()
            if block >= round_.close_block:
                return

            self.consider_update(round_, block)
            try:
                self.refresh_weights(block)
            except Exception as exc:
                log.warning("weight refresh failed, continuing the round: %s", exc)
            remaining = round_.close_block - block
            delay = min(self.poll_seconds, max(1, remaining * BLOCK_TIME_SECONDS))
            log.info("round %d open: %d blocks until submissions close", round_.index, remaining)
            self._sleep(delay)
        raise Stopped

    def step(self) -> RoundOutcome | None:
        round_ = self.round_at_head()

        if self.context.state.is_settled(round_.index):
            log.info("round %d is already settled; waiting for the next", round_.index)
            try:
                self.refresh_weights(self.context.client.block())
            except Exception as exc:
                log.warning("weight refresh failed while idle: %s", exc)
            self._sleep(self.poll_seconds)
            return None

        self.wait_for_close(round_)
        outcome = run_round(self.context, round_)
        self.rounds_run += 1
        if outcome is not None and outcome.settled:
            self._last_weight_block = self.context.client.block()
        return outcome

    def run(self, max_rounds: int | None = None) -> int:
        self._running = True
        from microtensor.cli.common import reclaim_logging

        reclaim_logging()
        log.info(
            "validator up on netuid %d across %d competitions",
            self.context.config.chain.netuid,
            len(self.context.competitions),
        )

        while self._running:
            if max_rounds is not None and self.rounds_run >= max_rounds:
                break
            try:
                self.step()
            except Restart as exc:
                log.warning("exiting to restart on %s", exc)
                break
            except Stopped:
                break
            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as exc:
                log.exception("round failed unexpectedly, retrying after backoff: %s", exc)
                self._sleep(self.poll_seconds)

        log.info("validator stopped after %d rounds", self.rounds_run)
        return self.rounds_run
