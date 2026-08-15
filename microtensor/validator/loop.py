from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable
from types import FrameType

from microtensor.chain.rounds import Round, round_for_block
from microtensor.core.constants import BLOCK_TIME_SECONDS, POLL_INTERVAL_SECONDS
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

    def wait_for_close(self, round_: Round) -> None:
        while self._running:
            block = self.context.client.block()
            if block >= round_.close_block:
                return

            self.consider_update(round_, block)
            remaining = round_.close_block - block
            delay = min(self.poll_seconds, max(1, remaining * BLOCK_TIME_SECONDS))
            log.info(
                "round %d open: %d blocks until submissions close", round_.index, remaining
            )
            self._sleep(delay)
        raise Stopped

    def step(self) -> RoundOutcome | None:
        round_ = self.round_at_head()

        if self.context.state.is_settled(round_.index):
            log.info("round %d is already settled; waiting for the next", round_.index)
            self._sleep(self.poll_seconds)
            return None

        self.wait_for_close(round_)
        outcome = run_round(self.context, round_)
        self.rounds_run += 1
        return outcome

    def run(self, max_rounds: int | None = None) -> int:
        self._running = True
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
