from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from microtensor.chain.rounds import Round
from microtensor.core.constants import (
    MECHANISM_VERSION,
    RELEASE_REPO,
    RELEASE_SIGNING_KEY,
    UPDATE_POLL_SECONDS,
)
from microtensor.update.apply import Applied, apply_release
from microtensor.update.policy import Action, Decision, decide
from microtensor.update.release import Release, ReleaseError, latest

log = logging.getLogger("microtensor.update.loop")


@dataclass(frozen=True, slots=True)
class UpdateSettings:
    enabled: bool = False
    channel: str = "stable"
    repo: str = RELEASE_REPO
    poll_seconds: int = UPDATE_POLL_SECONDS
    signing_key: str = RELEASE_SIGNING_KEY
    require_signature: bool = True
    allow_mechanism_change: bool = False
    allow_major: bool = False
    dry_run: bool = False


@dataclass
class UpdateChecker:
    settings: UpdateSettings
    fetch: Callable[..., Release | None] = latest
    apply: Callable[..., Applied] = apply_release
    now: Callable[[], float] = time.monotonic
    last_checked: float | None = field(default=None)
    last_decision: Decision | None = field(default=None)
    held: str = field(default="")

    @property
    def due(self) -> bool:
        if not self.settings.enabled:
            return False
        if self.last_checked is None:
            return True
        return self.now() - self.last_checked >= self.settings.poll_seconds

    def check(self, round_: Round, block: int) -> Decision:
        self.last_checked = self.now()
        try:
            release = self.fetch(self.settings.repo, channel=self.settings.channel)
        except ReleaseError as exc:
            log.warning("release check failed, staying on %s: %s", MECHANISM_VERSION, exc)
            return Decision(Action.NONE, str(exc))

        decision = decide(
            release,
            round_,
            block,
            allow_mechanism_change=self.settings.allow_mechanism_change,
            allow_major=self.settings.allow_major,
        )
        self.last_decision = decision

        if decision.action is Action.HOLD and decision.reason != self.held:
            self.held = decision.reason
            log.warning("update held for the operator: %s", decision.reason)
        elif decision.action is Action.DEFER:
            log.info("update deferred: %s", decision.reason)
        elif decision.action is Action.APPLY:
            log.info("update ready: %s", decision.reason)

        return decision

    def step(self, round_: Round, block: int) -> Applied | None:
        if not self.due:
            return None

        decision = self.check(round_, block)
        if not decision.should_apply or decision.release is None:
            return None

        applied = self.apply(
            decision.release,
            signing_key=self.settings.signing_key,
            require_signature=self.settings.require_signature,
            dry_run=self.settings.dry_run,
        )
        if applied.installed:
            log.warning(
                "%s installed; exiting so the supervisor restarts on the new build",
                decision.release.tag,
            )
        return applied
