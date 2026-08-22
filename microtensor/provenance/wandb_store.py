from __future__ import annotations

import logging
import os
from typing import Any

from microtensor.core.constants import PROVENANCE_ENTITY, PROVENANCE_PROJECT
from microtensor.provenance.record import ProvenanceUnavailable, Run

log = logging.getLogger("microtensor.provenance")

API_KEY_ENV = "WANDB_API_KEY"

# A run name is not an identity: anyone with an account can create a run
# called with somebody else's hotkey. So every run bearing the name is a
# candidate and the checks decide, rather than one being picked by recency and
# allowed to speak for the miner. Bounded because an attacker can create many.
MAX_CANDIDATES = 25


def credentials_present() -> bool:
    """Whether wandb can find credentials, by any route it accepts.

    Not only the environment variable. `wandb login` writes a netrc entry and
    the sdk reads env, netrc and its own config in turn, so testing the
    variable alone told a validator that had logged in the documented way
    that the store was unreachable. Under PROVENANCE_REQUIRED that is fatal,
    which made a correctly configured host look like a broken one.

    Asking the sdk rather than reimplementing its search order, because the
    order is its business and a second copy of it would drift.
    """
    if os.environ.get(API_KEY_ENV, "").strip():
        return True

    try:
        import wandb
    except ImportError:
        return False

    try:
        found = wandb.setup()
        settings = getattr(found, "settings", None)
        return bool(getattr(settings, "api_key", "") or "")
    except Exception:
        return False


def _api() -> Any:
    try:
        import wandb
    except ImportError as exc:
        raise ProvenanceUnavailable(
            'wandb is required to read training runs; pip install ".[provenance]"'
        ) from exc

    try:
        return wandb.Api(timeout=30)
    except Exception as exc:
        raise ProvenanceUnavailable(
            f"the run store rejected our credentials: {exc}; set {API_KEY_ENV} or "
            "run wandb login"
        ) from exc


class WandbStore:
    def __init__(
        self,
        entity: str = PROVENANCE_ENTITY,
        project: str = PROVENANCE_PROJECT,
        api: Any = None,
    ) -> None:
        self.entity = entity
        self.project = project
        self._api = api

    @property
    def api(self) -> Any:
        if self._api is None:
            self._api = _api()
        return self._api

    def reachable(self) -> tuple[bool, str]:
        try:
            self.api.runs(f"{self.entity}/{self.project}", per_page=1)
        except ProvenanceUnavailable as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"{self.entity}/{self.project} is not readable: {exc}"
        return True, ""

    def candidates(self, hotkey: str) -> list[Run]:
        """Every run bearing this hotkey, for the checks to judge.

        Not one run. The store cannot tell whose account created a run, so
        selecting a single one by recency lets anybody shadow a miner: create
        a run named with their hotkey, log a later finish block, and the real
        submission is never the one examined. Returning the whole set means an
        added run can only add a candidate, never hide one.

        A multi-component system needs this anyway — each component is
        checked against its own digest, and one run cannot carry three.
        """
        path = f"{self.entity}/{self.project}"
        try:
            found = list(
                self.api.runs(path, filters={"display_name": hotkey}, per_page=MAX_CANDIDATES)
            )[:MAX_CANDIDATES]
        except ProvenanceUnavailable:
            raise
        except Exception as exc:
            raise ProvenanceUnavailable(f"{path} could not be queried: {exc}") from exc

        return [
            Run(
                hotkey=hotkey,
                run_id=str(getattr(run, "id", "")),
                config=dict(getattr(run, "config", {}) or {}),
                summary=_summary(run),
                entity=self.entity,
                project=self.project,
            )
            for run in found
        ]

    def fetch(self, hotkey: str) -> Run | None:
        """One run, for reporting. The gate uses candidates()."""
        found = self.candidates(hotkey)
        if not found:
            return None
        return max(found, key=lambda r: r.finished_block)


def _summary(run: Any) -> dict[str, Any]:
    raw = getattr(run, "summary", None)
    if raw is None:
        return {}
    try:
        return dict(raw)
    except (TypeError, ValueError):
        return {}
