from __future__ import annotations

import logging
import os
from typing import Any

from microtensor.core.constants import PROVENANCE_ENTITY, PROVENANCE_PROJECT
from microtensor.provenance.record import ProvenanceUnavailable, Run

log = logging.getLogger("microtensor.provenance")

API_KEY_ENV = "WANDB_API_KEY"


def credentials_present() -> bool:
    return bool(os.environ.get(API_KEY_ENV, "").strip())


def _api() -> Any:
    try:
        import wandb
    except ImportError as exc:
        raise ProvenanceUnavailable(
            'wandb is required to read training runs; pip install ".[provenance]"'
        ) from exc

    if not credentials_present():
        raise ProvenanceUnavailable(
            f"{API_KEY_ENV} is not set; a validator cannot read the run store without it"
        )

    try:
        return wandb.Api(timeout=30)
    except Exception as exc:
        raise ProvenanceUnavailable(f"the run store rejected our credentials: {exc}") from exc


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

    def fetch(self, hotkey: str) -> Run | None:
        path = f"{self.entity}/{self.project}"
        try:
            found = list(
                self.api.runs(path, filters={"display_name": hotkey}, per_page=2)
            )
        except ProvenanceUnavailable:
            raise
        except Exception as exc:
            raise ProvenanceUnavailable(f"{path} could not be queried: {exc}") from exc

        if not found:
            return None

        latest = max(found, key=lambda r: _summary(r).get("mt_finished_at", 0) or 0)
        return Run(
            hotkey=hotkey,
            run_id=str(getattr(latest, "id", "")),
            config=dict(getattr(latest, "config", {}) or {}),
            summary=_summary(latest),
            entity=self.entity,
            project=self.project,
        )


def _summary(run: Any) -> dict[str, Any]:
    raw = getattr(run, "summary", None)
    if raw is None:
        return {}
    try:
        return dict(raw)
    except (TypeError, ValueError):
        return {}
