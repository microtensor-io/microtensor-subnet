from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from microtensor_subnet.serving import client
from microtensor_subnet.serving.probes.ask import Answered, ask
from microtensor_subnet.serving.probes.sets import ProbeSet
from microtensor_subnet.validator import grade

log = logging.getLogger("microtensor_subnet.serving.probes")

OPERATORS_PATH = "/v1/inference/validators/operators"
PROBES_PATH = "/v1/inference/validators/probes"
PROBED_STATES = ("collateralised", "probing", "admitted")
Lister = Callable[[str, Any], list[dict[str, Any]]]
Reporter = Callable[..., dict[str, Any]]
Asker = Callable[..., Answered]


@dataclass(frozen=True, slots=True)
class Probe:
    operator: str
    model: str
    artifact_digest: str
    item_id: str
    success: bool | None
    reason: str = ""

    @property
    def reported(self) -> bool:
        return self.success is not None


def operators_for(server: str, wallet: Any) -> list[dict[str, Any]]:
    found = client._signed(server, wallet, method="GET", path=OPERATORS_PATH)
    return [dict(row) for row in (found or {}).get("operators") or []]


def report(server: str, wallet: Any, *, operator: str, model: str, failed: bool) -> dict[str, Any]:
    payload = {"operator_hotkey": operator, "model": model, "failed": bool(failed)}
    return dict(client._signed(server, wallet, method="POST", path=PROBES_PATH, payload=payload))


def _pairs(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for pair in row.get("models") or []:
        if not isinstance(pair, Mapping):
            continue
        model = str(pair.get("model", ""))
        digest = str(pair.get("artifact_digest", "")).lower()
        if model and digest:
            found.append((model, digest))
    return found


def cycle(
    *,
    server: str,
    gateway: str,
    credential: str,
    wallet: Any,
    sets: Mapping[str, ProbeSet],
    floor: float,
    per_operator: int,
    rng: random.Random,
    pause: float = 0.0,
    lister: Lister = operators_for,
    asker: Asker = ask,
    reporter: Reporter = report,
) -> list[Probe]:
    results: list[Probe] = []
    models = {held.model for held in sets.values()}
    for row in lister(server, wallet):
        hotkey = str(row.get("hotkey", ""))
        if str(row.get("state", "")).lower() not in PROBED_STATES:
            continue
        for model, digest in _pairs(row):
            if model not in models:
                continue
            held = sets.get(digest)
            if held is None:
                results.append(Probe(hotkey, model, digest, "", None, "no withheld items"))
                continue
            items = held.right_at(floor)
            if not items:
                results.append(Probe(hotkey, model, digest, "", None, "no item passed the floor"))
                continue
            for item in rng.sample(items, min(per_operator, len(items))):
                answered = asker(
                    gateway, credential, hotkey=hotkey, model=model, task=item.task, floor=floor
                )
                if not answered.ok:
                    results.append(Probe(hotkey, model, digest, item.item_id, None, answered.error))
                    continue
                if answered.artifact_digest and answered.artifact_digest.lower() != digest:
                    success, reason = False, "answered with another artifact"
                else:
                    verdict = grade.grade(held.grader, answered.answer, item.label, held.databases)
                    success, reason = verdict.success, verdict.reason
                try:
                    reporter(server, wallet, operator=hotkey, model=model, failed=not success)
                except client.ServerError as exc:
                    log.warning("probe of %s not recorded: %s", hotkey[:12], exc)
                results.append(Probe(hotkey, model, digest, item.item_id, success, reason))
                if pause:
                    time.sleep(pause)
    return results
