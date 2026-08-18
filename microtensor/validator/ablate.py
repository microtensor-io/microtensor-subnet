from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from microtensor.core.constants import ROLE_BASELINES
from microtensor.core.protocol import Role
from microtensor.scoring.ablation import Contribution, baseline_for, settle, unset_roles
from microtensor.scoring.frontier import Point, quantise_point
from microtensor.tasks.selection import RoundTasks

log = logging.getLogger("microtensor.validator")

BASELINE_DIR = "baselines"
MISSING_BASELINE_ARTIFACT = "baseline artifact is not published with the corpus"


def task_set_digest(tasks: RoundTasks) -> str:
    """Identity of the exact task set a measurement was taken against.

    Half of the cache key. A cached component output is only reusable against
    the tasks it was produced on, so the digest covers the task refs and the
    seed that ordered them, not merely the round index.
    """
    body = {
        "track": tasks.track,
        "hardware_class": tasks.hardware_class,
        "corpus_version": tasks.corpus_version,
        "seed": tasks.seed,
        "tasks": [task.ref for task in tasks.all],
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return f"sha256:{sha256(canonical.encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class Measured:
    """What one ablated system scored and cost over the round's task set."""

    quality: float
    cost_ms: float


@dataclass(slots=True)
class OutputCache:
    """Component outputs keyed by (component_digest, task_set_digest).

    Section 5c. This caches *component* work, not system results. Ablating the
    router leaves the front untouched, so the front's outputs over this task set
    are reusable for every ablation of every system that carries it, and the
    same published baseline is executed once per epoch rather than once per
    system that substitutes it.

    It deliberately does not cache the ablated system's score. Each S' is a
    different composition, so a system-level cache would hand every system the
    same ablated measurement and the contributions would be uniform fiction.
    """

    entries: dict[tuple[str, str], Any] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, component_digest: str, task_digest: str) -> Any | None:
        found = self.entries.get((component_digest, task_digest))
        if found is None:
            self.misses += 1
        else:
            self.hits += 1
        return found

    def put(self, component_digest: str, task_digest: str, outputs: Any) -> None:
        self.entries[(component_digest, task_digest)] = outputs

    @property
    def reuse(self) -> str:
        total = self.hits + self.misses
        return f"{self.hits}/{total} component runs served from cache" if total else "unused"


@dataclass(frozen=True, slots=True)
class BaselineStore:
    """Role baselines ship with the corpus and are versioned with it."""

    root: Path
    digests: Mapping[str, str] = field(default_factory=lambda: ROLE_BASELINES)

    def digest(self, role: Role) -> str:
        return baseline_for(role, self.digests)

    def path(self, role: Role) -> Path:
        return self.root / BASELINE_DIR / role.value

    def available(self, role: Role) -> tuple[bool, str]:
        if not self.digest(role):
            return False, ""
        if not self.path(role).exists():
            return False, MISSING_BASELINE_ARTIFACT
        return True, ""


Rerun = Callable[[str, Role, str, Path, OutputCache], "Measured | None"]
"""Re-evaluate system `key` with `role` replaced by the baseline at that path.

Takes the system key because S' is a different composition for every S: the
substitution is into one particular system, not a measurement of the baseline
in isolation. Takes the cache so unchanged components are not re-executed.
"""


def _point(key: str, measured: Measured, cost_ceiling: float, template: Point) -> Point:
    qi, ci = quantise_point(measured.quality, measured.cost_ms, cost_ceiling)
    return Point(
        key=key,
        qi=qi,
        ci=ci,
        committed_at=template.committed_at,
        stale_rounds=template.stale_rounds,
    )


def ablations_for(
    key: str,
    roles: Sequence[Role],
    members: Sequence[Point],
    store: BaselineStore,
    rerun: Rerun,
    cache: OutputCache,
    cost_ceiling: float,
) -> dict[Role, Point | None]:
    """Measure the ablated twin of one system, once per role."""
    template = next((p for p in members if p.key == key), None)
    if template is None:
        return dict.fromkeys(roles)

    ablated: dict[Role, Point | None] = {}

    for role in roles:
        ready, reason = store.available(role)
        if not ready:
            if reason:
                log.warning("%s baseline unusable: %s", role.value, reason)
            ablated[role] = None
            continue

        measured = rerun(key, role, store.digest(role), store.path(role), cache)
        if measured is None:
            ablated[role] = None
            continue

        ablated[role] = _point(f"{key}~{role.value}", measured, cost_ceiling, template)

    return ablated


def contributions(
    keys: Sequence[str],
    roles_by_key: Mapping[str, Sequence[Role]],
    members: Sequence[Point],
    store: BaselineStore,
    rerun: Rerun,
    cost_ceiling: float,
    cache: OutputCache | None = None,
) -> dict[str, tuple[Contribution, ...]]:
    """Price every component of every frontier system.

    Only frontier members are ablated. A system off the frontier earns nothing,
    so dividing its value between components would be dividing zero, and the
    work would scale with submissions rather than with the frontier.
    """
    if len(unset_roles(store.digests)) == len(store.digests):
        log.info("ablation skipped: no role baseline is published yet")
        return {
            key: settle(members, key, dict.fromkeys(roles_by_key.get(key, ())))
            for key in keys
        }

    shared = cache if cache is not None else OutputCache()
    on_frontier = {point.key for point in members}
    priced: dict[str, tuple[Contribution, ...]] = {}

    for key in keys:
        roles = roles_by_key.get(key, ())
        if key not in on_frontier:
            priced[key] = settle(members, key, dict.fromkeys(roles))
            continue
        ablated = ablations_for(
            key, roles, members, store, rerun, shared, cost_ceiling
        )
        priced[key] = settle(members, key, ablated, store.digests)

    log.info("ablation cache: %s", shared.reuse)
    return priced
