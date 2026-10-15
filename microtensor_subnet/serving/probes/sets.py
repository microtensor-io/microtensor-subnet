from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from microtensor_subnet.validator.context import Artifact, EpochContext
    from microtensor_subnet.validator.results import ArtifactOutcome

log = logging.getLogger("microtensor_subnet.serving.probes")

PROBE_DIR = "probe"
DB_DIR = "db"
MODEL_PREFIX = "mt/"


def alias(arena_id: str) -> str:
    return f"{MODEL_PREFIX}{arena_id}"


def floor_key(floor: float) -> str:
    return f"{float(floor):.2f}"


@dataclass(frozen=True, slots=True)
class ProbeItem:
    item_id: str
    family: str
    task: dict[str, Any]
    label: dict[str, Any]


@dataclass(slots=True)
class ProbeSet:
    arena: str
    epoch: int
    artifact_sha: str
    model: str
    grader: str
    right: dict[str, list[str]]
    items: dict[str, ProbeItem]
    databases: dict[str, bytes] = field(default_factory=dict)

    def floors(self) -> list[float]:
        return sorted(float(k) for k in self.right)

    def right_at(self, floor: float) -> list[ProbeItem]:
        if not self.right:
            return []
        nearest = min(self.right, key=lambda k: (abs(float(k) - floor), k))
        return [self.items[i] for i in self.right[nearest] if i in self.items]

    def wire(self) -> dict[str, Any]:
        return {
            "arena": self.arena,
            "epoch": self.epoch,
            "artifact_sha": self.artifact_sha,
            "model": self.model,
            "grader": self.grader,
            "right": self.right,
            "items": {
                i: {"family": it.family, "task": it.task, "label": it.label}
                for i, it in self.items.items()
            },
            "databases": sorted(self.databases),
        }


def _needed(items: Mapping[str, ProbeItem]) -> set[str]:
    found: set[str] = set()
    for item in items.values():
        for sha in item.label.get("databases") or item.label.get("blobs") or []:
            found.add(str(sha))
    return found


def write(
    context: EpochContext,
    outcomes: Mapping[str, ArtifactOutcome],
    artifacts: Mapping[str, Artifact],
) -> list[Path]:
    folder = context.work_dir / PROBE_DIR
    (folder / DB_DIR).mkdir(parents=True, exist_ok=True)
    by_id = {item.item_id: item for item in context.items.items("system")}
    grader = str(context.definition.get("grader", ""))
    written: list[Path] = []
    for hotkey, outcome in outcomes.items():
        artifact = artifacts.get(hotkey)
        if artifact is None or outcome.system is None or outcome.failure:
            continue
        right = {
            floor_key(run.floor): sorted(r.item_id for r in run.runs if r.success and r.answered)
            for run in outcome.system.floors
        }
        chosen = {i for ids in right.values() for i in ids}
        items = {
            i: ProbeItem(
                item_id=i,
                family=by_id[i].family,
                task=dict(by_id[i].task),
                label=dict(by_id[i].label),
            )
            for i in sorted(chosen)
            if i in by_id
        }
        held = ProbeSet(
            arena=context.arena_id,
            epoch=context.epoch_index,
            artifact_sha=artifact.artifact_sha,
            model=alias(context.arena_id),
            grader=grader,
            right=right,
            items=items,
        )
        for sha in _needed(items):
            body = context.items.databases.get(sha)
            target = folder / DB_DIR / sha
            if body is not None and not target.exists():
                target.write_bytes(body)
            if body is not None:
                held.databases[sha] = body
        path = folder / f"{artifact.artifact_sha}.json"
        path.write_text(json.dumps(held.wire(), sort_keys=True), encoding="utf-8")
        written.append(path)
    log.info("wrote %d probe sets under %s", len(written), folder)
    return written


def load(path: Path) -> ProbeSet:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = {
        str(i): ProbeItem(
            item_id=str(i),
            family=str(entry.get("family", "")),
            task=dict(entry.get("task") or {}),
            label=dict(entry.get("label") or {}),
        )
        for i, entry in dict(raw.get("items") or {}).items()
    }
    held = ProbeSet(
        arena=str(raw.get("arena", "")),
        epoch=int(raw.get("epoch") or 0),
        artifact_sha=str(raw.get("artifact_sha", "")),
        model=str(raw.get("model") or alias(str(raw.get("arena", "")))),
        grader=str(raw.get("grader", "")),
        right={str(k): [str(i) for i in v] for k, v in dict(raw.get("right") or {}).items()},
        items=items,
    )
    for sha in raw.get("databases") or []:
        blob = path.parent / DB_DIR / str(sha)
        if blob.is_file():
            held.databases[str(sha)] = blob.read_bytes()
    return held


def latest(work_root: Path, arena: str) -> dict[str, ProbeSet]:
    found: dict[str, ProbeSet] = {}
    for path in sorted(work_root.rglob(f"{PROBE_DIR}/*.json")):
        try:
            held = load(path)
        except (OSError, ValueError) as exc:
            log.warning("probe set %s unreadable: %s", path, exc)
            continue
        if held.arena != arena:
            continue
        previous = found.get(held.artifact_sha)
        if previous is None or held.epoch >= previous.epoch:
            found[held.artifact_sha] = held
    return found
