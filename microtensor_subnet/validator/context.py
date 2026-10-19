from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from microtensor_subnet.artifact.validate import Manifest
from microtensor_subnet.chain.commitment import SubmissionCommitment
from microtensor_subnet.core.items import Item

SLM = "slm"
HARNESS = "harness"
ROUTER = "router"
ESCALATION = "escalation"
SYSTEM = "system"
ADVERSARIAL = "adversarial"
TESTS = (SLM, HARNESS, ROUTER, ESCALATION, SYSTEM)
SQL_STOP = ("\n\n", ";")
SQL_MAX_TOKENS = 256


@dataclass(slots=True)
class Artifact:
    hotkey: str
    coldkey: str
    commitment: SubmissionCommitment
    commit_block: int
    artifact_dir: Path
    slm_dir: Path | None
    manifest: Manifest
    manifest_sha: str
    config: dict[str, Any]
    slm_parameters: int = 0
    failure: str = ""

    @property
    def has(self) -> frozenset[str]:
        return self.manifest.has

    @property
    def artifact_sha(self) -> str:
        return self.commitment.artifact_sha


@dataclass(frozen=True, slots=True)
class Reference:
    slm_dir: Path
    harness_dir: Path
    variant_dirs: tuple[Path, ...]
    router_dir: Path | None
    escalation_dir: Path | None
    slm_parameters: int


@dataclass(slots=True)
class EpochItems:
    by_test: dict[str, list[Item]]
    databases: dict[str, bytes]
    adversarial: list[dict[str, Any]]
    families: dict[str, str]

    def items(self, test: str) -> list[Item]:
        return list(self.by_test.get(test, []))

    def twins_of(self, test: str) -> dict[str, Item]:
        return {item.twin_of: item for item in self.by_test.get(test, []) if item.twin_of}

    def originals(self, test: str) -> list[Item]:
        return [item for item in self.by_test.get(test, []) if not item.twin_of]

    def item_databases(self) -> dict[str, tuple[str, ...]]:
        found: dict[str, tuple[str, ...]] = {}
        for chosen in self.by_test.values():
            for item in chosen:
                shas = item.label.get("databases") or item.label.get("blobs") or []
                found[item.item_id] = tuple(str(s) for s in shas)
        for row in self.adversarial:
            shas = row.get("databases") or row.get("blobs") or []
            found[str(row.get("item_id", ""))] = tuple(str(s) for s in shas)
        return found


@dataclass(slots=True)
class EpochContext:
    arena_id: str
    epoch_index: int
    definition: dict[str, Any]
    seed: str
    open_block: int
    open_block_hash: str
    bundle_sha256: str
    release_digest: str
    snapshot_digest: str
    roster_digest: str
    previous_best: dict[tuple[str, str, str], float]
    roster: dict[str, dict[str, Any]]
    items: EpochItems
    reference: Reference
    work_dir: Path
    sample_database: bytes = b""
    r_class: dict[str, float] = field(default_factory=dict)
    floors: tuple[float, ...] = (0.8, 0.9, 0.95, 0.99)

    @property
    def grader(self) -> str:
        return str(self.definition.get("grader") or "sqlite_execution")

    @property
    def stop(self) -> tuple[str, ...]:
        block = self.definition.get("generation") or {}
        found = block.get("stop", SQL_STOP) if isinstance(block, Mapping) else SQL_STOP
        return tuple(str(s) for s in found)

    @property
    def max_tokens(self) -> int:
        block = self.definition.get("generation") or {}
        found = block.get("max_tokens") if isinstance(block, Mapping) else None
        return int(found or SQL_MAX_TOKENS)

    @property
    def slm_prompt(self) -> str:
        return str(
            self.definition.get("slm_prompt")
            or "Schema:\n{schema}\n\nQuestion: {question}\n\n"
            "Write one SQLite query that answers the question.\nSQL:"
        )

    def previous(self, hotkey: str, artifact_sha: str, pool: str) -> float | None:
        return self.previous_best.get((hotkey, artifact_sha, pool))

    def token_price(self, parameters: int) -> float:
        if parameters <= 1_000_000_000:
            return float(self.r_class.get("1b", 0.0))
        if parameters <= 3_000_000_000:
            return float(self.r_class.get("3b", 0.0))
        return float(self.r_class.get("8b", 0.0))


def order_for(seed: str, test: str, items: Iterable[Item]) -> list[Item]:
    draw = int.from_bytes(hashlib.sha256(f"{seed}:{test}".encode()).digest()[:8], "big")
    rng = random.Random(draw)  # noqa: S311
    ordered = sorted(items, key=lambda item: item.item_id)
    rng.shuffle(ordered)
    return ordered


def split_items(seed: str, items: Sequence[Item], limit: int = 0) -> dict[str, list[Item]]:
    by_test: dict[str, list[Item]] = {}
    for item in items:
        for test in item.tests:
            by_test.setdefault(test, []).append(item)
    for test, chosen in by_test.items():
        ordered = order_for(seed, test, chosen)
        by_test[test] = ordered[:limit] if limit else ordered
    return by_test


def render_prompt(template: str, task: Mapping[str, Any]) -> str:
    rendered = template
    for key, value in task.items():
        rendered = rendered.replace("{" + str(key) + "}", str(value))
    return rendered
