from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from microtensor.core.hashing import canonical_json
from microtensor.scoring.pools import POOLS

REPORT_VERSION: Final[str] = "e1"
FAILURE_LIMIT: Final[int] = 300


class ReportError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PoolScore:
    pool: str
    score: float
    lower_bound: float
    point: float
    n: int
    eligible: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pool not in POOLS:
            raise ReportError(f"unknown pool {self.pool!r}")
        if not 0.0 <= self.score <= 1.0:
            raise ReportError(f"pool {self.pool} score must lie in [0, 1]")
        if self.n < 0:
            raise ReportError("a pool score cannot cover a negative count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool": self.pool,
            "score": self.score,
            "lower_bound": self.lower_bound,
            "point": self.point,
            "n": self.n,
            "eligible": self.eligible,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PoolScore:
        return cls(
            pool=str(raw["pool"]),
            score=float(raw["score"]),
            lower_bound=float(raw.get("lower_bound", 0.0)),
            point=float(raw.get("point", 0.0)),
            n=int(raw.get("n", 0)),
            eligible=bool(raw.get("eligible", False)),
            detail=dict(raw.get("detail") or {}),
        )


@dataclass(frozen=True, slots=True)
class MinerReport:
    hotkey: str
    artifact_sha: str
    manifest_sha: str
    pools: tuple[PoolScore, ...] = ()
    failure: str = ""
    receipts_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hotkey": self.hotkey,
            "artifact_sha": self.artifact_sha,
            "manifest_sha": self.manifest_sha,
            "pools": [pool.to_dict() for pool in self.pools],
            "failure": self.failure[:FAILURE_LIMIT],
            "receipts_digest": self.receipts_digest,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> MinerReport:
        return cls(
            hotkey=str(raw["hotkey"]),
            artifact_sha=str(raw.get("artifact_sha", "")),
            manifest_sha=str(raw.get("manifest_sha", "")),
            pools=tuple(PoolScore.from_dict(p) for p in raw.get("pools") or []),
            failure=str(raw.get("failure", "")),
            receipts_digest=str(raw.get("receipts_digest", "")),
        )

    def score(self, pool: str) -> PoolScore | None:
        return next((p for p in self.pools if p.pool == pool), None)


@dataclass(frozen=True, slots=True)
class EpochReport:
    arena_id: str
    epoch_index: int
    validator_hotkey: str
    bundle_sha256: str
    open_block_hash: str
    seed: str
    item_ids_digest: str
    ledger_snapshot_digest: str
    roster_digest: str
    miners: tuple[MinerReport, ...] = ()
    mechanism_version: str = ""
    version: str = REPORT_VERSION
    signature: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.version != REPORT_VERSION:
            raise ReportError(f"report version {self.version!r} is not {REPORT_VERSION!r}")
        if not self.arena_id or not self.validator_hotkey:
            raise ReportError("a report names its arena and validator")
        seen: set[str] = set()
        for miner in self.miners:
            if miner.hotkey in seen:
                raise ReportError(f"miner {miner.hotkey} appears twice")
            seen.add(miner.hotkey)

    def body(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "arena_id": self.arena_id,
            "epoch_index": self.epoch_index,
            "validator_hotkey": self.validator_hotkey,
            "bundle_sha256": self.bundle_sha256,
            "open_block_hash": self.open_block_hash,
            "seed": self.seed,
            "item_ids_digest": self.item_ids_digest,
            "ledger_snapshot_digest": self.ledger_snapshot_digest,
            "roster_digest": self.roster_digest,
            "mechanism_version": self.mechanism_version,
            "miners": [miner.to_dict() for miner in self.miners],
        }

    def signed_message(self) -> bytes:
        return canonical_json(self.body())

    def digest(self) -> str:
        return hashlib.sha256(self.signed_message()).hexdigest()

    def signed_with(self, signature: str) -> EpochReport:
        return EpochReport(
            arena_id=self.arena_id,
            epoch_index=self.epoch_index,
            validator_hotkey=self.validator_hotkey,
            bundle_sha256=self.bundle_sha256,
            open_block_hash=self.open_block_hash,
            seed=self.seed,
            item_ids_digest=self.item_ids_digest,
            ledger_snapshot_digest=self.ledger_snapshot_digest,
            roster_digest=self.roster_digest,
            miners=self.miners,
            mechanism_version=self.mechanism_version,
            version=self.version,
            signature=signature,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "signature": self.signature}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EpochReport:
        try:
            return cls(
                arena_id=str(raw["arena_id"]),
                epoch_index=int(raw["epoch_index"]),
                validator_hotkey=str(raw["validator_hotkey"]),
                bundle_sha256=str(raw.get("bundle_sha256", "")),
                open_block_hash=str(raw.get("open_block_hash", "")),
                seed=str(raw.get("seed", "")),
                item_ids_digest=str(raw.get("item_ids_digest", "")),
                ledger_snapshot_digest=str(raw.get("ledger_snapshot_digest", "")),
                roster_digest=str(raw.get("roster_digest", "")),
                miners=tuple(MinerReport.from_dict(m) for m in raw.get("miners") or []),
                mechanism_version=str(raw.get("mechanism_version", "")),
                version=str(raw.get("version", REPORT_VERSION)),
                signature=str(raw.get("signature", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ReportError):
                raise
            raise ReportError(f"the report is malformed: {exc}") from exc


Verifier = Callable[[str, bytes, str], bool]


def read_report(raw: Mapping[str, Any], verify: Verifier) -> EpochReport:
    report = EpochReport.from_dict(raw)
    if not report.signature:
        raise ReportError("the report is unsigned")
    if not verify(report.validator_hotkey, report.signed_message(), report.signature):
        raise ReportError("the report signature does not verify against the validator hotkey")
    return report
