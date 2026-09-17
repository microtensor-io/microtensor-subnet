from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from microtensor.rigs.validator.scoring.formula import UPTIME_RAMP_DAYS

DAY_SECONDS = 86400.0
MAX_GAP_SECONDS = 120.0


@dataclass
class RigReliability:
    uptime_seconds: float = 0.0
    completed_jobs: int = 0
    failed_jobs: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = 0.0
    last_online: bool = False

    @property
    def uptime_days(self) -> float:
        return self.uptime_seconds / DAY_SECONDS

    @property
    def multiplier(self) -> float:
        return min(1.0, self.uptime_days / UPTIME_RAMP_DAYS)

    @property
    def success_rate(self) -> float:
        total = self.completed_jobs + self.failed_jobs
        return self.completed_jobs / total if total else 1.0

    def payload(self) -> dict[str, float | int | bool]:
        return {
            "uptime_seconds": round(self.uptime_seconds, 1),
            "uptime_days": round(self.uptime_days, 3),
            "multiplier": round(self.multiplier, 4),
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "success_rate": round(self.success_rate, 4),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "last_online": self.last_online,
        }


class UptimeLedger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.rigs: dict[str, RigReliability] = {}
        self.load()

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for rig_id, body in (data.get("rigs") or {}).items():
            self.rigs[str(rig_id)] = RigReliability(
                uptime_seconds=float(body.get("uptime_seconds", 0.0)),
                completed_jobs=int(body.get("completed_jobs", 0)),
                failed_jobs=int(body.get("failed_jobs", 0)),
                first_seen=float(body.get("first_seen", time.time())),
                last_seen=float(body.get("last_seen", 0.0)),
                last_online=bool(body.get("last_online", False)),
            )

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps({"rigs": {k: v.payload() for k, v in self.rigs.items()}}, indent=1),
            encoding="utf-8",
        )
        os.replace(temp, self.path)

    def observe(self, rig_id: str, online: bool, now: float | None = None) -> RigReliability:
        moment = time.time() if now is None else now
        rig = self.rigs.setdefault(rig_id, RigReliability(first_seen=moment))
        if rig.last_seen and rig.last_online and online:
            gap = moment - rig.last_seen
            if 0.0 < gap <= MAX_GAP_SECONDS:
                rig.uptime_seconds += gap
        rig.last_seen = moment
        rig.last_online = online
        return rig

    def record_job(self, rig_id: str, completed: bool) -> None:
        rig = self.rigs.setdefault(rig_id, RigReliability())
        if completed:
            rig.completed_jobs += 1
        else:
            rig.failed_jobs += 1

    def uptime_days(self, rig_id: str) -> float:
        rig = self.rigs.get(rig_id)
        return rig.uptime_days if rig else 0.0

    def forget(self, rig_id: str) -> None:
        self.rigs.pop(rig_id, None)

    def apply_server_uptime(self, uptime_seconds_by_rig: dict[str, float]) -> None:
        for rig_id, seconds in uptime_seconds_by_rig.items():
            rig = self.rigs.setdefault(rig_id, RigReliability())
            rig.uptime_seconds = max(rig.uptime_seconds, float(seconds))
