from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from microtensor.rigs.validator.session.runner import Runner
from microtensor.rigs.validator.storage import volumes
from microtensor.rigs.validator.verify.exclusivity import OwnContainer, parse_containers
from microtensor.rigs.validator.work import containers, power

LIST_TIMEOUT = 60.0
LIST_OURS = (
    f"docker ps -a --no-trunc --filter label={containers.OWNER_LABEL} "
    "--format '{{.ID}}\t{{.Names}}\t{{.Label \"mt.job\"}}\t{{.State}}'"
)


@dataclass
class SweepReport:
    stale_volumes: list[str] = field(default_factory=list)
    orphaned_containers: list[str] = field(default_factory=list)
    dangling_images: int = 0
    retried_teardowns: list[str] = field(default_factory=list)
    failed_teardowns: list[str] = field(default_factory=list)
    power_raised: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        return {
            "stale_volumes": list(self.stale_volumes),
            "orphaned_containers": list(self.orphaned_containers),
            "dangling_images": self.dangling_images,
            "retried_teardowns": list(self.retried_teardowns),
            "failed_teardowns": list(self.failed_teardowns),
            "power_raised": list(self.power_raised),
            "errors": list(self.errors),
        }


class TeardownRetries:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.pending: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.pending = {str(k): dict(v) for k, v in (data.get("pending") or {}).items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps({"pending": self.pending}, indent=1), encoding="utf-8")
        os.replace(temp, self.path)

    def add(
        self, rig_id: str, name: str, job_id: str = "", volume_names: tuple[str, ...] = ()
    ) -> None:
        self.pending[name] = {
            "rig_id": rig_id,
            "job_id": job_id,
            "volumes": list(volume_names),
            "attempts": int(self.pending.get(name, {}).get("attempts", 0)) + 1,
            "recorded_at": time.time(),
        }
        self.save()

    def forget(self, name: str) -> None:
        if self.pending.pop(name, None) is not None:
            self.save()

    def for_rig(self, rig_id: str) -> dict[str, dict[str, Any]]:
        return {
            name: entry for name, entry in self.pending.items() if entry.get("rig_id") == rig_id
        }


async def own_containers(runner: Runner) -> tuple[list[OwnContainer], str]:
    result = await runner.run(LIST_OURS, timeout=LIST_TIMEOUT)
    if not result.ok:
        return [], (result.error or result.stderr_tail(200)).strip()
    return parse_containers(result.stdout), ""


async def stale_volumes(runner: Runner) -> list[str]:
    return await containers.remove_stale_volumes(runner)


async def orphaned_containers(
    runner: Runner,
    running_jobs: set[str],
    records: power.PowerRecords | None = None,
    store: containers.JobStore | None = None,
) -> tuple[list[str], list[str]]:
    ours, error = await own_containers(runner)
    if error:
        return [], [error]
    removed: list[str] = []
    errors: list[str] = []
    for container in ours:
        if container.job and container.job in running_jobs:
            continue
        outcome = await containers.destroy(
            runner,
            container.name or container.id,
            grace=containers.STOP_GRACE_FILLER,
            job_id=container.job,
            records=records,
            volume_names=(volumes.volume_name(container.job),)
            if container.job and volumes.valid(f"{volumes.PREFIX}{container.job}")
            else (),
            store=store,
            prune_images=False,
        )
        if outcome.ok:
            removed.append(container.name or container.id)
        else:
            errors.append(f"{container.name or container.id}: {outcome.reason}")
    return removed, errors


async def dangling_images(runner: Runner) -> int:
    listing = await runner.run("docker images -q --filter dangling=true", timeout=LIST_TIMEOUT)
    if not listing.ok:
        return 0
    ids = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    if not ids:
        return 0
    await runner.run(
        "docker images -q --filter dangling=true | xargs -r docker image rm -f",
        timeout=LIST_TIMEOUT * 5,
    )
    return len(ids)


async def retry_teardowns(
    runner: Runner,
    rig_id: str,
    retries: TeardownRetries,
    records: power.PowerRecords | None = None,
    store: containers.JobStore | None = None,
) -> tuple[list[str], list[str]]:
    done: list[str] = []
    failed: list[str] = []
    for name, entry in list(retries.for_rig(rig_id).items()):
        outcome = await containers.destroy(
            runner,
            name,
            grace=containers.STOP_GRACE_FILLER,
            job_id=str(entry.get("job_id", "") or ""),
            records=records,
            volume_names=tuple(str(v) for v in entry.get("volumes") or []),
            store=store,
            prune_images=False,
        )
        if outcome.ok:
            retries.forget(name)
            done.append(name)
        else:
            retries.add(
                rig_id,
                name,
                str(entry.get("job_id", "") or ""),
                tuple(str(v) for v in entry.get("volumes") or []),
            )
            failed.append(f"{name}: {outcome.reason}")
    return done, failed


async def sweep(
    runner: Runner,
    rig_id: str,
    running_jobs: set[str],
    retries: TeardownRetries | None = None,
    records: power.PowerRecords | None = None,
    store: containers.JobStore | None = None,
    gpu_uuids: list[str] | None = None,
) -> SweepReport:
    report = SweepReport()
    if retries is not None:
        report.retried_teardowns, report.failed_teardowns = await retry_teardowns(
            runner, rig_id, retries, records, store
        )
    report.orphaned_containers, errors = await orphaned_containers(
        runner, running_jobs, records, store
    )
    report.errors.extend(errors)
    report.stale_volumes = await stale_volumes(runner)
    report.dangling_images = await dangling_images(runner)
    if gpu_uuids:
        report.power_raised = await power.raise_low_limits(runner, list(gpu_uuids))
    return report
