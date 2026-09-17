from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from microtensor.rigs.protocol.failures import Failure, FailureClass, Verdict
from microtensor.rigs.protocol.session import JobRecord
from microtensor.rigs.validator.session.client import Session

OWNER_LABEL = "mt.owner=validator"
JOB_LABEL = "mt.job"
LIST_COMMAND = (
    f"docker ps -a --no-trunc --filter label={OWNER_LABEL} "
    "--format '{{.ID}}\t{{.Names}}\t{{.Label \"mt.job\"}}\t{{.State}}'"
)
LIST_TIMEOUT = 30.0


@dataclass(frozen=True)
class OwnContainer:
    id: str
    name: str
    job: str
    state: str = ""


def parse_containers(stdout: str) -> list[OwnContainer]:
    found: list[OwnContainer] = []
    for line in stdout.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2 or not parts[0].strip():
            continue
        found.append(
            OwnContainer(
                id=parts[0].strip(),
                name=parts[1].strip().lstrip("/"),
                job=parts[2].strip() if len(parts) > 2 else "",
                state=parts[3].strip() if len(parts) > 3 else "",
            )
        )
    return found


async def own_containers(session: Session) -> tuple[list[OwnContainer], str]:
    result = await session.run(LIST_COMMAND, LIST_TIMEOUT)
    if not result.ok:
        return [], (result.error or result.stderr_tail(200)).strip()
    return parse_containers(result.stdout), ""


def jobs_of(rig: dict[str, Any]) -> list[JobRecord]:
    records: list[JobRecord] = []
    for job in rig.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        payload = dict(job)
        payload.setdefault("job_id", job.get("id", ""))
        records.append(JobRecord.from_payload(payload))
    return records


def container_matches(container: str, known: set[str]) -> bool:
    if not container:
        return False
    for candidate in known:
        if not candidate:
            continue
        if (
            container == candidate
            or container.startswith(candidate)
            or candidate.startswith(container)
        ):
            return True
    return False


def check(
    gpus: list[dict[str, Any]],
    jobs: list[JobRecord],
    ours: list[OwnContainer],
    seed: str = "",
    alive_pids: set[int] | None = None,
) -> Verdict:
    job_pids = {job.pid for job in jobs if job.pid > 0}
    job_containers = {job.container for job in jobs if job.container}
    our_names = (
        {c.id for c in ours} | {c.name for c in ours} | {f"mt-{c.job}" for c in ours if c.job}
    )
    known = job_containers | our_names
    seen_pids: set[int] = set()
    foreign: list[dict[str, Any]] = []
    authorised = 0
    for card in gpus:
        uuid = str(card.get("uuid", "") or "")
        for process in card.get("processes") or []:
            try:
                pid = int(process.get("pid", 0) or 0)
            except (TypeError, ValueError):
                pid = 0
            seen_pids.add(pid)
            container = str(process.get("container", "") or "")
            if pid in job_pids or container_matches(container, known):
                authorised += 1
                continue
            foreign.append(
                {
                    "uuid": uuid,
                    "pid": pid,
                    "comm": process.get("comm", ""),
                    "cmdline": str(process.get("cmdline", "") or "")[:120],
                    "container": container[:12],
                    "kind": process.get("kind", ""),
                    "memory_mb": process.get("memory_mb"),
                }
            )
    running_containers = {c.id for c in ours if c.state in ("", "running")} | {
        c.name for c in ours if c.state in ("", "running")
    }
    dead: list[dict[str, Any]] = []
    for job in jobs:
        if job.pid > 0:
            gone = job.pid not in seen_pids and (alive_pids is None or job.pid not in alive_pids)
            if gone:
                dead.append(
                    {"job_id": job.job_id, "kind": job.kind, "pid": job.pid, "reason": "no process"}
                )
        elif job.container and not container_matches(job.container, running_containers):
            dead.append(
                {
                    "job_id": job.job_id,
                    "kind": job.kind,
                    "container": job.container[:20],
                    "reason": "container not running",
                }
            )
    evidence: dict[str, Any] = {
        "jobs": len(jobs),
        "own_containers": len(ours),
        "authorised_processes": authorised,
        "foreign_processes": foreign[:20],
        "dead_jobs": dead,
        "summary": f"{authorised} placed, {len(foreign)} foreign, {len(dead)} dead jobs",
    }
    if foreign:
        first = foreign[0]
        detail = f"pid {first['pid']} ({first['comm'] or 'unknown'}) on {first['uuid'] or 'a card'} is not placed by the pool"
        return Verdict(
            "exclusivity",
            failure=Failure(
                FailureClass.UNAUTHORISED_WORK,
                detail,
                seed=seed,
                evidence={"processes": foreign[:10]},
            ),
            evidence=evidence,
        )
    return Verdict("exclusivity", evidence=evidence)


def command_quoted() -> str:
    return shlex.quote(LIST_COMMAND)
