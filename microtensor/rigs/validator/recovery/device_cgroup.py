from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any

from microtensor.rigs.validator.session.runner import Runner
from microtensor.rigs.validator.storage.sweep import own_containers
from microtensor.rigs.validator.verify.exclusivity import OwnContainer
from microtensor.rigs.validator.work import containers, power

PROBE_TIMEOUT = 60.0
LOSS_MARKERS = ("operation not permitted", "eperm", "unknown error", "failed to initialize nvml")


@dataclass
class RepairReport:
    checked: int = 0
    wiped: list[str] = field(default_factory=list)
    repaired: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "wiped": list(self.wiped),
            "repaired": list(self.repaired),
            "failed": dict(self.failed),
            "skipped": self.skipped,
        }


def looks_wiped(exit_code: int, stdout: str, stderr: str) -> bool:
    if exit_code == 0 and "GPU 0" in stdout:
        return False
    text = f"{stdout}\n{stderr}".lower()
    return any(marker in text for marker in LOSS_MARKERS)


async def probe(runner: Runner, container: OwnContainer) -> bool | None:
    target = container.name or container.id
    result = await runner.run(
        f"docker exec {shlex.quote(target)} nvidia-smi -L", timeout=PROBE_TIMEOUT
    )
    if result.transport_failed:
        return None
    return looks_wiped(result.exit_code, result.stdout, result.stderr)


async def detect(runner: Runner) -> tuple[list[OwnContainer], list[OwnContainer], str]:
    ours, error = await own_containers(runner)
    if error:
        return [], [], error
    running = [c for c in ours if c.state in ("", "running")]
    wiped: list[OwnContainer] = []
    for container in running:
        outcome = await probe(runner, container)
        if outcome is None:
            return running, wiped, "probe transport failed"
        if outcome:
            wiped.append(container)
    return running, wiped, ""


async def repair(
    runner: Runner,
    container: OwnContainer,
    store: containers.JobStore | None,
    allowlist: tuple[str, ...],
    records: power.PowerRecords | None = None,
) -> tuple[bool, str]:
    loaded = store.load(container.job) if store is not None and container.job else None
    if loaded is None:
        target = container.name or container.id
        restarted = await runner.run(f"docker restart -t 30 {shlex.quote(target)}", timeout=120.0)
        if not restarted.ok:
            return (
                False,
                f"restart failed: {(restarted.error or restarted.stderr_tail(200)).strip()}",
            )
        again = await probe(runner, container)
        return (again is False), "" if again is False else "devices still unreachable after restart"
    spec, rig_id = loaded
    outcome = await containers.recreate(runner, spec, allowlist, records, store, rig_id=rig_id)
    if not outcome.ok:
        return False, f"{outcome.step}: {outcome.reason}"
    again = await probe(
        runner, OwnContainer(id="", name=spec.container_name, job=spec.job_id, state="running")
    )
    return (again is False), "" if again is False else "devices still unreachable after recreation"


async def sweep(
    runner: Runner,
    store: containers.JobStore | None,
    allowlist: tuple[str, ...],
    records: power.PowerRecords | None = None,
) -> RepairReport:
    running, wiped, error = await detect(runner)
    report = RepairReport(
        checked=len(running), wiped=[c.name or c.id for c in wiped], skipped=error
    )
    if error:
        return report
    for container in wiped:
        ok, reason = await repair(runner, container, store, allowlist, records)
        name = container.name or container.id
        if ok:
            report.repaired.append(name)
        else:
            report.failed[name] = reason
    return report
