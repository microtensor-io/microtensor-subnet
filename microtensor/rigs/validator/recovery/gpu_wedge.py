from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass, field
from typing import Any

from microtensor.rigs.validator.session.runner import Runner

UTILISATION_MIN = 95.0
MEMORY_MAX_PERCENT = 1.0
SWEEP_MEMORY_MAX_MIB = 16.0
SETTLE_SECONDS = 5.0
QUERY_TIMEOUT = 30.0
CURE_TIMEOUT = 60.0
CURE_MARKER = "ctx open/close OK"

QUERY_WEDGE = (
    "nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader,nounits"
)
QUERY_APPS = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"

CURE_SNIPPET = (
    "import ctypes,sys;"
    "l=ctypes.CDLL('libcuda.so.1');"
    "assert l.cuInit(0)==0,'cuInit';"
    "d=ctypes.c_int();"
    "assert l.cuDeviceGet(ctypes.byref(d),0)==0,'cuDeviceGet';"
    "c=ctypes.c_void_p();"
    "assert l.cuCtxCreate_v2(ctypes.byref(c),0,d)==0,'cuCtxCreate';"
    "assert l.cuCtxDestroy_v2(c)==0,'cuCtxDestroy';"
    f"print('{CURE_MARKER}')"
)


@dataclass(frozen=True)
class SweepResult:
    wedged: list[str] = field(default_factory=list)
    cured: list[str] = field(default_factory=list)
    still_wedged: list[str] = field(default_factory=list)
    skipped: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "wedged": list(self.wedged),
            "cured": list(self.cured),
            "still_wedged": list(self.still_wedged),
            "skipped": self.skipped,
        }


def looks_wedged(utilisation: float, memory_used_mib: float) -> bool:
    return utilisation >= UTILISATION_MIN and memory_used_mib <= SWEEP_MEMORY_MAX_MIB


def candidate_from_scrape(gpu: dict[str, Any], processes: list[Any]) -> bool:
    if processes:
        return False
    uuid = str(gpu.get("uuid") or "")
    try:
        utilisation = float(gpu.get("utilization") or 0.0)
        memory_utilisation = float(gpu.get("memory_utilization") or 0.0)
    except (TypeError, ValueError):
        return False
    return (
        bool(uuid) and utilisation >= UTILISATION_MIN and memory_utilisation <= MEMORY_MAX_PERCENT
    )


def parse_query(stdout: str) -> list[str]:
    wedged: list[str] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or not parts[0].startswith("GPU-"):
            continue
        try:
            utilisation = float(parts[1])
            memory = float(parts[2])
        except ValueError:
            continue
        if looks_wedged(utilisation, memory):
            wedged.append(parts[0])
    return wedged


async def query_wedged(runner: Runner) -> tuple[list[str], str]:
    apps, sample = await asyncio.gather(
        runner.run(QUERY_APPS, timeout=QUERY_TIMEOUT),
        runner.run(QUERY_WEDGE, timeout=QUERY_TIMEOUT),
    )
    if not apps.ok or not sample.ok:
        return (
            [],
            f"query failed: {apps.error or apps.stderr_tail(200)} {sample.error or sample.stderr_tail(200)}".strip(),
        )
    if apps.stdout.strip():
        return [], "compute processes present, wedge signature cannot apply"
    return parse_query(sample.stdout), ""


async def wedged_gpus(runner: Runner) -> tuple[list[str], str]:
    first, skipped = await query_wedged(runner)
    if skipped or not first:
        return [], skipped
    await asyncio.sleep(SETTLE_SECONDS)
    second, skipped = await query_wedged(runner)
    if skipped:
        return [], skipped
    return [uuid for uuid in first if uuid in second], ""


async def cure(runner: Runner, uuid: str) -> bool:
    command = f"CUDA_VISIBLE_DEVICES={shlex.quote(uuid)} python3 -c {shlex.quote(CURE_SNIPPET)}"
    result = await runner.run(command, timeout=CURE_TIMEOUT)
    return result.ok and CURE_MARKER in result.stdout


async def sweep(runner: Runner) -> SweepResult:
    wedged, skipped = await wedged_gpus(runner)
    if skipped:
        return SweepResult(skipped=skipped)
    if not wedged:
        return SweepResult()
    for uuid in wedged:
        await cure(runner, uuid)
    remaining, _ = await query_wedged(runner)
    still = [uuid for uuid in wedged if uuid in remaining]
    cured = [uuid for uuid in wedged if uuid not in still]
    return SweepResult(wedged=wedged, cured=cured, still_wedged=still)
