from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any

from microtensor.rigs.validator.recovery import gpu_wedge
from microtensor.rigs.validator.session.runner import Runner
from microtensor.rigs.validator.storage import volumes

SCRUB_TIMEOUT = 120.0
CHUNK_BYTES = 256 * 1024 * 1024
SCRUB_MARKER = "scrubbed_bytes="
SCRUB_SNIPPET = (
    "import ctypes,sys\n"
    f"chunk={CHUNK_BYTES}\n"
    "def runtime():\n"
    "    for name in ('libcudart.so','libcudart.so.12','libcudart.so.11.0'):\n"
    "        try: return ctypes.CDLL(name)\n"
    "        except OSError: pass\n"
    "    return None\n"
    "total=0\n"
    "rt=runtime()\n"
    "if rt is not None:\n"
    "    assert rt.cudaSetDevice(0)==0,'cudaSetDevice'\n"
    "    held=[]\n"
    "    while True:\n"
    "        p=ctypes.c_void_p()\n"
    "        if rt.cudaMalloc(ctypes.byref(p),ctypes.c_size_t(chunk))!=0: break\n"
    "        assert rt.cudaMemset(p,0,ctypes.c_size_t(chunk))==0,'cudaMemset'\n"
    "        held.append(p); total+=chunk\n"
    "    assert rt.cudaDeviceSynchronize()==0,'cudaDeviceSynchronize'\n"
    "    for p in held: rt.cudaFree(p)\n"
    "    rt.cudaDeviceReset()\n"
    "else:\n"
    "    cu=ctypes.CDLL('libcuda.so.1')\n"
    "    assert cu.cuInit(0)==0,'cuInit'\n"
    "    d=ctypes.c_int(); assert cu.cuDeviceGet(ctypes.byref(d),0)==0,'cuDeviceGet'\n"
    "    c=ctypes.c_void_p(); assert cu.cuCtxCreate_v2(ctypes.byref(c),0,d)==0,'cuCtxCreate'\n"
    "    held=[]\n"
    "    while True:\n"
    "        p=ctypes.c_uint64()\n"
    "        if cu.cuMemAlloc_v2(ctypes.byref(p),ctypes.c_size_t(chunk))!=0: break\n"
    "        assert cu.cuMemsetD8_v2(p,ctypes.c_ubyte(0),ctypes.c_size_t(chunk))==0,'cuMemsetD8'\n"
    "        held.append(p); total+=chunk\n"
    "    assert cu.cuCtxSynchronize()==0,'cuCtxSynchronize'\n"
    "    for p in held: cu.cuMemFree_v2(p)\n"
    "    cu.cuCtxDestroy_v2(c)\n"
    f"print('{SCRUB_MARKER}%d'%total)\n"
)
SCRUB_PATTERN = re.compile(re.escape(SCRUB_MARKER) + r"(\d+)")


@dataclass
class TeardownReport:
    scrubbed: dict[str, int] = field(default_factory=dict)
    scrub_errors: dict[str, str] = field(default_factory=dict)
    volumes_removed: list[str] = field(default_factory=list)
    volume_errors: dict[str, str] = field(default_factory=dict)
    wedge: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "scrubbed": dict(self.scrubbed),
            "scrub_errors": dict(self.scrub_errors),
            "volumes_removed": list(self.volumes_removed),
            "volume_errors": dict(self.volume_errors),
            "wedge": dict(self.wedge),
        }


async def scrub_vram(runner: Runner, uuid: str, python: str = "python3") -> tuple[int, str]:
    if not re.match(r"^GPU-[0-9a-f-]{36}$", uuid or ""):
        return 0, f"uuid {uuid!r} is not accepted"
    command = f"CUDA_VISIBLE_DEVICES={shlex.quote(uuid)} {shlex.quote(python)} -c {shlex.quote(SCRUB_SNIPPET)}"
    result = await runner.run(command, timeout=SCRUB_TIMEOUT)
    match = SCRUB_PATTERN.search(result.stdout)
    if not result.ok or not match:
        return 0, (result.error or result.stderr_tail(300)).strip() or "no scrub marker"
    return int(match.group(1)), ""


async def after_teardown(
    runner: Runner,
    gpu_uuids: list[str],
    volume_names: list[str],
    python: str = "python3",
) -> TeardownReport:
    report = TeardownReport()
    for uuid in gpu_uuids:
        scrubbed, error = await scrub_vram(runner, uuid, python)
        if error:
            report.scrub_errors[uuid] = error
        else:
            report.scrubbed[uuid] = scrubbed
    for name in volume_names:
        ok, reason = await volumes.destroy(runner, name)
        if ok:
            report.volumes_removed.append(name)
        else:
            report.volume_errors[name] = reason
    report.wedge = (await gpu_wedge.sweep(runner)).payload()
    return report
