from __future__ import annotations

import shlex
from typing import Any

from microtensor.rigs.protocol.failures import Failure, FailureClass, Verdict
from microtensor.rigs.validator.session.client import Session
from microtensor.rigs.validator.session.ssh import SshError, classify
from microtensor.rigs.validator.verify.payloads import source

PAYLOAD = "host_reality.py"
NESTED_MARKERS = (
    "/docker/",
    "/lxc/",
    "/containerd/",
    "/kubepods",
    "/podman/",
    "/machine.slice/libpod",
    "docker-",
    "/garden/",
)
VIRT_CLEAR = ("none", "unavailable", "")


def judge(host: dict[str, Any], seed: str = "") -> Failure | None:
    if not host.get("init_root_reachable"):
        return Failure(
            FailureClass.NESTED_CONTAINER, "the init process root is not reachable", seed=seed
        )
    if not host.get("init_root_differs"):
        return Failure(
            FailureClass.NESTED_CONTAINER,
            "the init process root is the session root, not a host root",
            seed=seed,
        )
    if not host.get("init_os_release_readable"):
        return Failure(
            FailureClass.NESTED_CONTAINER,
            "the host os-release is not readable through the init root",
            seed=seed,
        )
    cgroup = str(host.get("init_cgroup", "") or "")
    nested = [marker for marker in NESTED_MARKERS if marker in cgroup]
    if nested:
        return Failure(
            FailureClass.NESTED_CONTAINER,
            f"the init cgroup shows a nested path ({nested[0]})",
            seed=seed,
            evidence={"init_cgroup": cgroup[:300]},
        )
    virt = str(host.get("detect_virt", "") or "").strip().lower()
    if virt not in VIRT_CLEAR:
        return Failure(
            FailureClass.NESTED_CONTAINER, f"systemd-detect-virt reports {virt}", seed=seed
        )
    dmi = host.get("dmi") or {}
    if (
        not str(dmi.get("sys_vendor", "") or "").strip()
        and not str(dmi.get("product_name", "") or "").strip()
    ):
        return Failure(
            FailureClass.NESTED_CONTAINER, "no DMI vendor or product identity", seed=seed
        )
    return None


async def run(
    session: Session, session_dir: str, timeout: float, seed: str = ""
) -> tuple[Verdict, dict[str, Any]]:
    remote = f"{session_dir}/{PAYLOAD}"
    try:
        await session.upload(source(PAYLOAD), remote, 0o600)
        result, parsed = await session.run_json(
            f"{shlex.quote(session.python)} -I {shlex.quote(remote)}", timeout
        )
    except SshError as exc:
        return Verdict("host reality", failure=exc.failure(seed)), {}
    crash = classify(result, seed)
    if crash is not None and parsed is None:
        return Verdict("host reality", failure=crash, elapsed_ms=result.elapsed_ms), {}
    if parsed is None:
        return (
            Verdict(
                "host reality",
                failure=Failure(
                    FailureClass.AGENT_CRASH,
                    "the host reality payload printed no result",
                    seed=seed,
                ),
                elapsed_ms=result.elapsed_ms,
            ),
            {},
        )
    failure = judge(parsed, seed)
    dmi = parsed.get("dmi") or {}
    evidence = {
        "summary": f"{dmi.get('sys_vendor', '')} {dmi.get('product_name', '')}".strip() or "host",
        "detect_virt": parsed.get("detect_virt"),
        "init_root_differs": parsed.get("init_root_differs"),
        "init_comm": parsed.get("init_comm"),
        "boot_id": parsed.get("boot_id"),
        "machine_id": parsed.get("machine_id"),
        "kernel": parsed.get("kernel"),
    }
    return Verdict(
        "host reality", failure=failure, evidence=evidence, elapsed_ms=result.elapsed_ms
    ), parsed
