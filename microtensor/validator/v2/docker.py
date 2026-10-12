from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger("microtensor.validator.v2.docker")

DOCKER = "docker"
COMMAND_SECONDS = 120.0


class DockerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ContainerSpec:
    name: str
    image: str
    network: str
    command: Sequence[str] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    volumes: Sequence[tuple[str, str, str]] = ()
    gpus: Sequence[str] = ()
    runtime: str = ""
    cpus: float = 0.0
    memory: str = ""
    pids: int = 0
    read_only: bool = False
    hardened: bool = False
    user: str = ""
    extra_networks: Sequence[str] = ()
    tmpfs: Sequence[str] = ()
    shm_size: str = ""
    ipc_none: bool = False


class Runner(Protocol):
    def run(self, argv: Sequence[str], *, timeout: float = COMMAND_SECONDS) -> str: ...


class SubprocessRunner:
    def run(self, argv: Sequence[str], *, timeout: float = COMMAND_SECONDS) -> str:
        try:
            found = subprocess.run(  # noqa: S603
                [DOCKER, *argv], capture_output=True, text=True, timeout=timeout, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DockerError(f"docker {argv[0]} failed to start: {exc}") from exc
        if found.returncode != 0:
            raise DockerError(f"docker {' '.join(argv[:2])}: {found.stderr.strip()[-400:]}")
        return found.stdout


def _flags(spec: ContainerSpec) -> list[str]:
    argv = ["run", "-d", "--name", spec.name, "--network", spec.network, "--init"]
    for key, value in spec.env.items():
        argv += ["-e", f"{key}={value}"]
    for host, guest, mode in spec.volumes:
        argv += ["-v", f"{host}:{guest}:{mode}"]
    for path in spec.tmpfs:
        argv += ["--tmpfs", f"{path}:rw,noexec,nosuid,size=256m"]
    if spec.gpus:
        argv += ["--gpus", f"device={','.join(spec.gpus)}"]
    if spec.runtime:
        argv += ["--runtime", spec.runtime]
    if spec.cpus:
        argv += ["--cpus", f"{spec.cpus:g}"]
    if spec.memory:
        argv += ["--memory", spec.memory, "--memory-swap", spec.memory]
    if spec.pids:
        argv += ["--pids-limit", str(spec.pids), "--ulimit", f"nproc={spec.pids}"]
    if spec.read_only:
        argv.append("--read-only")
    if spec.hardened:
        argv += [
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--sysctl",
            "net.ipv6.conf.all.disable_ipv6=1",
            "--dns",
            "0.0.0.0",  # noqa: S104
        ]
    if spec.ipc_none:
        argv += ["--ipc", "none"]
    if spec.user:
        argv += ["--user", spec.user]
    if spec.shm_size:
        argv += ["--shm-size", spec.shm_size]
    argv += ["--label", "microtensor=sandbox"]
    argv.append(spec.image)
    argv += list(spec.command)
    return argv


@dataclass(slots=True)
class Docker:
    runner: Runner = field(default_factory=SubprocessRunner)

    def create_network(self, name: str, *, internal: bool = True) -> None:
        argv = ["network", "create", "--label", "microtensor=sandbox"]
        if internal:
            argv += ["--internal", "--opt", "com.docker.network.bridge.gateway_mode_ipv4=isolated"]
        argv.append(name)
        try:
            self.runner.run(argv)
        except DockerError as exc:
            if "gateway_mode" in str(exc):
                self.runner.run(
                    ["network", "create", "--label", "microtensor=sandbox", "--internal", name]
                )
            elif "already exists" not in str(exc):
                raise

    def remove_network(self, name: str) -> None:
        try:
            self.runner.run(["network", "rm", name])
        except DockerError as exc:
            log.debug("network %s not removed: %s", name, exc)

    def connect(self, network: str, container: str, alias: str = "") -> None:
        argv = ["network", "connect"]
        if alias:
            argv += ["--alias", alias]
        self.runner.run([*argv, network, container])

    def start(self, spec: ContainerSpec) -> str:
        container = self.runner.run(_flags(spec)).strip()
        for network in spec.extra_networks:
            self.connect(network, spec.name)
        return container

    def kill(self, name: str) -> None:
        for argv in (["kill", name], ["rm", "-f", name]):
            try:
                self.runner.run(argv)
            except DockerError as exc:
                log.debug("%s: %s", " ".join(argv), exc)

    def address(self, name: str, network: str) -> str:
        raw = self.runner.run(["inspect", name])
        try:
            info = json.loads(raw)[0]
            return str(info["NetworkSettings"]["Networks"][network]["IPAddress"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise DockerError(f"no address for {name} on {network}") from exc

    def running(self, name: str) -> bool:
        try:
            raw = self.runner.run(["inspect", "-f", "{{.State.Running}}", name])
        except DockerError:
            return False
        return raw.strip() == "true"

    def exit_code(self, name: str) -> int | None:
        try:
            raw = self.runner.run(["inspect", "-f", "{{.State.ExitCode}}", name])
        except DockerError:
            return None
        return int(raw.strip()) if raw.strip().lstrip("-").isdigit() else None

    def logs(self, name: str, tail: int = 200) -> str:
        try:
            return self.runner.run(["logs", "--tail", str(tail), name])
        except DockerError:
            return ""

    def usage(self, name: str) -> dict[str, Any]:
        try:
            raw = self.runner.run(["inspect", "-f", "{{.Id}}", name]).strip()
        except DockerError:
            return {}
        if not raw:
            return {}
        try:
            text = self.runner.run(
                [
                    "exec",
                    name,
                    "sh",
                    "-c",
                    "cat /sys/fs/cgroup/cpu.stat /sys/fs/cgroup/memory.peak "
                    "/sys/fs/cgroup/pids.current 2>/dev/null",
                ]
            )
        except DockerError:
            return {}
        usage: dict[str, Any] = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "usage_usec":
                usage["cpu_seconds"] = int(parts[1]) / 1_000_000
            elif len(parts) == 1 and parts[0].isdigit():
                usage.setdefault("memory_peak_bytes", int(parts[0]))
        return usage

    def wait_running(self, name: str, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.running(name):
                return True
            if self.exit_code(name) not in (None, 0):
                return False
            time.sleep(1.0)
        return self.running(name)

    def sweep(self) -> None:
        try:
            names = self.runner.run(["ps", "-aq", "--filter", "label=microtensor=sandbox"]).split()
        except DockerError:
            return
        for name in names:
            self.kill(name)
        try:
            networks = self.runner.run(
                ["network", "ls", "-q", "--filter", "label=microtensor=sandbox"]
            ).split()
        except DockerError:
            return
        for network in networks:
            self.remove_network(network)
