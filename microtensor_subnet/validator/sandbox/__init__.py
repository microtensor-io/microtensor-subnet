from __future__ import annotations

import json
import logging
import os
import queue
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from microtensor_subnet.artifact import contract
from microtensor_subnet.artifact.gguf import gguf_in
from microtensor_subnet.validator.relay import TicketDesk
from microtensor_subnet.validator.sandbox import tools
from microtensor_subnet.validator.sandbox.docker import ContainerSpec, Docker, DockerError
from microtensor_subnet.validator.sandbox.services import ArtifactSession, Services
from microtensor_subnet.validator.sandbox.slm import (
    MEMORY_SHARE,
    SHARED_MEMORY_SHARE,
    SlmClient,
    llama_command,
    vllm_command,
)

log = logging.getLogger("microtensor_subnet.validator.sandbox")

GATEWAY_PORT = 9000
ROUTER_PORT_BASE = 9800
MAIN_PORT = 8080
READY_SECONDS = 180
ROUTER_INPUT_CEILING = contract.CONFIG_LIMITS["router"]["max_input_tokens"]


class SandboxError(RuntimeError):
    pass


class SandboxConfig(Protocol):
    @property
    def vllm_image(self) -> str: ...

    @property
    def llama_image(self) -> str: ...

    @property
    def sandbox_image(self) -> str: ...

    @property
    def gateway_image(self) -> str: ...

    @property
    def sandbox_runtime(self) -> str: ...

    @property
    def slm_memory_share(self) -> float: ...

    @property
    def sandbox_cpus(self) -> float: ...

    @property
    def sandbox_memory(self) -> str: ...

    @property
    def sandbox_pids(self) -> int: ...

    @property
    def slm_ready_seconds(self) -> int: ...


class GpuPool:
    def __init__(self, gpus: tuple[str, ...]) -> None:
        self._free: queue.Queue[tuple[int, str]] = queue.Queue()
        for slot, gpu in enumerate(gpus):
            self._free.put((slot, gpu))

    def take(self) -> tuple[int, str]:
        return self._free.get()

    def give(self, slot: int, gpu: str) -> None:
        self._free.put((slot, gpu))


def host_ip() -> str:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return str(probe.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


@dataclass(frozen=True, slots=True)
class Limits:
    max_model_len: int | None = None
    router_max_parameters: int = contract.ROUTER_CPU_PARAMETERS
    router_max_input_tokens: int = ROUTER_INPUT_CEILING
    max_model_calls: int = 0
    model_faults: bool = False
    device: str = "gpu"
    cpu_threads: int = 0
    memory_cap_gb: float = 0.0

    @property
    def on_cpu(self) -> bool:
        return self.device == "cpu"

    @property
    def router_on_gpu(self) -> bool:
        return self.router_max_parameters > contract.ROUTER_CPU_PARAMETERS

    @classmethod
    def from_definition(cls, definition: Mapping[str, Any], max_model_len: int | None) -> Limits:
        limits = definition.get("limits") or {}
        router = limits.get("router") or {} if isinstance(limits, Mapping) else {}
        harness = limits.get("harness") or {} if isinstance(limits, Mapping) else {}
        envelope = definition.get("envelope") or {}
        envelope = envelope if isinstance(envelope, Mapping) else {}
        reference = envelope.get("reference") or {}
        reference = reference if isinstance(reference, Mapping) else {}
        return cls(
            max_model_len=max_model_len,
            router_max_parameters=int(
                router.get("max_parameters") or contract.ROUTER_CPU_PARAMETERS
            ),
            router_max_input_tokens=int(router.get("max_input_tokens") or ROUTER_INPUT_CEILING),
            max_model_calls=int(harness.get("max_model_calls") or 0),
            model_faults=harness.get("faults") == "model",
            device=str(envelope.get("device") or "gpu"),
            cpu_threads=int(reference.get("threads") or 0),
            memory_cap_gb=float(reference.get("memory_cap_gb") or 0),
        )


def cpu_container(
    name: str,
    image: str,
    model_dir: Path,
    max_len: int,
    limits: Limits,
    network: str,
    ports: Sequence[tuple[str, int]] = (),
) -> ContainerSpec:
    found = gguf_in(model_dir)
    if not found:
        raise SandboxError("a CPU envelope needs one GGUF file in the model repository")
    threads = max(1, limits.cpu_threads)
    return ContainerSpec(
        name=name,
        image=image,
        network=network,
        command=llama_command(f"/model/{found[0].name}", threads=threads, max_model_len=max_len),
        volumes=((str(model_dir), "/model", "ro"),),
        cpus=float(threads),
        cpuset=f"0-{threads - 1}",
        memory=f"{limits.memory_cap_gb:g}g" if limits.memory_cap_gb else "",
        ports=ports,
    )


@dataclass(slots=True)
class Sandbox:
    key: str
    network: str
    slm_container: str
    gateway_container: str
    main_container: str
    router_process: subprocess.Popen[bytes] | None
    slm_url: str
    router_url: str
    main_url: str
    session: ArtifactSession
    gpu: str
    started_at: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at


@dataclass(slots=True)
class SandboxHost:
    config: SandboxConfig
    docker: Docker
    services: Services
    host_ip: str

    def _slm_container(
        self,
        key: str,
        model_dir: Path,
        gpu: str,
        network: str,
        config: dict[str, Any],
        limits: Limits,
    ) -> str:
        name = f"mt-slm-{key}"
        max_len = limits.max_model_len or int((config.get("slm") or {}).get("max_context") or 8192)
        if limits.on_cpu:
            self.docker.start(
                cpu_container(name, self.config.llama_image, model_dir, max_len, limits, network)
            )
            return name
        share = self.config.slm_memory_share or (
            SHARED_MEMORY_SHARE if limits.router_on_gpu else MEMORY_SHARE
        )
        self.docker.start(
            ContainerSpec(
                name=name,
                image=self.config.vllm_image,
                network=network,
                command=vllm_command("/model", max_model_len=max_len, memory_share=share),
                env={"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1"},
                volumes=((str(model_dir), "/model", "ro"),),
                gpus=(gpu,),
                shm_size="8g",
                ipc_none=False,
            )
        )
        return name

    def _router_process(
        self,
        key: str,
        router_dir: Path,
        port: int,
        config: dict[str, Any],
        gpu: str,
        limits: Limits,
    ) -> subprocess.Popen[bytes]:
        wanted = int((config.get("router") or {}).get("max_input_tokens") or 4096)
        max_input = min(wanted, limits.router_max_input_tokens)
        env = dict(os.environ)
        if limits.router_on_gpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        return subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-m",
                "microtensor_subnet.validator.sandbox.router_server",
                "--model",
                str(router_dir),
                "--port",
                str(port),
                "--listen",
                self.host_ip,
                "--max-input-tokens",
                str(max_input),
                "--max-parameters",
                str(limits.router_max_parameters),
                "--device",
                "cuda" if limits.router_on_gpu else "cpu",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )

    def _gateway_container(self, key: str, network: str) -> str:
        name = f"mt-gw-{key}"
        self.docker.start(
            ContainerSpec(
                name=name,
                image=self.config.gateway_image,
                network="bridge",
                command=(
                    "python",
                    "-m",
                    "microtensor_subnet.validator.sandbox.gateway",
                    "--upstream",
                    f"http://{self.host_ip}:{self.services.port}",
                    "--sandbox",
                    key,
                    "--port",
                    str(GATEWAY_PORT),
                ),
                cpus=1.0,
                memory="512m",
                pids=64,
                read_only=True,
                hardened=False,
                extra_networks=(network,),
                tmpfs=("/tmp",),  # noqa: S108
            )
        )
        return name

    def _main_container(self, key: str, artifact_dir: Path, network: str, gateway_ip: str) -> str:
        name = f"mt-run-{key}"
        base = f"http://{gateway_ip}:{GATEWAY_PORT}"
        self.docker.start(
            ContainerSpec(
                name=name,
                image=self.config.sandbox_image,
                network=network,
                command=("python", "/artifact/main.py"),
                env={
                    contract.ENV_PORT: str(MAIN_PORT),
                    contract.ENV_SLM_URL: base + "/slm",
                    contract.ENV_ROUTER_URL: base + "/router",
                    contract.ENV_TOOLS_URL: base,
                    contract.ENV_RELAY_URL: base + "/relay",
                    "HOME": "/tmp",  # noqa: S108
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                volumes=((str(artifact_dir), "/artifact", "ro"),),
                runtime=self.config.sandbox_runtime,
                cpus=self.config.sandbox_cpus,
                memory=self.config.sandbox_memory,
                pids=self.config.sandbox_pids,
                read_only=True,
                hardened=True,
                user="65534:65534",
                tmpfs=("/tmp",),  # noqa: S108
                ipc_none=True,
            )
        )
        return name

    def open(
        self,
        *,
        artifact_dir: Path,
        slm_dir: Path,
        router_dir: Path | None,
        gpu: str,
        config: dict[str, Any],
        sample_database: bytes,
        fault_plan: tools.FaultPlan | None,
        desk: TicketDesk | None,
        slot: int,
        databases: Mapping[str, bytes] | None = None,
        item_databases: Mapping[str, Sequence[str]] | None = None,
        limits: Limits | None = None,
    ) -> Sandbox:
        chosen = limits or Limits()
        key = secrets.token_hex(4)
        network = f"mt-net-{key}"
        self.docker.create_network(network)
        slm_name = gateway_name = main_name = ""
        router: subprocess.Popen[bytes] | None = None
        try:
            slm_name = self._slm_container(key, slm_dir, gpu, network, config, chosen)
            slm_ip = self.docker.address(slm_name, network)
            router_port = ROUTER_PORT_BASE + slot
            router_url = ""
            if router_dir is not None:
                router = self._router_process(key, router_dir, router_port, config, gpu, chosen)
                router_url = f"http://{self.host_ip}:{router_port}"
            gateway_name = self._gateway_container(key, network)
            gateway_ip = self.docker.address(gateway_name, network)
            session = ArtifactSession(
                sample_database=sample_database,
                fault_plan=fault_plan,
                slm_url=f"http://{slm_ip}:8000",
                router_url=router_url,
                desk=desk,
                databases=dict(databases or {}),
                item_databases={k: tuple(v) for k, v in (item_databases or {}).items()},
                max_model_calls=chosen.max_model_calls,
                model_faults=chosen.model_faults,
            )
            self.services.attach(key, session)
            self._wait_slm(session.slm_url)
            if router is not None:
                self._wait_router(router, router_url)
            main_name = self._main_container(key, artifact_dir, network, gateway_ip)
            main_ip = self.docker.address(main_name, network)
            main_url = f"http://{main_ip}:{MAIN_PORT}"
            self._wait_main(main_name, main_url)
        except Exception:
            self._teardown(key, network, slm_name, gateway_name, main_name, router)
            raise
        return Sandbox(
            key=key,
            network=network,
            slm_container=slm_name,
            gateway_container=gateway_name,
            main_container=main_name,
            router_process=router,
            slm_url=session.slm_url,
            router_url=router_url,
            main_url=main_url,
            session=session,
            gpu=gpu,
        )

    def _wait_slm(self, url: str) -> None:
        client = SlmClient(url)
        deadline = time.monotonic() + self.config.slm_ready_seconds
        while time.monotonic() < deadline:
            if client.ready():
                return
            time.sleep(5.0)
        raise SandboxError("the model server did not become ready")

    def _wait_router(self, process: subprocess.Popen[bytes], url: str) -> None:
        deadline = time.monotonic() + READY_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                detail = (
                    process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
                )[-300:]
                raise SandboxError(f"the router did not load: {detail}")
            try:
                with httpx.Client(timeout=5.0) as http:
                    if http.get(url + "/health").status_code == 200:
                        return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
        raise SandboxError("the router server did not become ready")

    def _wait_main(self, name: str, url: str) -> None:
        deadline = time.monotonic() + READY_SECONDS
        while time.monotonic() < deadline:
            if not self.docker.running(name):
                raise SandboxError(f"main.py exited: {self.docker.logs(name, 40)[-300:]}")
            try:
                with httpx.Client(timeout=5.0) as http:
                    if http.get(url + "/health").status_code < 500:
                        return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
        raise SandboxError("main.py did not start listening")

    def _teardown(
        self,
        key: str,
        network: str,
        slm_name: str,
        gateway_name: str,
        main_name: str,
        router: subprocess.Popen[bytes] | None,
    ) -> None:
        self.services.detach(key)
        for name in (main_name, gateway_name, slm_name):
            if name:
                self.docker.kill(name)
        if router is not None and router.poll() is None:
            router.kill()
        self.docker.remove_network(network)

    def close(self, sandbox: Sandbox) -> dict[str, Any]:
        usage = self.docker.usage(sandbox.main_container)
        self._teardown(
            sandbox.key,
            sandbox.network,
            sandbox.slm_container,
            sandbox.gateway_container,
            sandbox.main_container,
            sandbox.router_process,
        )
        return usage

    def open_router(
        self,
        router_dir: Path,
        *,
        slot: int,
        gpu: str,
        config: dict[str, Any],
        limits: Limits | None = None,
    ) -> tuple[subprocess.Popen[bytes], str]:
        port = ROUTER_PORT_BASE + slot
        process = self._router_process("", router_dir, port, config, gpu, limits or Limits())
        url = f"http://{self.host_ip}:{port}"
        try:
            self._wait_router(process, url)
        except Exception:
            self.close_router(process)
            raise
        return process, url

    def close_router(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)

    def restart_main(self, sandbox: Sandbox, artifact_dir: Path) -> None:
        self.docker.kill(sandbox.main_container)
        gateway_ip = self.docker.address(sandbox.gateway_container, sandbox.network)
        sandbox.main_container = self._main_container(
            sandbox.key, artifact_dir, sandbox.network, gateway_ip
        )
        main_ip = self.docker.address(sandbox.main_container, sandbox.network)
        sandbox.main_url = f"http://{main_ip}:{MAIN_PORT}"
        self._wait_main(sandbox.main_container, sandbox.main_url)


@dataclass(frozen=True, slots=True)
class Call:
    ok: bool
    body: dict[str, Any]
    latency_ms: float
    error: str = ""


def call_main(url: str, path: str, payload: dict[str, Any], *, timeout: float) -> Call:
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as http:
            response = http.post(url.rstrip("/") + path, json=payload)
    except httpx.TimeoutException:
        return Call(False, {}, (time.perf_counter() - started) * 1000, "timeout")
    except httpx.HTTPError as exc:
        return Call(False, {}, (time.perf_counter() - started) * 1000, f"unreachable: {exc}")
    latency = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        return Call(False, {}, latency, f"status {response.status_code}")
    try:
        body = response.json()
    except ValueError:
        return Call(False, {}, latency, "not json")
    if not isinstance(body, dict):
        return Call(False, {}, latency, "not an object")
    return Call(True, body, latency)


def write_config(artifact_dir: Path, config: dict[str, Any]) -> None:
    (artifact_dir / contract.CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")


def docker_available(docker: Docker) -> bool:
    try:
        docker.runner.run(["version", "--format", "{{.Server.Version}}"])
    except DockerError:
        return False
    return True
