from __future__ import annotations

import secrets
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx

from microtensor_subnet.validator.rerun.engine import EngineError
from microtensor_subnet.validator.rerun.pod import (
    POD_PORT,
    START_CODE,
    PodClient,
    worker_payload,
)
from microtensor_subnet.validator.sandbox.docker import ContainerSpec, Docker

SYSTEM_DIR: Final[str] = "/opt/system"
POD_HOME: Final[str] = "/root/mt"
SHM_SIZE: Final[str] = "8g"
UP_SECONDS: Final[float] = 300.0
HEALTH_SECONDS: Final[float] = 10.0


def answers(url: str, token: str) -> bool:
    try:
        with httpx.Client(timeout=HEALTH_SECONDS) as http:
            found = http.get(url + "/health", headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError:
        return False
    return bool(found.status_code == 200)


@dataclass(slots=True)
class LocalPod:
    docker: Docker
    image: str
    gpus: Sequence[str]
    cache_dir: Path
    name: str = ""
    network: str = ""

    def guest(self, path: Path) -> str:
        try:
            inner = Path(path).relative_to(self.cache_dir)
        except ValueError as exc:
            raise EngineError(f"{path} is not under the cache folder {self.cache_dir}") from exc
        return f"{SYSTEM_DIR}/{inner.as_posix()}"

    def spec(self, token: str) -> ContainerSpec:
        return ContainerSpec(
            name=self.name,
            image=self.image,
            network=self.network,
            entrypoint="python3",
            command=("-c", START_CODE),
            env={
                "MT_POD_TOKEN": token,
                "MT_POD_WORKER": worker_payload(),
                "MT_POD_PORT": str(POD_PORT),
                "MT_POD_HOME": POD_HOME,
                "HF_HUB_OFFLINE": "1",
            },
            volumes=((str(self.cache_dir), SYSTEM_DIR, "ro"),),
            gpus=tuple(self.gpus),
            shm_size=SHM_SIZE,
        )

    def up(self) -> PodClient:
        self.docker.ensure_image(self.image)
        key = secrets.token_hex(4)
        self.name = f"mt-spot-{key}"
        self.network = f"mt-snet-{key}"
        token = secrets.token_urlsafe(32)
        self.docker.create_network(self.network)
        self.docker.start(self.spec(token))
        address = self.docker.address(self.name, self.network)
        return self.wait(f"http://{address}:{POD_PORT}", token)

    def wait(self, url: str, token: str) -> PodClient:
        deadline = time.monotonic() + UP_SECONDS
        while time.monotonic() < deadline:
            if not self.docker.running(self.name):
                raise EngineError(
                    f"the spot check worker exited: {self.docker.exit_note(self.name)}"
                )
            if answers(url, token):
                return PodClient(url, token)
            time.sleep(2.0)
        raise EngineError("the spot check worker did not answer in time")

    def down(self) -> None:
        if self.name:
            self.docker.kill(self.name)
        if self.network:
            self.docker.remove_network(self.network)
        self.name = ""
        self.network = ""
