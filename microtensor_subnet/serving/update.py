from __future__ import annotations

import logging
import os
import re
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx

from microtensor_subnet.core.constants import RELEASE_VERSION
from microtensor_subnet.validator.sandbox.docker import Docker, DockerError, pinned

log = logging.getLogger("microtensor_subnet.serving.update")

AGENT_PATH: Final[str] = "/v2/agent"
DEFAULT_IMAGE: Final[str] = "ghcr.io/microtensor-io/validator"
COMPOSE_SERVICE: Final[str] = "operator"
IMAGE_KEY: Final[str] = "MT_AGENT_IMAGE"
HELPER_PREFIX: Final[str] = "mt-operator-update"
CONTAINER_ID: Final[re.Pattern[str]] = re.compile(r"/docker/containers/([0-9a-f]{64})/")
DIGEST: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
HELPER_SECONDS: Final[float] = 180.0
MIN_CHECK_SECONDS: Final[float] = 300.0


class UpdateError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Release:
    image: str
    digest: str
    version: str
    minimum: str

    @property
    def reference(self) -> str:
        return pinned(self.image, self.digest)


def parse_version(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in str(text or "").strip().split("."):
        digits = re.match(r"\d+", piece)
        parts.append(int(digits.group()) if digits else 0)
    return tuple(parts) or (0,)


def outdated(running: str, minimum: str) -> bool:
    if not minimum:
        return False
    return parse_version(running) < parse_version(minimum)


def release_of(document: Mapping[str, Any]) -> Release | None:
    inner = document.get("document")
    body: Mapping[str, Any] = inner if isinstance(inner, Mapping) else document
    image = str(body.get("image") or DEFAULT_IMAGE)
    digest = str(body.get("digest") or "").strip().lower()
    if not DIGEST.match(digest):
        return None
    return Release(
        image=image,
        digest=digest,
        version=str(body.get("version") or ""),
        minimum=str(body.get("minimum") or ""),
    )


def read_release(server: str, *, timeout: float = 30.0) -> Release | None:
    with httpx.Client(timeout=timeout) as http:
        response = http.get(server.rstrip("/") + AGENT_PATH)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return release_of(response.json())


def own_container_id() -> str:
    try:
        for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
            match = CONTAINER_ID.search(line)
            if match:
                return match.group(1)
    except OSError:
        pass
    name = socket.gethostname()
    return name if re.match(r"^[0-9a-f]{12,64}$", name) else ""


def running_digest(docker: Docker, container: str) -> str:
    if not container:
        return ""
    try:
        image = docker.runner.run(["inspect", "--format", "{{.Image}}", container]).strip()
        held = docker.image_digests(image) if image else []
    except DockerError:
        return ""
    return held[0] if held else ""


def compose_project(docker: Docker, container: str) -> str:
    if not container:
        return ""
    try:
        return docker.runner.run(
            [
                "inspect",
                "--format",
                '{{index .Config.Labels "com.docker.compose.project"}}',
                container,
            ]
        ).strip()
    except DockerError:
        return ""


def rewrite_env(path: Path, reference: str) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    kept = [line for line in lines if not line.startswith(f"{IMAGE_KEY}=")]
    kept.append(f"{IMAGE_KEY}={reference}")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    os.replace(temp, path)


@dataclass(slots=True)
class Updater:
    server: str
    docker: Docker
    compose_file: Path
    env_file: Path
    enabled: bool = True
    last_check: float = 0.0
    last_error: str = ""
    applied: str = ""

    def check(self) -> dict[str, Any]:
        self.last_check = time.monotonic()
        try:
            release = read_release(self.server)
        except (httpx.HTTPError, ValueError) as exc:
            self.last_error = f"release unreadable: {exc}"
            return {"applied": False, "reason": self.last_error}
        if release is None:
            return {"applied": False, "reason": "no agent release published"}
        if outdated(RELEASE_VERSION, release.minimum):
            log.warning(
                "this agent is %s and the network requires at least %s",
                RELEASE_VERSION,
                release.minimum,
            )
        container = own_container_id()
        running = running_digest(self.docker, container)
        if running == release.digest:
            return {"applied": False, "reason": "already on the published image", "digest": running}
        if not self.enabled:
            return {"applied": False, "reason": "auto update is off", "digest": release.digest}
        if not self.compose_file.is_file():
            self.last_error = f"compose file {self.compose_file} is not mounted"
            return {"applied": False, "reason": self.last_error}
        try:
            reference = self.docker.ensure_digest(release.image, release.digest)
            rewrite_env(self.env_file, reference)
            helper = self.restart(container, reference)
        except (DockerError, OSError, UpdateError) as exc:
            self.last_error = str(exc)
            return {"applied": False, "reason": self.last_error}
        self.applied = release.digest
        log.warning("moving to agent image %s through %s", reference, helper)
        return {"applied": True, "digest": release.digest, "helper": helper}

    def restart(self, container: str, reference: str) -> str:
        helper = f"{HELPER_PREFIX}-{int(time.time())}"
        argv = [
            "run",
            "-d",
            "--name",
            helper,
            "--entrypoint",
            "docker",
            "-v",
            "/var/run/docker.sock:/var/run/docker.sock",
        ]
        if container:
            argv += ["--volumes-from", container]
        argv.append(reference)
        argv += ["compose", "-f", str(self.compose_file), "--env-file", str(self.env_file)]
        project = compose_project(self.docker, container)
        if project:
            argv += ["-p", project]
        argv += ["up", "-d", "--no-deps", COMPOSE_SERVICE]
        self.docker.runner.run(argv, timeout=60.0)
        return helper

    def due(self, every: float) -> bool:
        return time.monotonic() - self.last_check >= max(MIN_CHECK_SECONDS, every)


def helper_outcome(docker: Docker, helper: str) -> tuple[bool, str]:
    try:
        code = docker.runner.run(["wait", helper], timeout=HELPER_SECONDS).strip()
        logs = docker.runner.run(["logs", "--tail", "40", helper], timeout=20.0)
        docker.runner.run(["rm", "-f", helper], timeout=20.0)
    except DockerError as exc:
        return False, str(exc)
    return code == "0", logs.strip()[-400:]


def wire(release: Release | None) -> dict[str, Any]:
    if release is None:
        return {"published": False}
    return {
        "published": True,
        "image": release.image,
        "digest": release.digest,
        "version": release.version,
        "minimum": release.minimum,
        "running": RELEASE_VERSION,
        "outdated": outdated(RELEASE_VERSION, release.minimum),
    }
