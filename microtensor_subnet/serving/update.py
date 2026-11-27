from __future__ import annotations

import json
import logging
import os
import posixpath
import re
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx

from microtensor_subnet.core.constants import RELEASE_VERSION
from microtensor_subnet.validator.sandbox.docker import Docker, DockerError, image_name, pinned

log = logging.getLogger("microtensor_subnet.serving.update")

AGENT_PATH: Final[str] = "/v2/agent"
TRUSTED_IMAGE: Final[str] = "ghcr.io/microtensor-io/validator"
TRUSTED_TITLE: Final[str] = "validator"
SIGNER_IDENTITY: Final[str] = (
    "https://github.com/microtensor-io/microtensor-subnet/.github/workflows/images.yml"
    "@refs/heads/main"
)
SIGNER_ISSUER: Final[str] = "https://token.actions.githubusercontent.com"
VERSION_LABEL: Final[str] = "org.opencontainers.image.version"
TITLE_LABEL: Final[str] = "org.opencontainers.image.title"
BUILD_LABEL: Final[str] = "io.microtensor.build"
DIGEST_LABEL: Final[str] = "io.microtensor.update.digest"
HELPER_LABEL: Final[str] = "microtensor=operator-update"
HELPER_PREFIX: Final[str] = "mt-operator-update"
ENTRYPOINT: Final[str] = "/usr/local/bin/mt-entrypoint"
COMPOSE_SERVICE: Final[str] = "operator"
IMAGE_KEY: Final[str] = "MT_AGENT_IMAGE"
PROJECT_LABEL: Final[str] = "com.docker.compose.project"
SERVICE_LABEL: Final[str] = "com.docker.compose.service"
WORKDIR_LABEL: Final[str] = "com.docker.compose.project.working_dir"
FILES_LABEL: Final[str] = "com.docker.compose.project.config_files"
ENV_LABEL: Final[str] = "com.docker.compose.project.environment_file"
WALLETS: Final[str] = "/wallets"
SOCKET: Final[str] = "/var/run/docker.sock"
HOME: Final[str] = "/var/lib/microtensor"
PACKAGE_DIR: Final[str] = "/opt/microtensor"
SYSTEM_DIRS: Final[frozenset[str]] = frozenset(
    {"/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib64", "/opt", "/proc"}
    | {"/root", "/run", "/sbin", "/srv", "/sys", "/usr", "/var", HOME}
)
REFUSED_FILE: Final[str] = "update-refused.json"
CONTAINER_ID: Final[re.Pattern[str]] = re.compile(r"/docker/containers/([0-9a-f]{64})/")
DIGEST: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
UNSTEADY_EXIT: Final[int] = 3
COSIGN_SECONDS: Final[float] = 120.0
COMPOSE_SECONDS: Final[float] = 900.0
HELPER_SECONDS: Final[float] = 900.0
SETTLE_SECONDS: Final[float] = 300.0
STEADY_SECONDS: Final[float] = 60.0
POLL_SECONDS: Final[float] = 3.0
RETRY_SECONDS: Final[float] = 3600.0
MIN_CHECK_SECONDS: Final[float] = 300.0


class UpdateError(RuntimeError):
    pass


class Unsteady(UpdateError):
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


@dataclass(frozen=True, slots=True, order=True)
class Build:
    version: tuple[int, ...]
    number: int

    def describe(self) -> str:
        return ".".join(str(part) for part in self.version) + f" build {self.number}"


@dataclass(frozen=True, slots=True)
class Project:
    name: str
    service: str
    workdir: str
    files: tuple[str, ...]
    env_file: str

    def argv(self, *tail: str) -> list[str]:
        argv = ["compose"]
        for name in self.files:
            argv += ["-f", name]
        argv += ["--project-directory", self.workdir, "--env-file", self.env_file]
        if self.name:
            argv += ["-p", self.name]
        return [*argv, *tail]

    def folders(self) -> list[str]:
        found: list[str] = []
        parents = [posixpath.dirname(name) for name in self.files]
        for path in (self.workdir, *parents, posixpath.dirname(self.env_file)):
            if path and path not in found:
                found.append(path)
        return found


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
    signed = body.get("signed")
    if not isinstance(signed, Mapping):
        return None
    digest = str(signed.get("digest") or "").strip().lower()
    if not DIGEST.match(digest):
        return None
    return Release(
        image=str(signed.get("image") or TRUSTED_IMAGE),
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


def trusted_reference(image: str, digest: str) -> str:
    named = image_name(image)
    if named != TRUSTED_IMAGE:
        raise UpdateError(f"the release names {named}; only {TRUSTED_IMAGE} is installed")
    try:
        return pinned(TRUSTED_IMAGE, digest)
    except DockerError as exc:
        raise UpdateError(str(exc)) from exc


def verify_signature(reference: str) -> None:
    argv = [
        os.environ.get("MT_COSIGN", "cosign"),
        "verify",
        "--certificate-identity",
        SIGNER_IDENTITY,
        "--certificate-oidc-issuer",
        SIGNER_ISSUER,
        "--output",
        "json",
        reference,
    ]
    try:
        done = subprocess.run(  # noqa: S603
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=COSIGN_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise UpdateError("cosign is not installed here, so no update can be checked") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise UpdateError(f"the signature on {reference} could not be checked: {exc}") from exc
    if done.returncode != 0:
        said = (done.stderr or done.stdout or "").strip().splitlines()
        detail = said[-1] if said else f"cosign exited {done.returncode}"
        raise UpdateError(
            f"{reference} is not signed by the subnet's images workflow on main: {detail}"
        )


def labels_of_image(docker: Docker, reference: str) -> dict[str, str]:
    raw = docker.runner.run(
        ["image", "inspect", "--format", "{{json .Config.Labels}}", reference]
    ).strip()
    try:
        found = json.loads(raw or "null")
    except ValueError:
        return {}
    if not isinstance(found, Mapping):
        return {}
    return {str(key): str(value) for key, value in found.items()}


def build_of(labels: Mapping[str, str], fallback: str = "") -> Build:
    number = str(labels.get(BUILD_LABEL) or "")
    version = str(labels.get(VERSION_LABEL) or fallback)
    return Build(parse_version(version), int(number) if number.isdigit() else 0)


def own_image(docker: Docker, container: str) -> str:
    if not container:
        return ""
    try:
        return docker.runner.run(["inspect", "--format", "{{.Image}}", container]).strip()
    except DockerError:
        return ""


def newer_than(docker: Docker, reference: str, container: str) -> Build:
    offered = labels_of_image(docker, reference)
    title = offered.get(TITLE_LABEL, "")
    if title != TRUSTED_TITLE:
        raise UpdateError(
            f"the published image is titled {title or 'nothing'}, not {TRUSTED_TITLE}"
        )
    if not offered.get(VERSION_LABEL):
        raise UpdateError(f"the published image carries no {VERSION_LABEL} label")
    current = own_image(docker, container)
    held = build_of(labels_of_image(docker, current) if current else {}, RELEASE_VERSION)
    found = build_of(offered)
    if found <= held:
        raise UpdateError(
            f"the published image is {found.describe()}, not newer than {held.describe()}"
        )
    return found


def vet(docker: Docker, image: str, digest: str, container: str) -> str:
    reference = trusted_reference(image, digest)
    verify_signature(reference)
    docker.ensure_digest(TRUSTED_IMAGE, digest)
    newer_than(docker, reference, container)
    return reference


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
    image = own_image(docker, container)
    if not image:
        return ""
    held = docker.image_digests(image)
    return held[0] if held else ""


def inspect_json(docker: Docker, target: str, template: str) -> Any:
    raw = docker.runner.run(["inspect", "--format", "{{json " + template + "}}", target]).strip()
    try:
        return json.loads(raw or "null")
    except ValueError:
        return None


def labels_of(docker: Docker, container: str) -> dict[str, str]:
    found = inspect_json(docker, container, ".Config.Labels")
    if not isinstance(found, Mapping):
        return {}
    return {str(key): str(value) for key, value in found.items()}


def project_of(labels: Mapping[str, str]) -> Project:
    workdir = labels.get(WORKDIR_LABEL, "")
    files = tuple(name for name in labels.get(FILES_LABEL, "").split(",") if name)
    if not workdir or not files:
        raise UpdateError("this agent was not started by docker compose, so update it by hand")
    envs = [name for name in labels.get(ENV_LABEL, "").split(",") if name]
    project = Project(
        name=labels.get(PROJECT_LABEL, ""),
        service=labels.get(SERVICE_LABEL, "") or COMPOSE_SERVICE,
        workdir=workdir,
        files=files,
        env_file=envs[0] if envs else posixpath.join(workdir, ".env"),
    )
    for folder in project.folders():
        inside = folder == PACKAGE_DIR or folder.startswith(PACKAGE_DIR + "/")
        if not folder.startswith("/") or folder in SYSTEM_DIRS or inside:
            raise UpdateError(
                f"the compose project lives in {folder}; move it to its own folder to update"
            )
    return project


def mount_source(docker: Docker, container: str, destination: str) -> str:
    mounts = inspect_json(docker, container, ".Mounts")
    for mount in mounts if isinstance(mounts, list) else []:
        if isinstance(mount, Mapping) and mount.get("Destination") == destination:
            return str(mount.get("Source") or "")
    return ""


def host_settings(docker: Docker, container: str) -> dict[str, str]:
    settings: dict[str, str] = {}
    wallets = mount_source(docker, container, WALLETS)
    if wallets:
        settings["MT_WALLET_DIR"] = wallets
    groups = inspect_json(docker, container, ".HostConfig.GroupAdd")
    if isinstance(groups, list) and groups:
        settings["DOCKER_GID"] = str(groups[0])
    return settings


def with_image(text: str, reference: str) -> str:
    kept = [line for line in text.splitlines() if not line.startswith(f"{IMAGE_KEY}=")]
    kept.append(f"{IMAGE_KEY}={reference}")
    return "".join(line + "\n" for line in kept)


def write_in_place(path: Path, text: str) -> None:
    with path.open("a+", encoding="utf-8") as handle:
        handle.seek(0)
        handle.truncate()
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def refused_file(home: str | Path | None = None) -> Path:
    return Path(home or os.environ.get("MT_HOME") or HOME) / "state" / REFUSED_FILE


def load_refused(path: Path) -> dict[str, str]:
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(found, Mapping):
        return {}
    return {str(key): str(value) for key, value in found.items() if DIGEST.match(str(key))}


def remember(path: Path, digest: str, reason: str) -> None:
    held = load_refused(path)
    held[digest] = reason[:400]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(held, indent=2, sort_keys=True), encoding="utf-8")
        owner = path.parent.stat()
        if os.geteuid() == 0:
            os.chown(path, owner.st_uid, owner.st_gid)
    except OSError as exc:
        log.warning("the refusal of %s could not be saved: %s", digest[:19], exc)


def image_id(docker: Docker, reference: str) -> str:
    found = docker.runner.run(["image", "inspect", "--format", "{{.Id}}", reference]).strip()
    if not found:
        raise UpdateError(f"{reference} is not on this host")
    return found


def settle_on(
    docker: Docker,
    project: Project,
    wanted: str,
    settle: float = SETTLE_SECONDS,
    steady: float = STEADY_SECONDS,
) -> None:
    deadline = time.monotonic() + settle
    first: tuple[str, str, float] | None = None
    last = "no container"
    while time.monotonic() < deadline:
        ids = docker.runner.run(
            [
                "ps",
                "-q",
                "--filter",
                f"label={PROJECT_LABEL}={project.name}",
                "--filter",
                f"label={SERVICE_LABEL}={project.service}",
            ]
        ).split()
        seen: tuple[str, str] | None = None
        for found in ids:
            template = "{{.State.Status}}|{{.State.Restarting}}|{{.Image}}|{{.State.StartedAt}}"
            parts = docker.runner.run(["inspect", "--format", template, found]).strip().split("|")
            last = "|".join(parts)
            if len(parts) == 4 and parts[:3] == ["running", "false", wanted]:
                seen = (found, parts[3])
                break
        now = time.monotonic()
        if seen is None:
            first = None
        elif first is None or (first[0], first[1]) != seen:
            first = (seen[0], seen[1], now)
        elif now - first[2] >= steady:
            return
        time.sleep(POLL_SECONDS)
    raise Unsteady(f"the new agent did not stay up for {steady:.0f}s within {settle:.0f}s ({last})")


def apply(
    docker: Docker,
    reference: str,
    container: str,
    *,
    settle: float = SETTLE_SECONDS,
    steady: float = STEADY_SECONDS,
    home: str | None = None,
) -> str:
    image, _, digest = reference.partition("@")
    reference = vet(docker, image, digest, container)
    project = project_of(labels_of(docker, container))
    os.environ.update(host_settings(docker, container))
    wanted = image_id(docker, reference)
    env_path = Path(project.env_file)
    before = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    write_in_place(env_path, with_image(before, reference))
    up = project.argv("up", "-d", "--no-deps", project.service)
    try:
        docker.runner.run(up, timeout=COMPOSE_SECONDS)
        settle_on(docker, project, wanted, settle, steady)
    except (DockerError, UpdateError) as exc:
        remember(refused_file(home), digest, f"the new agent did not stay up: {exc}")
        write_in_place(env_path, before)
        try:
            docker.runner.run(up, timeout=COMPOSE_SECONDS)
        except DockerError as again:
            log.error("the previous agent could not be started again: %s", again)
        raise Unsteady(
            f"the move to {reference} failed and {env_path} was restored: {exc}"
        ) from exc
    return wanted


def helper_state(docker: Docker, helper: str) -> str:
    try:
        return docker.runner.run(["inspect", "--format", "{{.State.Status}}", helper]).strip()
    except DockerError:
        return ""


def helper_outcome(docker: Docker, helper: str) -> tuple[int, str]:
    try:
        code = docker.runner.run(["wait", helper], timeout=HELPER_SECONDS).strip()
        logs = docker.runner.run(["logs", "--tail", "40", helper], timeout=20.0)
        docker.runner.run(["rm", "-f", helper], timeout=20.0)
    except DockerError as exc:
        return -1, str(exc)
    return int(code) if code.lstrip("-").isdigit() else -1, logs.strip()[-400:]


@dataclass(slots=True)
class Updater:
    server: str
    docker: Docker
    enabled: bool = False
    home: str | None = None
    last_check: float = 0.0
    last_error: str = ""
    applied: str = ""
    helper: str = ""
    ready: tuple[str, str] = ("", "")
    stopping: threading.Event = field(default_factory=threading.Event)
    retry: dict[str, tuple[str, float]] = field(default_factory=dict)

    def published(self) -> Release | None:
        self.last_check = time.monotonic()
        try:
            return read_release(self.server)
        except (httpx.HTTPError, ValueError) as exc:
            self.last_error = f"release unreadable: {exc}"
            return None

    def refusal(self, digest: str) -> str:
        lasting = load_refused(refused_file(self.home)).get(digest, "")
        if lasting:
            return lasting
        held = self.retry.get(digest)
        if held is None:
            return ""
        if time.monotonic() - held[1] >= RETRY_SECONDS:
            del self.retry[digest]
            return ""
        return held[0]

    def refuse(self, digest: str, reason: str, *, lasting: bool) -> None:
        if lasting:
            remember(refused_file(self.home), digest, reason)
        else:
            self.retry[digest] = (reason, time.monotonic())
        self.last_error = reason
        log.warning("not installing agent image %s: %s", digest[:19], reason)

    def settle_helper(self, helper: str) -> None:
        digest = ""
        try:
            template = '{{index .Config.Labels "' + DIGEST_LABEL + '"}}'
            digest = self.docker.runner.run(["inspect", "--format", template, helper]).strip()
        except DockerError:
            pass
        code, said = helper_outcome(self.docker, helper)
        log.warning("agent update helper %s exited %d: %s", helper[:24], code, said)
        if code != 0 and DIGEST.match(digest):
            reason = f"the update helper failed: {said[-200:]}"
            self.refuse(digest, reason, lasting=code == UNSTEADY_EXIT)

    def collect(self) -> None:
        try:
            found = self.docker.runner.run(
                ["ps", "-aq", "--filter", f"label={HELPER_LABEL}", "--filter", "status=exited"]
            ).split()
        except DockerError:
            return
        for helper in found:
            self.settle_helper(helper)

    def pending(self) -> Release | None:
        self.last_check = time.monotonic()
        self.ready = ("", "")
        if not self.enabled:
            return None
        self.collect()
        release = self.published()
        if release is None or self.refusal(release.digest):
            return None
        if outdated(RELEASE_VERSION, release.minimum):
            log.warning(
                "this agent is %s and the network requires at least %s",
                RELEASE_VERSION,
                release.minimum,
            )
        container = own_container_id()
        if running_digest(self.docker, container) == release.digest:
            return None
        try:
            reference = trusted_reference(release.image, release.digest)
            if not container:
                raise UpdateError("this agent cannot find its own container")
            project_of(labels_of(self.docker, container))
        except (UpdateError, DockerError) as exc:
            self.refuse(release.digest, str(exc), lasting=True)
            return None
        try:
            verify_signature(reference)
        except UpdateError as exc:
            self.refuse(release.digest, str(exc), lasting=False)
            return None
        try:
            self.docker.ensure_digest(TRUSTED_IMAGE, release.digest)
        except DockerError as exc:
            self.last_error = f"the published image could not be pulled: {exc}"
            return None
        try:
            newer_than(self.docker, reference, container)
        except UpdateError as exc:
            self.refuse(release.digest, str(exc), lasting=True)
            self.forget(reference)
            return None
        self.ready = (release.digest, reference)
        return release

    def forget(self, reference: str) -> None:
        try:
            self.docker.runner.run(["image", "rm", reference])
        except DockerError as exc:
            log.info("%s stays on disk: %s", reference, exc)

    def check(self) -> dict[str, Any]:
        digest, reference = self.ready
        self.ready = ("", "")
        if not digest:
            return {"applied": False, "reason": "no checked release is waiting"}
        if self.stopping.is_set():
            return {"applied": False, "reason": "a stop was requested", "digest": digest}
        release = self.published()
        if release is None or release.digest != digest:
            return {"applied": False, "reason": "the published release changed", "digest": digest}
        try:
            self.helper = self.restart(own_container_id(), reference, digest)
        except (DockerError, UpdateError) as exc:
            self.refuse(digest, f"the update helper could not start: {exc}", lasting=True)
            return {"applied": False, "reason": self.last_error, "digest": digest}
        self.applied = digest
        log.warning("moving to agent image %s through %s", reference, self.helper)
        return {"applied": True, "digest": digest, "helper": self.helper}

    def restart(self, container: str, reference: str, digest: str) -> str:
        if not container:
            raise UpdateError("this agent cannot find its own container")
        project = project_of(labels_of(self.docker, container))
        current = own_image(self.docker, container)
        if not current:
            raise UpdateError("this agent cannot read its own image")
        helper = f"{HELPER_PREFIX}-{int(time.time())}"
        argv = ["run", "-d", "--name", helper, "--label", HELPER_LABEL]
        argv += ["--label", f"{DIGEST_LABEL}={digest}", "--user", "0", "--entrypoint", ENTRYPOINT]
        argv += ["-v", f"{SOCKET}:{SOCKET}"]
        state = mount_source(self.docker, container, self.home or HOME)
        if state:
            argv += ["-v", f"{state}:{HOME}"]
        for folder in project.folders():
            argv += ["-v", f"{folder}:{folder}"]
        argv += [current, "operator", "apply-update"]
        argv += ["--image", reference, "--container", container]
        self.docker.runner.run(argv, timeout=60.0)
        return helper

    def await_replacement(
        self, halted: Callable[[], bool], timeout: float = HELPER_SECONDS
    ) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not halted():
            if self.helper and helper_state(self.docker, self.helper) in {"exited", "dead"}:
                helper, self.helper = self.helper, ""
                self.settle_helper(helper)
                return halted()
            time.sleep(POLL_SECONDS)
        return halted()

    def due(self, every: float) -> bool:
        return time.monotonic() - self.last_check >= max(MIN_CHECK_SECONDS, every)
