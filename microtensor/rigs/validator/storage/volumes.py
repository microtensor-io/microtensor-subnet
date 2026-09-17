from __future__ import annotations

import math
import re
import secrets
import shlex
from dataclasses import dataclass
from typing import Any

from microtensor.rigs.validator.session.runner import Runner
from microtensor.rigs.validator.storage.keys import volume_passphrase

PREFIX = "mt-vol-"
NAME = re.compile(r"^mt-vol-[A-Za-z0-9][A-Za-z0-9_.-]{0,60}$")
MOUNT_PATH = re.compile(r"^/[A-Za-z0-9_./-]{1,200}$")
CIPHER_PATH = "/mnt/mt-cipher"
CREATE_TIMEOUT = 10.0
CREATE_TIMEOUT_MAX = 180.0
COMMAND_TIMEOUT = 60.0
OVERHEAD_GB = 20.0
FREE_MARGIN_GB = 10.0
FLOOR_GB = 1.0
GB = 1e9


class VolumeNameError(ValueError):
    pass


def valid(name: str) -> bool:
    return bool(NAME.match(name or ""))


def require(name: str) -> str:
    if not valid(name):
        raise VolumeNameError(f"volume name {name!r} is not an accepted ephemeral volume name")
    return name


def volume_name(job_id: str) -> str:
    return require(f"{PREFIX}{job_id}")


def require_path(path: str) -> str:
    if not MOUNT_PATH.match(path or "") or ".." in path:
        raise VolumeNameError(f"mount path {path!r} is not accepted")
    return path


def create_timeout(size_gb: float) -> float:
    if size_gb <= 100:
        return CREATE_TIMEOUT
    return min(CREATE_TIMEOUT_MAX, 30.0 + math.ceil(size_gb / 10.0))


@dataclass
class VolumeSpec:
    name: str
    mount_path: str
    size_gb: float = 0.0
    driver: str = ""
    sparse: bool = False
    encrypted: bool = False
    create: bool = True

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mount_path": self.mount_path,
            "size_gb": self.size_gb,
            "driver": self.driver,
            "sparse": self.sparse,
            "encrypted": self.encrypted,
            "create": self.create,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> VolumeSpec:
        return cls(
            name=str(payload.get("name", "")),
            mount_path=str(payload.get("mount_path", "")),
            size_gb=float(payload.get("size_gb", 0.0) or 0.0),
            driver=str(payload.get("driver", "") or ""),
            sparse=bool(payload.get("sparse", False)),
            encrypted=bool(payload.get("encrypted", False)),
            create=bool(payload.get("create", True)),
        )

    def problems(self) -> list[str]:
        found: list[str] = []
        if not valid(self.name):
            found.append(f"volume name {self.name!r} is not accepted")
        if not MOUNT_PATH.match(self.mount_path or "") or ".." in self.mount_path:
            found.append(f"mount path {self.mount_path!r} is not accepted")
        if self.encrypted and self.mount_path == CIPHER_PATH:
            found.append("an encrypted volume cannot mount at the ciphertext path")
        return found

    @property
    def container_target(self) -> str:
        return CIPHER_PATH if self.encrypted else self.mount_path


def mount_args(spec: VolumeSpec) -> list[str]:
    require(spec.name)
    require_path(spec.mount_path)
    args = ["-v", f"{spec.name}:{spec.container_target}"]
    if spec.encrypted:
        args += ["--device", "/dev/fuse"]
    return args


def create_command(spec: VolumeSpec) -> str:
    parts = ["docker", "volume", "create", "--name", require(spec.name)]
    if spec.driver:
        parts += ["--driver", spec.driver]
        if spec.size_gb > 0:
            parts += ["--opt", f"size={max(1, int(math.ceil(spec.size_gb)))}G"]
        if spec.sparse:
            parts += ["--opt", "sparse=true"]
    parts += ["--label", "mt.owner=validator"]
    return " ".join(shlex.quote(part) for part in parts)


async def create(runner: Runner, spec: VolumeSpec) -> tuple[bool, str]:
    problems = spec.problems()
    if problems:
        return False, problems[0]
    result = await runner.run(create_command(spec), timeout=create_timeout(spec.size_gb))
    if not result.ok:
        return False, (result.error or result.stderr_tail(300)).strip()
    return True, ""


async def destroy(runner: Runner, name: str) -> tuple[bool, str]:
    if not valid(name):
        return False, f"refusing to remove {name!r}: not an ephemeral volume name"
    result = await runner.run(f"docker volume rm -f {shlex.quote(name)}", timeout=COMMAND_TIMEOUT)
    if result.ok or "no such volume" in result.stderr.lower():
        return True, ""
    return False, (result.error or result.stderr_tail(300)).strip()


async def list_ours(runner: Runner) -> list[str]:
    result = await runner.run(
        f"docker volume ls -q --filter name={shlex.quote('^' + PREFIX)}", timeout=COMMAND_TIMEOUT
    )
    if not result.ok:
        return []
    return [line.strip() for line in result.stdout.splitlines() if valid(line.strip())]


async def mounted_volumes(runner: Runner) -> set[str] | None:
    result = await runner.run("docker ps -a --format '{{.Mounts}}'", timeout=COMMAND_TIMEOUT)
    if not result.ok:
        return None
    found: set[str] = set()
    for line in result.stdout.splitlines():
        for item in line.split(","):
            clean = item.strip()
            if clean.startswith(PREFIX):
                found.add(clean)
    return found


def parse_size_bytes(text: str) -> float:
    try:
        return float(text.strip())
    except ValueError:
        return 0.0


async def size_for(
    runner: Runner, requested_gb: float, share: float = 1.0, quotas: bool = True
) -> tuple[float, float]:
    if not quotas or requested_gb <= 0:
        return requested_gb, 0.0
    root = await runner.run("docker info --format '{{.DockerRootDir}}'", timeout=COMMAND_TIMEOUT)
    if not root.ok or not root.stdout.strip():
        return requested_gb, 0.0
    path = root.stdout.strip().splitlines()[-1].strip()
    free = await runner.run(
        f"df -B1 --output=avail {shlex.quote(path)} | tail -n 1", timeout=COMMAND_TIMEOUT
    )
    if not free.ok:
        return requested_gb, 0.0
    free_gb = parse_size_bytes(free.stdout) / GB
    existing = 0.0
    listing = await runner.run(
        "docker volume ls -q --filter name=^mt-vol- | xargs -r docker volume inspect --format '{{index .Options \"size\"}}'",
        timeout=COMMAND_TIMEOUT,
    )
    if listing.ok:
        for line in listing.stdout.splitlines():
            clean = line.strip().rstrip("Gg")
            try:
                existing += float(clean)
            except ValueError:
                continue
    pool = max(0.0, free_gb + existing - OVERHEAD_GB)
    slice_gb = min(share * pool, requested_gb * 1.5, max(0.0, free_gb - FREE_MARGIN_GB) * 1.5)
    volume_gb = max(FLOOR_GB, slice_gb * 2.0 / 3.0)
    storage_gb = max(FLOOR_GB, slice_gb / 3.0)
    return volume_gb, storage_gb


def unwrap_script(passphrase: str, cipher_path: str, plain_path: str, fresh: bool) -> str:
    raw = passphrase.encode("ascii")
    pad = secrets.token_bytes(len(raw))
    wrapped = bytes(a ^ b for a, b in zip(raw, pad, strict=True))
    passfile = f"/tmp/.mt-{secrets.token_hex(8)}"
    awk = (
        'BEGIN{h="0123456789abcdef";n=length(w)/2;'
        "for(i=0;i<n;i++){a=(index(h,substr(w,2*i+1,1))-1)*16+index(h,substr(w,2*i+2,1))-1;"
        "b=(index(h,substr(k,2*i+1,1))-1)*16+index(h,substr(k,2*i+2,1))-1;"
        "r=0;p=1;while(a>0||b>0){if((a%2)!=(b%2))r+=p;a=int(a/2);b=int(b/2);p*=2}"
        'printf "%c",r}}'
    )
    init = ""
    if fresh:
        init = (
            f"if [ ! -f {shlex.quote(cipher_path)}/gocryptfs.conf ]; then "
            f'gocryptfs -q -init -passfile "$F" {shlex.quote(cipher_path)} || exit 3; fi\n'
        )
    else:
        init = f"[ -f {shlex.quote(cipher_path)}/gocryptfs.conf ] || exit 4\n"
    return (
        "#!/bin/sh\n"
        "set -e\n"
        "umask 077\n"
        f"F={shlex.quote(passfile)}\n"
        "trap 'rm -f \"$F\"' EXIT\n"
        f'awk -v w={wrapped.hex()} -v k={pad.hex()} {shlex.quote(awk)} > "$F"\n'
        'chmod 600 "$F"\n'
        f"mkdir -p {shlex.quote(cipher_path)} {shlex.quote(plain_path)}\n"
        f"{init}"
        f'gocryptfs -q -allow_other -passfile "$F" {shlex.quote(cipher_path)} {shlex.quote(plain_path)}\n'
    )


async def setup_encrypted(
    runner: Runner,
    container: str,
    spec: VolumeSpec,
    master_secret: bytes,
    fresh: bool,
) -> tuple[bool, str]:
    require(spec.name)
    require_path(spec.mount_path)
    passphrase = volume_passphrase(master_secret, spec.name)
    script = unwrap_script(passphrase, CIPHER_PATH, spec.mount_path, fresh)
    script_path = f"/tmp/.mt-{secrets.token_hex(8)}.sh"
    quoted_container = shlex.quote(container)
    written = await runner.run(
        f"docker exec -i {quoted_container} sh -c {shlex.quote(f'umask 077; cat > {script_path}')}",
        timeout=COMMAND_TIMEOUT,
        stdin=script,
    )
    if not written.ok:
        return (
            False,
            f"could not deliver the mount script: {(written.error or written.stderr_tail(200)).strip()}",
        )
    ran = await runner.run(
        f"docker exec {quoted_container} sh -c {shlex.quote(f'sh {script_path}; s=$?; rm -f {script_path}; exit $s')}",
        timeout=COMMAND_TIMEOUT * 2,
    )
    if not ran.ok:
        if ran.exit_code == 4:
            return (
                False,
                "gocryptfs.conf is missing: the volume data is gone, refusing to re-initialise",
            )
        return False, f"gocryptfs mount failed: {(ran.error or ran.stderr_tail(300)).strip()}"
    if not await verify_mount(runner, container, spec.mount_path):
        return False, "the encrypted mount is not present in /proc/mounts"
    return True, ""


async def verify_mount(runner: Runner, container: str, path: str) -> bool:
    result = await runner.run(
        f"docker exec {shlex.quote(container)} grep -F ' fuse.gocryptfs ' /proc/mounts",
        timeout=COMMAND_TIMEOUT,
    )
    if not result.ok:
        return False
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == path and parts[2] == "fuse.gocryptfs":
            return True
    return False


async def grant_workspace(runner: Runner, container: str, path: str, user: str) -> bool:
    if not re.match(r"^[A-Za-z0-9_.:-]{1,64}$", user or ""):
        return False
    result = await runner.run(
        f"docker exec {shlex.quote(container)} chown {shlex.quote(user)} {shlex.quote(require_path(path))}",
        timeout=COMMAND_TIMEOUT,
    )
    return result.ok
