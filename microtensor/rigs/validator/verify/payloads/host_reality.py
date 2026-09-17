from __future__ import annotations

import json
import os
import subprocess
import sys

HOST_ROOT = "/proc/1/root"
DMI_FIELDS = (
    "sys_vendor",
    "product_name",
    "product_version",
    "board_vendor",
    "board_name",
    "bios_vendor",
    "bios_version",
)
DETECT_VIRT_PATHS = (
    "/usr/bin/systemd-detect-virt",
    "/bin/systemd-detect-virt",
    "/usr/sbin/systemd-detect-virt",
)


def read(path, limit=4096):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read(limit).strip()
    except OSError:
        return ""


def stat_pair(path):
    try:
        info = os.stat(path)
    except OSError:
        return None
    return [int(info.st_dev), int(info.st_ino)]


def run(argv, timeout=10):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, "", str(exc)
    return done.returncode, (done.stdout or "").strip(), (done.stderr or "").strip()


def detect_virt():
    attempts = []
    for binary in DETECT_VIRT_PATHS:
        host_binary = HOST_ROOT + binary
        if os.path.exists(host_binary):
            code, out, err = run(["chroot", HOST_ROOT, binary, "--container"], timeout=15)
            attempts.append(
                {
                    "method": "chroot",
                    "binary": binary,
                    "code": code,
                    "out": out[:80],
                    "err": err[:200],
                }
            )
            if code is not None and (code == 0 or out):
                return out or "none", attempts
            if code == 1 and not out:
                return "none", attempts
    for binary in DETECT_VIRT_PATHS:
        if os.path.exists(binary):
            code, out, err = run([binary, "--container"], timeout=15)
            attempts.append(
                {
                    "method": "local",
                    "binary": binary,
                    "code": code,
                    "out": out[:80],
                    "err": err[:200],
                }
            )
            if code is not None and (code == 0 or out):
                return out or "none", attempts
            if code == 1 and not out:
                return "none", attempts
    return "unavailable", attempts


def dmi():
    values = {}
    for name in DMI_FIELDS:
        value = read(os.path.join(HOST_ROOT, "sys/class/dmi/id", name), 256)
        if not value:
            value = read(os.path.join("/sys/class/dmi/id", name), 256)
        values[name] = value
    return values


def init_environ_container():
    try:
        with open("/proc/1/environ", "rb") as handle:
            raw = handle.read(65536)
    except OSError:
        return ""
    for item in raw.split(b"\0"):
        if item.startswith(b"container="):
            return item.decode("utf-8", "replace")[len("container=") :][:40]
    return ""


def main():
    init_root = stat_pair(HOST_ROOT + "/")
    own_root = stat_pair("/")
    os_release = read(os.path.join(HOST_ROOT, "etc/os-release"), 4096)
    virt, attempts = detect_virt()
    result = {
        "init_root_stat": init_root,
        "own_root_stat": own_root,
        "init_root_reachable": init_root is not None,
        "init_root_differs": init_root is not None
        and own_root is not None
        and init_root != own_root,
        "init_os_release_readable": bool(os_release),
        "init_os_release": os_release[:1200],
        "init_hostname": read(os.path.join(HOST_ROOT, "etc/hostname"), 256),
        "init_comm": read("/proc/1/comm", 64),
        "init_cgroup": read("/proc/1/cgroup", 2000),
        "self_cgroup": read("/proc/self/cgroup", 2000),
        "init_environ_container": init_environ_container(),
        "dockerenv": os.path.exists("/.dockerenv"),
        "host_dockerenv": os.path.exists(HOST_ROOT + "/.dockerenv"),
        "detect_virt": virt,
        "detect_virt_attempts": attempts,
        "dmi": dmi(),
        "kernel": read("/proc/sys/kernel/osrelease", 128),
        "boot_id": read("/proc/sys/kernel/random/boot_id", 64),
        "machine_id": read(os.path.join(HOST_ROOT, "etc/machine-id"), 64)
        or read("/etc/machine-id", 64),
        "uid": os.getuid(),
        "python": sys.version.split()[0],
    }
    sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
