from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import time

CONTAINER_ID = re.compile(r"([0-9a-f]{64})")
MAX_PROCESSES = 4000
CMDLINE_LIMIT = 200


def read(path, limit=4096):
    try:
        with open(path, "rb") as handle:
            return handle.read(limit).decode("utf-8", "replace")
    except OSError:
        return ""


def handshake(secret_hex, nonce_hex):
    return hmac.new(
        bytes.fromhex(secret_hex), nonce_hex.encode("ascii"), hashlib.sha256
    ).hexdigest()


def container_of(cgroup_text):
    match = CONTAINER_ID.search(cgroup_text)
    return match.group(1) if match else ""


def uid_of(status_text):
    for line in status_text.splitlines():
        if line.startswith("Uid:"):
            parts = line.split()
            if len(parts) > 1 and parts[1].isdigit():
                return int(parts[1])
    return -1


def processes():
    found = []
    try:
        entries = sorted(int(name) for name in os.listdir("/proc") if name.isdigit())
    except OSError:
        return found
    for pid in entries[:MAX_PROCESSES]:
        base = f"/proc/{pid}"
        comm = read(base + "/comm", 64).strip()
        if not comm:
            continue
        cmdline = read(base + "/cmdline", 4096).replace("\0", " ").strip()[:CMDLINE_LIMIT]
        status = read(base + "/status", 2048)
        ppid = 0
        for line in status.splitlines():
            if line.startswith("PPid:"):
                parts = line.split()
                if len(parts) > 1 and parts[1].isdigit():
                    ppid = int(parts[1])
                break
        found.append(
            {
                "pid": pid,
                "ppid": ppid,
                "comm": comm,
                "cmdline": cmdline,
                "uid": uid_of(status),
                "container": container_of(read(base + "/cgroup", 2048)),
            }
        )
    return found


def containers():
    try:
        done = subprocess.run(
            ["docker", "ps", "-a", "--no-trunc", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"docker ps: {exc}"
    if done.returncode != 0:
        return [], f"docker ps exit {done.returncode}: {(done.stderr or '').strip()[:200]}"
    found = []
    for line in done.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        labels = {}
        for item in str(row.get("Labels", "") or "").split(","):
            if "=" in item:
                key, value = item.split("=", 1)
                labels[key.strip()] = value.strip()
        found.append(
            {
                "id": str(row.get("ID", "")),
                "name": str(row.get("Names", "")),
                "image": str(row.get("Image", "")),
                "state": str(row.get("State", "")),
                "status": str(row.get("Status", "")),
                "labels": labels,
                "runtime": "",
            }
        )
    return found, ""


def main(argv):
    if len(argv) != 3:
        sys.stdout.write(json.dumps({"error": "usage: inspector <secret hex> <nonce hex>"}) + "\n")
        return 2
    secret_hex, nonce_hex = argv[1], argv[2]
    try:
        reply = handshake(secret_hex, nonce_hex)
    except ValueError as exc:
        sys.stdout.write(json.dumps({"error": f"handshake: {exc}"}) + "\n")
        return 2
    started = time.time()
    found, error = containers()
    result = {
        "nonce": nonce_hex,
        "handshake": reply,
        "collected_at": round(time.time(), 3),
        "processes": processes(),
        "containers": found,
        "containers_error": error,
        "pid": os.getpid(),
        "elapsed_ms": round((time.time() - started) * 1000.0, 3),
    }
    sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
