from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final

COORDINATOR: Final[str] = "coordinator"
WORKER: Final[str] = "worker"
ROLE_ENV: Final[str] = "MT_ROLE"
ROLE_FILE: Final[str] = "role.json"


class RoleConflict(RuntimeError):
    """This host is configured for the other role.

    A coordinator runs no measurement and a worker publishes no settlement.
    Conflating them is how an owner ends up running a jail on a web server, or
    a public API on a machine holding certified reference hardware, so the two
    commands refuse each other rather than trusting the operator to remember.
    """


def role_path(home: Path) -> Path:
    return home / ROLE_FILE


def declared(home: Path) -> str:
    """What this host says it is. The environment wins so a container can
    declare its role without writing to a mounted volume."""
    override = os.environ.get(ROLE_ENV, "").strip().lower()
    if override:
        return override

    path = role_path(home)
    if not path.exists():
        return ""
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("role", ""))
    except (OSError, ValueError):
        return ""


def claim(home: Path, role: str) -> None:
    existing = declared(home)
    if existing and existing != role:
        raise RoleConflict(
            f"{home} is already initialised as a {existing}; "
            f"use a separate home directory for the {role}"
        )
    home.mkdir(parents=True, exist_ok=True)
    role_path(home).write_text(json.dumps({"role": role}, indent=2), encoding="utf-8")


def require(home: Path, role: str) -> None:
    """Refuse to run the wrong command on a host claimed by the other role."""
    existing = declared(home)
    if existing and existing != role:
        raise RoleConflict(
            f"this host is configured as a {existing}, so it will not run as a {role}; "
            f"point {ROLE_ENV} or --home at a {role} home"
        )
