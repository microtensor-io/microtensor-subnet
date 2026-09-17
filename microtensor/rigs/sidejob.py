from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import threading
import urllib.request
from pathlib import Path
from typing import Any

from microtensor.core.constants import PUBLIC_SERVER_URL
from microtensor.rigs.validator.config import Settings
from microtensor.rigs.validator.main import Validator

log = logging.getLogger("microtensor.rigs")

LIBRARY_URL = (
    "https://github.com/microtensor-io/microtensor-compute/releases/latest/download/"
    "libmtchallenge.so"
)
GATE_POLL_SECONDS = 5.0


class Measuring:
    def __init__(self) -> None:
        self._busy = threading.Event()

    def start(self) -> None:
        self._busy.set()

    def stop(self) -> None:
        self._busy.clear()

    @property
    def busy(self) -> bool:
        return self._busy.is_set()


def settings_for(work_dir: Path, hotkey: str, *, server_url: str = "") -> Settings:
    base = Settings()
    state = work_dir / "rigs"
    state.mkdir(parents=True, exist_ok=True)
    library = base.agent_library
    if not library.is_file():
        library = state / "libmtchallenge.so"
    return dataclasses.replace(
        base,
        server_url=server_url or base.server_url or PUBLIC_SERVER_URL,
        label=base.label or f"validator {hotkey[:8]}",
        state_dir=state,
        agent_library=library,
        set_weights=False,
    )


def ensure_library(path: Path) -> bool:
    if path.is_file():
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(LIBRARY_URL, timeout=120) as answer:
            raw = answer.read()
        with urllib.request.urlopen(LIBRARY_URL + ".sha256", timeout=60) as answer:
            expected = answer.read().decode().split()[0].strip().lower()
    except Exception as exc:
        log.warning("the rig challenge library could not be fetched (%s); deep passes wait", exc)
        return False
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        log.warning("the rig challenge library digest %s does not match %s", digest, expected)
        return False
    path.write_bytes(raw)
    log.info("rig challenge library fetched to %s", path)
    return True


def run_forever(settings: Settings, keypair: Any, measuring: Measuring | None) -> None:
    async def gate() -> None:
        while measuring is not None and measuring.busy:
            await asyncio.sleep(GATE_POLL_SECONDS)

    ensure_library(settings.agent_library)
    validator = Validator(settings, keypair=keypair, gate=gate)
    asyncio.run(validator.run())


def start_thread(settings: Settings, keypair: Any, measuring: Measuring) -> threading.Thread:
    def body() -> None:
        while True:
            try:
                run_forever(settings, keypair, measuring)
                return
            except Exception as exc:
                log.warning("rig verification stopped (%s); restarting in a minute", exc)
                threading.Event().wait(60.0)

    thread = threading.Thread(target=body, name="rigs", daemon=True)
    thread.start()
    log.info("rig verification runs beside this validator as idle time work")
    return thread
