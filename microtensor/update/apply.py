from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from microtensor.core.constants import (
    RELEASE_SIGNING_KEY,
    UPDATE_HTTP_TIMEOUT_SECONDS,
    UPDATE_MAX_ASSET_BYTES,
)
from microtensor.update.release import Release, ReleaseError
from microtensor.update.verify import Verification, verify_artifact

log = logging.getLogger("microtensor.update.apply")


class ApplyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Applied:
    release: Release
    verification: Verification
    installed: bool
    reason: str = ""

    @property
    def restart_required(self) -> bool:
        return self.installed


def download(url: str, destination: Path, *, timeout: int, cap: int) -> Path:
    if not url.startswith("https://"):
        raise ApplyError(f"refusing to download over a non-https url: {url}")

    request = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": "microtensor-updater"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            written = 0
            with destination.open("wb") as fh:
                while chunk := response.read(1024 * 1024):
                    written += len(chunk)
                    if written > cap:
                        raise ApplyError(f"{url} exceeded the {cap} byte asset cap")
                    fh.write(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ApplyError(f"{url} could not be downloaded: {exc}") from exc
    return destination


def _fetch_text(url: str, timeout: int) -> str:
    if not url:
        return ""
    with tempfile.TemporaryDirectory() as tmp:
        path = download(url, Path(tmp) / "asset", timeout=timeout, cap=UPDATE_MAX_ASSET_BYTES)
        return path.read_text(encoding="utf-8", errors="replace")


def _fetch_bytes(url: str, timeout: int) -> bytes:
    if not url:
        return b""
    with tempfile.TemporaryDirectory() as tmp:
        path = download(url, Path(tmp) / "asset", timeout=timeout, cap=UPDATE_MAX_ASSET_BYTES)
        return path.read_bytes()


def pip_install(wheel: Path, *, executable: str = "") -> tuple[bool, str]:
    python = executable or sys.executable
    command = [python, "-m", "pip", "install", "--upgrade", "--no-deps", str(wheel)]
    try:
        result = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"pip could not be run: {exc}"

    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        return False, "pip install failed: " + " | ".join(tail)
    return True, ""


def apply_release(
    release: Release,
    *,
    signing_key: str = RELEASE_SIGNING_KEY,
    require_signature: bool = True,
    timeout: int = UPDATE_HTTP_TIMEOUT_SECONDS,
    dry_run: bool = False,
    installer: object = None,
) -> Applied:
    if not release.sums_url and require_signature:
        return Applied(
            release,
            Verification(False, False, False, "release publishes no SHA256SUMS"),
            installed=False,
            reason="release publishes no SHA256SUMS",
        )

    staging = Path(tempfile.mkdtemp(prefix="mt-update-"))
    try:
        wheel = download(
            release.wheel_url,
            staging / Path(release.wheel_url).name,
            timeout=timeout,
            cap=UPDATE_MAX_ASSET_BYTES,
        )
        sums_text = _fetch_text(release.sums_url, timeout)
        signature = _fetch_bytes(release.signature_url, timeout)

        verification = verify_artifact(
            wheel,
            sums_text,
            signature,
            signing_key,
            require_signature=require_signature,
        )
        if not verification.trusted:
            log.error("refusing %s: %s", release.tag, verification.reason)
            return Applied(release, verification, installed=False, reason=verification.reason)

        if dry_run:
            return Applied(release, verification, installed=False, reason="dry run")

        install = installer or pip_install
        ok, reason = install(wheel)  # type: ignore[operator]
        if not ok:
            log.error("install of %s failed, staying on the running build: %s", release.tag, reason)
            return Applied(release, verification, installed=False, reason=reason)

        log.info("installed %s; a restart is required to run it", release.tag)
        return Applied(release, verification, installed=True)
    except (ApplyError, ReleaseError) as exc:
        return Applied(
            release,
            Verification(False, False, False, str(exc)),
            installed=False,
            reason=str(exc),
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
