from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from microtensor.core.constants import (
    RELEASE_CHANNELS,
    RELEASE_REPO,
    UPDATE_HTTP_TIMEOUT_SECONDS,
)

log = logging.getLogger("microtensor.update")

TAG_PATTERN = re.compile(r"^(?:(?P<channel>[a-z]+)-)?v?(?P<version>\d+\.\d+\.\d+)$")
WHEEL_SUFFIX = ".whl"
SUMS_NAME = "SHA256SUMS"
SIGNATURE_NAME = "SHA256SUMS.sig"
MECHANISM_MARKER = "mechanism:"
ACTIVATION_MARKER = "activation-block:"


class ReleaseError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, raw: str) -> Version:
        match = TAG_PATTERN.match(raw.strip())
        if match is None:
            raise ReleaseError(f"{raw!r} is not a release version")
        major, minor, patch = (int(p) for p in match.group("version").split("."))
        return cls(major, minor, patch)

    @property
    def parts(self) -> tuple[int, int, int]:
        return self.major, self.minor, self.patch

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: Version) -> bool:
        return self.parts < other.parts

    def __le__(self, other: Version) -> bool:
        return self.parts <= other.parts


@dataclass(frozen=True, slots=True)
class Release:
    tag: str
    channel: str
    version: Version
    wheel_url: str
    sums_url: str
    signature_url: str
    mechanism_version: str
    activation_block: int | None
    notes: str = ""

    @property
    def changes_mechanism(self) -> bool:
        from microtensor.core.constants import MECHANISM_VERSION

        return self.mechanism_version != MECHANISM_VERSION

    @property
    def is_signed(self) -> bool:
        return bool(self.signature_url)


def channel_of(tag: str) -> str:
    match = TAG_PATTERN.match(tag.strip())
    if match is None:
        raise ReleaseError(f"{tag!r} is not a release tag")
    return match.group("channel") or "mainnet"


def _get_json(url: str, timeout: int) -> Any:
    if not url.startswith("https://"):
        raise ReleaseError(f"refusing to read releases over a non-https url: {url}")

    request = urllib.request.Request(  # noqa: S310
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "microtensor-updater",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        raise ReleaseError(f"release index unreadable: {exc}") from exc


def _asset(assets: list[dict[str, Any]], predicate: Any) -> str:
    for asset in assets:
        name = str(asset.get("name", ""))
        if predicate(name):
            return str(asset.get("browser_download_url", ""))
    return ""


def _marker(body: str, marker: str) -> str:
    for line in body.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith(marker):
            return stripped[len(marker):].strip()
    return ""


def parse_release(payload: dict[str, Any]) -> Release | None:
    tag = str(payload.get("tag_name", ""))
    try:
        version = Version.parse(tag)
        channel = channel_of(tag)
    except ReleaseError:
        return None

    if payload.get("draft") or payload.get("prerelease"):
        return None

    assets = payload.get("assets") or []
    wheel = _asset(assets, lambda n: n.endswith(WHEEL_SUFFIX))
    if not wheel:
        return None

    body = str(payload.get("body", ""))
    activation = _marker(body, ACTIVATION_MARKER)

    return Release(
        tag=tag,
        channel=channel,
        version=version,
        wheel_url=wheel,
        sums_url=_asset(assets, lambda n: n == SUMS_NAME),
        signature_url=_asset(assets, lambda n: n == SIGNATURE_NAME),
        mechanism_version=_marker(body, MECHANISM_MARKER) or str(version),
        activation_block=int(activation) if activation.isdigit() else None,
        notes=body,
    )


def fetch_releases(
    repo: str = RELEASE_REPO,
    *,
    channel: str = "mainnet",
    timeout: int = UPDATE_HTTP_TIMEOUT_SECONDS,
    limit: int = 30,
) -> list[Release]:
    if channel not in RELEASE_CHANNELS:
        raise ReleaseError(f"unknown channel {channel!r}; known: {list(RELEASE_CHANNELS)}")

    payload = _get_json(
        f"https://api.github.com/repos/{repo}/releases?per_page={limit}", timeout
    )
    if not isinstance(payload, list):
        raise ReleaseError("release index was not a list")

    found: list[Release] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        release = parse_release(entry)
        if release is not None and release.channel == channel:
            found.append(release)
    return sorted(found, key=lambda r: r.version.parts, reverse=True)


def latest(
    repo: str = RELEASE_REPO,
    *,
    channel: str = "mainnet",
    timeout: int = UPDATE_HTTP_TIMEOUT_SECONDS,
) -> Release | None:
    releases = fetch_releases(repo, channel=channel, timeout=timeout)
    return releases[0] if releases else None
