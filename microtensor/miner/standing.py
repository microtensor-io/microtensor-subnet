from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("microtensor.miner")

TIMEOUT_SECONDS = 15


@dataclass(slots=True)
class Standing:
    """Where one system sits, as the public surface reports it.

    Everything here is read from endpoints anyone can read. A miner asking where
    it stands should not need a credential to find out, and a number it cannot
    verify independently is not worth printing.
    """

    on_frontier: bool = False
    rank_by_cost: int = 0
    of_total: int = 0
    quality: float = 0.0
    expected_ms: float = 0.0
    contribution: dict[str, float] = field(default_factory=dict)
    release_version: str = ""
    milestone: dict[str, Any] = field(default_factory=dict)
    reachable: bool = False
    reason: str = ""


def _get(base: str, path: str) -> Any:
    url = base.rstrip("/") + path
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"the server URL must be http or https: {base!r}")
    request = urllib.request.Request(  # noqa: S310 - scheme checked above
        url, headers={"accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
        return json.loads(response.read() or b"null")


def fetch(base: str, track: str, hardware_class: str, system_digest: str) -> Standing:
    """Read this system's position from the public frontier and release.

    Best effort. An unreachable server means the local half of the status output
    still prints, because a miner offline from the API still needs to see what
    it packaged.
    """
    standing = Standing()

    try:
        frontier = _get(base, f"/v1/public/frontier/{track}/{hardware_class}") or {}
    except urllib.error.HTTPError as exc:
        standing.reason = f"the frontier returned {exc.code}"
        return standing
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        standing.reason = str(exc)
        return standing

    standing.reachable = True
    points = list(frontier.get("points", ()))
    standing.of_total = len(points)

    ordered = sorted(points, key=lambda p: float(p.get("expected_ms") or 0.0))
    for position, point in enumerate(ordered, start=1):
        if str(point.get("system_digest", "")) != system_digest:
            continue
        standing.on_frontier = True
        standing.rank_by_cost = position
        standing.quality = float(point.get("quality") or 0.0)
        standing.expected_ms = float(point.get("expected_ms") or 0.0)
        standing.contribution = dict(point.get("contribution") or {})
        break

    query = urllib.parse.urlencode({"track": track, "hardware_class": hardware_class})
    try:
        release = _get(base, f"/v1/public/releases/latest?{query}") or {}
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError):
        return standing

    standing.release_version = str(release.get("version", ""))
    standing.milestone = dict(release.get("milestone") or {})
    return standing
