from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from microtensor_subnet.artifact import contract
from microtensor_subnet.cli.common import add_common_arguments, fail
from microtensor_subnet.core.constants import PUBLIC_SERVER_URL
from microtensor_subnet.core.profiles import PROFILES
from microtensor_subnet.validator import preflight

TIMEOUT_SECONDS = 20


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "check-model", help="dry-load a model on the CPU the way the mirror and the runner do"
    )
    parser.add_argument("--slm", type=Path, required=True, help="model repository root")
    parser.add_argument("--profile", default="", choices=("", *PROFILES))
    parser.add_argument("--arena", default="", help="read bounds and context from this arena")
    parser.add_argument("--server", default=PUBLIC_SERVER_URL)
    parser.add_argument("--min-parameters", type=int, default=contract.SLM_MIN_PARAMETERS)
    parser.add_argument("--max-parameters", type=int, default=contract.SLM_MAX_PARAMETERS)
    parser.add_argument("--max-len", type=int, default=preflight.DEFAULT_MAX_LEN)
    parser.add_argument("--json", action="store_true", help="print the full result as json")
    add_common_arguments(parser)
    parser.set_defaults(handler=run)


def _definition(server: str, arena_id: str) -> dict[str, Any]:
    url = f"{server.rstrip('/')}/v2/arenas/{urllib.parse.quote(arena_id)}"
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"the server URL must be http or https: {server!r}")
    request = urllib.request.Request(url, headers={"accept": "application/json"})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            found = json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        raise ValueError(f"the server has no arena {arena_id!r} ({exc.code})") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"the server could not be reached: {exc}") from exc
    return dict((found or {}).get("definition") or {})


def run(args: argparse.Namespace) -> int:
    bounds: tuple[int, int] | None = (int(args.min_parameters), int(args.max_parameters))
    max_len = int(args.max_len)
    if args.arena:
        try:
            definition = _definition(args.server, args.arena)
        except ValueError as exc:
            return fail(str(exc))
        bounds = preflight.bounds_of(definition) or bounds
        max_len = preflight.max_len_of(definition)
    found = preflight.check(args.slm, profile=args.profile, bounds=bounds, max_len=max_len)
    if args.json:
        print(
            json.dumps(
                {
                    "result": found.message,
                    "parameters": found.parameters,
                    "weights_sha256": found.weights_sha256,
                    "footprint_bytes": found.footprint_bytes,
                    "files": found.files,
                    "architectures": list(found.architectures),
                },
                indent=2,
            )
        )
    else:
        print(found.message)
        if found.parameters:
            print(f"parameters {found.parameters}")
        if found.weights_sha256:
            print(f"weights    {found.weights_sha256}")
        if found.footprint_bytes:
            print(f"footprint  {found.footprint_bytes} bytes at {max_len} tokens")
    return 0 if found.ok else 1
