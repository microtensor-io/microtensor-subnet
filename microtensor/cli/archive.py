from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

log = logging.getLogger("microtensor.cli.archive")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("archive", help="archive certified artifacts")
    inner = parser.add_subparsers(dest="archive_command", required=True)

    round_ = inner.add_parser("round", help="push one settled round's admitted artifacts")
    round_.add_argument("--track", default="code")
    round_.add_argument("--hardware-class", default="mt-3g")
    round_.add_argument("--server", default="https://api.microtensor.cloud")
    round_.add_argument("--org", default="microtensor-archive")
    round_.add_argument(
        "--cache",
        action="append",
        default=None,
        help="hub cache roots holding fetched snapshots; repeatable",
    )
    round_.add_argument("--staging", default="~/.microtensor/archive-staging")
    round_.add_argument("--dry-run", action="store_true")
    round_.set_defaults(handler=_round)


def _round(args: argparse.Namespace) -> int:
    from microtensor.archive.push import run

    token = os.environ.get("MT_HF_ARCHIVE_TOKEN", "").strip()
    if not token and not args.dry_run:
        print("MT_HF_ARCHIVE_TOKEN is unset; set it or pass --dry-run")
        return 1

    caches = args.cache or [
        "~/.cache/huggingface/hub",
        "~/archive-r1236-salvage",
    ]
    archived = run(
        server_url=args.server.rstrip("/"),
        track=args.track,
        hardware_class=args.hardware_class,
        org=args.org,
        token=token,
        cache_dirs=[Path(c).expanduser() for c in caches],
        staging_root=Path(args.staging).expanduser(),
        dry_run=args.dry_run,
    )
    print(f"archived {archived} artifacts")
    return 0
