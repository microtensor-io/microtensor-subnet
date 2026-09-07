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
    round_.add_argument("--coordinator", default="https://coordinator.microtensor.cloud")
    round_.add_argument("--org", default="microtensor-archive")
    round_.add_argument(
        "--cache",
        action="append",
        default=None,
        help="hub cache roots holding fetched snapshots; repeatable",
    )
    round_.add_argument("--staging", default="~/.microtensor/archive-staging")
    round_.add_argument("--dry-run", action="store_true")
    round_.add_argument("--state", default="")
    round_.add_argument("--network", default=os.environ.get("MT_NETWORK", "finney"))
    round_.add_argument("--netuid", type=int, default=int(os.environ.get("MT_NETUID", "92")))
    round_.add_argument("--endpoint", default=os.environ.get("MT_ENDPOINT", ""))
    round_.add_argument("--no-chain", action="store_true")
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
    objects = [
        "~/.microtensor/cache/objects",
        "~/archive-r1236-objects",
    ]
    from microtensor.archive.push import _get

    server_url = args.server.rstrip("/")
    board = _get(f"{server_url}/v1/arenas/{args.track}/{args.hardware_class}/leaderboard")
    round_index = int(board.get("round_index"))
    hotkeys = sorted(
        {str(s.get("hotkey", "")) for s in board.get("systems", ()) if s.get("hotkey")}
    )
    sources = _sources(args, round_index)
    keys = {} if args.no_chain else _reveal_keys(args, round_index, hotkeys)
    log.info(
        "round %d: %d committed sources and %d reveal keys at hand for %d systems",
        round_index,
        len(sources),
        len(keys),
        len(hotkeys),
    )
    archived = run(
        server_url=server_url,
        coordinator_url=args.coordinator.rstrip("/"),
        track=args.track,
        hardware_class=args.hardware_class,
        org=args.org,
        token=token,
        cache_dirs=[Path(c).expanduser() for c in caches],
        object_dirs=[Path(o).expanduser() for o in objects],
        staging_root=Path(args.staging).expanduser(),
        dry_run=args.dry_run,
        sources=sources,
        keys=keys,
        board=board,
    )
    print(f"archived {archived} artifacts")
    return 0


def _sources(args: argparse.Namespace, round_index: int) -> dict[str, str]:
    from microtensor.store.state import ValidatorState

    home = Path(os.environ.get("MT_HOME", "~/.microtensor")).expanduser()
    path = Path(args.state).expanduser() if args.state else home / "state" / "validator.sqlite"
    if not path.is_file():
        log.warning("no validator state at %s; sealed artifacts cannot be fetched", path)
        return {}
    observed = ValidatorState(path).observed_submissions(round_index)
    return {hotkey: found[3] for hotkey, found in observed.items() if found[3]}


def _reveal_keys(
    args: argparse.Namespace, round_index: int, hotkeys: list[str]
) -> dict[str, str]:
    from microtensor.chain.commitment import Reveal
    from microtensor.chain.config import ChainConfig
    from microtensor.cli.common import open_client

    if not hotkeys:
        return {}
    try:
        client = open_client(
            ChainConfig(netuid=args.netuid, network=args.network, endpoint=args.endpoint), None
        )
        raw = client.commitments(hotkeys)
    except Exception as exc:
        log.warning("reveal keys could not be read from chain: %s", exc)
        return {}
    keys: dict[str, str] = {}
    for hotkey, payload in raw.items():
        reveal = Reveal.decode(str(payload or ""))
        if reveal is not None and reveal.round_index == round_index:
            keys[hotkey] = reveal.key
    return keys
