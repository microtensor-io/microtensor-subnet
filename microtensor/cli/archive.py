from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from microtensor.core.constants import DEFAULT_NETUID, DEFAULT_NETWORK

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
    round_.add_argument("--network", default=os.environ.get("MT_NETWORK", DEFAULT_NETWORK))
    round_.add_argument("--netuid", type=int, default=DEFAULT_NETUID)
    round_.add_argument("--endpoint", default=os.environ.get("MT_ENDPOINT", ""))
    round_.add_argument("--no-chain", action="store_true")
    round_.add_argument("--frontier-only", action="store_true")
    round_.add_argument("--round", type=int, default=None)
    round_.set_defaults(handler=_round)

    cards = inner.add_parser("cards", help="state the base model licence on archived cards")
    cards.add_argument("--track", default="code")
    cards.add_argument("--hardware-class", default="mt-3g")
    cards.add_argument("--round", type=int, required=True)
    cards.add_argument("--server", default="https://api.microtensor.cloud")
    cards.add_argument("--org", default="microtensor-archive")
    cards.add_argument("--dry-run", action="store_true")
    cards.set_defaults(handler=_cards)

    intake = inner.add_parser(
        "intake", help="copy every submission into the private archive as it is committed"
    )
    intake.add_argument("--db", required=True, help="the coordinator's sqlite state")
    intake.add_argument("--round", type=int, default=None, help="defaults to the latest round")
    intake.add_argument("--store-dir", default="", help="archive into this directory")
    intake.add_argument("--org", default="", help="archive into private repos of this org")
    intake.add_argument("--staging", default="~/.microtensor/intake-staging")
    intake.add_argument("--watch", type=int, default=0, help="rescan every N seconds")
    intake.add_argument("--reveals", action="store_true", help="also record reveal keys")
    intake.add_argument("--network", default=os.environ.get("MT_NETWORK", DEFAULT_NETWORK))
    intake.add_argument("--netuid", type=int, default=DEFAULT_NETUID)
    intake.add_argument("--endpoint", default=os.environ.get("MT_ENDPOINT", ""))
    intake.set_defaults(handler=_intake)

    mirror = inner.add_parser(
        "mirror", help="mirror an allowlisted escalation model revision into our org"
    )
    mirror.add_argument("--model", required=True)
    mirror.add_argument("--revision", required=True)
    mirror.add_argument("--org", default="microtensor-archive")
    mirror.set_defaults(handler=_mirror)


def _intake_store(args: argparse.Namespace) -> object:
    from microtensor.archive.intake import HubStore, LocalStore

    if args.store_dir:
        return LocalStore(Path(args.store_dir).expanduser())
    token = os.environ.get("MT_HF_ARCHIVE_TOKEN", "").strip()
    if not args.org or not token:
        raise SystemExit("pass --store-dir, or --org with MT_HF_ARCHIVE_TOKEN set")
    return HubStore(args.org, token)


def _intake_once(args: argparse.Namespace, store: object) -> int:
    import time

    from microtensor.archive.intake import Submission, intake, record_reveal
    from microtensor.coordinator.store import CoordinatorStore

    with CoordinatorStore(Path(args.db).expanduser()) as coordinator:
        latest = coordinator.latest_round()
        round_index = args.round if args.round is not None else int((latest or {})["round_index"])
        found = coordinator.recorded_submissions(round_index)
    submissions = [
        Submission(round_index, hotkey, track, cls, digest, source, sealed)
        for hotkey, (track, cls, digest, source, sealed) in sorted(found.items())
        if source
    ]
    staging = Path(args.staging).expanduser()
    staging.mkdir(parents=True, exist_ok=True)
    archived = 0
    for submission in submissions:
        reason = intake(submission, store, staging)  # type: ignore[arg-type]
        if reason and reason != "already archived":
            log.warning("%s: %s", submission.key, reason)
        elif not reason:
            archived += 1
            log.info("%s archived at %d", submission.key, int(time.time()))
    if args.reveals and submissions:
        keys = _reveal_keys(args, round_index, [s.hotkey for s in submissions])
        for submission in submissions:
            if submission.hotkey in keys:
                record_reveal(store, submission, keys[submission.hotkey])  # type: ignore[arg-type]
    print(f"round {round_index}: {archived} new of {len(submissions)} submissions archived")
    return 0


def _intake(args: argparse.Namespace) -> int:
    import time

    store = _intake_store(args)
    while True:
        _intake_once(args, store)
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


def _mirror(args: argparse.Namespace) -> int:
    from microtensor.archive.intake import mirror

    token = os.environ.get("MT_HF_ARCHIVE_TOKEN", "").strip()
    if not token:
        print("MT_HF_ARCHIVE_TOKEN is unset")
        return 1
    print(mirror(args.model, args.revision, args.org, token))
    return 0


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
    url = f"{server_url}/v1/arenas/{args.track}/{args.hardware_class}/leaderboard"
    if args.round is not None:
        url += f"?round={args.round}"
    board = _get(url)
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
        frontier_only=args.frontier_only,
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


def _reveal_keys(args: argparse.Namespace, round_index: int, hotkeys: list[str]) -> dict[str, str]:
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


def _cards(args: argparse.Namespace) -> int:
    from microtensor.archive.push import refresh_cards

    token = os.environ.get("MT_HF_ARCHIVE_TOKEN", "").strip()
    if not token and not args.dry_run:
        print("MT_HF_ARCHIVE_TOKEN is unset; set it or pass --dry-run")
        return 1
    updated = refresh_cards(
        server_url=args.server.rstrip("/"),
        track=args.track,
        hardware_class=args.hardware_class,
        round_index=args.round,
        org=args.org,
        token=token,
        dry_run=args.dry_run,
    )
    print(f"updated {updated} cards")
    return 0
