from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from microtensor.cli.common import add_chain_arguments, add_common_arguments, fail
from microtensor.coordinator.api import Coordinator, Registry
from microtensor.coordinator.assign import System, Worker, assign, by_worker, under_replicated
from microtensor.coordinator.config import config_hash, served_config
from microtensor.coordinator.store import CoordinatorStore
from microtensor.core.constants import (
    COORDINATOR_PORT,
    COORDINATOR_REPLICATION,
    CORPUS_VERSION,
)
from microtensor.core.role import COORDINATOR, RoleConflict, claim, require

log = logging.getLogger("microtensor.coordinator")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "coordinator", help="run the owner-operated coordinator (not a validator)"
    )
    inner = parser.add_subparsers(dest="action", required=True)

    init = inner.add_parser("init", help="claim this host as a coordinator")
    add_common_arguments(init)
    add_chain_arguments(init)
    init.set_defaults(handler=_init)

    serve = inner.add_parser("serve", help="serve the coordinator API")
    add_common_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=COORDINATOR_PORT)
    serve.set_defaults(handler=_serve)

    opened = inner.add_parser(
        "open", help="read the round from chain, assign it, and store the catalogue"
    )
    add_common_arguments(opened)
    add_chain_arguments(opened)
    opened.add_argument("--replication", type=int, default=COORDINATOR_REPLICATION)
    opened.set_defaults(handler=_open)

    plan = inner.add_parser("assign", help="compute and store this round's assignment")
    add_common_arguments(plan)
    plan.add_argument("--round", type=int, required=True)
    plan.add_argument("--seed", required=True)
    plan.add_argument("--systems", type=Path, required=True, help="JSON list of systems")
    plan.add_argument("--workers", type=Path, required=True, help="JSON list of worker hotkeys")
    plan.add_argument("--replication", type=int, default=COORDINATOR_REPLICATION)
    plan.set_defaults(handler=_assign)

    settle = inner.add_parser("settle", help="reconcile and publish a round")
    add_common_arguments(settle)
    settle.add_argument("--round", type=int, required=True)
    settle.set_defaults(handler=_settle)

    cfg = inner.add_parser("config", help="print the served config and its hash")
    add_common_arguments(cfg)
    cfg.set_defaults(handler=_config)

    status = inner.add_parser("status", help="round state, quorum, worker standing")
    add_common_arguments(status)
    status.add_argument("--round", type=int)
    status.set_defaults(handler=_status)


def _home(args: argparse.Namespace) -> Path:
    return Path(args.home)


def _store(args: argparse.Namespace) -> CoordinatorStore:
    try:
        require(_home(args), COORDINATOR)
    except RoleConflict as exc:
        raise SystemExit(fail(str(exc))) from exc
    return CoordinatorStore(_home(args) / "coordinator.sqlite")


def _init(args: argparse.Namespace) -> int:
    home = _home(args)
    try:
        claim(home, COORDINATOR)
    except RoleConflict as exc:
        return fail(str(exc))

    store = CoordinatorStore(home / "coordinator.sqlite")
    store.close()

    config = served_config(CORPUS_VERSION)
    print(f"coordinator home   {home}")
    print(f"config hash        {config_hash(config)}")
    print()
    print("This host measures nothing. It fetches no artifact, loads no model, and")
    print("installs no jail. Commit the config hash on chain each round so workers")
    print("can prove the rules they measured against.")
    return 0


def _config(args: argparse.Namespace) -> int:
    config = served_config(CORPUS_VERSION)
    print(json.dumps(config, indent=2, sort_keys=True))
    print()
    print(f"hash  {config_hash(config)}")
    return 0


def _assign(args: argparse.Namespace) -> int:
    systems_raw = json.loads(Path(args.systems).read_text(encoding="utf-8"))
    workers_raw = json.loads(Path(args.workers).read_text(encoding="utf-8"))

    systems = [
        System(
            digest=str(s["digest"]),
            track=str(s["track"]),
            hardware_class=str(s["hardware_class"]),
            miner_hotkey=str(s.get("miner_hotkey", "")),
        )
        for s in systems_raw
    ]
    workers = [
        Worker(hotkey=str(w["hotkey"]), classes=tuple(w.get("classes", ())))
        if isinstance(w, dict)
        else Worker(hotkey=str(w))
        for w in workers_raw
    ]

    mapping = assign(systems, workers, args.seed, replication=args.replication)

    with _store(args) as store:
        store.record_assignment(
            args.round,
            mapping,
            {s.digest: (s.track, s.hardware_class, s.miner_hotkey) for s in systems},
        )

    thin = under_replicated(mapping, args.replication)
    print(f"{len(systems)} systems across {len(workers)} workers, seed {args.seed}")
    for hotkey, digests in sorted(by_worker(mapping).items()):
        print(f"  {hotkey:<50} {len(digests)}")
    if thin:
        print()
        print(f"under-replicated, measured by fewer than {args.replication}:")
        for digest in thin:
            print(f"  {digest}")
    return 0


def _open(args: argparse.Namespace) -> int:
    from microtensor.cli.common import chain_config, open_client, open_wallet
    from microtensor.coordinator.chain import ChainSource, observed

    chain = chain_config(args)
    wallet = open_wallet(chain, required=False)
    client = open_client(chain, wallet)
    source = ChainSource(client=client)

    round_ = source.open_round()
    systems, catalogue = source.systems(round_)
    workers = source.workers()
    seed = source.seed(round_)

    if not systems:
        print(f"round {round_.index}: nothing committed on chain yet")
        return 0
    if not workers:
        return fail("no worker holds a validator permit, so nothing can be assigned")

    mapping = assign(systems, workers, seed, replication=args.replication)

    with _store(args) as store:
        store.open_round(
            round_.index,
            seed_block=round_.seed_block,
            close_block=round_.close_block,
            block_hash=seed,
            config_hash=config_hash(served_config(CORPUS_VERSION)),
        )
        store.record_assignment(
            round_.index,
            mapping,
            {s.digest: (s.track, s.hardware_class, s.miner_hotkey) for s in systems},
        )
        store.record_catalogue(
            round_.index, observed(catalogue, store.observations(round_.index))
        )

    thin = under_replicated(mapping, args.replication)
    print(f"round {round_.index} opened, seed block {round_.seed_block}")
    print(f"  systems     {len(systems)}")
    print(f"  workers     {len(workers)}")
    print(f"  assignments {sum(len(v) for v in mapping.values())}")
    if thin:
        print(f"  under-replicated: {len(thin)}")
    print()
    print("Commit the config hash on chain before workers measure against it:")
    print(f"  {config_hash(served_config(CORPUS_VERSION))}")
    return 0


def _settle(args: argparse.Namespace) -> int:
    with _store(args) as store:
        service = Coordinator(
            store=store,
            registry=Registry({}),
            corpus_version=CORPUS_VERSION,
            catalogue=store.catalogue(args.round),
        )
        published = service.settle(args.round)

    if published is None:
        return fail(f"round {args.round} has not reached quorum yet")

    print(json.dumps(published, indent=2, sort_keys=True))
    return 0


def _status(args: argparse.Namespace) -> int:
    with _store(args) as store:
        service = Coordinator(
            store=store, registry=Registry({}), corpus_version=CORPUS_VERSION
        )
        health = service.health()
        standings = service.reputation()

    if health.get("round") is None:
        print("no round opened yet")
        return 0

    print(f"round            {health['round']}")
    print(f"reports          {health['received_reports']} / {health['expected_reports']}")
    print(f"quorum           {'reached' if health['quorum'] else 'waiting'}")
    print(f"settled          {'yes' if health['settled'] else 'no'}")
    print(f"config anchored  {'yes' if health['anchored'] else 'NO'}")
    print(f"divergence rate  {health['divergence_rate']:.2%}")

    if standings:
        print()
        print(f"{'worker':<50}{'agreed':>8}{'diverged':>10}{'rate':>8}  state")
        for s in standings:
            state = "advisory" if s["advisory"] else "deciding"
            print(
                f"{s['hotkey']:<50}{s['agreed']:>8}{s['diverged']:>10}"
                f"{s['rate']:>8.2f}  {state}"
            )
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        return fail('the API needs the web stack: pip install ".[coordinator]"')

    from microtensor.coordinator.api import build_app

    with _store(args) as store:
        service = Coordinator(
            store=store, registry=Registry({}), corpus_version=CORPUS_VERSION
        )
        app = build_app(service)
        log.info("serving the coordinator on %s:%d", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0
