from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

from microtensor_subnet.chain.client import ChainError
from microtensor_subnet.chain.wallet import hotkey_address, sign_bytes
from microtensor_subnet.cli.common import (
    add_chain_arguments,
    add_common_arguments,
    chain_config,
    fail,
    open_client,
    open_wallet,
)
from microtensor_subnet.core.constants import PUBLIC_SERVER_URL
from microtensor_subnet.core.items import key_registration_message, new_recipient_key
from microtensor_subnet.serving import probes
from microtensor_subnet.serving.client import ServerError as ServingError
from microtensor_subnet.validator.client import ServerClient, ServerError
from microtensor_subnet.validator.config import EpochConfig
from microtensor_subnet.validator.epoch import EpochError, run_epoch
from microtensor_subnet.validator.items import ReleaseError

log = logging.getLogger("microtensor_subnet.cli.epoch")
KEY_BLOCKS = 2_600_000
PUBLIC_GATEWAY_URL = "https://api.microtensor.cloud"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("epoch", help="validate one epoch of an arena")
    inner = parser.add_subparsers(dest="action", required=True)

    key = inner.add_parser("register-key", help="register the key hidden items are sealed to")
    key.add_argument("--server", default=os.environ.get("MT_SERVER_URL", PUBLIC_SERVER_URL))
    add_chain_arguments(key)
    add_common_arguments(key)
    key.set_defaults(handler=_register_key)

    token = inner.add_parser("relay-token", help="fetch this validator's relay token")
    token.add_argument("--server", default=os.environ.get("MT_SERVER_URL", PUBLIC_SERVER_URL))
    add_chain_arguments(token)
    add_common_arguments(token)
    token.set_defaults(handler=_relay_token)

    run = inner.add_parser("run", help="evaluate every submission of one epoch and set weights")
    run.add_argument("--arena", required=True)
    run.add_argument(
        "--epoch", type=int, help="epoch index; defaults to the one whose window is open"
    )
    run.add_argument("--server", default=os.environ.get("MT_SERVER_URL", PUBLIC_SERVER_URL))
    run.add_argument("--relay", default=os.environ.get("MT_RELAY_URL", ""))
    run.add_argument(
        "--gpus", default=os.environ.get("MT_GPUS", "0"), help="comma separated indices"
    )
    run.add_argument("--concurrency", type=int, default=3)
    run.add_argument("--sandbox-runtime", default=os.environ.get("MT_SANDBOX_RUNTIME", "runsc"))
    run.add_argument("--item-limit", type=int, default=0, help="development only")
    run.add_argument("--dry-run", action="store_true", help="evaluate and report, set no weights")
    add_chain_arguments(run)
    add_common_arguments(run)
    run.set_defaults(handler=_run)

    loop = inner.add_parser("loop", help="run every epoch of an arena as its window opens")
    loop.add_argument("--arena", required=True)
    loop.add_argument("--server", default=os.environ.get("MT_SERVER_URL", PUBLIC_SERVER_URL))
    loop.add_argument("--relay", default=os.environ.get("MT_RELAY_URL", ""))
    loop.add_argument(
        "--gpus", default=os.environ.get("MT_GPUS", "0"), help="comma separated indices"
    )
    loop.add_argument("--concurrency", type=int, default=3)
    loop.add_argument("--sandbox-runtime", default=os.environ.get("MT_SANDBOX_RUNTIME", "runsc"))
    loop.add_argument("--poll-seconds", type=int, default=600)
    loop.add_argument("--once", action="store_true", help="one poll, then exit")
    loop.add_argument("--item-limit", type=int, default=0, help="development only")
    loop.add_argument("--dry-run", action="store_true", help="evaluate and report, set no weights")
    add_chain_arguments(loop)
    add_common_arguments(loop)
    loop.set_defaults(handler=_loop)

    probe = inner.add_parser("probe", help="probe inference operators with withheld items")
    probe.add_argument("--arena", required=True)
    probe.add_argument("--server", default=PUBLIC_SERVER_URL)
    probe.add_argument("--gateway-url", default=PUBLIC_GATEWAY_URL)
    probe.add_argument("--credential", default=os.environ.get("MT_SERVE_CREDENTIAL", ""))
    probe.add_argument("--floor", type=float, default=0.9)
    probe.add_argument("--per-operator", type=int, default=8)
    probe.add_argument("--pause", type=float, default=30.0)
    probe.add_argument("--seed", type=int, default=0)
    probe.add_argument("--once", action="store_true")
    add_chain_arguments(probe)
    add_common_arguments(probe)
    probe.set_defaults(handler=_probe)


def _keys_dir(home: Path) -> Path:
    path = home / "keys"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _register_key(args: argparse.Namespace) -> int:
    config = chain_config(args)
    wallet = open_wallet(config)
    hotkey = hotkey_address(wallet)
    keys = _keys_dir(args.home)
    path = keys / "x25519.key"
    if path.is_file():
        private = bytes.fromhex(path.read_text(encoding="utf-8").strip())
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

        public = (
            X25519PrivateKey.from_private_bytes(private)
            .public_key()
            .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        )
    else:
        private, public = new_recipient_key()
        path.write_text(private.hex(), encoding="utf-8")
        path.chmod(0o600)
    client = open_client(config, wallet)
    try:
        expires = client.block() + KEY_BLOCKS
    except ChainError as exc:
        return fail(str(exc))
    message = key_registration_message(config.netuid, hotkey, public, expires)
    signature = sign_bytes(wallet, message)
    server = ServerClient(args.server, hotkey, wallet)
    try:
        found = server.register_key(public.hex(), expires, signature)
    except ServerError as exc:
        return fail(str(exc))
    print(f"registered  {found.get('public_key', public.hex())}")
    print(f"expires     block {found.get('expires_block', expires)}")
    print(f"stored      {path}")
    return 0


def _relay_token(args: argparse.Namespace) -> int:
    config = chain_config(args)
    wallet = open_wallet(config)
    server = ServerClient(args.server, hotkey_address(wallet), wallet)
    try:
        found = server.relay_token()
    except ServerError as exc:
        return fail(str(exc))
    path = _keys_dir(args.home) / "relay.token"
    path.write_text(str(found.get("token", "")), encoding="utf-8")
    path.chmod(0o600)
    print(f"token       stored at {path}")
    print(f"allowance   {int(found.get('relay_limit_micros') or 0) / 1_000_000:.2f} USD per epoch")
    return 0


def _run(args: argparse.Namespace) -> int:
    config = chain_config(args)
    wallet = open_wallet(config)
    client = open_client(config, wallet)
    server = ServerClient(args.server, hotkey_address(wallet), wallet)
    try:
        block = client.block()
    except ChainError as exc:
        return fail(str(exc))
    epoch_index = args.epoch
    if epoch_index is None:
        try:
            arena = server.arena(args.arena, block)
        except ServerError as exc:
            return fail(str(exc))
        current = (arena or {}).get("current_epoch") or {}
        if current.get("phase") not in ("window", "settling"):
            return fail(f"no evaluation window is open for {args.arena} at block {block}")
        epoch_index = int(current["index"])
    settings = _settings(args, config)
    try:
        outcome = run_epoch(settings, chain=client, wallet=wallet, epoch_index=epoch_index)
    except (EpochError, ReleaseError, ChainError, ServerError) as exc:
        return fail(str(exc))
    print(f"epoch       {epoch_index}")
    print(f"scored      {len(outcome.scores)} submissions, {len(outcome.failures)} refused")
    for hotkey, reason in sorted(outcome.failures.items()):
        print(f"refused     {hotkey[:12]}  {reason[:90]}")
    for hotkey, scores in sorted(outcome.scores.items()):
        parts = "  ".join(f"{s.pool} {s.score:.3f}{'*' if s.eligible else ''}" for s in scores)
        print(f"{hotkey[:12]}  {parts}")
    if outcome.burned:
        print(f"burned      {json.dumps(outcome.burned)}")
    print(f"weights     {len(outcome.weights)} hotkeys{' (dry run)' if args.dry_run else ''}")
    return 0


def _probe(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO)
    if not args.credential:
        return fail("set MT_SERVE_CREDENTIAL or pass --credential")
    config = chain_config(args)
    wallet = open_wallet(config)
    work_root = Path(args.home) / "work" / args.arena
    rng = random.Random(args.seed or None)  # noqa: S311
    while True:
        sets = probes.latest(work_root, args.arena)
        if not sets:
            log.warning("no probe sets under %s yet; run an epoch first", work_root)
        else:
            try:
                found = probes.cycle(
                    server=args.server,
                    gateway=args.gateway_url,
                    credential=args.credential,
                    wallet=wallet,
                    sets=sets,
                    floor=args.floor,
                    per_operator=args.per_operator,
                    rng=rng,
                    pause=min(float(args.pause), 5.0),
                )
            except ServingError as exc:
                log.error("probe cycle: %s", exc)
                found = []
            for item in found:
                state = "ok" if item.success else ("fail" if item.reported else "skip")
                print(f"{state:4s} {item.operator[:12]} {item.model} {item.item_id} {item.reason}")
            reported = [p for p in found if p.reported]
            failed = sum(1 for p in reported if not p.success)
            print(f"probed {len(reported)} answers, {failed} failed")
        if args.once:
            return 0
        time.sleep(max(30.0, float(args.pause)))


def _settings(args: argparse.Namespace, config: Any) -> EpochConfig:
    return EpochConfig(
        chain=config,
        home=args.home,
        arena_id=args.arena,
        server_url=args.server,
        relay_url=args.relay or args.server,
        gpus=tuple(g.strip() for g in str(args.gpus).split(",") if g.strip()),
        concurrency=args.concurrency,
        sandbox_runtime=args.sandbox_runtime,
        dry_run=args.dry_run,
        item_limit=args.item_limit,
    )


def _loop(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO)
    config = chain_config(args)
    wallet = open_wallet(config)
    client = open_client(config, wallet)
    server = ServerClient(args.server, hotkey_address(wallet), wallet)
    state = Path(args.home) / "state"
    state.mkdir(parents=True, exist_ok=True)
    record = state / f"{args.arena}.epochs.json"
    done: set[int] = set()
    if record.is_file():
        done = {int(i) for i in json.loads(record.read_text(encoding="utf-8"))}
    settings = _settings(args, config)
    while True:
        try:
            block = client.block()
            arena = server.arena(args.arena, block)
            current = dict((arena or {}).get("current_epoch") or {})
            index = current.get("index")
            if arena is None:
                log.warning("arena %s is unknown to %s", args.arena, args.server)
            elif current.get("phase") == "window" and index is not None and int(index) not in done:
                log.info("epoch %s of %s is open at block %d", index, args.arena, block)
                outcome = run_epoch(settings, chain=client, wallet=wallet, epoch_index=int(index))
                done.add(int(index))
                record.write_text(json.dumps(sorted(done)), encoding="utf-8")
                log.info(
                    "epoch %s done: %d scored, %d refused, %d weights",
                    index,
                    len(outcome.scores),
                    len(outcome.failures),
                    len(outcome.weights),
                )
            else:
                log.info(
                    "block %d: %s epoch %s is %s", block, args.arena, index, current.get("phase")
                )
        except (EpochError, ReleaseError, ChainError, ServerError) as exc:
            log.error("epoch loop: %s", exc)
        if args.once:
            return 0
        time.sleep(max(30, int(args.poll_seconds)))
