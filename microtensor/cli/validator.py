from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from microtensor.chain.wallet import hotkey_address
from microtensor.cli.common import (
    add_chain_arguments,
    add_common_arguments,
    chain_config,
    fail,
    open_client,
    open_wallet,
)
from microtensor.core.constants import (
    ARTIFACT_CACHE_CAP_BYTES,
    CORPUS_VERSION,
    GENESIS_BLOCK,
    RELEASE_CHANNELS,
    RELEASE_REPO,
    RELEASE_SIGNING_KEY,
    ROUND_BLOCKS,
    TASKS_PER_ROUND,
    UPDATE_POLL_SECONDS,
)
from microtensor.harness.limits import sandbox_available
from microtensor.update.loop import UpdateChecker, UpdateSettings
from microtensor.validator import loopback
from microtensor.validator.context import ValidatorConfig, ValidatorContext
from microtensor.validator.loop import RoundLoop
from microtensor.validator.round import current_round, run_round

log = logging.getLogger("microtensor.cli.validator")


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("validator", help="run or inspect a validator")
    inner = parser.add_subparsers(dest="action", required=True)

    run = inner.add_parser("run", help="run the round loop until stopped")
    _add_validator_arguments(run)
    run.add_argument("--max-rounds", type=int, help="stop after this many rounds")
    run.set_defaults(handler=_run)

    once = inner.add_parser("once", help="evaluate and settle exactly one round")
    _add_validator_arguments(once)
    once.set_defaults(handler=_once)

    status = inner.add_parser("status", help="show local validator state")
    _add_validator_arguments(status)
    status.set_defaults(handler=_status)

    loop = inner.add_parser(
        "loopback", help="run real rounds against a synthetic chain, no wallet needed"
    )
    add_common_arguments(loop)
    loop.add_argument("--rounds", type=int, default=2)
    loop.add_argument("--miners", type=int, default=3)
    loop.add_argument("--tasks", type=int, default=12)
    loop.set_defaults(handler=_loopback)


def _add_validator_arguments(parser: argparse.ArgumentParser) -> None:
    add_chain_arguments(parser)
    add_common_arguments(parser)
    parser.add_argument("--corpus", type=Path, help="directory of <track>.jsonl corpora")
    parser.add_argument("--corpus-version", default=CORPUS_VERSION)
    parser.add_argument("--round-blocks", type=int, default=ROUND_BLOCKS)
    parser.add_argument("--genesis-block", type=int, default=GENESIS_BLOCK)
    parser.add_argument("--tasks-per-round", type=int, default=TASKS_PER_ROUND)
    parser.add_argument("--cache-cap-bytes", type=int, default=ARTIFACT_CACHE_CAP_BYTES)
    parser.add_argument("--cpu-seconds", type=int, default=900)
    parser.add_argument("--profile-seconds", type=int, default=60)
    parser.add_argument(
        "--allow-unsandboxed",
        action="store_true",
        help="run artifacts without resource limits (development only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="compute weights but do not submit them"
    )
    parser.add_argument(
        "--auto-update",
        action="store_true",
        help="install signed releases between rounds and exit for the supervisor to restart",
    )
    parser.add_argument("--update-channel", default="mainnet", choices=list(RELEASE_CHANNELS))
    parser.add_argument("--update-repo", default=RELEASE_REPO)
    parser.add_argument("--update-poll-seconds", type=int, default=UPDATE_POLL_SECONDS)
    parser.add_argument("--signing-key", default=RELEASE_SIGNING_KEY)
    parser.add_argument("--allow-unsigned-updates", action="store_true")
    parser.add_argument("--allow-mechanism-change", action="store_true")


def _build(args: argparse.Namespace) -> ValidatorContext:
    chain = chain_config(args)
    home = Path(args.home)
    corpus = args.corpus or home / "corpus"

    if not sandbox_available() and not args.allow_unsandboxed:
        raise SystemExit(
            "this host cannot enforce cpu and memory limits; run the validator on Linux, "
            "or pass --allow-unsandboxed for local development"
        )

    config = ValidatorConfig(
        chain=chain,
        home=home,
        corpus_dir=corpus,
        corpus_version=args.corpus_version,
        round_blocks=args.round_blocks,
        genesis_block=args.genesis_block,
        tasks_per_round=args.tasks_per_round,
        cache_cap_bytes=args.cache_cap_bytes,
        cpu_seconds_per_artifact=args.cpu_seconds,
        profile_seconds=args.profile_seconds,
        allow_unsandboxed=args.allow_unsandboxed,
        dry_run=args.dry_run,
    )

    wallet = open_wallet(chain, required=not args.dry_run)
    client = open_client(chain, wallet)
    hotkey = hotkey_address(wallet) if wallet is not None else ""

    snapshot = client.snapshot()
    if hotkey and not snapshot.is_registered(hotkey):
        raise SystemExit(f"hotkey {hotkey} is not registered on netuid {chain.netuid}")
    if hotkey and not snapshot.has_permit(hotkey):
        log.warning("hotkey %s holds no validator permit; weights may be ignored", hotkey)

    return ValidatorContext.build(config, client, wallet=wallet, hotkey=hotkey)


def _updater(args: argparse.Namespace) -> UpdateChecker | None:
    if not args.auto_update:
        return None

    settings = UpdateSettings(
        enabled=True,
        channel=args.update_channel,
        repo=args.update_repo,
        poll_seconds=args.update_poll_seconds,
        signing_key=args.signing_key,
        require_signature=not args.allow_unsigned_updates,
        allow_mechanism_change=args.allow_mechanism_change,
    )
    if not settings.signing_key and settings.require_signature:
        raise SystemExit(
            "auto-update needs a release signing key; pass --signing-key, or "
            "--allow-unsigned-updates to accept unverified releases"
        )
    log.info("auto-update armed on the %s channel", settings.channel)
    return UpdateChecker(settings)


def _run(args: argparse.Namespace) -> int:
    context = _build(args)
    loop = RoundLoop(context, updater=_updater(args))
    loop.install_signal_handlers()
    try:
        loop.run(max_rounds=args.max_rounds)
    finally:
        context.close()
    return 75 if loop.restarting else 0


def _once(args: argparse.Namespace) -> int:
    context = _build(args)
    try:
        outcome = run_round(context, current_round(context))
        print(f"round {outcome.round_index}: {outcome.status}")
        if outcome.reason:
            print(f"  reason      {outcome.reason}")
        print(f"  participants {outcome.participants}")
        print(f"  scored       {outcome.scored}")
        if outcome.settlement is not None:
            print(f"  weights      {len(outcome.settlement.vector)}")
        return 0 if outcome.settled else 2
    finally:
        context.close()


def _loopback(args: argparse.Namespace) -> int:
    home = Path(args.home) / "loopback"
    shutil.rmtree(home, ignore_errors=True)

    log.warning("loopback mode: synthetic chain, reference engine, signatures not verified")
    world = loopback.build(home, miners=args.miners, tasks_per_round=args.tasks)

    try:
        outcomes = []
        for _ in range(max(1, args.rounds)):
            outcome = run_round(world.context, world.round)
            outcomes.append(outcome)
            world = loopback.advance(world)

        print()
        header = f"{'round':>8}  {'status':<11}{'participants':>13}{'scored':>8}"
        print(f"{header}{'weights':>9}  reason")
        for outcome in outcomes:
            weights = len(outcome.settlement.vector) if outcome.settlement else 0
            print(
                f"{outcome.round_index:>8}  {outcome.status:<11}{outcome.participants:>13}"
                f"{outcome.scored:>8}{weights:>9}  {outcome.reason}"
            )

        submitted = world.client.submitted
        print(f"\nweight vectors submitted: {len(submitted)}")
        for vector in submitted:
            print(f"  uids {list(vector.uids)}  values {list(vector.values)}  sum {vector.total}")

        settled = sum(1 for o in outcomes if o.settled)
        print(f"\n{settled}/{len(outcomes)} rounds settled")
        return 0 if settled == len(outcomes) else 2
    finally:
        world.context.close()


def _status(args: argparse.Namespace) -> int:
    context = _build(args)
    try:
        summary = context.state.summary()
        last = context.state.last_settled_round()
        print(f"netuid          {context.config.chain.netuid}")
        print(f"competitions    {len(context.competitions)}")
        print(f"tracks          {', '.join(context.tracks)}")
        print(f"last settled    {last if last is not None else 'never'}")
        for key, value in summary.items():
            print(f"{key:<15} {value}")
        print(f"cache           {context.cache.total_bytes / 1024**3:.2f} GiB")
        return 0
    except Exception as exc:
        return fail(str(exc))
    finally:
        context.close()
