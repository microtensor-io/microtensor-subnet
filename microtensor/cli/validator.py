from __future__ import annotations

import argparse
import json
import logging
import os
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
    reclaim_logging,
)
from microtensor.core.constants import (
    ARTIFACT_CACHE_CAP_BYTES,
    COORDINATOR_URL,
    CORPUS_VERSION,
    CPU_SECONDS_PER_ARTIFACT,
    GENESIS_BLOCK,
    PROVENANCE_REQUIRED,
    RELEASE_CHANNELS,
    RELEASE_REPO,
    RELEASE_SIGNING_KEY,
    ROUND_BLOCKS,
    TASKS_PER_ROUND,
    UPDATE_POLL_SECONDS,
)
from microtensor.core.role import WORKER, RoleConflict
from microtensor.core.role import require as require_role
from microtensor.envelope.certify import (
    DEFAULT_REPETITIONS,
    LAUNCH_CLASSES,
    CertificationError,
    band_verdict,
    fit_band,
)
from microtensor.envelope.certify import certify as certification_run
from microtensor.envelope.certify import save as certify_save
from microtensor.harness.jail import cpu_limit_binds
from microtensor.harness.limits import sandbox_available
from microtensor.provenance.record import CachedStore
from microtensor.provenance.wandb_store import WandbStore
from microtensor.update.loop import UpdateChecker, UpdateSettings
from microtensor.validator import loopback
from microtensor.validator.client import CoordinatorClient
from microtensor.validator.context import ValidatorConfig, ValidatorContext
from microtensor.validator.loop import RoundLoop
from microtensor.validator.round import current_round, run_round

log = logging.getLogger("microtensor.cli.validator")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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

    cert = inner.add_parser("certify", help="benchmark this host as a reference device for a class")
    add_common_arguments(cert)
    cert.add_argument("hardware_class", choices=list(LAUNCH_CLASSES))
    cert.add_argument("--cooling-mode", default=None)
    cert.add_argument("--power-mode", default=None)
    cert.add_argument("--warmup-policy", default=None)
    cert.add_argument("--idle-seconds", type=int, default=None)
    cert.add_argument("--repetitions", type=int, default=None)
    cert.add_argument(
        "--fit-band",
        type=int,
        metavar="RUNS",
        default=None,
        help="run the workload RUNS times and print a CERT_BANDS entry from the spread",
    )
    cert.set_defaults(handler=_certify)


def _add_validator_arguments(parser: argparse.ArgumentParser) -> None:
    add_chain_arguments(parser)
    add_common_arguments(parser)
    parser.add_argument(
        "--coordinator",
        default=os.environ.get("MT_COORDINATOR_URL", COORDINATOR_URL),
        help="coordinator base URL; without one this validator holds its last vector",
    )
    parser.add_argument("--corpus", type=Path, help="directory of <track>.jsonl corpora")
    parser.add_argument("--corpus-version", default=CORPUS_VERSION)
    parser.add_argument("--round-blocks", type=int, default=ROUND_BLOCKS)
    parser.add_argument("--genesis-block", type=int, default=GENESIS_BLOCK)
    parser.add_argument("--tasks-per-round", type=int, default=TASKS_PER_ROUND)
    parser.add_argument("--cache-cap-bytes", type=int, default=ARTIFACT_CACHE_CAP_BYTES)
    parser.add_argument("--cpu-seconds", type=int, default=CPU_SECONDS_PER_ARTIFACT)
    parser.add_argument(
        "--profile-seconds",
        type=int,
        default=int(os.environ.get("MT_PROFILE_SECONDS", "150")),
    )
    parser.add_argument(
        "--allow-unsandboxed",
        action="store_true",
        help="run artifacts without resource limits (development only)",
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="if the cpu limit does not bind, run abstain-only instead of exiting",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="compute weights but do not submit them"
    )
    parser.add_argument(
        "--auto-update",
        action="store_true",
        help="install signed releases between rounds and exit for the supervisor to restart",
    )
    parser.add_argument("--update-channel", default="stable", choices=list(RELEASE_CHANNELS))
    parser.add_argument("--update-repo", default=RELEASE_REPO)
    parser.add_argument("--update-poll-seconds", type=int, default=UPDATE_POLL_SECONDS)
    parser.add_argument("--signing-key", default=RELEASE_SIGNING_KEY)
    parser.add_argument("--allow-unsigned-updates", action="store_true")
    parser.add_argument("--allow-mechanism-change", action="store_true")


def _degraded(args: argparse.Namespace, *, probe: bool) -> bool:
    if not probe or args.allow_unsandboxed or not sandbox_available():
        return False
    if cpu_limit_binds():
        return False
    if not args.allow_degraded:
        raise SystemExit(
            "the cpu limit does not bind on this host, so budgets cannot be enforced "
            "and slow infrastructure would be misattributed as artifact fault; fix the "
            "host, or pass --allow-degraded to run abstain-only"
        )
    log.error("cpu limit does not bind; running degraded, no weights will be set")
    return True


def _run_store(args: argparse.Namespace, *, probe: bool) -> CachedStore | None:
    if not PROVENANCE_REQUIRED:
        log.warning("provenance is not required; submissions need no training run")
        return None

    store = WandbStore()
    if probe:
        reachable, reason = store.reachable()
        if not reachable:
            raise SystemExit(
                f"the training run store is unreachable, and a validator that cannot "
                f"check provenance would materialise a different participant set to its "
                f"peers and abstain every round: {reason}"
            )
        log.info("training run store reachable at %s/%s", store.entity, store.project)
    return CachedStore(store)


def _build(args: argparse.Namespace, *, probe: bool = False) -> ValidatorContext:
    chain = chain_config(args)
    home = Path(args.home)
    corpus = args.corpus or home / "corpus"

    try:
        require_role(home, WORKER)
    except RoleConflict as exc:
        raise SystemExit(str(exc)) from exc

    if not sandbox_available() and not args.allow_unsandboxed:
        raise SystemExit(
            "this host cannot enforce cpu and memory limits; run the validator on Linux, "
            "or pass --allow-unsandboxed for local development"
        )

    degraded = _degraded(args, probe=probe)
    runs = _run_store(args, probe=probe)

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
        degraded=degraded,
        coordinator_url=args.coordinator,
    )

    log.info(
        "wallet %s/%s on %s, netuid %d",
        chain.wallet_name,
        chain.wallet_hotkey,
        chain.network,
        chain.netuid,
    )
    wallet = open_wallet(chain, required=not args.dry_run)
    client = open_client(chain, wallet)
    reclaim_logging()
    hotkey = hotkey_address(wallet) if wallet is not None else ""
    if hotkey:
        log.info("hotkey %s", hotkey)

    log.info(
        "fetching the metagraph from %s; the first fetch can take a few minutes", chain.network
    )
    snapshot = client.snapshot()
    reclaim_logging()
    log.info("metagraph: %d neurons at block %d", len(snapshot), snapshot.block)
    if hotkey and not snapshot.is_registered(hotkey):
        raise SystemExit(f"hotkey {hotkey} is not registered on netuid {chain.netuid}")
    if hotkey and not snapshot.has_permit(hotkey):
        log.warning("hotkey %s holds no validator permit; weights may be ignored", hotkey)
    elif hotkey:
        log.info("registered with a validator permit")

    coordinator = None
    if config.coordinated:
        coordinator = CoordinatorClient(
            base_url=config.coordinator_url, hotkey=hotkey, wallet=wallet
        )
        log.info("taking assignments from the coordinator at %s", config.coordinator_url)

    return ValidatorContext.build(
        config, client, wallet=wallet, hotkey=hotkey, runs=runs, coordinator=coordinator
    )


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
        log.warning(
            "auto-update disabled: no signing key is pinned in this build; pass "
            "--signing-key, or --allow-unsigned-updates to accept unverified releases"
        )
        return None
    log.info("auto-update armed on the %s channel", settings.channel)
    return UpdateChecker(settings)


def _run(args: argparse.Namespace) -> int:
    context = _build(args, probe=True)
    loop = RoundLoop(context, updater=_updater(args))
    loop.install_signal_handlers()
    try:
        loop.run(max_rounds=args.max_rounds)
    finally:
        context.close()
    return 75 if loop.restarting else 0


def _once(args: argparse.Namespace) -> int:
    context = _build(args, probe=True)
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


def _certify(args: argparse.Namespace) -> int:
    policy = {
        key: value
        for key, value in {
            "cooling_mode": args.cooling_mode,
            "power_mode": args.power_mode,
            "warmup_policy": args.warmup_policy,
            "idle_seconds": args.idle_seconds,
        }.items()
        if value is not None
    }

    if args.fit_band:
        try:
            fitted, results = fit_band(
                args.hardware_class,
                policy,
                runs=args.fit_band,
                repetitions=args.repetitions or DEFAULT_REPETITIONS,
            )
        except CertificationError as exc:
            return fail(str(exc))

        certify_save(results[-1], Path(args.home))
        print(f"class            {fitted.class_id}")
        print(f"runs             {fitted.runs}")
        print(
            f"p50 observed     {fitted.p50_observed[0]:.1f} to {fitted.p50_observed[1]:.1f} ms"
            f"  (spread {fitted.p50_spread:.1%})"
        )
        print(
            f"p95 observed     {fitted.p95_observed[0]:.1f} to {fitted.p95_observed[1]:.1f} ms"
            f"  (spread {fitted.p95_spread:.1%})"
        )
        print(f"peak rss         {fitted.rss_observed / 1024**2:.0f} MiB")
        print(f"device profile   {results[-1].digest}")
        print("\npaste into CERT_BANDS in microtensor/envelope/certify.py:\n")
        print(fitted.as_constant())
        print(f"\nand set device_profile on the {fitted.class_id} class to:")
        print(f"    {results[-1].digest}")
        if fitted.p95_spread > 0.5:
            print(
                "\nWARNING: p95 varied by more than half across runs. Settle the host "
                "(cooling, background load) before trusting this band."
            )
        return 0

    try:
        certification = certification_run(
            args.hardware_class,
            policy,
            repetitions=args.repetitions or DEFAULT_REPETITIONS,
        )
    except CertificationError as exc:
        return fail(str(exc))

    path = certify_save(certification, Path(args.home))
    passed, verdict = band_verdict(certification)

    print(f"class            {certification.class_id}")
    print(f"workload         v{certification.workload_version}")
    print(f"p50 / p95        {certification.latency.p50:.1f} / {certification.latency.p95:.1f} ms")
    print(f"peak rss         {certification.peak_rss_bytes / 1024**2:.0f} MiB")
    print(f"device profile   {certification.digest}")
    print(f"policy           {json.dumps(certification.policy, sort_keys=True)}")
    print(f"band             {verdict}")
    print(f"\nsaved to {path}; envelope measurements will carry this profile")
    return 0 if passed is not False else 2


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
