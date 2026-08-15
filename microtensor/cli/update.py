from __future__ import annotations

import argparse

from microtensor.chain.rounds import round_for_block
from microtensor.cli.common import (
    add_chain_arguments,
    add_common_arguments,
    chain_config,
    fail,
    open_client,
)
from microtensor.core.constants import (
    GENESIS_BLOCK,
    MECHANISM_VERSION,
    RELEASE_CHANNELS,
    RELEASE_REPO,
    RELEASE_SIGNING_KEY,
    ROUND_BLOCKS,
)
from microtensor.update.apply import apply_release
from microtensor.update.policy import Action, decide
from microtensor.update.release import ReleaseError, fetch_releases, latest


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("update", help="check and apply signed releases")
    inner = parser.add_subparsers(dest="action", required=True)

    check = inner.add_parser("check", help="what would this validator do right now")
    _add_update_arguments(check)
    add_chain_arguments(check)
    check.set_defaults(handler=_check)

    apply_cmd = inner.add_parser("apply", help="verify and install the latest release")
    _add_update_arguments(apply_cmd)
    add_chain_arguments(apply_cmd)
    apply_cmd.add_argument("--dry-run", action="store_true", help="verify but do not install")
    apply_cmd.add_argument(
        "--force", action="store_true", help="apply even outside a submission window"
    )
    apply_cmd.set_defaults(handler=_apply)

    listing = inner.add_parser("list", help="published releases on a channel")
    _add_update_arguments(listing)
    listing.set_defaults(handler=_list)


def _add_update_arguments(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)
    parser.add_argument("--repo", default=RELEASE_REPO)
    parser.add_argument("--channel", default="mainnet", choices=list(RELEASE_CHANNELS))
    parser.add_argument("--signing-key", default=RELEASE_SIGNING_KEY)
    parser.add_argument(
        "--allow-unsigned",
        action="store_true",
        help="install a release whose SHA256SUMS carries no verifiable signature",
    )
    parser.add_argument("--allow-mechanism-change", action="store_true")
    parser.add_argument("--allow-major", action="store_true")
    parser.add_argument("--round-blocks", type=int, default=ROUND_BLOCKS)
    parser.add_argument("--genesis-block", type=int, default=GENESIS_BLOCK)


def _block(args: argparse.Namespace) -> int:
    client = open_client(chain_config(args))
    try:
        return client.block()
    finally:
        client.close()


def _decide(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    release = latest(args.repo, channel=args.channel)
    block = _block(args)
    round_ = round_for_block(block, length=args.round_blocks, genesis=args.genesis_block)
    return (
        decide(
            release,
            round_,
            block,
            allow_mechanism_change=args.allow_mechanism_change,
            allow_major=args.allow_major,
        ),
        block,
        round_,
    )


def _check(args: argparse.Namespace) -> int:
    try:
        decision, block, round_ = _decide(args)
    except ReleaseError as exc:
        return fail(str(exc))

    print(f"running      {MECHANISM_VERSION}")
    print(f"channel      {args.channel}")
    print(f"block        {block}  (round {round_.index})")
    if decision.release is not None:
        release = decision.release
        signed = "signed" if release.is_signed else "UNSIGNED"
        print(f"latest       {release.tag}  ({signed})")
        print(f"mechanism    {release.mechanism_version}")
        if release.activation_block is not None:
            print(f"activates    block {release.activation_block}")
    print(f"decision     {decision.action.value}")
    print(f"reason       {decision.reason}")
    if decision.ready_at_block is not None:
        print(f"ready at     block {decision.ready_at_block}")

    return 0 if decision.action in (Action.NONE, Action.APPLY) else 2


def _apply(args: argparse.Namespace) -> int:
    try:
        decision, _, _ = _decide(args)
    except ReleaseError as exc:
        return fail(str(exc))

    if decision.release is None:
        print(decision.reason)
        return 0

    if not decision.should_apply and not args.force:
        return fail(f"{decision.action.value}: {decision.reason}")

    applied = apply_release(
        decision.release,
        signing_key=args.signing_key,
        require_signature=not args.allow_unsigned,
        dry_run=args.dry_run,
    )

    verification = applied.verification
    print(f"release      {applied.release.tag}")
    print(f"digest       {'ok' if verification.digest_ok else 'FAILED'}")
    print(f"signature    {'ok' if verification.signature_ok else 'FAILED'}")
    if applied.reason:
        print(f"note         {applied.reason}")

    if not applied.installed:
        return 0 if args.dry_run and verification.trusted else 1

    print("\ninstalled — restart the validator to run it")
    return 0


def _list(args: argparse.Namespace) -> int:
    try:
        releases = fetch_releases(args.repo, channel=args.channel)
    except ReleaseError as exc:
        return fail(str(exc))

    if not releases:
        print(f"no releases published on {args.channel}")
        return 0

    print(f"{'tag':<20}{'mechanism':<12}{'activates':>12}  signed")
    for release in releases:
        activation = str(release.activation_block) if release.activation_block else "-"
        print(
            f"{release.tag:<20}{release.mechanism_version:<12}{activation:>12}  "
            f"{'yes' if release.is_signed else 'NO'}"
        )
    return 0
