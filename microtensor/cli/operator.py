from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

from microtensor.chain.wallet import hotkey_address
from microtensor.cli.common import (
    add_chain_arguments,
    add_common_arguments,
    chain_config,
    fail,
    open_wallet,
)
from microtensor.core.constants import PUBLIC_SERVER_URL
from microtensor.serving import audit as serving_audit
from microtensor.serving import client, plan
from microtensor.serving.client import ServerError
from microtensor.serving.host import HostConfig, HostError, ServingHost, read_catalogue
from microtensor.serving.host.config import DEFAULT_GPU_GB, REVIEW_SECONDS

log = logging.getLogger("microtensor.cli.operator")

GATEWAY_URL = "wss://api.microtensor.cloud/v1/operators/socket"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("operator", help="serve certified systems for the serving pool")
    inner = parser.add_subparsers(dest="action", required=True)

    enrol = inner.add_parser("register", help="register this hotkey as an inference operator")
    enrol.add_argument("--label", default="", help="a name for your node")
    _shared(enrol)
    enrol.set_defaults(handler=_register)

    collateral = inner.add_parser("collateral", help="report the transfer that posts collateral")
    collateral.add_argument("--reference", required=True, help="the transfer extrinsic hash")
    collateral.add_argument("--block", type=int, required=True, help="the block it landed in")
    _shared(collateral)
    collateral.set_defaults(handler=_collateral)

    declare = inner.add_parser("declare", help="declare the system and artifact you serve")
    declare.add_argument("--model", required=True, help="the model name, as clients call it")
    declare.add_argument("--artifact-digest", required=True, help="its artifact git sha")
    _shared(declare)
    declare.set_defaults(handler=_declare)

    status = inner.add_parser("status", help="what the server holds about this operator")
    _shared(status)
    status.set_defaults(handler=_status)

    checked = inner.add_parser(
        "audit", help="recompute the published serving settlement, no GPU needed"
    )
    checked.add_argument("--json", action="store_true", help="print the full report")
    _shared(checked)
    checked.set_defaults(handler=_audit)

    preview = inner.add_parser("plan", help="what auto mode would hold on this machine")
    _gpu_arguments(preview)
    _policy_arguments(preview)
    _shared(preview)
    preview.set_defaults(handler=_plan)

    serve = inner.add_parser("run", help="hold certified systems and answer the gateway")
    serve.add_argument("--auto", action="store_true", help="let the catalogue choose what to hold")
    serve.add_argument(
        "--serve",
        action="append",
        default=[],
        metavar="MODEL@SHA",
        help="hold one pinned system; repeat for each",
    )
    serve.add_argument("--gateway", default=GATEWAY_URL)
    serve.add_argument("--worker", default="", help="a name when one hotkey runs several hosts")
    serve.add_argument("--sandbox-runtime", default="runsc")
    serve.add_argument("--request-seconds", type=float, default=120.0)
    serve.add_argument("--review-seconds", type=float, default=REVIEW_SECONDS)
    _gpu_arguments(serve)
    _policy_arguments(serve)
    _shared(serve)
    serve.set_defaults(handler=_run)


def _gpu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpus", default="0", help="comma separated GPU indices")
    parser.add_argument("--gpu-gb", type=float, default=DEFAULT_GPU_GB, help="memory per GPU")


def _policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arena", action="append", default=[], help="only these arenas")
    parser.add_argument("--not", dest="excluded", action="append", default=[])


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server", default=PUBLIC_SERVER_URL)
    add_chain_arguments(parser)
    add_common_arguments(parser)


def _wallet(args: argparse.Namespace) -> Any:
    return open_wallet(chain_config(args))


def _show(found: dict[str, object]) -> int:
    print(json.dumps(found, indent=2, sort_keys=True))
    return 0


def _register(args: argparse.Namespace) -> int:
    wallet = _wallet(args)
    try:
        found = client.register(args.server, wallet, label=args.label)
    except ServerError as exc:
        return fail(str(exc))
    print(f"registered {hotkey_address(wallet)}")
    required = found.get("collateral_required_tao")
    pay_to = found.get("collateral_pay_to")
    if required:
        print(f"post {required} TAO to {pay_to}, then report it with `mt operator collateral`")
    return _show(found)


def _collateral(args: argparse.Namespace) -> int:
    wallet = _wallet(args)
    try:
        found = client.post_collateral(
            args.server, wallet, reference=args.reference, block=args.block
        )
    except ServerError as exc:
        return fail(str(exc))
    return _show(found)


def _declare(args: argparse.Namespace) -> int:
    wallet = _wallet(args)
    try:
        found = client.declare(
            args.server, wallet, model=args.model, artifact_digest=args.artifact_digest
        )
    except ServerError as exc:
        return fail(str(exc))
    return _show(found)


def _status(args: argparse.Namespace) -> int:
    wallet = _wallet(args)
    try:
        found = client.me(args.server, wallet)
    except ServerError as exc:
        return fail(str(exc))
    return _show(found)


def _audit(args: argparse.Namespace) -> int:
    try:
        found = serving_audit.audit(args.server)
    except ServerError as exc:
        return fail(str(exc))
    if args.json:
        return _show(found.to_dict())
    if not found.checked and not found.invented and not found.omitted:
        print("no serving epoch has settled yet; nothing to check")
        return 0
    if found.holds:
        print(f"the published settlement holds: {found.agreed} of {found.checked} rows recomputed")
        return 0
    print(f"the published settlement does not hold: {found.agreed} of {found.checked} agreed")
    for row in found.disagreements:
        print(f"  {row}")
    for key in found.invented:
        print(f"  {key} was published but does not follow from the evidence")
    for key in found.omitted:
        print(f"  {key} follows from the evidence but was never published")
    return 1


def _gpus(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(g.strip() for g in str(args.gpus).split(",") if g.strip())


def _policy(args: argparse.Namespace) -> plan.Policy:
    return plan.Policy(arenas=frozenset(args.arena), exclude=frozenset(args.excluded))


def parse_pins(entries: list[str]) -> dict[str, str]:
    pinned: dict[str, str] = {}
    for entry in entries:
        text = entry.strip()
        if "@" not in text:
            raise HostError(f"cannot read {entry!r}; expected MODEL@SHA")
        model, sha = text.split("@", 1)
        pinned[model.strip()] = sha.strip().lower()
    return pinned


def _plan(args: argparse.Namespace) -> int:
    try:
        rows = read_catalogue(args.server)
    except Exception as exc:
        return fail(f"could not read the catalogue: {exc}")
    try:
        capacity = plan.Capacity(gpus=_gpus(args), gpu_gb=args.gpu_gb)
    except plan.PlanError as exc:
        return fail(str(exc))
    return _show(plan.build(rows, capacity, policy=_policy(args)).to_dict())


def _run(args: argparse.Namespace) -> int:
    if not args.auto and not args.serve:
        return fail("pass --auto to let the catalogue choose, or --serve MODEL@SHA")
    wallet = _wallet(args)
    try:
        pinned = parse_pins(args.serve) if args.serve else None
    except HostError as exc:
        return fail(str(exc))
    config = HostConfig(
        home=args.home,
        gpus=_gpus(args),
        gpu_gb=args.gpu_gb,
        sandbox_runtime=args.sandbox_runtime,
        request_seconds=args.request_seconds,
        review_seconds=args.review_seconds,
    )

    def declare(model: str, digest: str) -> None:
        client.declare(args.server, wallet, model=model, artifact_digest=digest)

    host = ServingHost(
        config,
        hotkey=hotkey_address(wallet),
        server=args.server,
        gateway=args.gateway,
        policy=_policy(args),
        pinned=pinned,
        worker=args.worker,
        declare=declare,
    )
    print(f"serving as {hotkey_address(wallet)} from {args.server}")
    try:
        asyncio.run(host.run())
    except KeyboardInterrupt:
        print("stopped")
    except HostError as exc:
        return fail(str(exc))
    return 0
