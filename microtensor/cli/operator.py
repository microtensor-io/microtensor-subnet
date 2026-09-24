from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from microtensor.chain.wallet import hotkey_address
from microtensor.cli.common import (
    add_chain_arguments,
    add_common_arguments,
    chain_config,
    fail,
    open_wallet,
)
from microtensor.core.constants import GATEWAY_URL, PUBLIC_SERVER_URL
from microtensor.serving import client
from microtensor.serving import loop as probe_loop
from microtensor.serving.agent import AgentError, HttpEngine, Settings, run
from microtensor.serving.client import ServerError

log = logging.getLogger("microtensor.cli.operator")

DEFAULT_ENGINE_URL = "http://127.0.0.1:8080"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("operator", help="serve certified models for the serving pool")
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

    declare = inner.add_parser("declare", help="declare the model and artifact you serve")
    declare.add_argument("--model", required=True, help="the model name, as clients call it")
    declare.add_argument("--artifact-digest", required=True, help="its manifest digest")
    _shared(declare)
    declare.set_defaults(handler=_declare)

    status = inner.add_parser("status", help="what the server holds about this operator")
    _shared(status)
    status.set_defaults(handler=_status)

    serve = inner.add_parser("run", help="dial the gateway and answer requests")
    serve.add_argument("--model", required=True)
    serve.add_argument("--artifact-digest", required=True)
    serve.add_argument("--engine-url", default=DEFAULT_ENGINE_URL, help="your local llama.cpp")
    serve.add_argument("--gateway", default=GATEWAY_URL)
    serve.add_argument("--concurrency", type=int, default=8)
    serve.add_argument("--worker", default="", help="a name when one host runs several")
    _shared(serve)
    serve.set_defaults(handler=_run)

    verify = inner.add_parser("verify", help="probe operators and report verdicts, as a validator")
    verify.add_argument("--gateway-url", default="", help="the gateway's http base")
    verify.add_argument(
        "--credential", default="", help="the serving credential; or MT_SERVE_SECRET"
    )
    verify.add_argument("--artifacts", type=Path, required=True, help="model to artifact path map")
    verify.add_argument("--calibrations", type=Path, required=True, help="model to threshold map")
    verify.add_argument("--model", default="", help="only this model")
    verify.add_argument("--per-operator", type=int, default=8)
    verify.add_argument("--once", action="store_true", help="one pass, then stop")
    verify.add_argument("--pause", type=float, default=probe_loop.CYCLE_PAUSE_SECONDS)
    _shared(verify)
    verify.set_defaults(handler=_verify)


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server", default=PUBLIC_SERVER_URL)
    add_chain_arguments(parser)
    add_common_arguments(parser)


def _wallet(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    return open_wallet(chain_config(args))


def _show(found: dict[str, object]) -> int:
    print(json.dumps(found, indent=2, sort_keys=True))
    return 0


def _register(args: argparse.Namespace) -> int:
    wallet = _wallet(args)
    try:
        found = client.register(args.server, wallet, label=args.label)
    except client.ServerError as exc:
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
    except client.ServerError as exc:
        return fail(str(exc))
    return _show(found)


def _declare(args: argparse.Namespace) -> int:
    wallet = _wallet(args)
    try:
        found = client.declare(
            args.server, wallet, model=args.model, artifact_digest=args.artifact_digest
        )
    except client.ServerError as exc:
        return fail(str(exc))
    return _show(found)


def _status(args: argparse.Namespace) -> int:
    wallet = _wallet(args)
    try:
        found = client.me(args.server, wallet)
    except client.ServerError as exc:
        return fail(str(exc))
    return _show(found)


def _run(args: argparse.Namespace) -> int:
    wallet = _wallet(args)
    try:
        settings = Settings(
            gateway=args.gateway,
            hotkey=hotkey_address(wallet),
            model=args.model,
            artifact_digest=args.artifact_digest,
            engine_url=args.engine_url,
            concurrency=args.concurrency,
            worker=args.worker,
        )
    except AgentError as exc:
        return fail(str(exc))

    engine = HttpEngine(settings.engine_url)
    try:
        asyncio.run(_serve(settings, engine))
    except KeyboardInterrupt:
        print("stopped")
    except AgentError as exc:
        return fail(str(exc))
    return 0


async def _serve(settings: Settings, engine: HttpEngine) -> None:
    if not await engine.ready():
        raise AgentError(f"no engine answering at {settings.engine_url}; start it first")
    log.info("engine ready at %s", settings.engine_url)
    await run(settings, engine)


def _verify(args: argparse.Namespace) -> int:
    wallet = _wallet(args)
    credential = args.credential or os.environ.get("MT_SERVE_SECRET", "")
    if not credential:
        return fail("a validator needs the serving credential to reach operators")

    gateway = args.gateway_url or _http(args.gateway)
    artifacts = probe_loop.load_artifacts(args.artifacts)
    calibrations = probe_loop.load_calibrations(args.calibrations)
    if not artifacts:
        return fail(f"no artifact paths in {args.artifacts}")
    if not calibrations:
        return fail(f"no calibrations in {args.calibrations}")

    missing = sorted(set(artifacts) - set(calibrations))
    if missing:
        return fail(f"no calibration for {', '.join(missing)}; an uncalibrated model cannot judge")

    while True:
        try:
            tally = probe_loop.cycle(
                server=args.server,
                gateway=gateway,
                credential=credential,
                wallet=wallet,
                artifacts=artifacts,
                calibrations=calibrations,
                model=args.model,
                per_operator=args.per_operator,
            )
        except ServerError as exc:
            return fail(str(exc))
        print(json.dumps(tally.to_dict(), sort_keys=True))
        if args.once:
            return 0
        time.sleep(max(1.0, args.pause))


def _http(gateway: str) -> str:
    if gateway.startswith("wss://"):
        return "https://" + gateway[len("wss://") :].split("/v1/")[0]
    if gateway.startswith("ws://"):
        return "http://" + gateway[len("ws://") :].split("/v1/")[0]
    return gateway
