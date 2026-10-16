from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Any

from microtensor_subnet.chain.wallet import hotkey_address
from microtensor_subnet.cli.common import (
    add_chain_arguments,
    add_common_arguments,
    chain_config,
    fail,
    open_wallet,
)
from microtensor_subnet.core.constants import PUBLIC_SERVER_URL
from microtensor_subnet.serving import audit as serving_audit
from microtensor_subnet.serving import client, plan
from microtensor_subnet.serving.client import ServerError
from microtensor_subnet.serving.engines import LLAMA, LLAMA_IMAGE, SGLANG, SGLANG_IMAGE
from microtensor_subnet.serving.host import HostConfig, HostError, ServingHost, read_catalogue
from microtensor_subnet.serving.host.config import DEFAULT_GPU_GB, MEM_FRACTION, REVIEW_SECONDS

log = logging.getLogger("microtensor_subnet.cli.operator")

GATEWAY_URL = "wss://api.microtensor.cloud/v1/operators/socket"
KINDS = ("system", "model", "any")
NO_IMAGE = "none"


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "operator", help="serve certified systems and general models for the serving pool"
    )
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
    _engine_arguments(preview)
    _shared(preview)
    preview.set_defaults(handler=_plan)

    serve = inner.add_parser(
        "run", help="hold certified systems or general models and answer the gateway"
    )
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
    serve.add_argument(
        "--mem-fraction",
        type=float,
        default=MEM_FRACTION,
        help="share of each GPU the engine keeps for weights and cache",
    )
    serve.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face token for gated model repositories (env HF_TOKEN)",
    )
    serve.add_argument("--engine-url", default="", help=argparse.SUPPRESS)
    _gpu_arguments(serve)
    _policy_arguments(serve)
    _engine_arguments(serve)
    _shared(serve)
    serve.set_defaults(handler=_run)


def _gpu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpus", default="0", help="comma separated GPU indices")
    parser.add_argument("--gpu-gb", type=float, default=DEFAULT_GPU_GB, help="memory per GPU")


def _policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arena", action="append", default=[], help="only these arenas")
    parser.add_argument("--not", dest="excluded", action="append", default=[])
    parser.add_argument(
        "--kind",
        choices=KINDS,
        default="any",
        help="certified systems, general models or both",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=[],
        metavar="NAME",
        help="only these models; repeat for each",
    )
    parser.add_argument(
        "--family",
        dest="families",
        action="append",
        default=[],
        metavar="NAME",
        help="only general models of these families; repeat for each",
    )
    parser.add_argument(
        "--accept-licences",
        action="store_true",
        default=os.environ.get("MT_ACCEPT_LICENCES", "").strip().lower() in ("1", "true", "yes"),
        help="accept the licence of every general model the plan holds",
    )


def _engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sglang-image",
        default=os.environ.get("MT_SGLANG_IMAGE") or SGLANG_IMAGE,
        help="SGLang image for awq, fp8 and safetensors models; none to disable",
    )
    parser.add_argument(
        "--llama-image",
        default=os.environ.get("MT_LLAMA_IMAGE") or LLAMA_IMAGE,
        help="llama.cpp server image for gguf models; none to disable",
    )


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
    return plan.Policy(
        arenas=frozenset(args.arena),
        exclude=frozenset(args.excluded),
        kinds=frozenset() if args.kind == "any" else frozenset({args.kind}),
        include=frozenset(args.models),
        families=frozenset(f.strip().lower() for f in args.families),
        accept_licences=bool(args.accept_licences),
    )


def _image(value: str) -> str:
    found = str(value or "").strip()
    return "" if found.lower() == NO_IMAGE else found


def _images(args: argparse.Namespace) -> frozenset[str]:
    pairs = ((SGLANG, _image(args.sglang_image)), (LLAMA, _image(args.llama_image)))
    return frozenset(backend for backend, image in pairs if image)


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
        capacity = plan.Capacity(gpus=_gpus(args), gpu_gb=args.gpu_gb, images=_images(args))
    except plan.PlanError as exc:
        return fail(str(exc))
    offers = [plan.Offer.from_wire(row) for row in rows]
    found = plan.build(offers, capacity, policy=_policy(args))
    shown: dict[str, object] = found.to_dict()
    shown["hardware_class"] = capacity.hardware_class
    shown["licences"] = {
        offer.model: {"licence": offer.licence, "licence_url": offer.licence_url}
        for offer in offers
        if offer.general and offer.model in found.models
    }
    return _show(shown)


def _run(args: argparse.Namespace) -> int:
    if not args.auto and not args.serve and not args.models:
        return fail("pass --auto to let the catalogue choose, --serve MODEL@SHA, or --model NAME")
    if args.engine_url and len(args.models) != 1:
        return fail("--engine-url drives one running engine; name its model with one --model")
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
        sglang_image=_image(args.sglang_image),
        llama_image=_image(args.llama_image),
        mem_fraction=args.mem_fraction,
        hf_token=args.hf_token,
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
        accept_licences=args.accept_licences,
        engine_url=args.engine_url,
    )
    print(f"serving as {hotkey_address(wallet)} from {args.server}")
    try:
        asyncio.run(host.run())
    except KeyboardInterrupt:
        print("stopped")
    except HostError as exc:
        return fail(str(exc))
    return 0
