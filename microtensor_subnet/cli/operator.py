from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from microtensor_subnet.chain.wallet import hotkey_address, sign_bytes
from microtensor_subnet.cli import operator_config
from microtensor_subnet.cli.common import (
    add_chain_arguments,
    add_common_arguments,
    chain_config,
    default_home,
    fail,
    open_wallet,
)
from microtensor_subnet.core.constants import PUBLIC_SERVER_URL
from microtensor_subnet.serving import audit as serving_audit
from microtensor_subnet.serving import client, gpus, plan, update
from microtensor_subnet.serving.client import ServerError
from microtensor_subnet.serving.engines import LLAMA, LLAMA_IMAGE, SGLANG, SGLANG_IMAGE
from microtensor_subnet.serving.host import HostConfig, HostError, ServingHost, read_catalogue
from microtensor_subnet.serving.host.config import DEFAULT_GPU_GB, MEM_FRACTION, REVIEW_SECONDS
from microtensor_subnet.validator.sandbox.docker import Docker

log = logging.getLogger("microtensor_subnet.cli.operator")

GATEWAY_URL = "wss://api.microtensor.cloud/v1/operators/socket"
KINDS = ("system", "model", "any")
NO_IMAGE = "none"
GPU_COLUMNS = ("gpu", "name", "memory", "model", "status", "last seen")
CONFIG_DEFAULTS: dict[str, Any] = {
    "wallet_path": None,
    "wallet_name": None,
    "wallet_hotkey": None,
    "gateway": GATEWAY_URL,
    "gpus": gpus.ALL,
    "models": [],
    "auto": False,
    "max_concurrency": 0,
    "cache_dir": "",
}


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
        "--config",
        type=Path,
        default=None,
        help="an operator.yaml with hotkey_path, gateway, gpus, models and cache_dir",
    )
    serve.add_argument(
        "--max-concurrency",
        type=int,
        default=0,
        help="requests in flight per gpu; 0 keeps a general model's own, 8 for a certified image",
    )
    serve.add_argument("--cache-dir", default="", help="where fetched systems and models are kept")
    serve.add_argument(
        "--serve",
        action="append",
        default=[],
        metavar="MODEL@SHA",
        help="hold one pinned system; repeat for each",
    )
    serve.add_argument("--gateway", default=GATEWAY_URL)
    serve.add_argument("--worker", default="", help="a name when one hotkey runs several hosts")
    serve.add_argument("--sandbox-runtime", default=os.environ.get("MT_SANDBOX_RUNTIME", ""))
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
    serve.add_argument(
        "--shim-url", action="append", default=[], metavar="[GPU=]URL", help=argparse.SUPPRESS
    )
    serve.add_argument(
        "--no-auto-update",
        dest="auto_update",
        action="store_false",
        help="keep running this image when the network publishes a newer one",
    )
    serve.add_argument(
        "--compose-file",
        default=os.environ.get("MT_COMPOSE_FILE", "/opt/microtensor/compose.yml"),
        help="the compose file the updater restarts this service from",
    )
    serve.add_argument(
        "--env-file",
        default=os.environ.get("MT_ENV_FILE", "/opt/microtensor/.env"),
        help="the env file the updater pins the new image in",
    )
    _gpu_arguments(serve)
    _policy_arguments(serve)
    _engine_arguments(serve)
    _shared(serve)
    serve.set_defaults(handler=_run)


def _gpu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gpus",
        default=gpus.ALL,
        help="comma separated GPU indices, or all for every card nvidia-smi lists",
    )
    parser.add_argument(
        "--gpu-gb",
        type=float,
        default=0.0,
        help="memory per GPU when nvidia-smi cannot be asked (default 24 then)",
    )


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


def gpu_table(found: dict[str, Any]) -> str:
    rows = [list(GPU_COLUMNS)]
    for gpu in found.get("gpus") or []:
        memory = int(gpu.get("memory_mb") or 0)
        status = str(gpu.get("status") or ("online" if gpu.get("online") else "offline"))
        rows.append(
            [
                str(gpu.get("gpu") or gpu.get("label") or ""),
                str(gpu.get("name") or "unnamed"),
                f"{memory / 1024:.0f} GB" if memory else "",
                str(gpu.get("model") or "nothing loaded"),
                status,
                str(gpu.get("last_seen_at") or "never"),
            ]
        )
    if len(rows) == 1:
        return "no gpu has dialled in under this hotkey yet"
    widths = [max(len(row[i]) for row in rows) for i in range(len(GPU_COLUMNS))]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)


def _status(args: argparse.Namespace) -> int:
    wallet = _wallet(args)
    try:
        found = client.me(args.server, wallet)
    except ServerError as exc:
        return fail(str(exc))
    print(gpu_table(found))
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


def _cards(args: argparse.Namespace, worker: str = "") -> tuple[gpus.Card, ...]:
    found = gpus.choose(gpus.detect(), args.gpus)
    if found:
        return tuple(found)
    gpu_gb = float(args.gpu_gb) or DEFAULT_GPU_GB
    log.warning("nvidia-smi listed no card for --gpus %s; assuming %.0f GB each", args.gpus, gpu_gb)
    return tuple(gpus.stand_ins(args.gpus, gpu_gb, worker=worker))


def _gpu_gb(args: argparse.Namespace, cards: tuple[gpus.Card, ...]) -> float:
    return float(args.gpu_gb) or gpus.smallest_gb(cards) or DEFAULT_GPU_GB


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
    cards = _cards(args)
    try:
        capacity = plan.Capacity(
            gpus=tuple(card.device for card in cards),
            gpu_gb=_gpu_gb(args, cards),
            images=_images(args),
        )
    except plan.PlanError as exc:
        return fail(str(exc))
    offers = [plan.Offer.from_wire(row) for row in rows]
    found = plan.build(offers, capacity, policy=_policy(args))
    shown: dict[str, object] = found.to_dict()
    shown["hardware_class"] = capacity.hardware_class
    shown["gpus"] = [card.announced() for card in cards]
    shown["licences"] = {
        offer.model: {"licence": offer.licence, "licence_url": offer.licence_url}
        for offer in offers
        if offer.general and offer.model in found.models
    }
    return _show(shown)


def _configured(args: argparse.Namespace) -> list[str]:
    if args.config is None:
        return []
    defaults = dict(CONFIG_DEFAULTS, home=default_home())
    return operator_config.apply(args, operator_config.read(args.config), defaults)


def _run(args: argparse.Namespace) -> int:
    try:
        notes = _configured(args)
    except operator_config.ConfigError as exc:
        return fail(str(exc))
    for note in notes:
        print(note)
    if not args.auto and not args.serve and not args.models:
        return fail("pass --auto to let the catalogue choose, --serve MODEL@SHA, or --model NAME")
    if args.engine_url and len(args.models) != 1:
        return fail("--engine-url drives one running engine; name its model with one --model")
    wallet = _wallet(args)
    try:
        pinned = parse_pins(args.serve) if args.serve else None
    except HostError as exc:
        return fail(str(exc))
    cards = _cards(args, worker=args.worker)
    config = HostConfig(
        home=args.home,
        gpus=tuple(card.device for card in cards),
        gpu_gb=_gpu_gb(args, cards),
        sandbox_runtime=args.sandbox_runtime,
        request_seconds=args.request_seconds,
        review_seconds=args.review_seconds,
        sglang_image=_image(args.sglang_image),
        llama_image=_image(args.llama_image),
        mem_fraction=args.mem_fraction,
        hf_token=args.hf_token,
        cards=cards,
        cache=Path(args.cache_dir) if args.cache_dir else None,
        max_concurrency=max(0, int(args.max_concurrency or 0)),
    )

    def declare(model: str, digest: str) -> None:
        client.declare(args.server, wallet, model=model, artifact_digest=digest)

    def signer(message: bytes) -> str:
        return sign_bytes(wallet, message)

    def stats() -> list[dict[str, Any]]:
        return [card.reported() for card in gpus.refresh(cards)]

    shim_urls: dict[str, str] = {}
    for item in args.shim_url:
        device, sep, url = str(item).partition("=")
        if not sep:
            device, url = cards[0].device if cards else "0", device
        shim_urls[device] = url
    docker = Docker()
    updater = update.Updater(
        server=args.server,
        docker=docker,
        compose_file=Path(args.compose_file),
        env_file=Path(args.env_file),
        enabled=bool(args.auto_update),
    )
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
        signer=signer,
        stats=stats,
        docker=docker,
        updater=updater,
        shim_urls=shim_urls,
    )
    for card in cards:
        print(f"gpu {card.device}: {card.name or 'unnamed'} {card.memory_gb:.0f} GB ({card.id})")
    print(f"serving as {hotkey_address(wallet)} from {args.server}")
    try:
        asyncio.run(host.run())
    except KeyboardInterrupt:
        print("stopped")
    except HostError as exc:
        return fail(str(exc))
    return 0
