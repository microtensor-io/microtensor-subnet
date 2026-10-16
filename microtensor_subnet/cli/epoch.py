from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx

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
from microtensor_subnet.serving import engines, markers, probes
from microtensor_subnet.serving.client import ServerError as ServingError
from microtensor_subnet.serving.engines import LLAMA, SGLANG, EngineError
from microtensor_subnet.serving.host import GENERAL, read_catalogue
from microtensor_subnet.serving.host.config import ENGINE_READY_SECONDS, MEM_FRACTION
from microtensor_subnet.serving.host.fetch import FetchedModel, FetchError, fetch_model
from microtensor_subnet.serving.host.plain import PlainError, close_model
from microtensor_subnet.serving.tokens import TokenError
from microtensor_subnet.serving.verifier import (
    Calibrated,
    Reference,
    calibrate_model,
    close_reference,
    generate,
    load_calibration,
    open_engine,
    open_reference,
    save_calibration,
    stamped,
)
from microtensor_subnet.serving.verify import UNPROVEN, VerificationError
from microtensor_subnet.validator.client import ServerClient, ServerError
from microtensor_subnet.validator.config import EpochConfig
from microtensor_subnet.validator.epoch import EpochError, run_epoch
from microtensor_subnet.validator.items import ReleaseError
from microtensor_subnet.validator.sandbox.docker import Docker, DockerError

log = logging.getLogger("microtensor_subnet.cli.epoch")
KEY_BLOCKS = 2_600_000
PUBLIC_GATEWAY_URL = "https://api.microtensor.cloud"
SETUP_ERRORS = (
    VerificationError,
    FetchError,
    PlainError,
    EngineError,
    TokenError,
    DockerError,
    markers.MarkerError,
    httpx.HTTPError,
    OSError,
)


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

    probe = inner.add_parser(
        "probe", help="probe inference operators with withheld items and general models"
    )
    probe.add_argument("--arena", default="")
    probe.add_argument("--server", default=PUBLIC_SERVER_URL)
    probe.add_argument("--gateway-url", default=PUBLIC_GATEWAY_URL)
    probe.add_argument("--credential", default=os.environ.get("MT_SERVE_CREDENTIAL", ""))
    probe.add_argument("--floor", type=float, default=0.9)
    probe.add_argument("--per-operator", type=int, default=8)
    probe.add_argument("--pause", type=float, default=30.0)
    probe.add_argument("--seed", type=int, default=0)
    probe.add_argument("--once", action="store_true")
    probe.add_argument(
        "--general",
        action="append",
        default=[],
        metavar="MODEL",
        help="a general model to probe and verify by prefill; repeatable",
    )
    probe.add_argument(
        "--verify-backend",
        choices=(SGLANG, LLAMA),
        default=SGLANG,
        help="what prefills a general model's answers: an sglang engine or llama.cpp in process",
    )
    probe.add_argument(
        "--engine-url", default="", help="a running engine that holds the one --general model"
    )
    probe.add_argument(
        "--gpu",
        default="",
        help="GPUs for reference engines, comma separated; without it llama.cpp uses the CPU",
    )
    probe.add_argument(
        "--calibrate",
        action="store_true",
        help="calibrate the one --general model against --substitute, store it and exit",
    )
    probe.add_argument(
        "--substitute",
        default="",
        metavar="MODEL",
        help="the catalogue model whose answers calibration must tell apart",
    )
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


def _images() -> dict[str, str]:
    return {
        SGLANG: os.environ.get("MT_SGLANG_IMAGE") or engines.SGLANG_IMAGE,
        LLAMA: os.environ.get("MT_LLAMA_IMAGE") or engines.LLAMA_IMAGE,
    }


def _general_entries(server: str) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("model") or row.get("id") or ""): row
        for row in read_catalogue(server)
        if str(row.get("kind") or "") == GENERAL
    }


def _entry(catalogue: dict[str, dict[str, Any]], model: str) -> dict[str, Any]:
    entry = catalogue.get(model)
    if entry is None:
        raise VerificationError(f"{model} is not a general model in the catalogue")
    return entry


def _fetch(args: argparse.Namespace, entry: dict[str, Any]) -> FetchedModel:
    return fetch_model(entry, Path(args.home) / "cache", token=os.environ.get("HF_TOKEN", ""))


def _gpus(args: argparse.Namespace) -> list[str]:
    return [gpu.strip() for gpu in str(args.gpu or "").split(",") if gpu.strip()] or ["0"]


def _reference(
    args: argparse.Namespace,
    docker: Docker,
    entry: dict[str, Any],
    fetched: FetchedModel,
    gpu: str,
) -> Reference:
    if args.verify_backend == LLAMA and args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return open_reference(
        docker,
        fetched,
        entry,
        backend=args.verify_backend,
        gpu=gpu,
        images=_images(),
        mem_fraction=MEM_FRACTION,
        ready_seconds=ENGINE_READY_SECONDS,
        engine_url=args.engine_url,
        gpu_layers=engines.GPU_LAYERS if args.gpu else 0,
    )


def _targets(
    args: argparse.Namespace, docker: Docker, references: list[Reference]
) -> dict[str, probes.Target]:
    if not args.general:
        return {}
    catalogue = _general_entries(args.server)
    found: dict[str, probes.Target] = {}
    gpus = _gpus(args)
    for model in args.general:
        entry = _entry(catalogue, model)
        items = tuple(probes.items_for(entry))
        digest = str(entry.get("artifact_digest") or "")
        if not probes.verifiable(entry):
            log.info("%s: no validator class can prefill it; outcome and format only", model)
            found[model] = probes.Target(model, digest, items)
            continue
        calibrated = load_calibration(args.home, model, digest)
        if calibrated is None:
            raise VerificationError(
                f"no calibration for {model} at {digest[:12]}; run "
                f"mt epoch probe --general {model} --calibrate --substitute MODEL first"
            )
        gpu = gpus[len(references) % len(gpus)]
        reference = _reference(args, docker, entry, _fetch(args, entry), gpu)
        references.append(reference)
        found[model] = probes.Target(
            model, digest, items, reference.verifier, calibrated.calibration
        )
    return found


async def _calibration(
    args: argparse.Namespace,
    docker: Docker,
    entry: dict[str, Any],
    substitute: dict[str, Any],
) -> Calibrated:
    requests = [item.request() for item in probes.items_for(entry)]
    other = await asyncio.to_thread(_fetch, args, substitute)
    held = await asyncio.to_thread(
        open_engine,
        docker,
        other,
        substitute,
        gpu=_gpus(args)[0],
        images=_images(),
        mem_fraction=MEM_FRACTION,
        ready_seconds=ENGINE_READY_SECONDS,
    )
    try:
        substituted = await generate(held, requests)
    finally:
        close_model(docker, held)
    fetched = await asyncio.to_thread(_fetch, args, entry)
    reference = await asyncio.to_thread(_reference, args, docker, entry, fetched, _gpus(args)[0])
    try:
        if reference.engine is not None:
            honest = await generate(reference.engine, requests)
        else:
            engine = await asyncio.to_thread(
                open_engine,
                docker,
                fetched,
                entry,
                gpu=_gpus(args)[0],
                images=_images(),
                mem_fraction=MEM_FRACTION,
                ready_seconds=ENGINE_READY_SECONDS,
                engine_url=args.engine_url,
            )
            try:
                honest = await generate(engine, requests)
            finally:
                if not args.engine_url:
                    close_model(docker, engine)
        calibration = await calibrate_model(
            reference.verifier,
            list(zip(requests, honest, strict=True)),
            list(zip(requests, substituted, strict=True)),
        )
    finally:
        close_reference(docker, reference)
    return stamped(reference, calibration, substitute=args.substitute)


def _calibrate(args: argparse.Namespace) -> int:
    docker = Docker()
    try:
        catalogue = _general_entries(args.server)
        entry = _entry(catalogue, args.general[0])
        substitute = _entry(catalogue, args.substitute)
        found = asyncio.run(_calibration(args, docker, entry, substitute))
        path = save_calibration(args.home, found)
    except SETUP_ERRORS as exc:
        return fail(str(exc))
    calibration = found.calibration
    print(f"model       {found.model}  {found.artifact_digest[:12]}")
    print(f"backend     {found.backend}")
    print(f"honest      {calibration.honest_samples} samples")
    print(f"threshold   {calibration.threshold:.6g} at alpha {calibration.alpha}")
    print(f"substitute  {found.substitute} lowest {calibration.substituted_min:.6g}")
    print(f"separated   {'yes' if calibration.separated else 'no; verdicts stay unproven'}")
    print(f"stored      {path}")
    return 0


def _show(found: list[probes.Probe]) -> None:
    for item in found:
        state = item.verdict or ("ok" if item.success else ("fail" if item.reported else "skip"))
        print(f"{state:8s} {item.operator[:12]} {item.model} {item.item_id} {item.reason}")
    reported = [p for p in found if p.reported]
    failed = sum(1 for p in reported if p.success is False)
    unproven = sum(1 for p in reported if p.verdict == UNPROVEN)
    print(f"probed {len(reported)} answers, {failed} failed, {unproven} unproven")


def _probe(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO)
    if not args.arena and not args.general:
        return fail("name an arena with --arena or a general model with --general")
    if args.engine_url and len(args.general) != 1:
        return fail("--engine-url drives one running engine; name its model with one --general")
    if args.calibrate:
        if len(args.general) != 1 or not args.substitute:
            return fail("--calibrate takes one --general model and the --substitute it must catch")
        return _calibrate(args)
    if not args.credential:
        return fail("set MT_SERVE_CREDENTIAL or pass --credential")
    config = chain_config(args)
    wallet = open_wallet(config)
    work_root = Path(args.home) / "work" / args.arena
    rng = random.Random(args.seed or None)  # noqa: S311
    docker = Docker()
    references: list[Reference] = []
    try:
        targets = _targets(args, docker, references)
    except SETUP_ERRORS as exc:
        for reference in references:
            close_reference(docker, reference)
        return fail(str(exc))
    try:
        while True:
            sets = probes.latest(work_root, args.arena) if args.arena else {}
            if args.arena and not sets:
                log.warning("no probe sets under %s yet; run an epoch first", work_root)
            if sets or targets:
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
                        general=targets,
                    )
                except ServingError as exc:
                    log.error("probe cycle: %s", exc)
                    found = []
                _show(found)
            if args.once:
                return 0
            time.sleep(max(30.0, float(args.pause)))
    finally:
        for reference in references:
            close_reference(docker, reference)


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
