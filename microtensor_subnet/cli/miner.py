from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from microtensor_subnet.chain.client import ChainError
from microtensor_subnet.chain.wallet import hotkey_address
from microtensor_subnet.cli.common import (
    add_chain_arguments,
    add_common_arguments,
    chain_config,
    fail,
    open_client,
    open_wallet,
)
from microtensor_subnet.core.constants import PUBLIC_SERVER_URL
from microtensor_subnet.validator import runner
from microtensor_subnet.validator.client import ServerClient, ServerError
from microtensor_subnet.validator.sandbox.docker import DockerError

log = logging.getLogger("microtensor_subnet.cli.miner")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("miner", help="run and replay your own evaluation")
    inner = parser.add_subparsers(dest="action", required=True)

    run = inner.add_parser("run", help="evaluate your submission on this GPU and upload")
    _add_run_arguments(run)
    run.add_argument("--dry-run", action="store_true", help="write the transcript, skip upload")
    run.add_argument("--item-limit", type=int, default=0, help="development only")
    run.add_argument(
        "--commitment",
        default="",
        help="the on-chain commitment string; read from the chain when omitted",
    )
    run.set_defaults(handler=_run)

    replay = inner.add_parser("replay", help="replay a transcript on the CPU like a validator")
    _add_run_arguments(replay)
    replay.add_argument("--transcript", type=Path, required=True)
    replay.add_argument("--artifact", type=Path, default=None, help="artifact root to replay")
    replay.set_defaults(handler=_replay)


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arena", required=True)
    parser.add_argument("--epoch", type=int, default=None, help="defaults to the open epoch")
    parser.add_argument("--server", default=os.environ.get("MT_SERVER_URL", PUBLIC_SERVER_URL))
    parser.add_argument("--relay", default=os.environ.get("MT_RELAY_URL", ""))
    parser.add_argument(
        "--gpus", default=os.environ.get("MT_GPUS", "0"), help="comma separated indices"
    )
    parser.add_argument("--sandbox-runtime", default=os.environ.get("MT_SANDBOX_RUNTIME", ""))
    parser.add_argument(
        "--slm-memory-share",
        type=float,
        default=float(os.environ.get("MT_SLM_MEMORY_SHARE", "0") or 0),
        help="engine memory share; 0 derives it from the arena budget",
    )
    parser.add_argument("--services-port", type=int, default=9702)
    add_chain_arguments(parser)
    add_common_arguments(parser)


def _show_logs() -> None:
    package = logging.getLogger("microtensor_subnet")
    package.setLevel(logging.INFO)
    package.disabled = False
    for name, found in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(found, logging.Logger) and name.startswith("microtensor_subnet."):
            found.disabled = False
            found.setLevel(logging.NOTSET)


def _epoch_index(server: ServerClient, arena_id: str, wanted: int | None) -> int:
    if wanted is not None:
        return int(wanted)
    arena = server.arena(arena_id)
    current = dict((arena or {}).get("current_epoch") or {})
    if current.get("index") is None:
        raise runner.RunnerError(f"{arena_id} has no current epoch; pass --epoch")
    return int(current["index"])


def _settings(
    args: argparse.Namespace, epoch_index: int, commitment: str = ""
) -> runner.RunnerConfig:
    return runner.RunnerConfig(
        home=Path(args.home),
        arena_id=args.arena,
        epoch_index=epoch_index,
        server_url=args.server,
        relay_url=args.relay or args.server,
        gpus=tuple(g.strip() for g in str(args.gpus).split(",") if g.strip()) or ("0",),
        sandbox_runtime=args.sandbox_runtime,
        slm_memory_share=float(args.slm_memory_share or 0.0),
        host_services_port=int(args.services_port),
        dry_run=bool(getattr(args, "dry_run", False)),
        item_limit=int(getattr(args, "item_limit", 0) or 0),
        runner_digest=os.environ.get(runner.ENV_RUNNER_DIGEST, "").strip(),
        commitment=commitment,
    )


def _commitment(args: argparse.Namespace, config: Any, wallet: Any, hotkey: str) -> str:
    if args.commitment:
        return str(args.commitment)
    saved = Path(args.home) / "reveals" / f"{args.arena}.json"
    try:
        client = open_client(config, wallet)
        found = client.commitments([hotkey]).get(hotkey, "")
    except (ChainError, Exception) as exc:
        log.warning("the chain commitment could not be read: %s", exc)
        found = ""
    if found:
        return str(found)
    if saved.is_file():
        record = json.loads(saved.read_text(encoding="utf-8"))
        if record.get("hotkey") == hotkey and record.get("commitment"):
            return str(record["commitment"])
    raise runner.RunnerError(
        "no commitment found on chain for this hotkey; pass --commitment with the string "
        "mt submit commit printed"
    )


def _run(args: argparse.Namespace) -> int:
    config = chain_config(args)
    wallet = open_wallet(config)
    hotkey = hotkey_address(wallet)
    _show_logs()
    server = ServerClient(args.server, hotkey, wallet)
    try:
        epoch_index = _epoch_index(server, args.arena, args.epoch)
        commitment = _commitment(args, config, wallet, hotkey)
        settings = _settings(args, epoch_index, commitment)
        upload = runner.run(settings, client=server, wallet=wallet)
    except (runner.RunnerError, ServerError, DockerError) as exc:
        return fail(str(exc))
    print(f"epoch       {epoch_index}")
    print(f"items       {upload.items}")
    print(f"transcript  {upload.transcript_sha256}")
    print(f"saved       {upload.path}")
    if upload.receipt:
        print(f"received    {upload.receipt.get('received_at', '')}")
        print(f"receipt     {upload.receipt.get('receipt_sig', '')}")
    else:
        print("upload      skipped (dry run)")
    return 0


def _replay(args: argparse.Namespace) -> int:
    config = chain_config(args)
    wallet = open_wallet(config, required=False)
    hotkey = hotkey_address(wallet) if wallet is not None else ""
    _show_logs()
    server = ServerClient(args.server, hotkey, wallet)
    try:
        epoch_index = _epoch_index(server, args.arena, args.epoch)
        recorded = runner.load_transcript(Path(args.transcript))
        outcome = runner.replay(
            _settings(args, epoch_index),
            client=server,
            recorded=recorded,
            artifact_dir=args.artifact,
        )
    except (runner.RunnerError, ServerError, DockerError, OSError) as exc:
        return fail(str(exc))
    print(f"epoch       {epoch_index}")
    print(f"items       {outcome.items}")
    print(f"calls       {outcome.calls}")
    if outcome.ok:
        print("replay      ok")
        return 0
    print(f"replay      failed: {outcome.reason}")
    return 1
