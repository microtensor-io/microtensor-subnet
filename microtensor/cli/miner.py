from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from microtensor.chain.client import ChainError
from microtensor.chain.rounds import (
    release_cutoff_block,
    release_index,
    release_version,
    rounds_until_release,
)
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
    COORDINATOR_URL,
    CORPUS_VERSION,
    GENESIS_BLOCK,
    PROVENANCE_REQUIRED,
    PUBLIC_SERVER_URL,
    ROUND_BLOCKS,
)
from microtensor.core.protocol import ArtifactFormat, DeclaredEnvelope, LoadManifest
from microtensor.core.system import SystemManifest
from microtensor.miner import fee as fee_client
from microtensor.miner import provenance
from microtensor.miner.config import MinerConfig, MinerConfigError
from microtensor.miner.package import (
    PackageError,
    load_packaged,
    package,
    publishable_files,
    upload_checklist,
)
from microtensor.miner.provenance import ProvenanceMissing
from microtensor.miner.publish import PublishError, PublishLoop, current_round, publish
from microtensor.miner.selfcheck import SelfCheckError, selfcheck
from microtensor.miner.standing import fetch as fetch_standing
from microtensor.miner.upload import UploadError, plan_upload, upload
from microtensor.provenance.record import ProvenanceUnavailable, RunStore

log = logging.getLogger("microtensor.cli.miner")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("miner", help="package, check and publish a submission")
    inner = parser.add_subparsers(dest="action", required=True)

    init = inner.add_parser("init", help="save your competition settings once")
    _add_settings_arguments(init, required=True)
    init.set_defaults(handler=_init)

    check = inner.add_parser(
        "selfcheck", aliases=["check"], help="measure your own envelope before declaring it"
    )
    _add_settings_arguments(check)
    check.add_argument("--profile-seconds", type=_at_least_one_second, default=60)
    check.set_defaults(handler=_selfcheck)

    sim = inner.add_parser("simulate", help="run the cascade over the public training split")
    _add_settings_arguments(sim)
    sim.add_argument("--corpus", type=Path, required=True, help="corpus directory")
    sim.add_argument("--limit", type=int, default=0, help="stop after this many tasks")
    sim.set_defaults(handler=_simulate)

    pack = inner.add_parser("package", help="digest, sign and write manifest.json")
    _add_settings_arguments(pack)
    pack.add_argument("--round", type=int, help="round to target; defaults to the open round")
    pack.add_argument("--size-bytes", type=int)
    pack.add_argument("--peak-rss-bytes", type=int)
    pack.add_argument("--p95-latency-ms", type=int)
    pack.add_argument(
        "--declare",
        action="store_true",
        help="declare what selfcheck measured instead of explicit ceilings",
    )
    pack.add_argument(
        "--sealed",
        action="store_true",
        help="encrypt the artifact; publish only ciphertext, reveal the key at close",
    )
    pack.set_defaults(handler=_package)

    up = inner.add_parser("upload", help="push the packaged artifact to your source")
    _add_settings_arguments(up)
    up.set_defaults(handler=_upload)

    push = inner.add_parser("publish", help="commit the pointer on chain for one round")
    _add_settings_arguments(push)
    push.add_argument("--upload", action="store_true", help="upload before committing")
    _add_fee_arguments(push)
    push.add_argument(
        "--recommit",
        action="store_true",
        help="restore a pointer a reveal overwrote, after the window closed",
    )
    push.set_defaults(handler=_publish)

    ship = inner.add_parser("ship", help="package, upload and publish in one step")
    _add_settings_arguments(ship)
    ship.add_argument("--profile-seconds", type=_at_least_one_second, default=60)
    ship.add_argument("--no-selfcheck", action="store_true")
    ship.add_argument("--size-bytes", type=int)
    ship.add_argument("--peak-rss-bytes", type=int)
    ship.add_argument("--p95-latency-ms", type=int)
    ship.add_argument(
        "--sealed",
        action="store_true",
        help="encrypt the artifact; publish only ciphertext, reveal the key at close",
    )
    _add_fee_arguments(ship)
    ship.add_argument(
        "--pay-fee",
        action="store_true",
        help="transfer the submission fee from your coldkey before committing",
    )
    ship.set_defaults(handler=_ship)

    unveil = inner.add_parser("reveal", help="post the key for a sealed submission")
    _add_settings_arguments(unveil)
    unveil.add_argument("--round", type=int, help="round to reveal; defaults to the packaged round")
    unveil.set_defaults(handler=_reveal)

    serve = inner.add_parser("run", help="re-commit automatically every round")
    _add_settings_arguments(serve)
    serve.add_argument("--max-rounds", type=int)
    _add_fee_arguments(serve)
    serve.set_defaults(handler=_run)

    fee = inner.add_parser("fee", help="the per submission fee: quote it, pay it, report it")
    fee_inner = fee.add_subparsers(dest="fee_command", required=True)
    for name, helptext, handler in (
        ("quote", "print what a submission costs and where it is paid", _fee_quote),
        ("status", "whether the packaged artifact's fee is paid", _fee_status),
        ("pay", "transfer the fee from your coldkey and report it", _fee_pay),
        ("report", "report a transfer you already made", _fee_report),
    ):
        sub = fee_inner.add_parser(name, help=helptext)
        _add_settings_arguments(sub)
        sub.add_argument("--server", default=PUBLIC_SERVER_URL, help="public API to talk to")
        sub.set_defaults(handler=handler)
    fee_inner.choices["pay"].add_argument(
        "--yes", action="store_true", help="send without asking first"
    )
    fee_inner.choices["report"].add_argument(
        "--extrinsic", required=True, help="hash of the transfer extrinsic"
    )
    fee_inner.choices["report"].add_argument(
        "--block", type=int, required=True, help="block the transfer landed in"
    )

    serve = inner.add_parser(
        "serve", help="train and submit unattended, reporting progress each epoch"
    )
    _add_settings_arguments(serve)
    serve.add_argument(
        "--coordinator", default=COORDINATOR_URL, help="where to report training progress"
    )
    serve.add_argument(
        "--train",
        dest="train_entrypoint",
        help="module:function that runs training and calls the hook, for example train:run",
    )
    serve.add_argument("--epochs", type=int, help="expected epoch count, used for the estimate")
    serve.add_argument(
        "--no-telemetry", action="store_true", help="submit unattended but report nothing"
    )
    serve.set_defaults(handler=_serve)

    status = inner.add_parser("status", help="show what this miner would publish")
    _add_settings_arguments(status)
    status.add_argument(
        "--server", default=PUBLIC_SERVER_URL, help="public API to read standing from"
    )
    status.add_argument(
        "--offline", action="store_true", help="print the local manifest without asking the API"
    )
    status.set_defaults(handler=_status)

    prov = inner.add_parser(
        "provenance", help="check your training run resolves and binds to the artifact"
    )
    _add_settings_arguments(prov)
    prov.set_defaults(handler=_provenance)


def _at_least_one_second(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number of seconds") from exc
    if seconds < 1:
        raise argparse.ArgumentTypeError("profiling needs at least one second")
    return seconds


def _add_settings_arguments(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    add_chain_arguments(parser)
    add_common_arguments(parser)
    parser.add_argument("--artifact", type=Path, required=required, help="artifact directory")
    parser.add_argument("--track", required=required)
    parser.add_argument("--hardware-class", "--class", dest="hardware_class", required=required)
    parser.add_argument("--source", required=required, help="scheme:locator validators fetch from")
    parser.add_argument("--entrypoint")
    parser.add_argument("--format", dest="artifact_format")
    parser.add_argument("--quantization")
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--tokenizer")
    parser.add_argument("--base-model")
    parser.add_argument("--round-blocks", type=int)
    parser.add_argument("--genesis-block", type=int)
    parser.add_argument("--allow-unsandboxed", action="store_true", default=None)


def _settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "artifact_dir": args.artifact,
        "track": args.track,
        "hardware_class": args.hardware_class,
        "source": args.source,
        "entrypoint": args.entrypoint,
        "artifact_format": args.artifact_format,
        "quantization": args.quantization,
        "max_input_tokens": args.max_input_tokens,
        "tokenizer": args.tokenizer,
        "base_model": args.base_model,
        "round_blocks": args.round_blocks,
        "genesis_block": args.genesis_block,
        "allow_unsandboxed": args.allow_unsandboxed,
    }


def _wallet_from_saved(args: argparse.Namespace, home: Path) -> None:
    saved = home / "miner.json"
    if not saved.is_file():
        return
    try:
        stored = json.loads(saved.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for attr, env in (("wallet_name", "MT_WALLET_NAME"), ("wallet_hotkey", "MT_WALLET_HOTKEY")):
        if getattr(args, attr, None) is None and not os.environ.get(env) and stored.get(attr):
            setattr(args, attr, stored[attr])


def _config(args: argparse.Namespace) -> MinerConfig:
    home = Path(args.home)
    _wallet_from_saved(args, home)
    chain = chain_config(args)
    settings = _settings(args)

    if (home / "miner.json").is_file():
        return MinerConfig.load(home, chain, **settings)

    defaults: dict[str, object] = {
        "entrypoint": "model.onnx",
        "artifact_format": ArtifactFormat.ONNX.value,
        "quantization": "",
        "max_input_tokens": 4096,
        "tokenizer": "tokenizer.json",
        "base_model": "",
        "round_blocks": ROUND_BLOCKS,
        "genesis_block": GENESIS_BLOCK,
        "allow_unsandboxed": False,
    }
    for key, value in defaults.items():
        if settings.get(key) is None:
            settings[key] = value
    return MinerConfig.build(home, chain, **settings)


def _warn_weak_source(config: MinerConfig) -> None:
    if config.scheme == "https":
        log.warning(
            "https source: the digest is your only binding and the host can vanish; "
            "prefer hf:<org>/<repo>@<commit-sha> or ipfs:"
        )


def _load_manifest_spec(config: MinerConfig) -> LoadManifest:
    return LoadManifest(
        format=ArtifactFormat(config.artifact_format),
        quantization=config.quantization,
        entrypoint=config.entrypoint,
        max_input={"tokens": config.max_input_tokens},
        preprocessing={"tokenizer": config.tokenizer},
        base_model=config.base_model,
    )


def _init(args: argparse.Namespace) -> int:
    saved = (Path(args.home) / "miner.json").is_file()
    missing = [
        flag
        for flag, value in (
            ("--artifact", args.artifact),
            ("--track", args.track),
            ("--hardware-class", args.hardware_class),
            ("--source", args.source),
        )
        if value is None
    ]
    if missing and not saved:
        return fail(
            "mt miner init needs " + " ".join(missing) + " the first time; nothing was saved"
        )
    if missing:
        print("keeping the saved values for " + ", ".join(missing))
    try:
        config = _config(args)
        path = config.save()
    except MinerConfigError as exc:
        return fail(str(exc))

    print(f"saved {path}\n")
    print(f"competition  {config.track}/{config.hardware_class}")
    print(f"artifact     {config.artifact_dir}")
    print(f"source       {config.source}")
    print(f"netuid       {config.chain.netuid}")
    print("\nevery `mt miner` command now runs with no flags:")
    print("  mt miner selfcheck")
    print("  mt miner ship")
    return 0


def _selfcheck(args: argparse.Namespace) -> int:
    try:
        config = _config(args)
        result = selfcheck(
            config.artifact_dir,
            _load_manifest_spec(config),
            config.hardware_class,
            profile_seconds=args.profile_seconds,
            allow_unsandboxed=config.allow_unsandboxed,
        )
    except (MinerConfigError, SelfCheckError) as exc:
        return fail(str(exc))

    print(result.report())
    config.home.mkdir(parents=True, exist_ok=True)
    config.selfcheck_path.write_text(
        json.dumps(
            {
                "size_bytes": result.proposed.size_bytes,
                "peak_rss_bytes": result.proposed.peak_rss_bytes,
                "p95_latency_ms": result.proposed.p95_latency_ms,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved to {config.selfcheck_path}")
    return 0 if result.admissible else 2


def _simulate(args: argparse.Namespace) -> int:
    from microtensor.envelope.certify import environment_root
    from microtensor.miner.simulate import SimulationError, simulate
    from microtensor.scoring import execution
    from microtensor.tasks.corpus import load_all

    try:
        config = _config(args)
        execution.configure(
            allow_unsandboxed=config.allow_unsandboxed,
            env_root=environment_root(config.home),
        )
        corpora = load_all(args.corpus, CORPUS_VERSION)
        corpus = corpora.get(config.track)
        if corpus is None:
            return fail(f"no corpus for {config.track} under {args.corpus}")

        system = _system_for(config)
        result = simulate(
            config.artifact_dir,
            _load_manifest_spec(config),
            system,
            corpus.fixed,
            config.hardware_class,
            metric=corpus.metric,
            track=config.track,
            limit=args.limit,
        )
    except (MinerConfigError, SimulationError) as exc:
        return fail(str(exc))

    print(result.report())
    return 0


def _system_for(config: MinerConfig) -> SystemManifest:
    path = config.artifact_dir / "system.json"
    if not path.is_file():
        return SystemManifest.single("sha256:local", config.hardware_class)
    return SystemManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _declared(args: argparse.Namespace, config: MinerConfig) -> DeclaredEnvelope:
    explicit = (args.size_bytes, args.peak_rss_bytes, args.p95_latency_ms)
    if all(v is not None for v in explicit):
        return DeclaredEnvelope(
            size_bytes=args.size_bytes,
            peak_rss_bytes=args.peak_rss_bytes,
            p95_latency_ms=args.p95_latency_ms,
        )
    if any(v is not None for v in explicit):
        raise PackageError(
            "declare all three of --size-bytes, --peak-rss-bytes and --p95-latency-ms, "
            "or none and let selfcheck supply them"
        )

    path = config.selfcheck_path
    if not path.is_file():
        raise PackageError(f"no selfcheck at {path}; run `mt miner selfcheck` first")
    return DeclaredEnvelope(**json.loads(path.read_text(encoding="utf-8")))


def _do_package(args: argparse.Namespace, config: MinerConfig, round_index: int | None = None):  # type: ignore[no-untyped-def]
    wallet = open_wallet(config.chain)
    client = open_client(config.chain, wallet)
    index = round_index if round_index is not None else current_round(config, client).index
    manifest = package(
        config,
        index,
        _load_manifest_spec(config),
        _declared(args, config),
        wallet,
        seal=bool(getattr(args, "sealed", False)),
    )
    return manifest, client


def _package(args: argparse.Namespace) -> int:
    try:
        config = _config(args)
        _warn_weak_source(config)
        manifest, client = _do_package(args, config, args.round)
    except (MinerConfigError, PackageError, ValueError) as exc:
        return fail(str(exc))

    head = client.block()
    client.close()
    print(f"manifest    {config.manifest_path}")
    print(f"round       {manifest.round_index}")
    print(f"digest      {manifest.digest()}")
    print(f"files       {len(manifest.files)}  ({manifest.total_bytes / 1024**3:.2f} GiB)")

    if PROVENANCE_REQUIRED:
        print(f"\nartifact digest  {manifest.artifact_digest}")
        print("log this to your training run before shipping:\n")
        for line in provenance.digest_snippet(manifest.artifact_digest, head).splitlines():
            print(f"  {line}")

    print("\nnext:  mt miner upload   (or `mt miner publish --upload`)")
    for target in upload_checklist(config, manifest):
        print(f"  {target}")
    return 0


def _store() -> RunStore:
    from microtensor.provenance.wandb_store import WandbStore

    return WandbStore()


def _require_provenance(
    config: MinerConfig, hotkey: str, artifact_digest: str, commit_block: int = 0
) -> None:
    if not PROVENANCE_REQUIRED:
        return
    provenance.require(_store(), config, hotkey, artifact_digest, commit_block=commit_block)


def _provenance(args: argparse.Namespace) -> int:
    try:
        config = _config(args)
        manifest = load_packaged(config)
        wallet = open_wallet(config.chain)
        client = open_client(config.chain, wallet)
        head = client.block()
        client.close()
        report = provenance.verify(
            _store(),
            config,
            hotkey_address(wallet),
            manifest.artifact_digest,
            commit_block=head,
        )
    except ProvenanceUnavailable as exc:
        # Told apart from "your run is missing" on purpose: one is the
        # operator's outage and the other is the miner's to fix, and a miner
        # who cannot tell them apart will hunt for a mistake they did not make.
        print("run store   UNREACHABLE")
        print(f"reason      {exc}")
        print()
        print("This is not your submission. The store the network reads is not")
        print("answering, so nobody can be checked right now. Report it to the")
        print("operator and retry later.")
        return 3
    except (MinerConfigError, PackageError) as exc:
        return fail(str(exc))

    print(report.render())
    return 0 if report.verdict.admissible else 2


def _do_upload(config: MinerConfig) -> str:
    manifest = load_packaged(config)
    plan = plan_upload(
        config.artifact_dir, config.scheme, config.locator, publishable_files(manifest)
    )
    locator = upload(plan, config.artifact_dir)
    print(f"uploaded {len(plan.files)} files ({plan.total_bytes / 1024**3:.2f} GiB)")
    pinned = f"{config.scheme}:{locator}"
    if pinned != config.source:
        config.with_overrides(source=pinned).save()
        print(f"pinned      {pinned}")
    return pinned


def _upload(args: argparse.Namespace) -> int:
    try:
        _do_upload(_config(args))
        return 0
    except (MinerConfigError, PackageError, UploadError) as exc:
        return fail(str(exc))


def _add_fee_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--server", default=PUBLIC_SERVER_URL, help="public API to talk to")
    parser.add_argument(
        "--skip-fee-check",
        action="store_true",
        help="commit even when the submission fee is unpaid",
    )


def _fee_gate(args: argparse.Namespace, hotkey: str, manifest_digest: str) -> str:
    if getattr(args, "skip_fee_check", False):
        return ""
    return fee_client.gate(str(getattr(args, "server", PUBLIC_SERVER_URL)), hotkey, manifest_digest)


def _print_fee_status(manifest, status: fee_client.FeeStatus) -> None:  # type: ignore[no-untyped-def]
    print(f"artifact    {fee_client.short(manifest.digest())}  (round {manifest.round_index})")
    state = "paid" if status.paid else status.state
    print(f"fee         {state}{': ' + status.reason if status.reason else ''}")
    if status.payment_reference:
        print(f"transfer    {status.payment_reference} in block {status.payment_block}")
    if not status.required:
        print("required    no; nothing to pay")


def _pay_fee(  # type: ignore[no-untyped-def]
    args: argparse.Namespace, config: MinerConfig, wallet, manifest
) -> fee_client.FeeStatus:
    """Transfer the fee and report it, printing the hash before the report so a
    failed report never loses the payment."""
    server = str(getattr(args, "server", PUBLIC_SERVER_URL))
    policy = fee_client.fetch_policy(server)
    hotkey = hotkey_address(wallet)
    if not policy.enabled:
        return fee_client.FeeStatus(state="not charged", paid=True, required=False)
    current = fee_client.fetch_status(server, hotkey, manifest.digest())
    if current.paid:
        return current

    print(f"artifact    {fee_client.short(manifest.digest())}  (round {manifest.round_index})")
    print(f"fee         {policy.fee_tao:g} TAO per submission, not refundable")
    print(f"pay to      {policy.pay_to}")
    if not getattr(args, "yes", False):
        answer = input("send this transfer from your coldkey? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            raise fee_client.FeeError("not sent")

    client = open_client(config.chain, wallet)
    receipt = client.transfer(policy.pay_to, policy.fee_tao)
    print(f"sent        {receipt.extrinsic_hash} in block {receipt.block}")
    try:
        return fee_client.report(
            server,
            wallet,
            manifest_digest=manifest.digest(),
            round_index=manifest.round_index,
            extrinsic_hash=receipt.extrinsic_hash,
            block=receipt.block,
        )
    except fee_client.FeeError as exc:
        raise fee_client.FeeError(
            f"the transfer went through but could not be reported ({exc}); run "
            f"`mt miner fee report --extrinsic {receipt.extrinsic_hash} --block {receipt.block}`"
        ) from exc


def _fee_context(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    config = _config(args)
    wallet = open_wallet(config.chain)
    manifest = load_packaged(config)
    return config, wallet, manifest


def _fee_quote(args: argparse.Namespace) -> int:
    try:
        policy = fee_client.fetch_policy(args.server)
    except fee_client.FeeError as exc:
        return fail(str(exc))
    if not policy.enabled:
        print("no submission fee is charged right now")
        return 0
    print(f"fee         {policy.fee_tao:g} TAO per submission, not refundable")
    print(f"pay to      {policy.pay_to}")
    print("pay with    mt miner fee pay   (after mt miner package)")
    return 0


def _fee_status(args: argparse.Namespace) -> int:
    try:
        _, wallet, manifest = _fee_context(args)
        status = fee_client.fetch_status(args.server, hotkey_address(wallet), manifest.digest())
    except (MinerConfigError, PackageError, fee_client.FeeError) as exc:
        return fail(str(exc))
    _print_fee_status(manifest, status)
    return 0 if status.paid or not status.required else 1


def _fee_pay(args: argparse.Namespace) -> int:
    try:
        config, wallet, manifest = _fee_context(args)
        status = _pay_fee(args, config, wallet, manifest)
    except (MinerConfigError, PackageError, fee_client.FeeError, ChainError) as exc:
        return fail(str(exc))
    _print_fee_status(manifest, status)
    return 0 if status.paid else 1


def _fee_report(args: argparse.Namespace) -> int:
    try:
        _, wallet, manifest = _fee_context(args)
        status = fee_client.report(
            args.server,
            wallet,
            manifest_digest=manifest.digest(),
            round_index=manifest.round_index,
            extrinsic_hash=args.extrinsic,
            block=args.block,
        )
    except (MinerConfigError, PackageError, fee_client.FeeError) as exc:
        return fail(str(exc))
    _print_fee_status(manifest, status)
    return 0 if status.paid else 1


def _publish(args: argparse.Namespace) -> int:
    try:
        config = _config(args)
        _warn_weak_source(config)
        wallet = open_wallet(config.chain)
        client = open_client(config.chain, wallet)
        hotkey = hotkey_address(wallet)

        if not client.snapshot().is_registered(hotkey):
            return fail(f"hotkey {hotkey} is not registered on netuid {config.chain.netuid}")

        round_ = current_round(config, client)
        if not round_.accepts_submissions(client.block()):
            if not getattr(args, "recommit", False):
                return fail(
                    f"round {round_.index} closed at block {round_.close_block}; "
                    f"package for round {round_.index + 1} instead"
                )
            print(
                f"round {round_.index} closed at block {round_.close_block}; "
                "re-committing a pointer that was overwritten by its own reveal"
            )

        if args.upload:
            config = config.with_overrides(source=_do_upload(config))

        manifest = load_packaged(config)
        _require_provenance(config, hotkey, manifest.artifact_digest, client.block())
        blocked = _fee_gate(args, hotkey, manifest.digest())
        if blocked:
            return fail(blocked)
        published = publish(config, client, round_.index, manifest)
    except (
        MinerConfigError,
        PackageError,
        PublishError,
        UploadError,
        ProvenanceMissing,
    ) as exc:
        return fail(str(exc))

    print(f"committed {published.bytes_used} bytes for round {published.round_index}")
    print(published.payload)
    return 0


def _reveal(args: argparse.Namespace) -> int:
    from microtensor.miner.package import load_packaged
    from microtensor.miner.publish import reveal

    try:
        config = _config(args)
        wallet = open_wallet(config.chain)
        client = open_client(config.chain, wallet)
        manifest = load_packaged(config)
        index = args.round if args.round is not None else manifest.round_index
        payload = reveal(config, client, index, manifest)
    except (MinerConfigError, PackageError, PublishError) as exc:
        return fail(str(exc))
    print(f"revealed   round {index}: {payload.split('|')[0]}|…|<key>")
    return 0


def _ship(args: argparse.Namespace) -> int:
    try:
        config = _config(args)
    except MinerConfigError as exc:
        return fail(str(exc))
    _warn_weak_source(config)

    if not args.no_selfcheck and not config.selfcheck_path.is_file():
        code = _selfcheck(args)
        if code != 0:
            return code

    try:
        wallet = open_wallet(config.chain)
        client = open_client(config.chain, wallet)
        round_ = current_round(config, client)

        if not round_.accepts_submissions(client.block()):
            return fail(
                f"round {round_.index} has closed; wait for round {round_.index + 1} to open"
            )

        manifest = package(
            config,
            round_.index,
            _load_manifest_spec(config),
            _declared(args, config),
            wallet,
            seal=bool(getattr(args, "sealed", False)),
        )
        print(f"packaged   {len(manifest.files)} files, {manifest.total_bytes / 1024**3:.2f} GiB")

        config = config.with_overrides(source=_do_upload(config))
        _require_provenance(
            config, hotkey_address(wallet), manifest.artifact_digest, client.block()
        )
        if getattr(args, "pay_fee", False):
            paid = _pay_fee(args, config, wallet, manifest)
            if paid.required and not paid.paid:
                return fail(f"the submission fee is {paid.state}: {paid.reason}")
        blocked = _fee_gate(args, hotkey_address(wallet), manifest.digest())
        if blocked:
            return fail(blocked)
        published = publish(config, client, round_.index, manifest)
    except (
        MinerConfigError,
        PackageError,
        PublishError,
        UploadError,
        ProvenanceMissing,
        fee_client.FeeError,
        ChainError,
    ) as exc:
        return fail(str(exc))

    print(f"published  round {published.round_index}: {published.payload}")
    print("\nkeep competing every round with:  mt miner run")
    return 0


def _run(args: argparse.Namespace) -> int:
    try:
        config = _config(args)
        wallet = open_wallet(config.chain)
        client = open_client(config.chain, wallet)
    except MinerConfigError as exc:
        return fail(str(exc))

    hotkey = hotkey_address(wallet)

    def check(manifest, round_index: int) -> str:  # type: ignore[no-untyped-def]
        return _fee_gate(args, hotkey, manifest.digest())

    loop = PublishLoop(config, client, fee_check=check)
    try:
        loop.run(max_rounds=args.max_rounds)
    finally:
        client.close()
    return 0


def _entrypoint(spec: str):  # type: ignore[no-untyped-def]
    """Resolve `module:function` to something callable.

    The participant's training code stays theirs and stays out of this package.
    All the daemon needs is a callable that accepts the hook.
    """
    import importlib

    if ":" not in spec:
        raise MinerConfigError(f"an entrypoint looks like module:function, not {spec!r}")

    module_name, _, attribute = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise MinerConfigError(f"could not import {module_name!r}: {exc}") from exc

    found = getattr(module, attribute, None)
    if not callable(found):
        raise MinerConfigError(f"{spec!r} is not callable")
    return found


def _serve(args: argparse.Namespace) -> int:
    """Train, then submit, without anyone typing a command.

    The submission half is the same code `mt miner ship` runs. It is called
    rather than reimplemented so the unattended path and the manual one cannot
    drift apart.
    """
    from microtensor.miner.daemon import Daemon
    from microtensor.miner.telemetry import TelemetryClient

    try:
        config = _config(args)
        wallet = open_wallet(config.chain)
        client = open_client(config.chain, wallet)
        round_ = current_round(config, client)
        hotkey = hotkey_address(wallet)
    except (MinerConfigError, PublishError) as exc:
        return fail(str(exc))

    if not args.train_entrypoint:
        return fail("give --train module:function so the daemon knows what to run")

    try:
        train = _entrypoint(args.train_entrypoint)
    except MinerConfigError as exc:
        return fail(str(exc))

    url = "" if args.no_telemetry else str(args.coordinator or "")
    telemetry = TelemetryClient(url, wallet)
    telemetry.start()

    daemon = Daemon(
        hotkey=hotkey,
        round_index=round_.index,
        client=telemetry,
        block_of=client.block,
    )
    if daemon.reporter is not None and args.epochs:
        daemon.reporter.epochs_total = int(args.epochs)

    print(f"round {round_.index}: training as {hotkey[:12]}")
    if not url:
        print("telemetry off; this run reports nothing")

    def submit() -> None:
        ship = argparse.Namespace(**vars(args))
        ship.no_selfcheck = getattr(args, "no_selfcheck", False)
        code = _ship(ship)
        if code != 0:
            raise PublishError("submission failed; see the error above")

    ok = daemon.run(train, submit)
    print(f"round {round_.index}: {daemon.phase.value}")
    return 0 if ok else 1


def _status(args: argparse.Namespace) -> int:
    try:
        config = _config(args)
        manifest = load_packaged(config)
    except (MinerConfigError, PackageError) as exc:
        return fail(str(exc))

    print(f"competition {manifest.track}/{manifest.hardware_class}")
    print(f"round       {manifest.round_index}")
    print(f"hotkey      {manifest.hotkey}")
    print(f"source      {manifest.source}")
    print(f"files       {len(manifest.files)}  ({manifest.total_bytes / 1024**3:.2f} GiB)")
    print(f"signed      {'yes' if manifest.signature else 'NO — validators will reject this'}")
    print(
        f"declared    size {manifest.declared.size_bytes}, "
        f"rss {manifest.declared.peak_rss_bytes}, p95 {manifest.declared.p95_latency_ms}ms"
    )

    index = manifest.round_index
    print()
    version = release_version(manifest.track, manifest.hardware_class, release_index(index))
    print(f"release     {version}")
    print(
        f"cutoff      {rounds_until_release(index)} round(s), block {release_cutoff_block(index)}"
    )

    if getattr(args, "offline", False):
        return 0

    standing = fetch_standing(
        args.server, manifest.track, manifest.hardware_class, manifest.digest()
    )
    if not standing.reachable:
        print(f"standing    unavailable ({standing.reason})")
        return 0

    if standing.on_frontier:
        print(f"frontier    yes, rank {standing.rank_by_cost} of {standing.of_total} by cost")
        print(f"measured    quality {standing.quality:.4f}, {standing.expected_ms:.1f} ms")
    else:
        print(f"frontier    no ({standing.of_total} system(s) on it)")

    if standing.contribution:
        parts = " · ".join(
            f"{role} {value:.2f}" for role, value in sorted(standing.contribution.items())
        )
        print(f"contribution {parts}")

    if standing.milestone:
        target_quality = standing.milestone.get("target_quality")
        target_cost = standing.milestone.get("target_cost")
        met = standing.milestone.get("met_by")
        print(
            f"milestone   {target_quality} quality under {target_cost} ms  ·  "
            f"{'met by ' + str(met)[:12] if met else 'unmet'}"
        )

    return 0
