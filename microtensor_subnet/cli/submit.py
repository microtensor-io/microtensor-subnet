from __future__ import annotations

import argparse
import json
import logging
import secrets
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from microtensor_subnet.artifact import contract
from microtensor_subnet.artifact.validate import (
    ManifestError,
    build_manifest,
    check_artifact,
    check_slm_repo,
    load_manifest,
    manifest_sha256,
    write_manifest,
)
from microtensor_subnet.chain.client import ChainError
from microtensor_subnet.chain.commitment import (
    CommitmentError,
    SealedCommitment,
    SubmissionCommitment,
    reveal_message,
)
from microtensor_subnet.chain.wallet import coldkey_address, hotkey_address, sign_bytes
from microtensor_subnet.cli.common import (
    add_chain_arguments,
    add_common_arguments,
    chain_config,
    fail,
    open_client,
    open_wallet,
)
from microtensor_subnet.core.constants import PUBLIC_SERVER_URL

log = logging.getLogger("microtensor_subnet.cli.submit")
TIMEOUT_SECONDS = 20


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("submit", help="validate, pay for and commit a submission")
    inner = parser.add_subparsers(dest="action", required=True)

    manifest = inner.add_parser("manifest", help="write manifest.json for the artifact repository")
    manifest.add_argument("--artifact", type=Path, required=True, help="artifact repository root")
    manifest.add_argument("--arena", required=True)
    manifest.add_argument("--components", default="slm,harness,router,escalation")
    manifest.add_argument("--slm", type=Path, help="slm repository root, when slm is submitted")
    manifest.add_argument("--slm-repo", default="", help="<user>/mt_slm_<arena>")
    manifest.add_argument("--slm-revision", default="", help="the slm repository's git commit sha")
    manifest.add_argument(
        "--hardware-class",
        default=contract.DEFAULT_HARDWARE_CLASS,
        choices=contract.HARDWARE_CLASSES,
    )
    add_common_arguments(manifest)
    manifest.set_defaults(handler=_manifest)

    validate = inner.add_parser("validate", help="check both repositories against the contract")
    validate.add_argument("--artifact", type=Path, required=True)
    validate.add_argument("--slm", type=Path, help="slm repository root")
    validate.add_argument("--allowed-bases", default="", help="comma separated, from the arena")
    add_common_arguments(validate)
    validate.set_defaults(handler=_validate)

    pay = inner.add_parser("pay", help="transfer the submission fee from your coldkey")
    pay.add_argument("--arena", required=True)
    pay.add_argument("--server", default=PUBLIC_SERVER_URL)
    pay.add_argument("--yes", action="store_true")
    add_chain_arguments(pay)
    add_common_arguments(pay)
    pay.set_defaults(handler=_pay)

    commit = inner.add_parser("commit", help="commit the submission on chain")
    commit.add_argument("--arena", required=True)
    commit.add_argument("--hf-user", required=True, help="the hugging face user or org")
    commit.add_argument("--artifact-sha", required=True, help="artifact repository commit sha")
    commit.add_argument("--fee-block", type=int, default=0, help="0 when the arena is free")
    commit.add_argument("--fee-index", type=int, default=0, help="0 when the arena is free")
    commit.add_argument(
        "--open", action="store_true", help="commit the repository in the open instead of sealed"
    )
    add_chain_arguments(commit)
    add_common_arguments(commit)
    commit.set_defaults(handler=_commit)

    reveal = inner.add_parser("reveal", help="reveal a sealed commitment once the window closes")
    reveal.add_argument("--arena", required=True)
    reveal.add_argument("--epoch", type=int, default=None, help="defaults to the latest epoch")
    reveal.add_argument("--server", default=PUBLIC_SERVER_URL)
    add_chain_arguments(reveal)
    add_common_arguments(reveal)
    reveal.set_defaults(handler=_reveal)

    status = inner.add_parser("status", help="where the arena and your submission stand")
    status.add_argument("--arena", required=True)
    status.add_argument("--server", default=PUBLIC_SERVER_URL)
    status.add_argument("--hotkey", default="", help="defaults to the wallet hotkey")
    add_chain_arguments(status)
    add_common_arguments(status)
    status.set_defaults(handler=_status)


def _get(base: str, path: str) -> Any:
    url = base.rstrip("/") + path
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"the server URL must be http or https: {base!r}")
    request = urllib.request.Request(url, headers={"accept": "application/json"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
        return json.loads(response.read() or b"null")


def _arena(server: str, arena_id: str) -> dict[str, Any]:
    try:
        found = _get(server, f"/v2/arenas/{urllib.parse.quote(arena_id)}")
    except urllib.error.HTTPError as exc:
        raise ValueError(f"the server has no arena {arena_id!r} ({exc.code})") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"the server could not be reached: {exc}") from exc
    return dict(found or {})


def _manifest(args: argparse.Namespace) -> int:
    components = [c.strip() for c in str(args.components).split(",") if c.strip()]
    try:
        manifest = build_manifest(
            args.artifact,
            arena=args.arena,
            components=components,
            hardware_class=args.hardware_class,
            slm_repo=args.slm_repo,
            slm_revision=args.slm_revision,
            slm_root=args.slm,
        )
    except (ManifestError, OSError) as exc:
        return fail(str(exc))
    path = write_manifest(args.artifact, manifest)
    print(f"wrote      {path}")
    print(f"files      {len(manifest.files)}")
    if manifest.slm is not None:
        print(f"slm        {manifest.slm.repo}@{manifest.slm.revision[:12]}")
    print(f"manifest   {manifest_sha256(args.artifact)}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    try:
        manifest = load_manifest(args.artifact)
    except ManifestError as exc:
        return fail(str(exc))
    findings = check_artifact(args.artifact, manifest)
    if "slm" in manifest.has:
        if args.slm is None:
            findings.problems.append("pass --slm with the slm repository to check it")
        else:
            bases = [b.strip() for b in str(args.allowed_bases).split(",") if b.strip()]
            more = check_slm_repo(args.slm, manifest.slm, bases)
            findings.problems += more.problems
            findings.notes += more.notes
    for note in findings.notes:
        print(f"note       {note}")
    for problem in findings.problems:
        print(f"problem    {problem}")
    if findings.ok:
        print(f"ok         {manifest.arena}: {', '.join(manifest.components)}")
        print(f"manifest   {manifest_sha256(args.artifact)}")
        return 0
    return 1


def _pay(args: argparse.Namespace) -> int:
    try:
        arena = _arena(args.server, args.arena)
    except ValueError as exc:
        return fail(str(exc))
    definition = dict(arena.get("definition") or {})
    fee_tao = float(definition.get("fee_tao") or 0.0)
    pay_to = str(definition.get("fee_coldkey") or "")
    if fee_tao <= 0 or not pay_to:
        print(
            f"{args.arena} charges no fee; commit with mt submit commit --arena {args.arena} "
            "--hf-user <user> --artifact-sha <sha>"
        )
        return 0
    config = chain_config(args)
    wallet = open_wallet(config)
    print(f"arena      {args.arena}")
    print(f"fee        {fee_tao:g} TAO, not refundable")
    print(f"pay to     {pay_to}")
    print(f"from       {coldkey_address(wallet)}")
    if not args.yes:
        answer = input("send this transfer from your coldkey? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return fail("not sent")
    client = open_client(config, wallet)
    try:
        receipt = client.transfer(pay_to, fee_tao)
    except ChainError as exc:
        return fail(str(exc))
    print(f"sent       {receipt.extrinsic_hash}")
    print(f"fee block  {receipt.block}")
    print(f"fee index  {receipt.extrinsic_index}")
    if receipt.extrinsic_index < 0:
        print("the index could not be read; find it with a block explorer before committing")
    print(
        "next       mt submit commit --arena "
        f"{args.arena} --hf-user <user> --artifact-sha <sha> "
        f"--fee-block {receipt.block} --fee-index {receipt.extrinsic_index}"
    )
    return 0


def _reveal_path(home: Path, arena_id: str) -> Path:
    return Path(home) / "reveals" / f"{arena_id}.json"


def _post(base: str, path: str, body: dict[str, Any]) -> Any:
    url = base.rstrip("/") + path
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"the server URL must be http or https: {base!r}")
    request = urllib.request.Request(  # noqa: S310
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as answer:  # noqa: S310
        return json.loads(answer.read())


def _commit(args: argparse.Namespace) -> int:
    artifact_sha = str(args.artifact_sha).lower()
    salt = secrets.token_hex(16)
    try:
        if args.open:
            commitment: SubmissionCommitment | SealedCommitment = SubmissionCommitment(
                arena_id=args.arena,
                hf_user=args.hf_user,
                artifact_sha=artifact_sha,
                fee_block=int(args.fee_block),
                fee_index=int(args.fee_index),
            )
        else:
            commitment = SealedCommitment.seal(
                args.arena,
                args.hf_user,
                artifact_sha,
                salt,
                int(args.fee_block),
                int(args.fee_index),
            )
        payload = commitment.encode()
    except CommitmentError as exc:
        return fail(str(exc))
    config = chain_config(args)
    wallet = open_wallet(config)
    client = open_client(config, wallet)
    try:
        ok = client.publish(payload)
    except ChainError as exc:
        return fail(str(exc))
    if not ok:
        return fail("the chain refused the commitment")
    hotkey = hotkey_address(wallet)
    print(f"committed  {payload}")
    print(f"hotkey     {hotkey}")
    if isinstance(commitment, SealedCommitment):
        path = _reveal_path(args.home, args.arena)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "arena_id": args.arena,
                    "hotkey": hotkey,
                    "hf_user": args.hf_user,
                    "artifact_sha": artifact_sha,
                    "salt": salt,
                    "digest": commitment.digest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"sealed     {commitment.digest}")
        print(f"reveal     saved to {path}; keep the repositories private until the window closes,")
        print(f"           then run: mt submit reveal --arena {args.arena}")
    else:
        print(
            f"artifact   https://huggingface.co/{commitment.artifact_repo}/tree/{commitment.artifact_sha}"
        )
    return 0


def _reveal(args: argparse.Namespace) -> int:
    path = _reveal_path(args.home, args.arena)
    if not path.is_file():
        return fail(f"no sealed commitment saved at {path}; commit first")
    saved = json.loads(path.read_text(encoding="utf-8"))
    config = chain_config(args)
    wallet = open_wallet(config)
    hotkey = hotkey_address(wallet)
    if saved.get("hotkey") and saved["hotkey"] != hotkey:
        return fail(f"the saved commitment belongs to {saved['hotkey']}, not this hotkey")
    epoch = args.epoch
    if epoch is None:
        try:
            epochs = _get(args.server, f"/v2/arenas/{args.arena}/epochs").get("epochs") or []
        except (urllib.error.URLError, ValueError) as exc:
            return fail(f"could not read the arena's epochs: {exc}")
        if not epochs:
            return fail("the arena has no epoch to reveal into yet")
        epoch = int(epochs[0]["epoch_index"])
    message = reveal_message(
        args.arena, epoch, hotkey, saved["hf_user"], saved["artifact_sha"], saved["salt"]
    )
    body = {
        "hotkey": hotkey,
        "hf_user": saved["hf_user"],
        "artifact_sha": saved["artifact_sha"],
        "salt": saved["salt"],
        "signature": sign_bytes(wallet, message),
    }
    try:
        found = _post(args.server, f"/v2/arenas/{args.arena}/epochs/{epoch}/reveal", body)
    except urllib.error.HTTPError as exc:
        return fail(f"the server refused the reveal: {exc.read().decode('utf-8', 'replace')[:300]}")
    except (urllib.error.URLError, ValueError) as exc:
        return fail(f"the reveal could not be sent: {exc}")
    print(f"revealed   epoch {epoch}  {found.get('reveal_state')}")
    print(
        f"artifact   https://huggingface.co/{found.get('artifact_repo')}/tree/{saved['artifact_sha']}"
    )
    if found.get("reveal_state") == "unsealed":
        print("make both repositories public now; validators fetch them at release")
    return 0


def _status(args: argparse.Namespace) -> int:
    try:
        arena = _arena(args.server, args.arena)
    except ValueError as exc:
        return fail(str(exc))
    print(f"arena      {arena.get('id')}  {arena.get('network')}  {arena.get('state')}")
    current = arena.get("current_epoch")
    if current:
        print(
            f"epoch      {current['index']}  {current['phase']}  "
            f"commits close at block {current['open_block']}, "
            f"window closes {current['close_block']}"
        )
    hotkey = args.hotkey
    if not hotkey:
        try:
            hotkey = hotkey_address(open_wallet(chain_config(args)))
        except Exception as exc:
            return fail(f"no hotkey to look up: {exc}")
    epoch = current["index"] if current else None
    if epoch is None:
        return 0
    try:
        ledger = _get(args.server, f"/v2/arenas/{args.arena}/epochs/{epoch}/ledger")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError):
        ledger = None
    if not ledger:
        print("ledger     nothing published for this epoch yet")
        return 0
    mine = [row for row in ledger.get("standings") or [] if row.get("hotkey") == hotkey]
    if not mine:
        print("ledger     your hotkey has no published score this epoch")
        return 0
    for row in mine:
        print(
            f"{row.get('pool'):<10} published {float(row.get('published') or 0):.4f}  "
            f"consensus {float(row.get('consensus') or 0):.4f}  "
            f"{'eligible' if row.get('eligible') else 'not eligible'}"
        )
    return 0
