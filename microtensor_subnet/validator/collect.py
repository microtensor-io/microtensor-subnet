from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microtensor_subnet.artifact import contract
from microtensor_subnet.artifact.params import (
    ParameterError,
    model_parameters,
)
from microtensor_subnet.artifact.validate import (
    ManifestError,
    check_artifact,
    check_slm_repo,
    load_manifest,
    manifest_sha256,
)
from microtensor_subnet.chain.client import ChainError
from microtensor_subnet.chain.commitment import (
    CommitmentError,
    SubmissionCommitment,
    decode_sealed,
    decode_submissions,
)
from microtensor_subnet.validator.context import Artifact

log = logging.getLogger("microtensor_subnet.validator.collect")
RAO_PER_TAO = 1_000_000_000


class CollectError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Candidate:
    hotkey: str
    coldkey: str
    commitment: SubmissionCommitment
    commit_block: int


def candidates(
    client: Any,
    arena_id: str,
    *,
    commit_close_block: int,
    roster: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Candidate]:
    snapshot = client.snapshot(refresh=True)
    hotkeys = [n.hotkey for n in snapshot.neurons]
    raw = client.commitments(hotkeys)
    decoded = decode_submissions(raw)
    for hotkey, sealed in decode_sealed(raw).items():
        entry = (roster or {}).get(hotkey) or {}
        if str(entry.get("reveal_state") or "") != "unsealed":
            log.info("%s is sealed and not revealed on the roster", hotkey)
            continue
        hf_user = str(entry.get("hf_user") or "")
        artifact_sha = str(entry.get("artifact_sha") or "").lower()
        salt = str(entry.get("salt") or "")
        if not sealed.matches(hf_user, artifact_sha, salt):
            log.info("%s revealed values do not match the sealed digest", hotkey)
            continue
        try:
            decoded[hotkey] = SubmissionCommitment(
                sealed.arena_id, hf_user, artifact_sha, sealed.fee_block, sealed.fee_index
            )
        except CommitmentError as exc:
            log.info("%s revealed values are invalid: %s", hotkey, exc)
    blocks = client.commitment_blocks(list(decoded))
    coldkeys = snapshot.coldkeys() if hasattr(snapshot, "coldkeys") else {}
    found: list[Candidate] = []
    seen_fees: set[tuple[int, int]] = set()
    for hotkey, commitment in sorted(decoded.items()):
        if commitment.arena_id != arena_id:
            continue
        block = int(blocks.get(hotkey, commit_close_block))
        if block >= commit_close_block:
            log.info("%s committed at block %d, after the commit window closed", hotkey, block)
            continue
        if roster is not None and hotkey not in roster:
            log.info("%s is not on the signed roster", hotkey)
            continue
        key = (commitment.fee_block, commitment.fee_index)
        if key in seen_fees:
            log.info("%s cites a fee transfer another commitment already cited", hotkey)
            continue
        seen_fees.add(key)
        found.append(Candidate(hotkey, str(coldkeys.get(hotkey, "")), commitment, block))
    return found


def fee_ok(client: Any, candidate: Candidate, *, pay_to: str, fee_tao: float) -> str:
    if fee_tao <= 0 or not pay_to:
        return ""
    try:
        event = client.transfer_at(candidate.commitment.fee_block, candidate.commitment.fee_index)
    except ChainError as exc:
        return f"fee transfer could not be read: {exc}"
    if not event.ok:
        return f"fee transfer invalid: {event.reason}"
    if event.recipient != pay_to:
        return "fee transfer did not pay the arena coldkey"
    if event.amount_rao < round(fee_tao * RAO_PER_TAO):
        return "fee transfer is below the arena fee"
    if candidate.coldkey and event.payer != candidate.coldkey:
        return "fee transfer was not sent by the hotkey's coldkey"
    return ""


def download(repo: str, revision: str, into: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise CollectError("huggingface_hub is required to download submissions") from exc
    target = into / repo.replace("/", "__") / revision
    if (target / ".complete").is_file():
        return target
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(target), max_workers=4)
    (target / ".complete").write_text("ok", encoding="utf-8")
    return target


def _config(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / contract.CONFIG_FILE
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(contract.DEFAULT_CONFIG)
    return parsed if isinstance(parsed, dict) else dict(contract.DEFAULT_CONFIG)


def materialise(
    candidate: Candidate,
    *,
    cache_dir: Path,
    arena_id: str,
    allowed_bases: list[str],
    allowed_licenses: list[str],
    bounds: tuple[int, int] | None = None,
    formats: list[str] | None = None,
) -> Artifact:
    commitment = candidate.commitment
    try:
        artifact_dir = download(commitment.artifact_repo, commitment.artifact_sha, cache_dir)
    except Exception as exc:
        raise CollectError(f"artifact repository could not be fetched: {exc}") from exc
    try:
        manifest = load_manifest(artifact_dir)
    except ManifestError as exc:
        raise CollectError(str(exc)) from exc
    if manifest.arena != arena_id:
        raise CollectError(f"manifest names arena {manifest.arena!r}, not {arena_id!r}")
    findings = check_artifact(artifact_dir, manifest)
    slm_dir: Path | None = None
    parameters = 0
    if manifest.slm is not None:
        if manifest.slm.repo != commitment.slm_repo:
            raise CollectError(
                "manifest pins an slm repository that does not follow the convention"
            )
        try:
            slm_dir = download(manifest.slm.repo, manifest.slm.revision, cache_dir)
        except Exception as exc:
            raise CollectError(f"slm repository could not be fetched: {exc}") from exc
        more = check_slm_repo(
            slm_dir, manifest.slm, allowed_bases, bounds=bounds, formats=formats or ()
        )
        findings.problems += more.problems
        licence = _licence(slm_dir)
        if allowed_licenses and licence not in [x.lower() for x in allowed_licenses]:
            findings.problems.append(f"slm licence {licence!r} is not allowed in this arena")
        try:
            parameters = model_parameters(slm_dir)
        except ParameterError as exc:
            findings.problems.append(str(exc))
    if findings.problems:
        raise CollectError("; ".join(findings.problems[:6]))
    return Artifact(
        hotkey=candidate.hotkey,
        coldkey=candidate.coldkey,
        commitment=commitment,
        commit_block=candidate.commit_block,
        artifact_dir=artifact_dir,
        slm_dir=slm_dir,
        manifest=manifest,
        manifest_sha=manifest_sha256(artifact_dir),
        config=_config(artifact_dir),
        slm_parameters=parameters,
    )


def _licence(root: Path) -> str:
    card = root / "README.md"
    if not card.is_file():
        return ""
    text = card.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        return ""
    front = text.split("---", 2)[1] if text.count("---") >= 2 else ""
    for line in front.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "license":
            return value.strip().strip("'\"").lower()
    return ""
