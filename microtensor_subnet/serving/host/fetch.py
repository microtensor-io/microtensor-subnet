from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microtensor_subnet.chain.commitment import CommitmentError, SubmissionCommitment
from microtensor_subnet.validator import collect
from microtensor_subnet.validator.context import Artifact


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Fetched:
    model: str
    artifact_digest: str
    artifact_dir: Path
    slm_dir: Path
    router_dir: Path | None
    config: dict[str, Any]
    artifact: Artifact


def fetch(entry: Mapping[str, Any], cache_dir: Path) -> Fetched:
    arena = str(entry.get("arena", ""))
    hotkey = str(entry.get("hotkey", ""))
    sha = str(entry.get("artifact_digest", ""))
    repos = dict(entry.get("repos") or {})
    artifact_repo = str((repos.get("artifact") or {}).get("repo", ""))
    hf_user = artifact_repo.split("/", 1)[0] if "/" in artifact_repo else ""
    if not (arena and sha and hf_user):
        raise FetchError(f"the catalogue entry for {entry.get('model', '?')} names no repository")
    try:
        commitment = SubmissionCommitment(
            arena_id=arena, hf_user=hf_user, artifact_sha=sha, fee_block=0, fee_index=0
        )
    except CommitmentError as exc:
        raise FetchError(str(exc)) from exc
    candidate = collect.Candidate(hotkey=hotkey, coldkey="", commitment=commitment, commit_block=0)
    try:
        artifact = collect.materialise(
            candidate, cache_dir=cache_dir, arena_id=arena, allowed_bases=[], allowed_licenses=[]
        )
    except collect.CollectError as exc:
        raise FetchError(str(exc)) from exc
    if artifact.slm_dir is None:
        raise FetchError("a served system carries its own model")
    router_dir: Path | None = None
    if "router" in artifact.has:
        router_dir = artifact.artifact_dir / "router"
    else:
        reference = repos.get("router") or {}
        if reference.get("repo"):
            try:
                router_dir = collect.download(
                    str(reference["repo"]), str(reference.get("revision") or "main"), cache_dir
                )
            except Exception as exc:
                raise FetchError(f"the reference router could not be fetched: {exc}") from exc
    return Fetched(
        model=str(entry.get("model") or entry.get("id") or ""),
        artifact_digest=sha,
        artifact_dir=artifact.artifact_dir,
        slm_dir=artifact.slm_dir,
        router_dir=router_dir,
        config=dict(artifact.config),
        artifact=artifact,
    )
