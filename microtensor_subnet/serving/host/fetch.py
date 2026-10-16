from __future__ import annotations

import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final

from microtensor_subnet.chain.commitment import CommitmentError, SubmissionCommitment
from microtensor_subnet.core.hashing import DIGEST_PREFIX, digest_entries, digest_file
from microtensor_subnet.validator import collect
from microtensor_subnet.validator.context import Artifact

HEX: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
COMPLETE: Final[str] = ".complete"
GGUF_SUFFIX: Final[str] = ".gguf"
PROJECTOR: Final[str] = "mmproj"
DOWNLOAD_WORKERS: Final[int] = 8


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


@dataclass(frozen=True, slots=True)
class FetchedModel:
    model: str
    artifact_digest: str
    model_dir: Path
    files: dict[str, str] = field(default_factory=dict)
    template: str = ""

    @property
    def weights(self) -> str:
        found = sorted(
            path
            for path in self.files
            if path.lower().endswith(GGUF_SUFFIX) and PROJECTOR not in path.lower()
        )
        return found[0] if found else ""


def digest_of(files: Mapping[str, str]) -> str:
    entries = [(path, DIGEST_PREFIX + str(hexdigest).lower()) for path, hexdigest in files.items()]
    return digest_entries(entries).removeprefix(DIGEST_PREFIX)


def _relative(path: str) -> str:
    found = PurePosixPath(str(path))
    if not path or found.is_absolute() or ".." in found.parts or "\\" in path:
        raise FetchError(f"the manifest names {path!r}, which is not a path inside the repository")
    return found.as_posix()


def manifest_of(entry: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    raw = entry.get("manifest")
    manifest: dict[str, Any] = dict(raw) if isinstance(raw, Mapping) else {}
    listed = manifest.get("files")
    if not isinstance(listed, Mapping) or not listed:
        raise FetchError(f"the catalogue entry for {entry.get('model', '?')} carries no manifest")
    files: dict[str, str] = {}
    for path, hexdigest in listed.items():
        value = str(hexdigest).lower().removeprefix(DIGEST_PREFIX)
        if not HEX.match(value):
            raise FetchError(f"the manifest hash for {path!r} is not a sha256")
        files[_relative(str(path))] = value
    template = str(manifest.get("template") or "")
    if template and _relative(template) not in files:
        raise FetchError("the manifest names a chat template it does not pin")
    return files, template


def check_manifest(root: Path, files: Mapping[str, str]) -> None:
    problems: list[str] = []
    for path, hexdigest in sorted(files.items()):
        found = root / _relative(path)
        if not found.is_file():
            problems.append(f"{path} is missing")
        elif digest_file(found) != DIGEST_PREFIX + hexdigest.lower():
            problems.append(f"{path} does not match its manifest hash")
    if problems:
        raise FetchError("; ".join(problems[:6]))


def fetch_model(entry: Mapping[str, Any], cache_dir: Path, *, token: str = "") -> FetchedModel:
    model = str(entry.get("model") or entry.get("id") or "")
    repos = entry.get("repos")
    reference = repos.get("model") if isinstance(repos, Mapping) else None
    pinned: dict[str, Any] = dict(reference) if isinstance(reference, Mapping) else {}
    repo = str(pinned.get("repo") or "")
    revision = str(pinned.get("revision") or "")
    if not (model and repo and revision):
        raise FetchError(f"the catalogue entry for {model or '?'} names no model repository")
    files, template = manifest_of(entry)
    digest = digest_of(files)
    expected = str(entry.get("artifact_digest") or "").lower().removeprefix(DIGEST_PREFIX)
    if expected and expected != digest:
        raise FetchError(f"the catalogue digest for {model} does not match its manifest")
    target = cache_dir / repo.replace("/", "__") / revision
    marker = target / COMPLETE
    fetched = FetchedModel(
        model=model, artifact_digest=digest, model_dir=target, files=files, template=template
    )
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == digest:
        return fetched
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise FetchError("huggingface_hub is required to download models") from exc
    target.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=str(target),
            allow_patterns=sorted(files),
            token=token or None,
            max_workers=DOWNLOAD_WORKERS,
        )
    except Exception as exc:
        raise FetchError(f"{repo} at {revision} could not be downloaded: {exc}") from exc
    try:
        check_manifest(target, files)
    except FetchError:
        shutil.rmtree(target, ignore_errors=True)
        raise
    marker.write_text(digest, encoding="utf-8")
    return fetched
