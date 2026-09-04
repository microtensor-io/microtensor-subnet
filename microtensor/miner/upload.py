from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("microtensor.miner.upload")


class UploadError(RuntimeError):
    pass


class UploadUnsupported(UploadError):
    pass


Uploader = Callable[[str, Path, Sequence[str]], "str | None"]


@dataclass(frozen=True, slots=True)
class UploadPlan:
    scheme: str
    locator: str
    files: tuple[str, ...]
    total_bytes: int

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(f"{self.scheme}:{self.locator}/{name}" for name in self.files)


_SHA = re.compile(r"^[0-9a-f]{7,64}$")


def _upload_hf(locator: str, root: Path, files: Sequence[str]) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise UploadUnsupported(
            "huggingface-hub is required to upload to an hf source; "
            'pip install ".[huggingface]"'
        ) from exc

    repo_id, _, revision = locator.partition("@")
    branch = None if not revision or _SHA.match(revision.lower()) else revision
    if revision and branch is None:
        log.info("hf:%s pins a commit; uploading to the default branch and re-pinning", locator)
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        commit = api.upload_folder(
            folder_path=str(root),
            repo_id=repo_id,
            repo_type="model",
            revision=branch,
            allow_patterns=list(files),
        )
    except Exception as exc:
        raise UploadError(f"upload to hf:{locator} failed: {exc}") from exc
    sha = str(getattr(commit, "oid", "") or "")
    if not sha:
        try:
            sha = str(
                api.list_repo_commits(repo_id, repo_type="model", revision=branch)[0].commit_id
            )
        except Exception as exc:
            raise UploadError(
                f"uploaded to hf:{repo_id} but could not read back the commit to pin: {exc}"
            ) from exc
    if not _SHA.match(sha.lower()):
        raise UploadError(f"hf:{repo_id} returned an unusable commit id {sha!r}")
    return f"{repo_id}@{sha}"


def _upload_object_store(locator: str, root: Path, files: Sequence[str]) -> str | None:
    try:
        import boto3
    except ImportError as exc:
        raise UploadUnsupported(
            'boto3 is required to upload to an object store; pip install ".[s3]"'
        ) from exc

    bucket, _, prefix = locator.partition("/")
    client = boto3.client("s3")
    try:
        for name in files:
            key = f"{prefix.rstrip('/')}/{name}" if prefix else name
            client.upload_file(str(root / name), bucket, key)
    except Exception as exc:
        raise UploadError(f"upload to {bucket} failed: {exc}") from exc


def _upload_https(locator: str, root: Path, files: Sequence[str]) -> str | None:
    raise UploadUnsupported(
        f"https://{locator} is a plain web host; publish the files with your own "
        "tooling, then run `mt miner publish` without --upload"
    )


UPLOADERS: dict[str, Uploader] = {
    "hf": _upload_hf,
    "s3": _upload_object_store,
    "r2": _upload_object_store,
    "https": _upload_https,
}


def uploader_for(scheme: str) -> Uploader:
    uploader = UPLOADERS.get(scheme)
    if uploader is None:
        raise UploadUnsupported(
            f"no uploader for scheme {scheme!r}; upload by hand and publish without --upload"
        )
    return uploader


def plan_upload(root: Path, scheme: str, locator: str, files: Sequence[str]) -> UploadPlan:
    missing = [name for name in files if not (root / name).is_file()]
    if missing:
        raise UploadError(f"not in the artifact directory: {', '.join(missing[:3])}")
    return UploadPlan(
        scheme=scheme,
        locator=locator,
        files=tuple(files),
        total_bytes=sum((root / name).stat().st_size for name in files),
    )


def upload(plan: UploadPlan, root: Path) -> str:
    log.info(
        "uploading %d files (%.2f GiB) to %s:%s",
        len(plan.files),
        plan.total_bytes / 1024**3,
        plan.scheme,
        plan.locator,
    )
    pinned = uploader_for(plan.scheme)(plan.locator, root, plan.files)
    log.info("upload complete")
    return pinned or plan.locator
