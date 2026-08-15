from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("microtensor.miner.upload")


class UploadError(RuntimeError):
    pass


class UploadUnsupported(UploadError):
    pass


Uploader = Callable[[str, Path, Sequence[str]], None]


@dataclass(frozen=True, slots=True)
class UploadPlan:
    scheme: str
    locator: str
    files: tuple[str, ...]
    total_bytes: int

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(f"{self.scheme}:{self.locator}/{name}" for name in self.files)


def _upload_hf(locator: str, root: Path, files: Sequence[str]) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise UploadUnsupported(
            "huggingface-hub is required to upload to an hf source; "
            'pip install ".[huggingface]"'
        ) from exc

    repo_id, _, revision = locator.partition("@")
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        for name in files:
            api.upload_file(
                path_or_fileobj=str(root / name),
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="model",
                revision=revision or None,
            )
    except Exception as exc:
        raise UploadError(f"upload to hf:{locator} failed: {exc}") from exc


def _upload_object_store(locator: str, root: Path, files: Sequence[str]) -> None:
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


def _upload_https(locator: str, root: Path, files: Sequence[str]) -> None:
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


def upload(plan: UploadPlan, root: Path) -> None:
    log.info(
        "uploading %d files (%.2f GiB) to %s:%s",
        len(plan.files),
        plan.total_bytes / 1024**3,
        plan.scheme,
        plan.locator,
    )
    uploader_for(plan.scheme)(plan.locator, root, plan.files)
    log.info("upload complete")
