from __future__ import annotations

import logging
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from microtensor.chain.commitment import Commitment
from microtensor.core.constants import (
    ARTIFACT_FETCH_RETRIES,
    ARTIFACT_FETCH_TIMEOUT_SECONDS,
    MAX_MANIFEST_BYTES,
)
from microtensor.registry.cache import ArtifactCache, CacheError
from microtensor.registry.manifest import ArtifactManifest, ManifestError, verify_tree

log = logging.getLogger("microtensor.registry")

CHUNK_BYTES = 4 * 1024 * 1024


class FetchError(RuntimeError):
    pass


class Unfetchable(FetchError):
    pass


class ArtifactMismatch(FetchError):
    pass


Fetcher = Callable[[str, str, Path, int], None]


def parse_source(source: str) -> tuple[str, str]:
    scheme, _, locator = source.partition(":")
    if not scheme or not locator:
        raise FetchError(f"source {source!r} is not scheme:locator")
    return scheme, locator


def _fetch_https(locator: str, relpath: str, destination: Path, timeout: int) -> None:
    url = f"https://{locator.lstrip('/')}/{relpath}"
    if not url.startswith("https://"):
        raise Unfetchable(f"refusing to fetch over a non-https url: {url}")

    request = urllib.request.Request(url, headers={"User-Agent": "microtensor-validator"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            _stream(response, destination)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise Unfetchable(f"{url} could not be fetched: {exc}") from exc


def _fetch_hf(locator: str, relpath: str, destination: Path, timeout: int) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise Unfetchable("huggingface_hub is required to fetch an hf source") from exc

    repo_id, _, revision = locator.partition("@")
    try:
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=relpath,
            revision=revision or None,
            etag_timeout=timeout,
        )
    except Exception as exc:
        raise Unfetchable(f"hf:{locator}/{relpath} could not be fetched: {exc}") from exc
    shutil.copyfile(cached, destination)


def _fetch_object_store(locator: str, relpath: str, destination: Path, timeout: int) -> None:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise Unfetchable("boto3 is required to fetch an object store source") from exc

    bucket, _, prefix = locator.partition("/")
    key = f"{prefix.rstrip('/')}/{relpath}" if prefix else relpath
    try:
        client = boto3.client("s3", config=Config(connect_timeout=timeout, read_timeout=timeout))
        client.download_file(bucket, key, str(destination))
    except Exception as exc:
        raise Unfetchable(f"s3://{bucket}/{key} could not be fetched: {exc}") from exc


FETCHERS: dict[str, Fetcher] = {
    "https": _fetch_https,
    "hf": _fetch_hf,
    "s3": _fetch_object_store,
    "r2": _fetch_object_store,
}


def fetcher_for(scheme: str) -> Fetcher:
    fetcher = FETCHERS.get(scheme)
    if fetcher is None:
        raise Unfetchable(f"no fetcher for scheme {scheme!r}; known: {sorted(FETCHERS)}")
    return fetcher


def _stream(response: object, destination: Path) -> None:
    with destination.open("wb") as fh:
        while True:
            chunk = response.read(CHUNK_BYTES)  # type: ignore[attr-defined]
            if not chunk:
                break
            fh.write(chunk)


def _fetch_one(
    fetcher: Fetcher,
    locator: str,
    relpath: str,
    destination: Path,
    *,
    attempts: int,
    timeout: int,
    sleep: Callable[[float], None],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            fetcher(locator, relpath, destination, timeout)
            return
        except Unfetchable as exc:
            last = exc
            destination.unlink(missing_ok=True)
            if attempt < attempts:
                log.warning("fetch of %s failed (%d/%d): %s", relpath, attempt, attempts, exc)
                sleep(2.0 * attempt)
    raise Unfetchable(f"{relpath} was unfetchable after {attempts} attempts: {last}")


MANIFEST_NAME = "manifest.json"


def fetch_manifest(
    commitment: Commitment,
    *,
    workdir: Path,
    attempts: int = ARTIFACT_FETCH_RETRIES,
    timeout: int = ARTIFACT_FETCH_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> ArtifactManifest:
    scheme, locator = parse_source(commitment.source)
    fetcher = fetcher_for(scheme)

    workdir.mkdir(parents=True, exist_ok=True)
    destination = workdir / f"{commitment.manifest_digest}.json"
    _fetch_one(
        fetcher,
        locator,
        MANIFEST_NAME,
        destination,
        attempts=attempts,
        timeout=timeout,
        sleep=sleep,
    )

    try:
        if destination.stat().st_size > MAX_MANIFEST_BYTES:
            raise ArtifactMismatch(f"manifest is over the {MAX_MANIFEST_BYTES} byte limit")
        manifest = ArtifactManifest.from_json(destination.read_bytes())
    except ManifestError as exc:
        raise ArtifactMismatch(str(exc)) from exc
    finally:
        destination.unlink(missing_ok=True)

    ok, reason = manifest.matches(commitment)
    if not ok:
        raise ArtifactMismatch(reason)
    return manifest


def materialise(
    manifest: ArtifactManifest,
    cache: ArtifactCache,
    *,
    workdir: Path,
    attempts: int = ARTIFACT_FETCH_RETRIES,
    timeout: int = ARTIFACT_FETCH_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    key: str | None = None,
) -> Path:
    cached = cache.get(manifest.artifact_digest)
    if cached is not None:
        return cached

    fits, reason = manifest.fits_class()
    if not fits:
        raise ArtifactMismatch(reason)

    scheme, locator = parse_source(manifest.source)
    fetcher = fetcher_for(scheme)

    staging = workdir / f"staging-{manifest.hotkey[:12]}-{manifest.round_index}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    try:
        try:
            cache.reserve(manifest.total_bytes)
        except CacheError as exc:
            raise Unfetchable(str(exc)) from exc

        if manifest.sealed is not None:
            from microtensor.core.hashing import digest_bytes
            from microtensor.core.sealing import SealError, open_sealed

            if not key:
                raise Unfetchable("the submission is sealed and no key was revealed")
            blob_name = str(manifest.sealed.get("blob", "artifact.enc"))
            blob_path = staging / blob_name
            _fetch_one(
                fetcher,
                locator,
                blob_name,
                blob_path,
                attempts=attempts,
                timeout=timeout,
                sleep=sleep,
            )
            blob = blob_path.read_bytes()
            expected = str(manifest.sealed.get("digest", ""))
            if expected and digest_bytes(blob) != expected:
                raise ArtifactMismatch("the ciphertext does not match the committed blob digest")
            try:
                open_sealed(blob, key, staging)
            except SealError as exc:
                raise ArtifactMismatch(str(exc)) from exc
            blob_path.unlink(missing_ok=True)
        else:
            for entry in manifest.files:
                _fetch_one(
                    fetcher,
                    locator,
                    entry.path,
                    staging / entry.path,
                    attempts=attempts,
                    timeout=timeout,
                    sleep=sleep,
                )

        ok, reason = verify_tree(staging, manifest)
        if not ok:
            raise ArtifactMismatch(reason)

        return cache.adopt(manifest.artifact_digest, staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
