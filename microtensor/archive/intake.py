from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from microtensor.archive.push import _manifest_of
from microtensor.chain.commitment import digest_matches
from microtensor.core.hashing import digest_bytes
from microtensor.registry.fetch import Unfetchable, _fetch_one, fetcher_for, parse_source
from microtensor.registry.manifest import verify_tree

log = logging.getLogger("microtensor.archive.intake")

INTAKE_FILE = "intake.json"


class Store(Protocol):
    def has(self, key: str) -> bool: ...

    def put_tree(self, key: str, root: Path, message: str) -> None: ...

    def put_json(self, key: str, name: str, payload: Any, message: str) -> None: ...


class LocalStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def has(self, key: str) -> bool:
        return (self.root / key / INTAKE_FILE).is_file()

    def put_tree(self, key: str, root: Path, message: str) -> None:
        target = self.root / key
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, target, dirs_exist_ok=True)

    def put_json(self, key: str, name: str, payload: Any, message: str) -> None:
        path = self.root / key / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class HubStore:
    def __init__(self, org: str, token: str) -> None:
        from huggingface_hub import HfApi

        self.org = org
        self.api = HfApi(token=token)

    def _repo(self, key: str) -> str:
        return f"{self.org}/{key}"

    def has(self, key: str) -> bool:
        try:
            return bool(self.api.file_exists(self._repo(key), INTAKE_FILE))
        except Exception:
            return False

    def put_tree(self, key: str, root: Path, message: str) -> None:
        self.api.create_repo(repo_id=self._repo(key), private=True, exist_ok=True)
        self.api.upload_folder(
            repo_id=self._repo(key), folder_path=str(root), commit_message=message
        )

    def put_json(self, key: str, name: str, payload: Any, message: str) -> None:
        self.api.upload_file(
            path_or_fileobj=(json.dumps(payload, sort_keys=True, indent=2) + "\n").encode(),
            path_in_repo=name,
            repo_id=self._repo(key),
            commit_message=message,
        )


@dataclass(frozen=True, slots=True)
class Submission:
    round_index: int
    hotkey: str
    track: str
    hardware_class: str
    manifest_digest: str
    source: str
    sealed: bool

    @property
    def key(self) -> str:
        return f"intake-{self.track}-{self.hardware_class}-r{self.round_index}-{self.hotkey[:12]}"


def _fetch(source: str, name: str, destination: Path) -> None:
    scheme, locator = parse_source(source)
    _fetch_one(
        fetcher_for(scheme),
        locator,
        name,
        destination,
        attempts=3,
        timeout=600,
        sleep=time.sleep,
    )


def intake(submission: Submission, store: Store, workdir: Path) -> str:
    if store.has(submission.key):
        return "already archived"
    target = workdir / submission.key
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    manifest, raw = _manifest_of(None, submission.source, workdir)
    if manifest is None or raw is None:
        return "the manifest could not be fetched"
    committed = submission.manifest_digest
    if committed and not (
        digest_matches(committed, manifest.digest())
        or digest_matches(committed, manifest.legacy_digest())
    ):
        return "the manifest at the source no longer matches the committed digest"
    (target / "manifest.json").write_bytes(raw)
    try:
        if manifest.sealed is not None:
            blob = str(manifest.sealed.get("blob", "artifact.enc"))
            _fetch(submission.source, blob, target / "sealed" / blob)
            expected = str(manifest.sealed.get("digest", ""))
            if expected and digest_bytes((target / "sealed" / blob).read_bytes()) != expected:
                return "the sealed blob does not match the committed digest"
        else:
            files = target / "files"
            for entry in manifest.files:
                _fetch(submission.source, entry.path, files / entry.path)
            ok, reason = verify_tree(files, manifest)
            if not ok:
                return f"the fetched files do not match the manifest: {reason}"
    except Unfetchable as exc:
        return f"the artifact could not be fetched: {exc}"
    system = manifest.system.body() if manifest.system is not None else None
    record = {
        "round_index": submission.round_index,
        "hotkey": submission.hotkey,
        "track": submission.track,
        "hardware_class": submission.hardware_class,
        "manifest_digest": submission.manifest_digest,
        "artifact_digest": manifest.artifact_digest,
        "source": submission.source,
        "sealed": manifest.sealed is not None,
        "system": system,
        "escalation": system.get("escalation") if isinstance(system, dict) else None,
        "archived_at": int(time.time()),
    }
    (target / INTAKE_FILE).write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    store.put_tree(submission.key, target, f"intake round {submission.round_index}")
    shutil.rmtree(target, ignore_errors=True)
    return ""


def record_reveal(store: Store, submission: Submission, key: str) -> None:
    store.put_json(submission.key, "reveal.json", {"key": key}, "reveal key")


def record_traces(store: Store, submission: Submission, traces: list[dict[str, Any]]) -> None:
    store.put_json(
        submission.key,
        f"traces/round-{submission.round_index}.json",
        traces,
        f"{len(traces)} traces",
    )


def record_certificate(store: Store, submission: Submission, certificate: dict[str, Any]) -> None:
    store.put_json(submission.key, "certificate.json", certificate, "certificate")


def mirror(model: str, revision: str, org: str, token: str) -> str:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    repo_id = f"{org}/{model.split('/', 1)[-1]}"
    tag = revision[:12]
    try:
        refs = api.list_repo_refs(repo_id)
        if any(t.name == tag for t in refs.tags):
            return f"{repo_id} already mirrors {model}@{revision}"
    except Exception as exc:
        log.info("%s has no mirror yet (%s); creating it", repo_id, exc)
    local = snapshot_download(repo_id=model, revision=revision, token=token or None)
    api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id, folder_path=local, commit_message=f"mirror {model}@{revision}"
    )
    api.create_tag(repo_id=repo_id, tag=tag, revision=commit.oid, tag_message=revision)
    return f"mirrored {model}@{revision} to {repo_id} (tag {tag})"
