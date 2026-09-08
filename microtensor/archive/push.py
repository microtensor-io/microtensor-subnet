"""Archive certified artifacts so the record outlives miner hosting.

The bytes were fetched and digest-verified during measurement, so archival
retains what the network already holds rather than requesting anything from a
participant. One public repository per admitted system, self-describing: the
weights, the manifest as submitted, the certificate as measured, and a card a
stranger can read without this database.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from microtensor.core.hashing import digest_bytes
from microtensor.core.sealing import SealError, open_sealed
from microtensor.registry.fetch import Unfetchable, _fetch_one, fetcher_for, parse_source
from microtensor.registry.manifest import ArtifactManifest, verify_tree

log = logging.getLogger("microtensor.archive")

ARCHIVED_STATES = ("confirmed", "unmeasured")
NO_CERTIFICATE = (
    "No certificate was available when this artifact was archived; the weights "
    "are retained as fetched and digest-verified, without a measured record."
)


@dataclass(slots=True)
class Candidate:
    system_id: str
    hotkey: str
    state: str
    snapshot: Path | None
    certificate: dict[str, Any] | None = None
    quality: float | None = None
    expected_ms: float | None = None
    has_manifest: bool = True
    tags: list[str] = field(default_factory=list)
    manifest: ArtifactManifest | None = None
    manifest_bytes: bytes | None = None
    source: str = ""
    key: str = ""


def _get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as answer:  # noqa: S310
        return json.load(answer)


def repo_name(track: str, hardware_class: str, round_index: int, hotkey: str) -> str:
    klass = hardware_class.replace("mt-", "", 1)
    return f"mt-{track}-{klass}-r{round_index}-{hotkey[:8]}"


def snapshots(cache_dirs: list[Path]) -> dict[str, Path]:
    skipped = 0
    """Every cached snapshot that carries a manifest, keyed by manifest digest.

    The digest is recomputed from the manifest bytes, never trusted from a
    path, so the mapping from settled system to archived bytes is verified
    rather than assumed.
    """
    found: dict[str, Path] = {}
    for root in cache_dirs:
        if not root.is_dir():
            continue
        for manifest_path in root.glob("models--*/snapshots/*/manifest.json"):
            try:
                manifest = ArtifactManifest.from_json(manifest_path.read_bytes())
            except Exception as exc:
                log.debug("unreadable manifest at %s: %s", manifest_path, exc)
                skipped += 1
                continue
            digest = manifest.digest().split(":", 1)[-1]
            found[digest] = manifest_path.parent
    if skipped:
        log.info("skipped %d cached snapshot(s) whose manifest could not be read", skipped)
    return found


def rewarded_systems(coordinator_url: str, round_index: int) -> set[str]:
    published = _get(f"{coordinator_url}/v1/settlement/{round_index}")
    body = published.get("body", published)
    return {
        str(entry.get("system"))
        for entry in body.get("frontier", ())
        if float(entry.get("share") or 0.0) > 0.0
    }


def components_by_system(coordinator_url: str, round_index: int) -> dict[str, str]:
    published = _get(f"{coordinator_url}/v1/settlement/{round_index}")
    out: dict[str, str] = {}
    for entry in published.get("frontier", ()):
        front = dict(entry.get("components") or {}).get("front", "")
        if front:
            out[str(entry.get("system"))] = str(front).split(":", 1)[-1]
    return out


def _manifest_of(
    snapshot: Path | None, source: str, workdir: Path
) -> tuple[ArtifactManifest | None, bytes | None]:
    raw: bytes | None = None
    if snapshot is not None and (snapshot / "manifest.json").is_file():
        raw = (snapshot / "manifest.json").read_bytes()
    elif source:
        scheme, locator = parse_source(source)
        destination = workdir / f"{abs(hash(source))}.manifest.json"
        try:
            _fetch_one(
                fetcher_for(scheme),
                locator,
                "manifest.json",
                destination,
                attempts=3,
                timeout=60,
                sleep=time.sleep,
            )
            raw = destination.read_bytes()
        except Unfetchable as exc:
            log.warning("manifest for %s could not be fetched: %s", source, exc)
    if raw is None:
        return None, None
    try:
        return ArtifactManifest.from_json(raw), raw
    except Exception as exc:
        log.warning("manifest from %s is unreadable: %s", source or snapshot, exc)
        return None, None


def _decrypted(manifest: ArtifactManifest, object_dirs: list[Path]) -> Path | None:
    digest = str(manifest.artifact_digest).split(":", 1)[-1]
    for root in object_dirs:
        if (root / digest).is_dir():
            return root / digest
    return None


def candidates(
    server_url: str,
    coordinator_url: str,
    track: str,
    hardware_class: str,
    cache_dirs: list[Path],
    object_dirs: list[Path],
    sources: Mapping[str, str] | None = None,
    keys: Mapping[str, str] | None = None,
    board: Mapping[str, Any] | None = None,
    workdir: Path | None = None,
    only: set[str] | None = None,
) -> tuple[list[Candidate], list[str], int]:
    if board is None:
        board = _get(f"{server_url}/v1/arenas/{track}/{hardware_class}/leaderboard")
    round_index = int(board["round_index"])
    by_digest = snapshots(cache_dirs)
    fronts = components_by_system(coordinator_url, round_index)
    sources = dict(sources or {})
    keys = dict(keys or {})
    workdir = workdir or Path(tempfile.mkdtemp(prefix="mt-archive-"))

    kept: list[Candidate] = []
    missing: list[str] = []
    for system in board.get("systems", ()):
        system_id = str(system["system_id"])
        if only is not None:
            if system_id not in only:
                continue
        elif system.get("state") not in ARCHIVED_STATES:
            continue
        hotkey = str(system["hotkey"])
        has_manifest = True
        source = ""
        key = ""
        snapshot = next(
            (path for digest, path in by_digest.items() if digest.startswith(system_id)),
            None,
        )
        manifest, manifest_bytes = _manifest_of(snapshot, sources.get(hotkey, ""), workdir)
        if manifest is not None and manifest.sealed is not None:
            decrypted = _decrypted(manifest, object_dirs)
            if decrypted is not None:
                snapshot = decrypted
            elif sources.get(hotkey) and keys.get(hotkey):
                snapshot = None
                source = sources[hotkey]
                key = keys[hotkey]
            else:
                log.warning(
                    "%s is sealed and neither decrypted bytes nor a reveal key are at hand",
                    system_id,
                )
                missing.append(system_id)
                continue
        elif snapshot is None:
            front = fronts.get(system_id, "")
            for root in object_dirs:
                if front and (root / front).is_dir():
                    snapshot = root / front
                    has_manifest = False
                    break
        if snapshot is None and not key:
            missing.append(system_id)
            continue

        certificate = None
        try:
            certificate = _get(f"{server_url}/v1/certificates/{system_id}").get("document")
        except Exception as exc:
            log.warning("no certificate for %s: %s", system_id, exc)

        kept.append(
            Candidate(
                system_id=system_id,
                hotkey=hotkey,
                state=str(system["state"]),
                snapshot=snapshot,
                certificate=certificate,
                quality=system.get("quality"),
                expected_ms=system.get("expected_ms"),
                has_manifest=has_manifest,
                manifest=manifest,
                manifest_bytes=manifest_bytes,
                source=source,
                key=key,
            )
        )
    return kept, missing, round_index


def card(candidate: Candidate, track: str, hardware_class: str, round_index: int) -> str:
    cert = candidate.certificate
    tags = [
        "microtensor",
        f"arena-{track}-{hardware_class}",
        f"round-{round_index}",
        f"state-{candidate.state}",
    ]
    if cert and cert.get("quality") is not None:
        tags.append(f"quality-{cert['quality']}")

    lines = [
        "---",
        "tags:",
        *[f"- {tag}" for tag in tags],
        "---",
        "",
        f"# Microtensor archive · {track}/{hardware_class} · round {round_index}",
        "",
        "This repository is an archival copy of a system submitted to the",
        "Microtensor subnet (Bittensor netuid 92) and certified by its",
        "validators. The figures below were measured by the network on",
        "reference hardware. They are not self-reported.",
        "",
        f"- Miner hotkey: `{candidate.hotkey}`",
        f"- System digest: `{candidate.system_id}`",
        f"- Arena: {track} / {hardware_class}",
        f"- Round: {round_index}",
        f"- Standing this round: {candidate.state}",
        "",
    ]
    if cert:
        lines += [
            "## Measured record",
            "",
            f"- Quality: {cert.get('quality')}",
            f"- Expected cost: {cert.get('expected_ms')} ms per query",
            f"- Replication: {cert.get('replication')}",
            f"- Config hash: `{cert.get('config_hash')}`",
            f"- Reports root: `{cert.get('reports_root')}`",
            "",
            "The full signed record is in `certificate.json`. It is",
            "recomputable from the round's published reports.",
        ]
    else:
        lines += ["## Measured record", "", NO_CERTIFICATE]
    if candidate.has_manifest:
        lines += [
            "",
            "The manifest in `manifest.json` is the submission exactly as the",
            "miner shipped it; this repository's contents hash to the digest",
            "committed on chain for this round.",
            "",
        ]
    else:
        lines += [
            "",
            "The artifact was fetched over a scheme that left no manifest file",
            "on disk; the bytes here are the component the network verified by",
            "digest and measured, retained without the submitted manifest.",
            "",
        ]
    return "\n".join(lines)


def stage(
    candidate: Candidate,
    track: str,
    hardware_class: str,
    round_index: int,
    staging_root: Path,
) -> Path:
    name = repo_name(track, hardware_class, round_index, candidate.hotkey)
    target = staging_root / name
    if target.exists():
        shutil.rmtree(target)
    if candidate.snapshot is not None:
        shutil.copytree(candidate.snapshot, target, copy_function=_link_or_copy)
    else:
        _unseal_into(candidate, target)
    if candidate.manifest_bytes is not None and not (target / "manifest.json").is_file():
        (target / "manifest.json").write_bytes(candidate.manifest_bytes)

    if candidate.certificate is not None:
        (target / "certificate.json").write_text(
            json.dumps(candidate.certificate, indent=2, sort_keys=True), encoding="utf-8"
        )
    (target / "README.md").write_text(
        card(candidate, track, hardware_class, round_index), encoding="utf-8"
    )
    return target


def _unseal_into(candidate: Candidate, target: Path) -> None:
    manifest = candidate.manifest
    if manifest is None or manifest.sealed is None:
        raise RuntimeError(f"{candidate.system_id} has no sealed manifest to open")
    target.mkdir(parents=True, exist_ok=True)
    scheme, locator = parse_source(candidate.source)
    blob_name = str(manifest.sealed.get("blob", "artifact.enc"))
    blob_path = target / blob_name
    _fetch_one(
        fetcher_for(scheme),
        locator,
        blob_name,
        blob_path,
        attempts=3,
        timeout=600,
        sleep=time.sleep,
    )
    blob = blob_path.read_bytes()
    expected = str(manifest.sealed.get("digest", ""))
    if expected and digest_bytes(blob) != expected:
        raise RuntimeError(
            f"{candidate.system_id}: the ciphertext does not match the committed blob digest"
        )
    try:
        open_sealed(blob, candidate.key, target)
    except SealError as exc:
        raise RuntimeError(f"{candidate.system_id}: {exc}") from exc
    blob_path.unlink(missing_ok=True)
    ok, reason = verify_tree(target, manifest)
    if not ok:
        raise RuntimeError(
            f"{candidate.system_id}: decrypted tree does not match its manifest: {reason}"
        )


def _link_or_copy(src: str, dst: str) -> None:
    try:
        import os

        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def push(target: Path, org: str, token: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = f"{org}/{target.name}"
    api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(target))
    return repo_id


def run(
    *,
    server_url: str,
    coordinator_url: str,
    track: str,
    hardware_class: str,
    org: str,
    token: str,
    cache_dirs: list[Path],
    object_dirs: list[Path],
    staging_root: Path,
    dry_run: bool = False,
    sources: Mapping[str, str] | None = None,
    keys: Mapping[str, str] | None = None,
    board: Mapping[str, Any] | None = None,
    frontier_only: bool = False,
) -> int:
    kept, missing, round_index = candidates(
        server_url,
        coordinator_url,
        track,
        hardware_class,
        cache_dirs,
        object_dirs,
        sources=sources,
        keys=keys,
        board=board,
        only=(
            rewarded_systems(coordinator_url, int((board or {}).get("round_index") or 0))
            if frontier_only and board is not None
            else None
        ),
    )
    for system_id in missing:
        log.warning("no bytes for %s; it cannot be archived from here", system_id)

    archived = 0
    for candidate in kept:
        try:
            target = stage(candidate, track, hardware_class, round_index, staging_root)
        except (RuntimeError, Unfetchable) as exc:
            log.warning("%s was not archived: %s", candidate.system_id, exc)
            continue
        if dry_run:
            log.info("staged %s (dry run, not pushed)", target.name)
            continue
        repo_id = push(target, org, token)
        log.info(
            "archived %s as %s%s",
            candidate.system_id,
            repo_id,
            "" if candidate.certificate else " without a certificate",
        )
        archived += 1
    return archived
