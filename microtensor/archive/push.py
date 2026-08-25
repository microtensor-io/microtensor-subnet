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
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from microtensor.registry.manifest import ArtifactManifest

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
    snapshot: Path
    certificate: dict[str, Any] | None = None
    quality: float | None = None
    expected_ms: float | None = None
    tags: list[str] = field(default_factory=list)


def _get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as answer:  # noqa: S310
        return json.load(answer)


def repo_name(track: str, hardware_class: str, round_index: int, hotkey: str) -> str:
    klass = hardware_class.replace("mt-", "", 1)
    return f"mt-{track}-{klass}-r{round_index}-{hotkey[:8]}"


def snapshots(cache_dirs: list[Path]) -> dict[str, Path]:
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
                log.warning("unreadable manifest at %s: %s", manifest_path, exc)
                continue
            digest = manifest.digest().split(":", 1)[-1]
            found[digest] = manifest_path.parent
    return found


def candidates(
    server_url: str, track: str, hardware_class: str, cache_dirs: list[Path]
) -> tuple[list[Candidate], list[str], int]:
    board = _get(f"{server_url}/v1/arenas/{track}/{hardware_class}/leaderboard")
    round_index = int(board.get("round_index"))
    by_digest = snapshots(cache_dirs)

    kept: list[Candidate] = []
    missing: list[str] = []
    for system in board.get("systems", ()):
        if system.get("state") not in ARCHIVED_STATES:
            continue
        system_id = str(system["system_id"])
        snapshot = next(
            (path for digest, path in by_digest.items() if digest.startswith(system_id)),
            None,
        )
        if snapshot is None:
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
                hotkey=str(system["hotkey"]),
                state=str(system["state"]),
                snapshot=snapshot,
                certificate=certificate,
                quality=system.get("quality"),
                expected_ms=system.get("expected_ms"),
            )
        )
    return kept, missing, round_index


def card(
    candidate: Candidate, track: str, hardware_class: str, round_index: int
) -> str:
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
    lines += [
        "",
        "The manifest in `manifest.json` is the submission exactly as the",
        "miner shipped it; this repository's contents hash to the digest",
        "committed on chain for this round.",
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
    shutil.copytree(candidate.snapshot, target, copy_function=_link_or_copy)

    if candidate.certificate is not None:
        (target / "certificate.json").write_text(
            json.dumps(candidate.certificate, indent=2, sort_keys=True), encoding="utf-8"
        )
    (target / "README.md").write_text(
        card(candidate, track, hardware_class, round_index), encoding="utf-8"
    )
    return target


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
    track: str,
    hardware_class: str,
    org: str,
    token: str,
    cache_dirs: list[Path],
    staging_root: Path,
    dry_run: bool = False,
) -> int:
    kept, missing, round_index = candidates(server_url, track, hardware_class, cache_dirs)
    for system_id in missing:
        log.warning("no cached bytes for %s; it cannot be archived from here", system_id)

    archived = 0
    for candidate in kept:
        target = stage(candidate, track, hardware_class, round_index, staging_root)
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
