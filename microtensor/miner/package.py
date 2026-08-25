from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from microtensor.chain.wallet import hotkey_address, sign_payload
from microtensor.core.protocol import DeclaredEnvelope, LoadManifest
from microtensor.miner.config import MinerConfig
from microtensor.registry.manifest import ArtifactManifest, ManifestError, build_manifest

log = logging.getLogger("microtensor.miner.package")

MANIFEST_NAME = "manifest.json"


class PackageError(RuntimeError):
    pass


def package(
    config: MinerConfig,
    round_index: int,
    load: LoadManifest,
    declared: DeclaredEnvelope,
    wallet: Any,
    seal: bool = False,
) -> ArtifactManifest:
    if not config.artifact_dir.is_dir():
        raise PackageError(f"{config.artifact_dir} is not a directory")

    hotkey = hotkey_address(wallet)
    staged = config.artifact_dir / MANIFEST_NAME
    staged.unlink(missing_ok=True)

    sealed = None
    if seal:
        from microtensor.core.hashing import digest_bytes
        from microtensor.core.sealing import ALGORITHM, BLOB_NAME, seal_tree

        blob, key = seal_tree(config.artifact_dir)
        (config.artifact_dir / BLOB_NAME).write_bytes(blob)
        sealed = {
            "algorithm": ALGORITHM,
            "blob": BLOB_NAME,
            "digest": digest_bytes(blob),
        }

    try:
        manifest = build_manifest(
            config.artifact_dir,
            hotkey=hotkey,
            round_index=round_index,
            track=config.track,
            hardware_class=config.hardware_class,
            source=config.source,
            load=load,
            declared=declared,
            sealed=sealed,
        )
    except ManifestError as exc:
        raise PackageError(str(exc)) from exc

    fits, reason = manifest.fits_class()
    if not fits:
        raise PackageError(reason)

    signed = manifest.signed_with(sign_payload(wallet, manifest.body()))
    staged.write_bytes(signed.to_json())

    if seal:
        key_path = save_key(signed.digest(), round_index, key)
        log.info("sealed under a key held at %s; reveal it at the close block", key_path)

    log.info(
        "packaged %d files, %.2f GiB, manifest %s",
        len(signed.files),
        signed.total_bytes / 1024**3,
        signed.digest()[:23],
    )
    return signed


def reveal_dir() -> Path:
    home = Path(os.environ.get("MT_HOME", "~/.microtensor")).expanduser()
    return home / "reveal"


def save_key(manifest_digest: str, round_index: int, key: str) -> Path:
    from microtensor.chain.commitment import short_digest

    target = reveal_dir()
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"r{round_index}-{short_digest(manifest_digest)}.key"
    path.write_text(key, encoding="utf-8")
    path.chmod(0o600)
    return path


def load_key(manifest_digest: str, round_index: int) -> str | None:
    from microtensor.chain.commitment import short_digest

    path = reveal_dir() / f"r{round_index}-{short_digest(manifest_digest)}.key"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip()


def load_packaged(config: MinerConfig) -> ArtifactManifest:
    if not config.manifest_path.is_file():
        raise PackageError(
            f"no manifest at {config.manifest_path}; run `mt miner package` first"
        )
    try:
        return ArtifactManifest.from_json(config.manifest_path.read_bytes())
    except ManifestError as exc:
        raise PackageError(str(exc)) from exc


def publishable_files(manifest: ArtifactManifest) -> list[str]:
    if manifest.sealed:
        return [MANIFEST_NAME, str(manifest.sealed.get("blob", ""))]
    return [entry.path for entry in manifest.files] + [MANIFEST_NAME]


def upload_checklist(config: MinerConfig, manifest: ArtifactManifest) -> list[str]:
    return [f"{config.source}/{name}" for name in publishable_files(manifest)]
