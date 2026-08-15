from __future__ import annotations

import logging
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
) -> ArtifactManifest:
    if not config.artifact_dir.is_dir():
        raise PackageError(f"{config.artifact_dir} is not a directory")

    hotkey = hotkey_address(wallet)
    staged = config.artifact_dir / MANIFEST_NAME
    staged.unlink(missing_ok=True)

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
        )
    except ManifestError as exc:
        raise PackageError(str(exc)) from exc

    fits, reason = manifest.fits_class()
    if not fits:
        raise PackageError(reason)

    signed = manifest.signed_with(sign_payload(wallet, manifest.body()))
    staged.write_bytes(signed.to_json())

    log.info(
        "packaged %d files, %.2f GiB, manifest %s",
        len(signed.files),
        signed.total_bytes / 1024**3,
        signed.digest()[:23],
    )
    return signed


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
    return [entry.path for entry in manifest.files] + [MANIFEST_NAME]


def upload_checklist(config: MinerConfig, manifest: ArtifactManifest) -> list[str]:
    return [f"{config.source}/{name}" for name in publishable_files(manifest)]
