from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from microtensor.chain.commitment import Commitment, digest_matches
from microtensor.core.constants import MAX_MANIFEST_BYTES, MECHANISM_VERSION
from microtensor.core.hashing import canonical_hash, canonical_json, digest_entries, digest_file
from microtensor.core.protocol import ArtifactFormat, DeclaredEnvelope, LoadManifest
from microtensor.core.system import SystemManifest
from microtensor.core.tracks import get_class, is_competable


class ManifestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FileEntry:
    path: str
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.path or self.path.startswith("/") or ".." in Path(self.path).parts:
            raise ManifestError(f"file path {self.path!r} must be relative and contained")
        if not self.digest.startswith("sha256:"):
            raise ManifestError(f"file {self.path!r} carries no sha256 digest")
        if self.size_bytes < 0:
            raise ManifestError(f"file {self.path!r} declares a negative size")


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    hotkey: str
    round_index: int
    track: str
    hardware_class: str
    source: str
    files: tuple[FileEntry, ...]
    load: LoadManifest
    declared: DeclaredEnvelope
    system: SystemManifest | None = None
    sealed: dict[str, str] | None = None
    version: str = MECHANISM_VERSION
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.files:
            raise ManifestError("a manifest must list at least one file")
        if not is_competable(self.track, self.hardware_class):
            raise ManifestError(f"{self.track}/{self.hardware_class} is not an open competition")
        if len({f.path for f in self.files}) != len(self.files):
            raise ManifestError("a manifest must not list a path twice")
        if self.load.entrypoint not in {f.path for f in self.files}:
            raise ManifestError(
                f"entrypoint {self.load.entrypoint!r} is not among the listed files"
            )

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    @property
    def artifact_digest(self) -> str:
        return digest_entries([(f.path, f.digest) for f in self.files])

    def body(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "hotkey": self.hotkey,
            "round_index": self.round_index,
            "track": self.track,
            "hardware_class": self.hardware_class,
            "source": self.source,
            "artifact_digest": self.artifact_digest,
            "files": [asdict(f) for f in self.files],
            "load": self.load.to_dict(),
            "declared": asdict(self.declared),
            "system": self.system.body() if self.system else None,
            "sealed": dict(self.sealed) if self.sealed else None,
        }

    def digest(self) -> str:
        return canonical_hash(self.body())

    def to_json(self) -> bytes:
        payload = self.body()
        payload["signature"] = self.signature
        encoded = canonical_json(payload)
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise ManifestError(
                f"manifest is {len(encoded)} bytes, over the {MAX_MANIFEST_BYTES} byte limit"
            )
        return encoded

    def signed_with(self, signature: str) -> ArtifactManifest:
        return ArtifactManifest(
            hotkey=self.hotkey,
            round_index=self.round_index,
            track=self.track,
            hardware_class=self.hardware_class,
            source=self.source,
            files=self.files,
            load=self.load,
            declared=self.declared,
            system=self.system,
            version=self.version,
            signature=signature,
        )

    @property
    def system_digest(self) -> str:
        return self.system.digest() if self.system else self.artifact_digest

    def fits_class(self) -> tuple[bool, str]:
        hardware = get_class(self.hardware_class)
        if self.total_bytes > hardware.max_size_bytes:
            return False, (
                f"declared files total {self.total_bytes} bytes, over the "
                f"{hardware.max_size_bytes} byte ceiling for {self.hardware_class}"
            )
        if self.declared.size_bytes < self.total_bytes:
            return False, "declared size is below the sum of the listed files"
        if self.system is not None:
            placed, reason = self.system.fits_class(self.hardware_class)
            if not placed:
                return False, reason
        return True, ""

    def matches(self, commitment: Commitment) -> tuple[bool, str]:
        if commitment.round_index != self.round_index:
            return False, "commitment names a different round"
        if commitment.competition != (self.track, self.hardware_class):
            return False, "commitment names a different competition"
        if commitment.source != self.source:
            return False, "commitment names a different source"
        if not digest_matches(commitment.manifest_digest, self.digest()):
            return False, "manifest does not hash to the committed digest"
        return True, ""

    @classmethod
    def from_json(cls, raw: bytes | str) -> ArtifactManifest:
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ManifestError(f"manifest exceeds the {MAX_MANIFEST_BYTES} byte limit")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest is not valid json: {exc}") from exc
        if not isinstance(payload, dict):
            raise ManifestError("manifest must be a json object")

        try:
            load = payload["load"]
            declared = payload["declared"]
            manifest = cls(
                hotkey=str(payload["hotkey"]),
                round_index=int(payload["round_index"]),
                track=str(payload["track"]),
                hardware_class=str(payload["hardware_class"]),
                source=str(payload["source"]),
                sealed=(dict(payload["sealed"]) if payload.get("sealed") else None),
                files=tuple(
                    FileEntry(
                        path=str(f["path"]),
                        digest=str(f["digest"]),
                        size_bytes=int(f["size_bytes"]),
                    )
                    for f in payload["files"]
                ),
                load=LoadManifest(
                    format=ArtifactFormat(load["format"]),
                    quantization=str(load.get("quantization", "")),
                    entrypoint=str(load["entrypoint"]),
                    max_input=dict(load["max_input"]),
                    preprocessing=dict(load.get("preprocessing", {})),
                    base_model=str(load.get("base_model", "")),
                ),
                declared=DeclaredEnvelope(
                    size_bytes=int(declared["size_bytes"]),
                    peak_rss_bytes=int(declared["peak_rss_bytes"]),
                    p95_latency_ms=int(declared["p95_latency_ms"]),
                ),
                system=(
                    SystemManifest.from_dict(payload["system"])
                    if payload.get("system")
                    else None
                ),
                version=str(payload.get("version", MECHANISM_VERSION)),
                signature=str(payload.get("signature", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"manifest is malformed: {exc}") from exc

        declared_digest = payload.get("artifact_digest")
        if declared_digest and declared_digest != manifest.artifact_digest:
            raise ManifestError("artifact digest does not match the listed files")
        return manifest


MANIFEST_NAME = "manifest.json"
SEALED_BLOB_NAME = "artifact.enc"
EXCLUDED_PREFIXES = (".",)


def is_artifact_file(root: Path, path: Path) -> bool:
    if not path.is_file():
        return False
    parts = path.relative_to(root).parts
    if parts in ((MANIFEST_NAME,), (SEALED_BLOB_NAME,)):
        return False
    return not any(part.startswith(EXCLUDED_PREFIXES) for part in parts)


def build_manifest(
    root: Path,
    *,
    hotkey: str,
    round_index: int,
    track: str,
    hardware_class: str,
    source: str,
    load: LoadManifest,
    declared: DeclaredEnvelope,
    sealed: dict[str, str] | None = None,
) -> ArtifactManifest:
    if not root.is_dir():
        raise ManifestError(f"{root} is not an artifact directory")

    files = tuple(
        FileEntry(
            path=path.relative_to(root).as_posix(),
            digest=digest_file(path),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(p for p in root.rglob("*") if is_artifact_file(root, p))
    )
    if not files:
        raise ManifestError(f"{root} holds no artifact files to publish")
    return ArtifactManifest(
        hotkey=hotkey,
        round_index=round_index,
        track=track,
        hardware_class=hardware_class,
        source=source,
        files=files,
        load=load,
        declared=declared,
        sealed=sealed,
    )


def verify_tree(root: Path, manifest: ArtifactManifest) -> tuple[bool, str]:
    for entry in manifest.files:
        path = root / entry.path
        if not path.is_file():
            return False, f"{entry.path} is missing from the fetched artifact"
        if path.stat().st_size != entry.size_bytes:
            return False, f"{entry.path} has the wrong size"
        if digest_file(path) != entry.digest:
            return False, f"{entry.path} does not match its declared digest"

    extra = {
        p.relative_to(root).as_posix() for p in root.rglob("*") if is_artifact_file(root, p)
    }
    listed = {f.path for f in manifest.files}
    if extra - listed:
        return False, f"artifact carries unlisted files: {sorted(extra - listed)[:3]}"
    return True, ""
