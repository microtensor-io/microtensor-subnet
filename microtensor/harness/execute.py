from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from microtensor.core.protocol import ArtifactFormat, LoadManifest
from microtensor.harness.contract import Request, Response, batch
from microtensor.harness.registry import engine_for, load_builtin


def rebuild_manifest(payload: dict[str, Any]) -> LoadManifest:
    return LoadManifest(
        format=ArtifactFormat(payload["format"]),
        quantization=str(payload.get("quantization", "")),
        entrypoint=str(payload["entrypoint"]),
        max_input=dict(payload["max_input"]),
        preprocessing=dict(payload.get("preprocessing", {})),
        base_model=str(payload.get("base_model", "")),
    )


def run_tasks(
    artifact_path: str,
    manifest_payload: dict[str, Any],
    requests: Sequence[Request],
) -> list[Response]:
    load_builtin()
    manifest = rebuild_manifest(manifest_payload)
    engine = engine_for(manifest.format)
    engine.load(Path(artifact_path), manifest)
    try:
        return batch(requests, engine)
    finally:
        engine.unload()
