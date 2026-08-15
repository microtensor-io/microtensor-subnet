from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from microtensor.core.protocol import ArtifactFormat, LoadManifest
from microtensor.harness.contract import (
    EngineInfo,
    EngineLoadError,
    Request,
    Response,
)

INFO = EngineInfo(
    format=ArtifactFormat.SAFETENSORS,
    name="reference",
    version="0.1.0",
    deterministic=True,
    notes="deterministic stand-in used for dry runs and mechanism tests",
)


class ReferenceEngine:
    format = ArtifactFormat.SAFETENSORS

    def __init__(self, *, tokens_per_call: int = 32, delay_ms: float = 0.0) -> None:
        self._tokens = tokens_per_call
        self._delay_ms = delay_ms
        self._manifest: LoadManifest | None = None
        self._fingerprint = ""

    def load(self, artifact: Path, manifest: LoadManifest) -> None:
        if not artifact.exists():
            raise EngineLoadError(f"artifact {artifact} does not exist")
        self._manifest = manifest
        self._fingerprint = hashlib.sha256(
            f"{artifact.name}:{manifest.entrypoint}".encode()
        ).hexdigest()

    def generate(self, request: Request) -> Response:
        if self._manifest is None:
            return Response.failed(request.task_ref, "engine was asked to generate before load")

        started = time.perf_counter()
        if self._delay_ms:
            time.sleep(self._delay_ms / 1000.0)
        first = time.perf_counter()

        digest = hashlib.sha256(
            f"{self._fingerprint}:{request.nonce}:{request.prompt}".encode()
        ).hexdigest()
        tokens = min(self._tokens, request.max_output_tokens)
        total = time.perf_counter()

        return Response(
            task_ref=request.task_ref,
            output=digest[: max(1, tokens)],
            ttft_ms=(first - started) * 1000.0,
            total_ms=(total - started) * 1000.0,
            output_tokens=tokens,
        )

    def unload(self) -> None:
        self._manifest = None
        self._fingerprint = ""


def factory() -> Any:
    return ReferenceEngine()


def load() -> tuple[Any, EngineInfo]:
    return factory, INFO
