from collections.abc import Callable
from typing import Any

from microtensor.core.protocol import ArtifactFormat
from microtensor.harness.engines import onnx, reference

BUILTIN: tuple[tuple[ArtifactFormat, Callable[[], Any]], ...] = (
    (ArtifactFormat.ONNX, onnx.load),
)

__all__ = ["BUILTIN", "onnx", "reference"]
