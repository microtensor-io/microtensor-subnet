from collections.abc import Callable
from typing import Any

from microtensor.core.protocol import ArtifactFormat
from microtensor.harness.engines import gguf, onnx, reference

# Each loader raises ImportError when its extra is not installed, and the
# registry logs and moves on. A validator installs only the formats it
# intends to measure; `mt inspect engines` reports what actually registered.
BUILTIN: tuple[tuple[ArtifactFormat, Callable[[], Any]], ...] = (
    (ArtifactFormat.ONNX, onnx.load),
    (ArtifactFormat.GGUF, gguf.load),
)

__all__ = ["BUILTIN", "gguf", "onnx", "reference"]
