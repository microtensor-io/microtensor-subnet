from __future__ import annotations

import re
import struct
import time
from pathlib import Path
from typing import Any, Final

from microtensor.core.protocol import ArtifactFormat, LoadManifest
from microtensor.core.tracks import Decoding
from microtensor.harness.contract import (
    EngineInfo,
    EngineLoadError,
    Request,
    Response,
    UnsupportedArtifact,
)

"""GGUF execution through llama.cpp.

Determinism is the whole difficulty. Two workers measuring the same system
must produce the same quality to the published quantum, so every source of
variation is pinned here rather than left to a default:

  * greedy decoding only, temperature zero, top_k one, every penalty off
  * one thread, so floating-point reductions happen in a fixed order
  * no GPU offload, because kernel selection varies by driver and card
  * a seed on every context, so a build that consults the RNG still agrees
  * a cache reset before every generation, so an answer never depends on
    which task ran before it
  * an exact llama-cpp-python pin in the `gguf` extra, which pins the
    llama.cpp build the numerics come from

What remains is CPU SIMD dispatch: llama.cpp selects AVX2 or AVX-512 kernels
by host, and the two do not reduce identically. That is the same exposure
onnxruntime carries and is answered the same way — the arena pins a certified
reference device, and quality is only compared between workers measuring on
it. An arena without a certified device has no business running either
engine.
"""

MAGIC: Final[bytes] = b"GGUF"
MIN_VERSION: Final[int] = 2
MAX_VERSION: Final[int] = 3
THREADS: Final[int] = 1
SEED: Final[int] = 0
GPU_LAYERS: Final[int] = 0
DEFAULT_CONTEXT: Final[int] = 2048

ARCH_KEY: Final[str] = "general.architecture"
MAX_KV_SCAN: Final[int] = 64

# What this engine will run, checked before llama.cpp opens the file.
#
# Declared rather than inferred, because "the pinned build happens to know
# this one" is not a promise anybody can read. A pin that cannot load an
# architecture named here fails tests/test_gguf_architectures.py, which is
# the whole reason that file exists: `unknown model architecture: 'qwen3'`
# was invisible until someone tried a real model.
SUPPORTED_ARCHITECTURES: Final[frozenset[str]] = frozenset(
    {
        "qwen2",
        "qwen3",
        "qwen3moe",
        "llama",
        "gemma2",
        "gemma3",
        "phi3",
    }
)

# GGUF metadata value types, by the tag written before each value.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_FIXED_WIDTH: Final[dict[int, int]] = {
    _UINT8: 1,
    _INT8: 1,
    _BOOL: 1,
    _UINT16: 2,
    _INT16: 2,
    _UINT32: 4,
    _INT32: 4,
    _FLOAT32: 4,
    _UINT64: 8,
    _INT64: 8,
    _FLOAT64: 8,
}

INFO = EngineInfo(
    format=ArtifactFormat.GGUF,
    name="llama-cpp",
    version="0.2.0",
    deterministic=True,
    notes=(
        f"gguf v{MIN_VERSION}-{MAX_VERSION}, {len(SUPPORTED_ARCHITECTURES)} architectures, "
        f"greedy only, {THREADS} thread, no gpu offload, seeded; numerics are "
        "pinned to the certified device"
    ),
)


def _llama() -> Any:
    import llama_cpp

    return llama_cpp


def _read(handle: Any, size: int) -> bytes:
    raw: bytes = handle.read(size)
    if len(raw) != size:
        raise UnsupportedArtifact("artifact ends inside its own header")
    return raw


def _u32(handle: Any) -> int:
    return int(struct.unpack("<I", _read(handle, 4))[0])


def _u64(handle: Any) -> int:
    return int(struct.unpack("<Q", _read(handle, 8))[0])


def _string(handle: Any) -> str:
    length = _u64(handle)
    if length > 1 << 20:
        raise UnsupportedArtifact("artifact declares an implausible metadata string")
    return _read(handle, length).decode("utf-8", "replace")


def _skip(handle: Any, kind: int) -> None:
    """Step over one metadata value without materialising it."""
    if kind in _FIXED_WIDTH:
        _read(handle, _FIXED_WIDTH[kind])
        return
    if kind == _STRING:
        _string(handle)
        return
    if kind == _ARRAY:
        element = _u32(handle)
        count = _u64(handle)
        if element == _ARRAY:
            raise UnsupportedArtifact("artifact nests metadata arrays")
        if element == _STRING:
            for _ in range(count):
                _string(handle)
            return
        width = _FIXED_WIDTH.get(element)
        if width is None:
            raise UnsupportedArtifact(f"artifact uses metadata type {element}")
        handle.seek(width * count, 1)
        return
    raise UnsupportedArtifact(f"artifact uses metadata type {kind}")


def read_architecture(path: Path) -> str:
    """The `general.architecture` the file declares, or "" if it names none.

    Read here rather than left to the loader: llama.cpp answers an
    architecture its build does not know with `unknown model architecture`
    from deep inside the C++, which reaches a miner as a crash rather than as
    a reason. Only the first few keys are scanned; the architecture is written
    near the front and a full metadata walk would read the tensor table too.
    """
    with path.open("rb") as handle:
        _read(handle, 8)
        _u64(handle)
        count = _u64(handle)

        for _ in range(min(count, MAX_KV_SCAN)):
            key = _string(handle)
            kind = _u32(handle)
            if key == ARCH_KEY:
                if kind != _STRING:
                    raise UnsupportedArtifact(f"{ARCH_KEY} is not a string")
                return _string(handle)
            _skip(handle, kind)

    return ""


def validate_artifact(path: Path) -> None:
    """Reject anything that is not a GGUF of a version the engine reads.

    Cheap and done before llama.cpp touches the file: a clear rejection at
    submission is worth more to a miner than a loader crash inside the jail.
    """
    try:
        with path.open("rb") as handle:
            header = handle.read(8)
    except OSError as exc:
        raise UnsupportedArtifact(f"artifact could not be read: {exc}") from exc

    if len(header) < 8 or header[:4] != MAGIC:
        raise UnsupportedArtifact("artifact is not a GGUF file")

    version = struct.unpack("<I", header[4:8])[0]
    if not MIN_VERSION <= version <= MAX_VERSION:
        raise UnsupportedArtifact(
            f"artifact is GGUF v{version}, outside the pinned v{MIN_VERSION}-v{MAX_VERSION} range"
        )

    architecture = read_architecture(path)
    if not architecture:
        raise UnsupportedArtifact(f"artifact declares no {ARCH_KEY}")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise UnsupportedArtifact(
            f"architecture {architecture!r} is not one this engine runs; "
            f"supported: {sorted(SUPPORTED_ARCHITECTURES)}"
        )


class GgufEngine:
    format = ArtifactFormat.GGUF

    def __init__(self, *, threads: int = THREADS, validate: bool = True) -> None:
        self._threads = threads
        self._validate = validate
        self._model: Any = None
        self._manifest: LoadManifest | None = None

    def load(self, artifact: Path, manifest: LoadManifest) -> None:
        weights = artifact / manifest.entrypoint if artifact.is_dir() else artifact
        if not weights.is_file():
            raise EngineLoadError(
                f"entrypoint {manifest.entrypoint!r} is missing from the artifact"
            )

        if self._validate:
            validate_artifact(weights)

        llama_cpp = _llama()
        context = int(manifest.max_input.get("tokens", DEFAULT_CONTEXT))

        try:
            self._model = llama_cpp.Llama(
                model_path=str(weights),
                n_ctx=context,
                n_threads=self._threads,
                n_threads_batch=self._threads,
                n_gpu_layers=GPU_LAYERS,
                seed=SEED,
                logits_all=False,
                embedding=False,
                verbose=False,
            )
        except Exception as exc:
            raise EngineLoadError(f"model could not be loaded: {exc}") from exc

        self._manifest = manifest

    # Qwen3 and other hybrid models emit a reasoning block before the answer.
    # It consumes the output budget and is not the answer, so it is removed
    # before scoring; thinking is also disabled at generation where the build
    # honours it, so the budget goes to the answer rather than the reasoning.
    _THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        cleaned = GgufEngine._THINK.sub("", text)
        # An unclosed think block (budget ran out mid-reasoning) leaves a
        # dangling open tag; drop everything from it, since no answer followed.
        head, _, _ = cleaned.partition("<think>")
        return head.strip() if "<think>" in cleaned else cleaned.strip()

    def generate(self, request: Request) -> Response:
        if self._model is None:
            return Response.failed(request.task_ref, "engine was asked to generate before load")
        if request.decoding is not Decoding.GREEDY:
            return Response.failed(
                request.task_ref,
                f"engine only supports greedy decoding, got {request.decoding.value}",
            )

        started = time.perf_counter()
        first_token_at = 0.0
        produced: list[str] = []
        count = 0

        try:
            # Clear the KV cache first. llama.cpp keeps it between calls and
            # prefix-matches against it, so a generation would otherwise
            # depend on whatever ran before it — the same prompt answered
            # differently on its second evaluation, and a system's score
            # depending on the order its tasks happened to run in. Two
            # validators drawing tasks in different orders would then disagree
            # for a reason no miner could act on.
            self._model.reset()

            # Streamed so the first token can be timed. Sampling is fixed to
            # argmax: temperature zero with top_k one leaves nothing for the
            # RNG to decide, whatever the build's sampler chain does.
            sampler = {
                "max_tokens": request.max_output_tokens,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "min_p": 0.0,
                "typical_p": 1.0,
                "repeat_penalty": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "mirostat_mode": 0,
                "seed": SEED,
                "stream": True,
            }

            # Instruction tasks go through the model's chat template so an
            # instruct model follows the prompt instead of continuing it.
            # Completion tasks (code) stay raw, which is the correct interface
            # for a function-completion benchmark.
            if request.chat:
                stream = self._chat_stream(request.prompt, sampler)
                delta_key = "delta"
            else:
                stream = self._model.create_completion(request.prompt, **sampler)
                delta_key = "text"

            for chunk in stream:
                choice = chunk["choices"][0]
                piece = (
                    choice.get("delta", {}).get("content", "")
                    if delta_key == "delta"
                    else choice.get("text", "")
                )
                if not piece:
                    continue
                if count == 0:
                    first_token_at = time.perf_counter()
                produced.append(piece)
                count += 1

            total = time.perf_counter()
        except Exception as exc:
            return Response.failed(
                request.task_ref,
                f"{type(exc).__name__}: {exc}",
                (time.perf_counter() - started) * 1000.0,
            )

        output = "".join(produced)
        if request.chat:
            output = self._strip_thinking(output)
        return Response(
            task_ref=request.task_ref,
            output=output,
            ttft_ms=(first_token_at - started) * 1000.0 if first_token_at else 0.0,
            total_ms=(total - started) * 1000.0,
            output_tokens=count,
        )

    def _chat_stream(self, prompt: str, sampler: dict[str, Any]) -> Any:
        """Chat completion with thinking disabled where the build supports it.

        `chat_template_kwargs` is passed when the installed llama-cpp-python
        accepts it, so Qwen3 skips its reasoning block and spends the budget on
        the answer. Older builds ignore the switch; the output is stripped of
        any think block regardless, so the answer is scored either way.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            return self._model.create_chat_completion(
                messages=messages,
                chat_template_kwargs={"enable_thinking": False},
                **sampler,
            )
        except TypeError:
            return self._model.create_chat_completion(messages=messages, **sampler)

    def unload(self) -> None:
        self._model = None
        self._manifest = None


def factory() -> Any:
    return GgufEngine()


def load() -> tuple[Any, EngineInfo]:
    _llama()
    return factory, INFO
