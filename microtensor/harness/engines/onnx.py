from __future__ import annotations

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

MIN_OPSET: Final[int] = 13
MAX_OPSET: Final[int] = 21
ALLOWED_DOMAINS: Final[frozenset[str]] = frozenset({"", "ai.onnx", "ai.onnx.ml"})
LOGITS_OUTPUT: Final[str] = "logits"
INTRA_OP_THREADS: Final[int] = 1

INFO = EngineInfo(
    format=ArtifactFormat.ONNX,
    name="onnxruntime",
    version="0.1.0",
    deterministic=True,
    notes=f"opset {MIN_OPSET}-{MAX_OPSET}, standard domains only, single intra-op thread",
)


def _onnxruntime() -> Any:
    import onnxruntime

    return onnxruntime


def _numpy() -> Any:
    import numpy

    return numpy


def validate_graph(path: Path) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise EngineLoadError("onnx is required to validate a graph before execution") from exc

    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:
        raise UnsupportedArtifact(f"graph could not be parsed: {exc}") from exc

    for entry in model.opset_import:
        domain = entry.domain or ""
        if domain not in ALLOWED_DOMAINS:
            raise UnsupportedArtifact(
                f"graph imports custom operator domain {domain!r}; only standard domains run"
            )
        if domain in ("", "ai.onnx") and not MIN_OPSET <= entry.version <= MAX_OPSET:
            raise UnsupportedArtifact(
                f"graph targets opset {entry.version}, outside the pinned "
                f"{MIN_OPSET}-{MAX_OPSET} range"
            )

    for node in model.graph.node:
        if (node.domain or "") not in ALLOWED_DOMAINS:
            raise UnsupportedArtifact(
                f"node {node.name or node.op_type} comes from custom domain {node.domain!r}"
            )


class OnnxEngine:
    format = ArtifactFormat.ONNX

    def __init__(self, *, threads: int = INTRA_OP_THREADS, validate: bool = True) -> None:
        self._threads = threads
        self._validate = validate
        self._session: Any = None
        self._tokenizer: Any = None
        self._manifest: LoadManifest | None = None
        self._input_names: tuple[str, ...] = ()
        self._past_inputs: tuple[str, ...] = ()
        self._eos_id: int | None = None

    def load(self, artifact: Path, manifest: LoadManifest) -> None:
        graph = artifact / manifest.entrypoint if artifact.is_dir() else artifact
        if not graph.is_file():
            raise EngineLoadError(
                f"entrypoint {manifest.entrypoint!r} is missing from the artifact"
            )

        if self._validate:
            validate_graph(graph)

        onnxruntime = _onnxruntime()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = self._threads
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.enable_mem_pattern = False

        try:
            self._session = onnxruntime.InferenceSession(
                str(graph), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise EngineLoadError(f"session could not be created: {exc}") from exc

        self._manifest = manifest
        self._input_names = tuple(i.name for i in self._session.get_inputs())
        self._past_inputs = tuple(n for n in self._input_names if n.startswith("past_key_values"))
        self._tokenizer = self._load_tokenizer(artifact, manifest)
        self._eos_id = self._resolve_eos(manifest)

    def _load_tokenizer(self, artifact: Path, manifest: LoadManifest) -> Any:
        spec = manifest.preprocessing.get("tokenizer")
        if not spec:
            raise EngineLoadError("manifest must name a tokenizer under preprocessing")

        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise EngineLoadError("tokenizers is required to run a text artifact") from exc

        candidate = artifact / spec if artifact.is_dir() else artifact.parent / spec
        if not candidate.is_file():
            raise EngineLoadError(f"tokenizer {spec!r} is missing from the artifact")
        try:
            return Tokenizer.from_file(str(candidate))
        except Exception as exc:
            raise EngineLoadError(f"tokenizer could not be loaded: {exc}") from exc

    def _resolve_eos(self, manifest: LoadManifest) -> int | None:
        value = manifest.preprocessing.get("eos_token_id")
        return int(value) if isinstance(value, int) else None

    def _empty_past(self, batch: int) -> dict[str, Any]:
        numpy = _numpy()
        past: dict[str, Any] = {}
        for entry in self._session.get_inputs():
            if not entry.name.startswith("past_key_values"):
                continue
            shape = [d if isinstance(d, int) else 0 for d in entry.shape]
            if shape:
                shape[0] = batch
            past[entry.name] = numpy.zeros(shape, dtype=numpy.float32)
        return past

    def generate(self, request: Request) -> Response:
        if self._session is None or self._tokenizer is None:
            return Response.failed(request.task_ref, "engine was asked to generate before load")
        if request.decoding is not Decoding.GREEDY:
            return Response.failed(
                request.task_ref,
                f"engine only supports greedy decoding, got {request.decoding.value}",
            )

        numpy = _numpy()
        started = time.perf_counter()
        first_token_at = 0.0

        try:
            tokens = list(self._tokenizer.encode(request.prompt).ids)
            produced: list[int] = []
            past = self._empty_past(1) if self._past_inputs else {}
            cursor = tokens

            for step in range(request.max_output_tokens):
                feed: dict[str, Any] = {}
                if "input_ids" in self._input_names:
                    feed["input_ids"] = numpy.asarray([cursor], dtype=numpy.int64)
                if "attention_mask" in self._input_names:
                    length = len(tokens) + len(produced)
                    feed["attention_mask"] = numpy.ones((1, length), dtype=numpy.int64)
                if "position_ids" in self._input_names:
                    offset = len(tokens) + len(produced) - len(cursor)
                    feed["position_ids"] = numpy.asarray(
                        [list(range(offset, offset + len(cursor)))], dtype=numpy.int64
                    )
                feed.update(past)

                outputs = self._session.run(None, feed)
                logits = outputs[0]
                next_id = int(numpy.argmax(logits[0, -1]))

                if step == 0:
                    first_token_at = time.perf_counter()

                produced.append(next_id)
                if self._eos_id is not None and next_id == self._eos_id:
                    break

                past = self._carry_past(outputs)
                cursor = [next_id] if past else tokens + produced

            total = time.perf_counter()
            text = self._tokenizer.decode(produced)
        except Exception as exc:
            return Response.failed(
                request.task_ref,
                f"{type(exc).__name__}: {exc}",
                (time.perf_counter() - started) * 1000.0,
            )

        return Response(
            task_ref=request.task_ref,
            output=text,
            ttft_ms=(first_token_at - started) * 1000.0 if first_token_at else 0.0,
            total_ms=(total - started) * 1000.0,
            output_tokens=len(produced),
        )

    def _carry_past(self, outputs: list[Any]) -> dict[str, Any]:
        if not self._past_inputs:
            return {}
        names = [o.name for o in self._session.get_outputs()]
        present = {
            name.replace("present", "past_key_values", 1): value
            for name, value in zip(names, outputs, strict=False)
            if name.startswith("present")
        }
        return {name: present[name] for name in self._past_inputs if name in present}

    def unload(self) -> None:
        self._session = None
        self._tokenizer = None
        self._manifest = None
        self._input_names = ()
        self._past_inputs = ()


def factory() -> Any:
    return OnnxEngine()


def load() -> tuple[Any, EngineInfo]:
    _onnxruntime()
    return factory, INFO
