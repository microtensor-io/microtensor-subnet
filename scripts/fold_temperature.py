from __future__ import annotations

import argparse
import math
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from microtensor.core.protocol import ArtifactFormat, LoadManifest
from microtensor.harness import decision_prompt
from microtensor.harness.engines.gguf import GgufEngine
from microtensor.scoring.metrics import expected_answers, gold_label
from microtensor.tasks.corpus import load_corpus

SAFE_ARCHITECTURES = frozenset({"qwen2", "qwen3", "llama"})
NORM_TENSOR = "output_norm.weight"
LOWEST = 0.05
HIGHEST = 20.0
STEPS = 80


class FoldError(RuntimeError):
    pass


def _normalise(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def collect(engine: GgufEngine, corpus: Path, track: str) -> list[tuple[list[float], int]]:
    samples: list[tuple[list[float], int]] = []
    for task in load_corpus(corpus, track).tasks:
        spec = task.inputs.get("decision")
        if spec is None:
            continue
        context, questions = decision_prompt.parse(spec)
        expected = expected_answers(task.gold)
        for question in questions:
            if question.name not in expected:
                continue
            target = gold_label(question.kind, expected[question.name])
            labels = [_normalise(label) for label in question.labels]
            if target not in labels:
                continue
            row = engine._decision_tokens(context, question)
            engine._model.reset()
            engine._model.eval(row)
            samples.append((engine._answer_scores(question), labels.index(target)))
    return samples


def log_loss(samples: Sequence[tuple[Sequence[float], int]], temperature: float) -> float:
    total = 0.0
    for scores, target in samples:
        scaled = [score / temperature for score in scores]
        top = max(scaled)
        log_partition = top + math.log(math.fsum(math.exp(value - top) for value in scaled))
        total += log_partition - scaled[target]
    return total / len(samples)


def fit(samples: Sequence[tuple[Sequence[float], int]]) -> float:
    if not samples:
        raise FoldError("no decision question in the train split had a usable gold answer")
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    low, high = math.log(LOWEST), math.log(HIGHEST)
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    at_left = log_loss(samples, math.exp(left))
    at_right = log_loss(samples, math.exp(right))
    for _ in range(STEPS):
        if at_left <= at_right:
            high, right, at_right = right, left, at_left
            left = high - ratio * (high - low)
            at_left = log_loss(samples, math.exp(left))
        else:
            low, left, at_left = left, right, at_right
            right = low + ratio * (high - low)
            at_right = log_loss(samples, math.exp(right))
    return math.exp((low + high) / 2.0)


def _field_text(reader: Any, key: str) -> str:
    field = reader.fields.get(key)
    if field is None:
        return ""
    return bytes(field.parts[field.data[0]]).decode("utf-8", "replace")


def fold(source: Path, target: Path, temperature: float) -> None:
    import gguf
    import numpy as np

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise FoldError(f"temperature {temperature} is not a positive number")

    reader = gguf.GGUFReader(str(source))
    architecture = _field_text(reader, "general.architecture")
    if architecture not in SAFE_ARCHITECTURES:
        raise FoldError(
            f"architecture {architecture!r} is not known to fold exactly; "
            f"folding is confirmed for {sorted(SAFE_ARCHITECTURES)}"
        )
    if any(name.endswith("final_logit_softcapping") for name in reader.fields):
        raise FoldError("this model caps its logits, so scaling the norm is not a temperature")
    names = {tensor.name for tensor in reader.tensors}
    if "output.bias" in names:
        raise FoldError("this model adds an output bias, so scaling the norm is not a temperature")

    norm = next((tensor for tensor in reader.tensors if tensor.name == NORM_TENSOR), None)
    if norm is None:
        raise FoldError(f"the model has no {NORM_TENSOR} to fold the temperature into")
    kinds = {gguf.GGMLQuantizationType.F32: np.float32, gguf.GGMLQuantizationType.F16: np.float16}
    kind = kinds.get(norm.tensor_type)
    if kind is None:
        stored = norm.tensor_type.name
        raise FoldError(f"{NORM_TENSOR} is stored as {stored}; only F32 and F16 fold")

    scaled = (np.asarray(norm.data, dtype=np.float64) / temperature).astype(kind)
    offset = int(norm.data_offset)
    del reader

    shutil.copyfile(source, target)
    with target.open("r+b") as handle:
        handle.seek(offset)
        handle.write(scaled.tobytes())

    check = gguf.GGUFReader(str(target))
    written = next(tensor for tensor in check.tensors if tensor.name == NORM_TENSOR)
    if not np.array_equal(np.asarray(written.data), scaled):
        raise FoldError("the folded norm did not read back as written")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit one temperature on the train split and fold it into a GGUF."
    )
    parser.add_argument("--model", type=Path, required=True, help="the GGUF front to calibrate")
    parser.add_argument("--out", type=Path, required=True, help="where to write the folded GGUF")
    parser.add_argument("--corpus", type=Path, help="train split JSONL with decision tasks")
    parser.add_argument("--track", default="classify", help="track the corpus belongs to")
    parser.add_argument("--context", type=int, default=4096, help="context window to load with")
    parser.add_argument("--temperature", type=float, help="skip fitting and fold this value")
    args = parser.parse_args(argv)

    if args.out.resolve() == args.model.resolve():
        parser.error("--out must differ from --model; the original is never modified")

    temperature = args.temperature
    if temperature is None:
        if args.corpus is None:
            parser.error("give --corpus to fit a temperature, or --temperature to fold one")
        engine = GgufEngine()
        engine.load(
            args.model,
            LoadManifest(
                format=ArtifactFormat.GGUF,
                quantization="",
                entrypoint=args.model.name,
                max_input={"tokens": args.context},
            ),
        )
        if engine._answer_fault:
            print(f"cannot calibrate: {engine._answer_fault}", file=sys.stderr)
            return 1
        samples = collect(engine, args.corpus, args.track)
        engine.unload()
        temperature = fit(samples)
        print(f"questions        {len(samples)}")
        print(f"log loss at T=1  {log_loss(samples, 1.0):.5f}")
        print(f"log loss at T    {log_loss(samples, temperature):.5f}")

    try:
        fold(args.model, args.out, temperature)
    except FoldError as exc:
        print(f"cannot fold: {exc}", file=sys.stderr)
        return 1
    print(f"temperature      {temperature:.5f}")
    print(f"written          {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
