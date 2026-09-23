"""Canonical logits for a served sequence, from the certified artifact.

One prefill of prompt and completion through the validator's own copy of the
artifact yields the logits at every generated position. Prefill is parallel
and therefore far cheaper than the generation it checks, which is what makes
auditing a sample of live traffic affordable.

Alignment is the whole of the care here. Evaluating the sequence
p_0..p_{n-1} y_0..y_{m-1} gives one logit row per token, and row i predicts
token i+1. The row that decides y_0 is therefore the last prompt row, and the
row that decides y_k is the row of y_{k-1}. Off by one here does not fail
loudly: it silently scores every honest operator against the wrong position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from microtensor.serving.verify import VerificationError

THREADS: Final[int] = 1
SEED: Final[int] = 0
GPU_LAYERS: Final[int] = 0
DEFAULT_CONTEXT: Final[int] = 4096


def _llama() -> Any:
    import llama_cpp

    return llama_cpp


def open_artifact(path: Path, *, context: int = DEFAULT_CONTEXT, threads: int = THREADS) -> Any:
    """The certified artifact, opened to return logits at every position.

    `logits_all` is the one setting that differs from the measurement engine,
    which asks for the last row only. Everything else matches it, so the
    numbers the audit compares against come from the same configuration the
    certificate was measured under.
    """
    weights = Path(path)
    if not weights.is_file():
        raise VerificationError(f"no artifact at {weights}")
    llama_cpp = _llama()
    try:
        return llama_cpp.Llama(
            model_path=str(weights),
            n_ctx=context,
            n_threads=threads,
            n_threads_batch=threads,
            n_gpu_layers=GPU_LAYERS,
            seed=SEED,
            logits_all=True,
            embedding=False,
            verbose=False,
        )
    except Exception as exc:
        raise VerificationError(f"the artifact could not be opened: {exc}") from exc


def aligned_rows(scores: Any, prompt_length: int, completion_length: int) -> list[list[float]]:
    """The logit rows that decide each generated token, in order.

    `scores` is one row per evaluated token. Row `prompt_length - 1` decides
    the first generated token, and the rows for the generated tokens themselves
    decide the ones that follow, so the last generated token's own row is never
    read: nothing comes after it.
    """
    if prompt_length < 1:
        raise VerificationError("a prompt must carry at least one token to condition on")
    if completion_length < 1:
        return []
    first = prompt_length - 1
    last = first + completion_length
    total = len(scores)
    if last > total:
        raise VerificationError(
            f"the sequence needs {last} logit rows and the engine returned {total}; "
            "the context is probably shorter than prompt plus completion"
        )
    return [list(scores[index]) for index in range(first, last)]


def canonical_logits(
    model: Any, prompt_tokens: list[int], completion_tokens: list[int]
) -> list[list[float]]:
    """One prefill over the whole served sequence, returning the rows to score."""
    if not completion_tokens:
        return []
    if not prompt_tokens:
        raise VerificationError("a prompt must carry at least one token to condition on")
    sequence = list(prompt_tokens) + list(completion_tokens)
    try:
        model.reset()
        model.eval(sequence)
    except Exception as exc:
        raise VerificationError(f"the artifact could not evaluate the sequence: {exc}") from exc
    scores = getattr(model, "scores", None)
    if scores is None:
        raise VerificationError("the engine exposed no logits; it was not opened with logits_all")
    return aligned_rows(scores, len(prompt_tokens), len(completion_tokens))
