from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Final

CHARS_PER_TOKEN: Final[int] = 4

# What a scoring round actually sends, measured across the corpus. The probe
# is floored here rather than at some small round number because prefill is
# most of the cost on single-threaded cpu inference, so a shorter probe does
# not measure a faster machine, it measures a smaller job.
#
# The floor was 256 chars, about 64 tokens, against this. A miner profiled a
# job an eighth the size of the one it would be scored on, declared the p95
# that came back, and was then held to it at full length. The error ran in
# the direction that zeroes honest miners.
SCORING_PROMPT_TOKENS: Final[int] = 541

CHARS_PER_TOKEN_SLACK: Final[int] = CHARS_PER_TOKEN
MIN_PROBE_CHARS: Final[int] = SCORING_PROMPT_TOKENS * CHARS_PER_TOKEN_SLACK
MAX_PROBE_CHARS: Final[int] = 1_000_000

_WORDS: Final[tuple[str, ...]] = (
    "ledger",
    "tensor",
    "quantise",
    "bandwidth",
    "residual",
    "throughput",
    "checkpoint",
    "kernel",
    "latency",
    "manifest",
    "envelope",
    "gradient",
    "attention",
    "embedding",
    "operator",
    "partition",
    "consensus",
    "artifact",
)


def declared_tokens(max_input: Mapping[str, Any]) -> int:
    for key in ("tokens", "context", "sequence_length", "max_tokens"):
        value = max_input.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return 0


def max_input_prompt(
    seed: str,
    max_input: Mapping[str, Any],
    *,
    chars_per_token: int = CHARS_PER_TOKEN,
) -> str:
    tokens = declared_tokens(max_input)
    target = max(MIN_PROBE_CHARS, min(tokens * chars_per_token, MAX_PROBE_CHARS))

    stream = hashlib.sha256(f"probe:{seed}".encode()).digest()
    parts: list[str] = []
    length = 0
    counter = 0

    while length < target:
        for byte in stream:
            word = _WORDS[byte % len(_WORDS)]
            parts.append(word)
            length += len(word) + 1
            if length >= target:
                break
        counter += 1
        stream = hashlib.sha256(f"probe:{seed}:{counter}".encode()).digest()

    return " ".join(parts)
