from __future__ import annotations

import hashlib
import logging
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from microtensor_subnet.validator import transcripts
from microtensor_subnet.validator.context import SLM, EpochContext, render_prompt
from microtensor_subnet.validator.sandbox import SandboxError

log = logging.getLogger("microtensor_subnet.validator.rerun.calls")

REPLAY_SECONDS: Final[float] = 900.0
SAMPLE_BLOCKS: Final[int] = 10


@dataclass(frozen=True, slots=True)
class Drawn:
    row: int
    item_id: str
    condition: str
    pool: str
    seq: int
    call: Mapping[str, Any]

    @property
    def direct(self) -> bool:
        return self.pool == SLM or self.condition == transcripts.CONDITION_BARE


@dataclass(frozen=True, slots=True)
class Call:
    item_id: str
    condition: str
    seq: int
    prompt: str
    max_tokens: int
    stop: tuple[str, ...]
    seed: int
    claimed: tuple[int, ...]
    claimed_text: str = ""

    @property
    def prompt_sha256(self) -> str:
        return str(transcripts.prompt_sha256(self.prompt))

    def wire(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "condition": self.condition,
            "seq": self.seq,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "stop": list(self.stop),
            "seed": self.seed,
            "claimed": list(self.claimed),
            "claimed_text": self.claimed_text,
        }

    @classmethod
    def from_wire(cls, row: Mapping[str, Any]) -> Call:
        return cls(
            item_id=str(row.get("item_id", "")),
            condition=str(row.get("condition", "")),
            seq=int(row.get("seq") or 0),
            prompt=str(row.get("prompt", "")),
            max_tokens=int(row.get("max_tokens") or 0),
            stop=tuple(str(s) for s in row.get("stop") or []),
            seed=int(row.get("seed") or 0),
            claimed=tuple(int(t) for t in row.get("claimed") or []),
            claimed_text=str(row.get("claimed_text") or ""),
        )


@dataclass(slots=True)
class PacedServices:
    inner: Any
    deadline: float = 0.0

    def attach(self, key: str, session: Any) -> None:
        if self.deadline and time.monotonic() >= self.deadline:
            raise SandboxError("the replay ran past its time limit")
        self.inner.attach(key, session)

    def detach(self, key: str) -> None:
        self.inner.detach(key)


def seeded(*parts: str) -> random.Random:
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))  # noqa: S311


def faulted(call: Mapping[str, Any]) -> bool:
    return bool(call.get("fault")) or not call.get("output_tokens")


def eligible(transcript: Mapping[str, Any]) -> list[Drawn]:
    found: list[Drawn] = []
    for position, row in enumerate(transcripts.items_of(transcript)):
        for call in transcripts.slm_calls(row):
            if faulted(call):
                continue
            found.append(
                Drawn(
                    row=position,
                    item_id=str(row.get("item_id", "")),
                    condition=str(row.get("condition", "")),
                    pool=str(row.get("pool", "")),
                    seq=int(call.get("seq") or 0),
                    call=call,
                )
            )
    return found


def draw(
    transcript: Mapping[str, Any], transcript_sha256: str, block_hash: str, count: int
) -> list[Drawn]:
    pool = eligible(transcript)
    picked = seeded(transcript_sha256, block_hash).sample(pool, min(count, len(pool)))
    return sorted(picked, key=lambda drawn: (drawn.row, drawn.seq))


def harness_rows(transcript: Mapping[str, Any], drawn: Sequence[Drawn]) -> list[dict[str, Any]]:
    rows = transcripts.items_of(transcript)
    return [rows[position] for position in sorted({d.row for d in drawn if not d.direct})]


def replay_prompts(
    replayer: Any,
    transcript: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    artifact_dir: Path,
    has: frozenset[str],
    seconds: float = REPLAY_SECONDS,
) -> dict[tuple[Any, ...], str]:
    deadline = time.monotonic() + seconds
    replayer.services.deadline = deadline
    found: dict[tuple[Any, ...], str] = {}
    remaining = list(rows)
    while remaining and time.monotonic() < deadline:
        result = replayer.replay(
            {**transcript, "items": remaining}, artifact_dir=artifact_dir, has=has
        )
        found.update(result.prompts)
        if result.ok or not result.items:
            if not result.ok:
                log.warning("the replay stopped: %s", result.reason)
            break
        log.warning("a sampled row did not replay: %s", result.reason)
        remaining = remaining[result.items :]
    return found


def paired(
    context: EpochContext, drawn: Sequence[Drawn], replayed: Mapping[tuple[Any, ...], str]
) -> tuple[list[Call], int]:
    known = transcripts.items_by_id(context)
    found: list[Call] = []
    skipped = 0
    for sampled in drawn:
        item = known.get(sampled.item_id)
        prompt: str | None
        if sampled.direct:
            prompt = render_prompt(context.slm_prompt, item.task) if item is not None else None
        else:
            prompt = replayed.get(
                (sampled.item_id, sampled.condition, sampled.seq),
                replayed.get((sampled.item_id, sampled.seq)),
            )
        call = sampled.call
        if prompt is None or transcripts.prompt_sha256(prompt) != call.get("prompt_sha256"):
            skipped += 1
            continue
        found.append(
            Call(
                item_id=sampled.item_id,
                condition=sampled.condition,
                seq=sampled.seq,
                prompt=prompt,
                max_tokens=int(call.get("max_tokens") or 0),
                stop=tuple(str(s) for s in call.get("stop") or []),
                seed=int(call.get("seed") or 0),
                claimed=tuple(int(t) for t in call.get("output_tokens") or []),
                claimed_text=str(call.get("output_text") or ""),
            )
        )
    return found, skipped
