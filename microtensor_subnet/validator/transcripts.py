from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from microtensor_subnet.core.items import Item
from microtensor_subnet.scoring.stats import percentile
from microtensor_subnet.validator.context import (
    ESCALATION,
    HARNESS,
    SLM,
    SYSTEM,
    EpochContext,
)
from microtensor_subnet.validator.results import (
    ArtifactOutcome,
    Condition,
    EscalationRun,
    FloorRun,
    HarnessRun,
    ItemRun,
    SlmRun,
    SystemRun,
)

if TYPE_CHECKING:
    from microtensor_subnet.validator.items import LabelsBundle, RedoRow
    from microtensor_subnet.validator.replay import ReplayResult

log = logging.getLogger("microtensor_subnet.validator.transcripts")

VERSION: Final[str] = "t1"
KIND_SLM: Final[str] = "slm"
KIND_TOOL: Final[str] = "tool"
KIND_RELAY: Final[str] = "relay"
CONDITION_MINER: Final[str] = "miner"
CONDITION_REPAIR: Final[str] = "repair"
CONDITION_BARE: Final[str] = "B0"
SYSTEM_PREFIX: Final[str] = "system@"
MINER_CONDITION_NAME: Final[str] = "H"
GZIP_MAGIC: Final[bytes] = b"\x1f\x8b"
TRANSCRIPT_MODULE: Final[str] = "microtensor_subnet.core.transcript"
RECEIPTS_MODULE: Final[str] = "microtensor_subnet.validator.receipts"
PROFILES_MODULE: Final[str] = "microtensor_subnet.core.profiles"
BYTES_PER_WEIGHT: Final[dict[str, float]] = {
    "bf16": 2,
    "fp16": 2,
    "awq": 0.5,
    "gptq": 0.5,
    "gguf_q4": 0.5,
    "gguf_q8": 1,
}
TRANSCRIPTS_DIR: Final[str] = "transcripts"


def resolve(module: str, name: str, shim: str = "") -> Any:
    return getattr(importlib.import_module(module), name)


canonical_json: Callable[[Any], bytes] = resolve(TRANSCRIPT_MODULE, "canonical_json")
transcript_sha256: Callable[[Mapping[str, Any]], str] = resolve(
    TRANSCRIPT_MODULE, "transcript_sha256"
)
verify_signature: Callable[[Mapping[str, Any]], bool] = resolve(
    TRANSCRIPT_MODULE, "verify_signature"
)
under_root: Callable[[str, str, Mapping[str, str], str], bool] = resolve(
    TRANSCRIPT_MODULE, "verify_leaf"
)
merkle_root: Callable[[Mapping[str, str]], str] = resolve(TRANSCRIPT_MODULE, "merkle_root")
prompt_sha256: Callable[[str], str] = resolve(TRANSCRIPT_MODULE, "prompt_sha256")
item_seed: Callable[[str, str], int] = resolve(TRANSCRIPT_MODULE, "item_seed")
verify_receipt: Callable[[Mapping[str, Any], str], bool] = resolve(
    RECEIPTS_MODULE, "verify", "verify_receipt"
)
receipt_cost: Callable[[Sequence[Mapping[str, Any]]], int] = resolve(RECEIPTS_MODULE, "total_cost")
receipt_latency: Callable[[Sequence[Mapping[str, Any]]], int] = resolve(
    RECEIPTS_MODULE, "total_latency"
)


class TranscriptError(RuntimeError):
    pass


def client_method(client: Any, name: str) -> Callable[..., Any]:
    found = getattr(client, name)
    if not callable(found):
        raise TranscriptError(f"the server client has no {name}")
    return cast(Callable[..., Any], found)


def fetch(client: Any, sha256: str, into: Path) -> dict[str, Any]:
    target = into / TRANSCRIPTS_DIR / f"{sha256.lower()}.bin"
    try:
        path = Path(client_method(client, "mirror_blob")(sha256, target))
        raw = path.read_bytes()
    except Exception as exc:
        raise TranscriptError(f"transcript {sha256[:12]} could not be fetched: {exc}") from exc
    return parse(raw, sha256)


def parse(raw: bytes, sha256: str) -> dict[str, Any]:
    body = raw
    if body[:2] == GZIP_MAGIC:
        try:
            body = gzip.decompress(body)
        except (OSError, EOFError) as exc:
            raise TranscriptError("the transcript is not valid gzip") from exc
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TranscriptError("the transcript is not valid json") from exc
    if not isinstance(parsed, dict):
        raise TranscriptError("the transcript is not an object")
    if transcript_sha256(parsed) != sha256.lower():
        raise TranscriptError("the transcript does not hash to its receipt")
    return parsed


def verify(
    transcript: Mapping[str, Any],
    *,
    hotkey: str,
    arena_id: str,
    epoch_index: int,
    leaves: Mapping[str, str],
    root: str,
) -> str:
    if str(transcript.get("version", "")) != VERSION:
        return f"transcript version {transcript.get('version')!r} is not {VERSION!r}"
    if str(transcript.get("miner_hotkey", "")) != hotkey:
        return "transcript names another miner"
    if str(transcript.get("arena_id", "")) != arena_id:
        return "transcript names another arena"
    if int(transcript.get("epoch") or -1) != epoch_index:
        return "transcript names another epoch"
    if not isinstance(transcript.get("items"), list):
        return "transcript carries no items"
    if not verify_signature(transcript):
        return "miner signature does not verify"
    if not under_root(hotkey, transcript_sha256(transcript), leaves, root):
        return "transcript is not under the receipts root"
    return ""


def items_of(transcript: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in transcript.get("items") or [] if isinstance(row, Mapping)]


def calls_of(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(c) for c in item.get("calls") or [] if isinstance(c, Mapping)]


def slm_calls(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [c for c in calls_of(item) if str(c.get("kind", "")) == KIND_SLM]


def relay_receipts(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(c["receipt"])
        for c in calls_of(item)
        if str(c.get("kind", "")) == KIND_RELAY and isinstance(c.get("receipt"), Mapping)
    ]


def tokens_of(item: Mapping[str, Any]) -> tuple[int, int]:
    tokens_in = 0
    tokens_out = 0
    for call in slm_calls(item):
        tokens_in += int(call.get("prompt_tokens") or 0)
        tokens_out += len(call.get("output_tokens") or [])
    return tokens_in, tokens_out


def output_of(item: Mapping[str, Any]) -> Any:
    output = item.get("output")
    if isinstance(output, Mapping):
        return output.get("answer")
    return output


def output_text(item: Mapping[str, Any]) -> str:
    output = item.get("output")
    if isinstance(output, Mapping) and output.get("answer") is not None:
        return str(output.get("answer"))
    calls = slm_calls(item)
    return str(calls[0].get("output_text") or "") if calls else ""


def items_by_id(context: EpochContext) -> dict[str, Item]:
    found: dict[str, Item] = {}
    for chosen in context.items.by_test.values():
        for item in chosen:
            found.setdefault(item.item_id, item)
    return found


def floor_of(condition: str) -> float | None:
    if not condition.startswith(SYSTEM_PREFIX):
        return None
    try:
        return float(condition[len(SYSTEM_PREFIX) :])
    except ValueError:
        return None


def slm_drafts(transcript: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(row.get("item_id", "")): output_text(row)
        for row in items_of(transcript)
        if str(row.get("pool", "")) == SLM and str(row.get("condition", "")) == CONDITION_MINER
    }


def weight_format(slm_dir: Path) -> str:
    from microtensor_subnet.validator import checks

    kind = checks.weight_format(slm_dir).lower()
    if kind.startswith("gguf"):
        return "gguf_q8" if "q8" in kind else "gguf_q4"
    for method in ("awq", "gptq"):
        if method in kind:
            return method
    try:
        config = json.loads((slm_dir / "config.json").read_text(encoding="utf-8"))
        dtype = str(config.get("torch_dtype") or config.get("dtype") or "").lower()
    except (OSError, ValueError, AttributeError):
        dtype = ""
    return "fp16" if "float16" in dtype and "bfloat16" not in dtype else "bf16"


def computed_memory(
    parameters: int, kind: str, kv_cache: int, definition: Mapping[str, Any]
) -> int:
    block = definition.get("evaluation") or {}
    cost = block.get("computed_cost") if isinstance(block, Mapping) else None
    table = dict(BYTES_PER_WEIGHT)
    if isinstance(cost, Mapping) and isinstance(cost.get("bytes_per_weight"), Mapping):
        table.update({str(k): float(v) for k, v in cost["bytes_per_weight"].items()})
    return int(parameters * table.get(kind, 2.0)) + int(kv_cache)


def _graded(
    context: EpochContext, item: Item, output: Any, failure: str = "", **extra: Any
) -> ItemRun:
    if failure:
        return ItemRun(item.item_id, item.family, output, False, 0.0, failure, **extra)
    verdict = context.grade(item, output)
    return ItemRun(
        item.item_id, item.family, output, verdict.success, verdict.metric, verdict.reason, **extra
    )


def slm_run(
    context: EpochContext,
    transcript: Mapping[str, Any],
    replay: ReplayResult,
    *,
    parameters: int,
    memory_bytes: int,
    timing: Mapping[tuple[str, int], tuple[float, float]] | None = None,
    ttft_p95_ms: float = 0.0,
    tpot_p95_ms: float = 0.0,
    vram_peak_bytes: int = 0,
) -> SlmRun:
    known = items_by_id(context)
    runs: list[ItemRun] = []
    twins: list[ItemRun] = []
    measured = dict(timing or {})
    for row in items_of(transcript):
        if str(row.get("pool", "")) != SLM or str(row.get("condition", "")) != CONDITION_MINER:
            continue
        item = known.get(str(row.get("item_id", "")))
        if item is None:
            continue
        calls = slm_calls(row)
        first = calls[0] if calls else {}
        produced = len(first.get("output_tokens") or [])
        ttft, tpot = measured.get((item.item_id, int(first.get("seq") or 0)), (0.0, 0.0))
        run = _graded(
            context,
            item,
            output_text(row),
            replay.item_failures.get((item.item_id, CONDITION_MINER), ""),
            prompt_tokens=int(first.get("prompt_tokens") or 0),
            completion_tokens=produced,
            ttft_ms=ttft,
            tpot_ms=tpot,
            latency_ms=ttft + tpot * max(0, produced - 1),
        )
        (twins if item.twin_of else runs).append(run)
    return SlmRun(
        runs=runs,
        twins=twins,
        ttft_p95_ms=ttft_p95_ms,
        tpot_p95_ms=tpot_p95_ms,
        vram_peak_bytes=vram_peak_bytes or memory_bytes,
        parameters=parameters,
    )


def harness_run(
    context: EpochContext, transcript: Mapping[str, Any], replay: ReplayResult
) -> HarnessRun | None:
    known = items_by_id(context)
    conditions: dict[str, Condition] = {}
    for row in items_of(transcript):
        if str(row.get("pool", "")) != HARNESS:
            continue
        item = known.get(str(row.get("item_id", "")))
        if item is None:
            continue
        name = str(row.get("condition", ""))
        label = MINER_CONDITION_NAME if name == CONDITION_MINER else name
        condition = conditions.setdefault(label, Condition(name=label, runs=[], tokens=0))
        tokens_in, tokens_out = tokens_of(row)
        condition.tokens += tokens_in + tokens_out
        wall = replay.wall_ms.get((item.item_id, name), 0.0)
        condition.runs.append(
            _graded(
                context,
                item,
                output_of(row),
                replay.item_failures.get((item.item_id, name), ""),
                latency_ms=wall,
            )
        )
        if name == CONDITION_MINER:
            condition.added_ms.append(replay.added_ms.get(item.item_id, wall))
    miner = conditions.pop(MINER_CONDITION_NAME, None)
    if miner is None:
        return None
    baselines = [conditions[name] for name in sorted(conditions)]
    limit = float(context.harness_limits.get("added_wall_p95_ms") or 0)
    added = percentile(miner.added_ms, 0.95)
    limits_ok = not limit or added <= limit
    reason = "" if limits_ok else f"added wall time p95 {added:.0f} ms is over {limit:.0f} ms"
    over = sorted(replay.over_calls)
    if over:
        limits_ok = False
        reason = reason or f"model calls over the limit on {len(over)} items"
    return HarnessRun(miner=miner, baselines=baselines, limits_ok=limits_ok, limits_reason=reason)


def _cost(row: Mapping[str, Any], token_price: float) -> tuple[int, int, float]:
    receipts = relay_receipts(row)
    relay_cost = receipt_cost(receipts)
    relay_latency = float(receipt_latency(receipts))
    tokens_in, tokens_out = tokens_of(row)
    return relay_cost + round((tokens_in + tokens_out) * token_price), relay_cost, relay_latency


def escalation_run(
    context: EpochContext,
    transcript: Mapping[str, Any],
    labels: LabelsBundle,
    replay: ReplayResult,
    *,
    token_price: float,
) -> EscalationRun | None:
    known = items_by_id(context)
    redo = labels.rows(ESCALATION)
    repairs: list[ItemRun] = []
    covered: list[str] = []
    for row in items_of(transcript):
        if str(row.get("condition", "")) != CONDITION_REPAIR:
            continue
        item = known.get(str(row.get("item_id", "")))
        if item is None:
            continue
        cost, _, _ = _cost(row, token_price)
        repairs.append(
            _graded(
                context,
                item,
                output_of(row),
                replay.item_failures.get((item.item_id, CONDITION_REPAIR), ""),
                latency_ms=replay.wall_ms.get((item.item_id, CONDITION_REPAIR), 0.0),
                cost_micros=cost,
            )
        )
        covered.append(item.item_id)
    if not repairs:
        return None
    rows = [redo[i] for i in covered if i in redo]
    redo_pass = sum(1 for r in rows if r.correct) / len(rows) if rows else 0.0
    return EscalationRun(
        repairs=repairs,
        redo_pass_rate=redo_pass,
        redo_cost_micros=sum(r.cost_micros for r in rows),
    )


def system_run(
    context: EpochContext,
    transcript: Mapping[str, Any],
    labels: LabelsBundle,
    replay: ReplayResult,
    *,
    token_price: float,
) -> SystemRun | None:
    known = items_by_id(context)
    floors: dict[float, list[ItemRun]] = {}
    for row in items_of(transcript):
        condition = str(row.get("condition", ""))
        floor = floor_of(condition)
        if floor is None:
            continue
        item = known.get(str(row.get("item_id", "")))
        if item is None:
            continue
        cost, relay_cost, relay_latency = _cost(row, token_price)
        output = row.get("output")
        route = str(output.get("route", "")) if isinstance(output, Mapping) else ""
        probability = output.get("p") if isinstance(output, Mapping) else None
        floors.setdefault(floor, []).append(
            _graded(
                context,
                item,
                output_of(row),
                replay.item_failures.get((item.item_id, condition), ""),
                latency_ms=replay.wall_ms.get((item.item_id, condition), 0.0) + relay_latency,
                cost_micros=cost,
                escalated=route == "escalated" or relay_cost > 0,
                probability=float(probability)
                if isinstance(probability, (int, float)) and not isinstance(probability, bool)
                else None,
            )
        )
    if not floors:
        return None
    reference = labels.rows(SYSTEM)
    answered = list(reference.values())
    return SystemRun(
        floors=[FloorRun(floor=f, runs=floors[f]) for f in sorted(floors)],
        reference_cost_micros=sum(r.cost_micros for r in answered),
        reference_latency_p95_ms=percentile([r.latency_ms for r in answered], 0.95),
        reference_items=len(answered),
    )


def reference_slm_run(
    context: EpochContext,
    rows: Sequence[RedoRow],
    *,
    parameters: int,
    memory_bytes: int,
    ttft_p95_ms: float = 0.0,
    tpot_p95_ms: float = 0.0,
) -> SlmRun:
    known = items_by_id(context)
    runs: list[ItemRun] = []
    twins: list[ItemRun] = []
    for row in rows:
        item = known.get(str(row.item_id))
        if item is None:
            continue
        run = _graded(context, item, row.answer, ttft_ms=row.ttft_ms, tpot_ms=row.tpot_ms)
        (twins if item.twin_of else runs).append(run)
    return SlmRun(
        runs=runs,
        twins=twins,
        ttft_p95_ms=ttft_p95_ms or percentile([r.ttft_ms for r in runs if r.ttft_ms > 0], 0.95),
        tpot_p95_ms=tpot_p95_ms or percentile([r.tpot_ms for r in runs if r.tpot_ms > 0], 0.95),
        vram_peak_bytes=memory_bytes,
        parameters=parameters,
    )


def reference_escalation_run(context: EpochContext, labels: LabelsBundle) -> EscalationRun | None:
    known = items_by_id(context)
    repairs: list[ItemRun] = []
    redo = labels.rows(ESCALATION)
    covered: list[str] = []
    for item_id, row in labels.rows(CONDITION_REPAIR).items():
        item = known.get(item_id)
        if item is None:
            continue
        repairs.append(_graded(context, item, row.answer, cost_micros=row.cost_micros))
        covered.append(item_id)
    if not repairs:
        return None
    rows = [redo[i] for i in covered if i in redo]
    return EscalationRun(
        repairs=repairs,
        redo_pass_rate=sum(1 for r in rows if r.correct) / len(rows) if rows else 0.0,
        redo_cost_micros=sum(r.cost_micros for r in rows),
    )


def to_outcome(
    context: EpochContext,
    hotkey: str,
    transcript: Mapping[str, Any],
    labels: LabelsBundle,
    replay: ReplayResult,
    *,
    parameters: int,
    memory_bytes: int,
    token_price: float,
    timing: Mapping[tuple[str, int], tuple[float, float]] | None = None,
    ttft_p95_ms: float = 0.0,
    tpot_p95_ms: float = 0.0,
    vram_peak_bytes: int = 0,
) -> ArtifactOutcome:
    outcome = ArtifactOutcome(hotkey=hotkey)
    outcome.slm = slm_run(
        context,
        transcript,
        replay,
        parameters=parameters,
        memory_bytes=memory_bytes,
        timing=timing,
        ttft_p95_ms=ttft_p95_ms,
        tpot_p95_ms=tpot_p95_ms,
        vram_peak_bytes=vram_peak_bytes,
    )
    outcome.harness = harness_run(context, transcript, replay)
    outcome.escalation = escalation_run(
        context, transcript, labels, replay, token_price=token_price
    )
    outcome.system = system_run(context, transcript, labels, replay, token_price=token_price)
    outcome.receipts = [r for item in items_of(transcript) for r in relay_receipts(item)]
    return outcome


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
