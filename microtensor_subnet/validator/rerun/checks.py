from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Final, cast

from microtensor_subnet.scoring.stats import percentile
from microtensor_subnet.validator import transcripts
from microtensor_subnet.validator.context import SLM, EpochContext, render_prompt
from microtensor_subnet.validator.rerun.calls import Call
from microtensor_subnet.validator.rerun.engine import EngineError, EngineHang, Generated
from microtensor_subnet.validator.sandbox.slm import SlmError
from microtensor_subnet.validator.spotcheck import CANARY, SPOT_CHECK, ReferenceTiming, SpotCheck

log = logging.getLogger("microtensor_subnet.validator.rerun.checks")

CANARY_TOKENS: Final[int] = 48
MAX_CALL_ERRORS: Final[int] = 5
MAX_RESTARTS: Final[int] = 1
WINDOW_TOKENS: Final[int] = 64
TIMED_CALLS: Final[int] = 10


class EngineHung(EngineError):
    pass


def never() -> bool:
    return False


@dataclass(slots=True)
class Guard:
    engine: Any
    restarts: int = 0

    def hung(self) -> None:
        if self.restarts >= MAX_RESTARTS:
            raise EngineHung("the engine hung twice")
        self.restarts += 1
        log.warning("the engine hung; restarting it and trying again")
        self.engine.restart()

    def generate(
        self, prompt: str, *, max_tokens: int, stop: Sequence[str], seed: int
    ) -> Generated:
        while True:
            try:
                return cast(
                    Generated,
                    self.engine.generate(prompt, max_tokens=max_tokens, stop=stop, seed=seed),
                )
            except EngineHang:
                self.hung()


def each(work: Callable[[Any], None], tasks: Sequence[Any], *, width: int) -> None:
    pool = ThreadPoolExecutor(max_workers=max(1, width))
    try:
        for future in [pool.submit(work, task) for task in tasks]:
            future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _one_round(
    tasks: Sequence[Any], work: Callable[[Any], None], *, width: int, stop: Callable[[], bool]
) -> tuple[list[Any], bool]:
    hung = threading.Event()
    finished: set[int] = set()

    def one(position: int) -> None:
        if hung.is_set() or stop():
            return
        try:
            work(tasks[position])
        except EngineHang:
            hung.set()
        else:
            finished.add(position)

    each(one, list(range(len(tasks))), width=width)
    left = [task for position, task in enumerate(tasks) if position not in finished]
    return left, hung.is_set()


def over_all(
    guard: Guard,
    tasks: Sequence[Any],
    work: Callable[[Any], None],
    *,
    width: int,
    stop: Callable[[], bool] = never,
) -> None:
    pending = list(tasks)
    while pending and not stop():
        pending, hung = _one_round(pending, work, width=width, stop=stop)
        if hung:
            guard.hung()


@dataclass(slots=True)
class Passed:
    spot: SpotCheck
    mismatches: list[tuple[Call, Generated]] = field(default_factory=list)
    errors: int = 0
    e2e_p95_ms: float = 0.0
    memory_ok: bool = True


@dataclass(slots=True)
class Checked:
    spot: SpotCheck
    canary: dict[str, int]
    evidence: list[dict[str, Any]] = field(default_factory=list)
    flagged: list[Call] = field(default_factory=list)
    flagged_canaries: list[dict[str, Any]] = field(default_factory=list)
    unsettled: int = 0


def memory_cap(budgets: Mapping[str, Any]) -> int:
    return int(budgets.get("memory_bytes") or budgets.get("vram_bytes") or 0)


def _over_budget(generated: Generated, budgets: Mapping[str, Any]) -> bool:
    ttft = float(budgets.get("ttft_ms") or 0)
    tpot = float(budgets.get("tpot_ms") or 0)
    rate = float(budgets.get("min_tokens_per_s") or 0)
    held = budgets.get("long_input")
    long: Mapping[str, Any] = held if isinstance(held, Mapping) else {}
    above = int(long.get("above_tokens") or 0)
    long_ttft = float(long.get("ttft_p95_ms") or 0)
    long_e2e = float(long.get("e2e_p95_ms") or 0)
    slowest = 1000.0 / rate if rate > 0 else 0.0
    is_long = bool(above) and generated.prompt_tokens > above
    limit = long_ttft if is_long and long_ttft else ttft
    late = bool(limit) and generated.ttft_ms > limit
    late = late or (is_long and bool(long_e2e) and generated.total_ms > long_e2e)
    slow = len(generated.ids) > 1 and (
        (bool(tpot) and generated.tpot_ms > tpot) or (bool(slowest) and generated.tpot_ms > slowest)
    )
    return late or slow


def _run(guard: Guard, call: Call) -> Generated:
    try:
        return guard.generate(
            call.prompt, max_tokens=call.max_tokens, stop=call.stop, seed=call.seed
        )
    except SlmError as exc:
        raise EngineError(f"the engine failed on {call.item_id}: {exc}") from exc


def same_answer(call: Call, generated: Generated) -> bool:
    return generated.ids == call.claimed and generated.text == call.claimed_text


def timed_order(call: Call) -> str:
    return hashlib.sha256(f"{call.item_id}#{call.condition}#{call.seq}".encode()).hexdigest()


def first_pass(
    guard: Guard,
    sampled: Sequence[Call],
    *,
    budgets: Mapping[str, Any],
    parallel: int,
    measure: bool,
) -> Passed:
    engine = guard.engine
    found = SpotCheck()
    done = Passed(found)
    ttfts: list[float] = []
    tpots: list[float] = []
    totals: list[float] = []
    lock = threading.Lock()
    timed = sorted(sampled, key=timed_order)[:TIMED_CALLS] if measure else []
    rest = [call for call in sampled if call not in timed]

    def miss(call: Call, exc: Exception, counted: bool) -> None:
        log.warning("%s#%d not checked: %s", call.item_id, call.seq, exc)
        with lock:
            done.errors += 1
            if counted:
                found.uniform_calls += 1
                found.over_budget += 1
            if done.errors > MAX_CALL_ERRORS:
                raise EngineError(f"the engine failed {done.errors} calls") from exc

    def attempt(call: Call, counted: bool) -> Generated | None:
        try:
            return cast(
                Generated,
                engine.generate(
                    call.prompt, max_tokens=call.max_tokens, stop=call.stop, seed=call.seed
                ),
            )
        except EngineHang:
            raise
        except (SlmError, EngineError) as exc:
            miss(call, exc, counted)
            return None

    def tally(call: Call, generated: Generated) -> None:
        found.calls += 1
        if same_answer(call, generated):
            found.matched += 1
        else:
            done.mismatches.append((call, generated))

    def timed_work(call: Call) -> None:
        generated = attempt(call, True)
        if generated is None:
            return
        measured: Generated | None = generated
        if _over_budget(generated, budgets):
            measured = attempt(call, True)
        with lock:
            compared = generated
            if measured is not None and same_answer(call, generated):
                compared = measured
            tally(call, compared)
            if measured is None:
                return
            first = found.uniform_calls == 0
            found.uniform_calls += 1
            found.timing[(call.item_id, call.seq)] = (measured.ttft_ms, measured.tpot_ms)
            ttfts.append(measured.ttft_ms)
            totals.append(measured.total_ms)
            if len(measured.ids) > 1:
                tpots.append(measured.tpot_ms)
            if _over_budget(measured, budgets):
                found.over_budget += 1
        if first:
            engine.sample_vram()

    def plain_work(call: Call) -> None:
        generated = attempt(call, False)
        if generated is not None:
            with lock:
                tally(call, generated)

    over_all(guard, timed, timed_work, width=1)
    over_all(guard, rest, plain_work, width=parallel)
    if measure:
        found.vram_peak_bytes = engine.sample_vram()
        found.ttft_p95_ms = percentile(ttfts, 0.95)
        found.tpot_p95_ms = percentile(tpots, 0.95)
        done.e2e_p95_ms = percentile(totals, 0.95)
        cap = memory_cap(budgets)
        done.memory_ok = not cap or found.vram_peak_bytes <= cap
        found.envelope_ok = found.over_budget == 0 and done.memory_ok
    return done


def marker_of(canary: Mapping[str, Any]) -> str:
    fact = str(canary.get("fact", ""))
    return fact.rsplit(" ", 1)[-1].strip().lower() if fact else ""


def canary_pass(
    guard: Guard, canaries: Sequence[Mapping[str, Any]]
) -> tuple[int, list[Mapping[str, Any]]]:
    hits: list[Mapping[str, Any]] = []
    checked = 0
    for canary in canaries:
        prefix = str(canary.get("prefix", ""))
        marker = marker_of(canary)
        if not prefix or not marker:
            continue
        try:
            generated = guard.generate(prefix, max_tokens=CANARY_TOKENS, stop=(), seed=0)
        except SlmError as exc:
            log.warning("canary %s not checked: %s", canary.get("item_id"), exc)
            continue
        checked += 1
        if marker in generated.text.lower():
            hits.append(canary)
    return checked, hits


def token_sha256(ids: Sequence[int]) -> str:
    return hashlib.sha256(transcripts.canonical_json(list(ids))).hexdigest()


def answer_sha256(ids: Sequence[int], text: str) -> str:
    if not ids:
        return token_sha256(ids)
    return hashlib.sha256(transcripts.canonical_json([list(ids), text])).hexdigest()


def divergence(claimed: Sequence[int], first: Sequence[int]) -> int:
    for position, (left, right) in enumerate(zip(claimed, first, strict=False)):
        if left != right:
            return position
    return min(len(claimed), len(first))


def spot_entry(
    call: Call, generated: Generated, second: Generated, weights_sha256: str
) -> dict[str, Any]:
    first = generated.ids
    at = divergence(call.claimed, first)
    return {
        "kind": SPOT_CHECK,
        "item_id": call.item_id,
        "call_seq": call.seq,
        "condition": call.condition,
        "seed": call.seed,
        "prompt_sha256": call.prompt_sha256,
        "claimed_sha256": answer_sha256(call.claimed, call.claimed_text),
        "first_sha256": answer_sha256(first, generated.text),
        "second_sha256": answer_sha256(second.ids, second.text),
        "claimed_len": len(call.claimed),
        "first_len": len(first),
        "divergence": at,
        "claimed_window": list(call.claimed[at : at + WINDOW_TOKENS]),
        "first_window": list(first[at : at + WINDOW_TOKENS]),
        "weights_sha256": weights_sha256,
    }


def canary_entry(canary: Mapping[str, Any], weights_sha256: str) -> dict[str, Any]:
    return {
        "kind": CANARY,
        "item_id": str(canary.get("item_id", "")),
        "call_seq": 0,
        "seed": 0,
        "prefix_sha256": str(transcripts.sha256_text(str(canary.get("prefix", "")))),
        "first_hit": True,
        "second_hit": True,
        "weights_sha256": weights_sha256,
    }


def settle(
    guard: Guard,
    spot: SpotCheck,
    mismatches: Sequence[tuple[Call, Generated]],
    hits: Sequence[Mapping[str, Any]],
    *,
    checked_canaries: int,
    weights_sha256: str,
) -> Checked:
    done = Checked(spot, {"checked": checked_canaries, "hits": 0})
    if not mismatches and not hits:
        return done
    guard.engine.restart()
    for call, first in mismatches:
        try:
            second = _run(guard, call)
        except EngineHung:
            raise
        except EngineError as exc:
            log.warning("%s#%d not rerun: %s", call.item_id, call.seq, exc)
            done.unsettled += 1
            continue
        if same_answer(call, second):
            spot.matched += 1
            done.unsettled += 1
            continue
        if not first.ids or second.ids != first.ids or second.text != first.text:
            done.unsettled += 1
            continue
        spot.mismatched += 1
        done.flagged.append(call)
        done.evidence.append(spot_entry(call, first, second, weights_sha256))
    for canary in hits:
        prefix = str(canary.get("prefix", ""))
        try:
            second = guard.generate(prefix, max_tokens=CANARY_TOKENS, stop=(), seed=0)
        except SlmError as exc:
            log.warning("canary %s not rerun: %s", canary.get("item_id"), exc)
            done.unsettled += 1
            continue
        if marker_of(canary) not in second.text.lower():
            done.unsettled += 1
            continue
        done.canary["hits"] += 1
        done.flagged_canaries.append(dict(canary))
        done.evidence.append(canary_entry(canary, weights_sha256))
    return done


def time_reference(
    guard: Guard,
    context: EpochContext,
    *,
    count: int,
    drafts: Mapping[str, str],
    seeds: Mapping[str, int],
    stop: Callable[[], bool] = never,
) -> ReferenceTiming:
    ttfts: list[float] = []
    tpots: list[float] = []
    checked = 0
    mismatched = 0
    for item in context.items.originals(SLM)[: max(0, count)]:
        if stop():
            break
        prompt = render_prompt(context.slm_prompt, item.task)
        try:
            generated = guard.generate(
                prompt,
                max_tokens=context.max_tokens,
                stop=context.stop,
                seed=int(seeds.get(item.item_id, 0)),
            )
        except SlmError as exc:
            raise EngineError(f"the reference engine failed on {item.item_id}: {exc}") from exc
        ttfts.append(generated.ttft_ms)
        if len(generated.ids) > 1:
            tpots.append(generated.tpot_ms)
        draft = drafts.get(item.item_id)
        if draft is None:
            continue
        checked += 1
        if generated.text != draft:
            mismatched += 1
            log.warning("the reference draft for %s does not reproduce here", item.item_id)
    return ReferenceTiming(
        ttft_p95_ms=percentile(ttfts, 0.95),
        tpot_p95_ms=percentile(tpots, 0.95),
        vram_peak_bytes=guard.engine.sample_vram(),
        drafts_checked=checked,
        drafts_mismatched=mismatched,
    )


def wire_spot(
    spot: SpotCheck,
    *,
    unsettled: int = 0,
    errors: int = 0,
    skipped_prompt: int = 0,
    e2e_p95_ms: float = 0.0,
    memory_ok: bool = True,
    router_latency_p95_ms: float | None = None,
    router_reason: str = "",
) -> dict[str, Any]:
    return {
        "calls": spot.calls,
        "matched": spot.matched,
        "mismatched": spot.mismatched,
        "unsettled": unsettled,
        "errors": errors,
        "skipped_prompt": skipped_prompt,
        "ttft_p95_ms": round(spot.ttft_p95_ms, 1),
        "tpot_p95_ms": round(spot.tpot_p95_ms, 2),
        "e2e_p95_ms": round(e2e_p95_ms, 1),
        "vram_peak_bytes": spot.vram_peak_bytes,
        "uniform_calls": spot.uniform_calls,
        "over_budget": spot.over_budget,
        "envelope_ok": spot.envelope_ok,
        "memory_ok": memory_ok,
        "router_latency_p95_ms": router_latency_p95_ms,
        "router_reason": router_reason,
        "timing": [
            [item_id, seq, round(ttft, 3), round(tpot, 3)]
            for (item_id, seq), (ttft, tpot) in sorted(spot.timing.items())
        ],
    }
