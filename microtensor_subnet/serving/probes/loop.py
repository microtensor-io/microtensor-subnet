from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from microtensor_subnet.core.hashing import DIGEST_PREFIX
from microtensor_subnet.serving import client, verify
from microtensor_subnet.serving.probes.ask import Answered, Chatted, ask, ask_chat
from microtensor_subnet.serving.probes.general import GeneralItem, Target, format_failure
from microtensor_subnet.serving.probes.sets import ProbeItem, ProbeSet
from microtensor_subnet.validator import grade

log = logging.getLogger("microtensor_subnet.serving.probes")

OPERATORS_PATH = "/v1/inference/validators/operators"
PROBES_PATH = "/v1/inference/validators/probes"
ITEMS_PATH = "/v1/inference/validators/probe-items"
ITEMS_PER_FETCH: Final[int] = 8
PROBED_STATES = ("collateralised", "probing", "admitted")
GPU_PROBED: Final[tuple[str, ...]] = ("serving", "probing", "paused")
ADMISSION: Final[tuple[str, ...]] = ("probing", "paused")
FORMAT: Final[str] = "format"
HIDDEN: Final[str] = "hidden"
EXACT: Final[str] = "exact"
CANARY: Final[str] = "canary"
KIND_WEIGHTS: Final[tuple[tuple[str, float], ...]] = ((HIDDEN, 0.6), (EXACT, 0.2), (CANARY, 0.2))
ADMISSION_PROBES: Final[int] = 5
QUALITY_FLOOR: Final[float] = 0.60
PROBES_PER_GPU_PER_DAY: Final[float] = 10.0
DAY_SECONDS: Final[float] = 86400.0
FAIL: Final[str] = "fail"
FAILING: Final[tuple[str, ...]] = (verify.CHEAT, FORMAT)
OUTCOME_ONLY: Final[str] = "outcome and format only; no validator class can prefill this model"
Lister = Callable[[str, Any], list[dict[str, Any]]]
Reporter = Callable[..., dict[str, Any]]
Asker = Callable[..., Answered]
ChatAsker = Callable[..., Chatted]


@dataclass(frozen=True, slots=True)
class Probe:
    operator: str
    model: str
    artifact_digest: str
    item_id: str
    success: bool | None
    reason: str = ""
    verdict: str = ""
    gpu: str = ""
    kind: str = ""
    quality: float = 0.0
    within_envelope: bool = True

    @property
    def reported(self) -> bool:
        return self.success is not None or bool(self.verdict)


def operators_for(server: str, wallet: Any) -> list[dict[str, Any]]:
    found = client._signed(server, wallet, method="GET", path=OPERATORS_PATH)
    return [dict(row) for row in (found or {}).get("operators") or []]


def _item(row: Mapping[str, Any], kind: str) -> ProbeItem:
    reference = row.get("reference")
    return ProbeItem(
        item_id=str(row["item_id"]),
        family=str(row.get("family", "")),
        task=dict(row.get("task") or {}),
        label=dict(row.get("label") or {}),
        kind=str(row.get("kind") or kind),
        reference=dict(reference) if isinstance(reference, Mapping) else {},
    )


def withheld_for(
    server: str, wallet: Any, model: str, count: int = ITEMS_PER_FETCH, kind: str = ""
) -> ProbeSet:
    query: dict[str, Any] = {"model": model, "count": count}
    if kind:
        query["kind"] = kind
    found = client._signed(server, wallet, method="GET", path=ITEMS_PATH, query=query)
    body = dict(found or {})
    held: dict[str, ProbeItem] = {}
    for row in body.get("items") or []:
        if isinstance(row, Mapping) and row.get("item_id"):
            held[str(row["item_id"])] = _item(row, HIDDEN)
    for row in body.get("canaries") or []:
        if isinstance(row, Mapping) and row.get("item_id"):
            held[str(row["item_id"])] = _item(row, CANARY)
    canaries = [i for i, item in held.items() if item.kind == CANARY]
    hidden = [i for i, item in held.items() if item.kind == HIDDEN]
    return ProbeSet(
        arena=str(body.get("arena", "")),
        epoch=int(body.get("epoch") or 0),
        artifact_sha="",
        model=model,
        grader=str(body.get("grader") or ""),
        right={"1.00": hidden},
        items=held,
        canaries=canaries,
        envelope=dict(body.get("envelope") or {}),
    )


def report(
    server: str,
    wallet: Any,
    *,
    operator: str,
    model: str,
    failed: bool,
    verdict: str = "",
    gpu: str = "",
    kind: str = "",
    reason: str = "",
    input_tokens: int = 0,
    ttft_ms: float = 0.0,
    total_ms: float = 0.0,
    quality: float = 0.0,
    within_envelope: bool = True,
    artifact_digest: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operator_hotkey": operator,
        "model": model,
        "failed": bool(failed),
        "verdict": verdict,
        "gpu": gpu,
        "kind": kind,
        "reason": reason[:200],
        "input_tokens": max(0, int(input_tokens)),
        "ttft_ms": max(0.0, float(ttft_ms)),
        "total_ms": max(0.0, float(total_ms)),
        "quality": round(min(1.0, max(0.0, float(quality))), 4),
        "within_envelope": bool(within_envelope),
        "artifact_digest": artifact_digest,
    }
    return dict(client._signed(server, wallet, method="POST", path=PROBES_PATH, payload=payload))


def pick_kind(rng: random.Random) -> str:
    draw = rng.random() * sum(weight for _, weight in KIND_WEIGHTS)
    for kind, weight in KIND_WEIGHTS:
        draw -= weight
        if draw < 0:
            return kind
    return KIND_WEIGHTS[-1][0]


def within(
    envelope: Mapping[str, Any], *, input_tokens: int, ttft_ms: float, total_ms: float
) -> bool:
    long_above = int(envelope.get("long_above_tokens") or 0)
    if long_above and input_tokens > long_above:
        ttft_limit = float(envelope.get("long_ttft_ms") or envelope.get("ttft_ms") or 0.0)
        total_limit = float(envelope.get("long_e2e_ms") or 0.0)
    else:
        ttft_limit = float(envelope.get("ttft_ms") or 0.0)
        total_limit = 0.0
    if ttft_limit and ttft_ms > ttft_limit:
        return False
    return not (total_limit and total_ms > total_limit)


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def judge(kind: str, item: ProbeItem, answered: Answered, held: ProbeSet) -> tuple[str, float, str]:
    if kind == EXACT:
        reference = item.reference
        if not reference:
            return verify.UNPROVEN, 0.0, "no reference run for this item"
        expected = tuple(int(t) for t in reference.get("completion_ids") or ())
        if expected and answered.completion_ids:
            same = tuple(answered.completion_ids) == expected
            return (verify.PASS if same else FAIL), float(same), ("" if same else "tokens differ")
        same = _text(answered.answer) == _text(reference.get("answer"))
        return (verify.PASS if same else FAIL), float(same), ("" if same else "the answer differs")
    if kind == CANARY:
        wanted = _text((item.reference or {}).get("answer") or item.label.get("answer"))
        if not wanted:
            return verify.UNPROVEN, 0.0, "the canary carries no known reply"
        same = _text(answered.answer) == wanted
        return (verify.PASS if same else FAIL), float(same), ("" if same else "the reply differs")
    if not held.grader:
        return verify.UNPROVEN, 0.0, "the server named no grader for this model"
    try:
        graded = grade.grade(
            held.grader, answered.answer, item.label, held.databases, task=item.task
        )
    except grade.GradeError as exc:
        return FAIL, 0.0, f"the answer could not be graded: {exc}"
    quality = float(graded.metric)
    if graded.success or quality >= QUALITY_FLOOR:
        return verify.PASS, max(quality, 1.0 if graded.success else quality), ""
    return FAIL, quality, graded.reason or f"quality {quality:.2f} is below {QUALITY_FLOOR:.2f}"


def _pairs(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for pair in row.get("models") or []:
        if not isinstance(pair, Mapping):
            continue
        model = str(pair.get("model", ""))
        digest = str(pair.get("artifact_digest", "")).lower()
        if model and digest:
            found.append((model, digest))
    return found


def _send(
    reporter: Reporter,
    server: str,
    wallet: Any,
    *,
    operator: str,
    model: str,
    failed: bool,
    verdict: str,
    **fields: Any,
) -> dict[str, Any] | None:
    try:
        return dict(
            reporter(
                server,
                wallet,
                operator=operator,
                model=model,
                failed=failed,
                verdict=verdict,
                **fields,
            )
            or {}
        )
    except client.ServerError as exc:
        log.warning("probe of %s not recorded: %s", operator[:12], exc)
        return None


def pool_for(held: ProbeSet, kind: str, floor: float) -> list[ProbeItem]:
    if kind == CANARY:
        return held.canary_items()
    if kind == EXACT:
        return held.exact_items()
    return [item for item in held.right_at(floor) if item.kind in ("", HIDDEN)]


def probe_gpu(
    *,
    server: str,
    gateway: str,
    credential: str,
    wallet: Any,
    hotkey: str,
    gpu: str,
    model: str,
    digest: str,
    held: ProbeSet,
    floor: float,
    rng: random.Random,
    kind: str = HIDDEN,
    asker: Asker = ask,
    reporter: Reporter = report,
) -> Probe:
    pool = pool_for(held, kind, floor)
    if not pool and kind == EXACT:
        kind, pool = HIDDEN, pool_for(held, HIDDEN, floor)
    if not pool:
        return Probe(hotkey, model, digest, "", None, "no item for this probe", gpu=gpu, kind=kind)
    item = rng.choice(pool)
    answered = asker(
        gateway,
        credential,
        hotkey=hotkey,
        model=model,
        task=item.task,
        floor=floor,
        gpu=gpu,
        kind=kind,
    )
    if not answered.ok:
        return Probe(hotkey, model, digest, item.item_id, None, answered.error, gpu=gpu, kind=kind)
    inside = within(
        held.envelope,
        input_tokens=answered.prompt_tokens,
        ttft_ms=answered.ttft_ms,
        total_ms=answered.latency_ms,
    )
    timings = {
        "gpu": gpu,
        "kind": kind,
        "input_tokens": answered.prompt_tokens,
        "ttft_ms": answered.ttft_ms,
        "total_ms": answered.latency_ms,
        "within_envelope": inside,
        "artifact_digest": answered.artifact_digest,
    }
    if answered.artifact_digest and digest and answered.artifact_digest.lower() != digest:
        reason = "answered with another artifact"
        _send(
            reporter,
            server,
            wallet,
            operator=hotkey,
            model=model,
            failed=True,
            verdict=FAIL,
            reason=reason,
            **timings,
        )
        return Probe(hotkey, model, digest, item.item_id, False, reason, FAIL, gpu, kind)
    verdict, quality, reason = judge(kind, item, answered, held)
    if verdict == verify.PASS and kind != EXACT and not inside:
        verdict, reason = FAIL, "outside the envelope"
    if verdict == verify.UNPROVEN:
        return Probe(
            hotkey, model, digest, item.item_id, None, reason, verdict, gpu, kind, quality, inside
        )
    failed = verdict == FAIL
    _send(
        reporter,
        server,
        wallet,
        operator=hotkey,
        model=model,
        failed=failed,
        verdict=verdict,
        reason=reason,
        quality=quality,
        **timings,
    )
    return Probe(
        hotkey, model, digest, item.item_id, not failed, reason, verdict, gpu, kind, quality, inside
    )


def admission_kinds(count: int = ADMISSION_PROBES) -> list[str]:
    return [HIDDEN] * max(0, count - 1) + [CANARY]


def _plan_card(card: Mapping[str, Any], rng: random.Random, chance: float) -> list[str]:
    status = str(card.get("status", ""))
    if card.get("due"):
        if status in ADMISSION:
            return admission_kinds()
        return [HIDDEN]
    if rng.random() >= chance:
        return []
    return [pick_kind(rng)]


def cycle_gpus(
    *,
    server: str,
    gateway: str,
    credential: str,
    wallet: Any,
    sets: Mapping[str, ProbeSet] | None,
    floor: float,
    chance: float,
    rng: random.Random,
    lister: Lister = operators_for,
    asker: Asker = ask,
    reporter: Reporter = report,
    fetcher: Callable[..., ProbeSet] = withheld_for,
) -> list[Probe]:
    results: list[Probe] = []
    fetched: dict[str, ProbeSet] = {}
    for row in lister(server, wallet):
        hotkey = str(row.get("hotkey", ""))
        if str(row.get("state", "")).lower() not in PROBED_STATES:
            continue
        digests = dict(_pairs(row))
        for card in row.get("gpus") or []:
            if not isinstance(card, Mapping):
                continue
            label = str(card.get("label", ""))
            model = str(card.get("model", ""))
            if not label or not model or str(card.get("status", "")) not in GPU_PROBED:
                continue
            digest = str(card.get("artifact_digest") or digests.get(model, "")).lower()
            kinds = _plan_card(card, rng, chance)
            if not kinds:
                continue
            held: ProbeSet | None
            if sets is not None:
                held = sets.get(digest)
            elif model in fetched:
                held = fetched[model]
            else:
                try:
                    held = fetcher(server, wallet, model)
                except client.ServerError as exc:
                    log.warning("no probe items for %s: %s", model, exc)
                    held = None
                else:
                    fetched[model] = held
            if held is None or not held.items:
                results.append(
                    Probe(hotkey, model, digest, "", None, "no withheld items", gpu=label)
                )
                continue
            for kind in kinds:
                found = probe_gpu(
                    server=server,
                    gateway=gateway,
                    credential=credential,
                    wallet=wallet,
                    hotkey=hotkey,
                    gpu=label,
                    model=model,
                    digest=digest,
                    held=held,
                    floor=floor,
                    rng=rng,
                    kind=kind,
                    asker=asker,
                    reporter=reporter,
                )
                results.append(found)
                if found.success is False or found.success is None:
                    break
    return results


def _judged(target: Target, item: GeneralItem, answered: Chatted) -> tuple[str, str]:
    reason = format_failure(item, answered)
    if reason:
        return FORMAT, reason
    if target.verifier is None:
        return verify.PASS, OUTCOME_ONLY
    judgement = asyncio.run(target.verifier.judge(item.request(), answered, target.calibration))
    return judgement.verdict, judgement.reason


def probe_general(
    *,
    server: str,
    gateway: str,
    credential: str,
    wallet: Any,
    hotkey: str,
    target: Target,
    digest: str,
    per_operator: int,
    rng: random.Random,
    pause: float = 0.0,
    chat_asker: ChatAsker = ask_chat,
    reporter: Reporter = report,
) -> list[Probe]:
    declared = digest.lower().removeprefix(DIGEST_PREFIX)
    if declared != target.digest:
        reason = "declared a digest the catalogue does not list"
        return [Probe(hotkey, target.model, digest, "", None, reason)]
    if not target.items:
        return [Probe(hotkey, target.model, digest, "", None, "no probe item fits the model")]
    results: list[Probe] = []
    for item in rng.sample(list(target.items), min(per_operator, len(target.items))):
        request = item.request()
        answered = chat_asker(
            gateway,
            credential,
            hotkey=hotkey,
            model=target.model,
            messages=request["messages"],
            tools=request["tools"] or None,
            max_tokens=item.max_tokens,
        )
        if not answered.ok:
            results.append(Probe(hotkey, target.model, digest, item.item_id, None, answered.error))
            continue
        served = answered.artifact_digest.lower().removeprefix(DIGEST_PREFIX)
        if served and served != target.digest:
            verdict, reason = verify.CHEAT, "answered with another artifact"
        else:
            verdict, reason = _judged(target, item, answered)
        failed = verdict in FAILING
        _send(
            reporter,
            server,
            wallet,
            operator=hotkey,
            model=target.model,
            failed=failed,
            verdict=verdict,
        )
        success = None if verdict == verify.UNPROVEN else not failed
        results.append(Probe(hotkey, target.model, digest, item.item_id, success, reason, verdict))
        if pause:
            time.sleep(pause)
    return results


def cycle(
    *,
    server: str,
    gateway: str,
    credential: str,
    wallet: Any,
    sets: Mapping[str, ProbeSet],
    floor: float,
    per_operator: int,
    rng: random.Random,
    pause: float = 0.0,
    lister: Lister = operators_for,
    asker: Asker = ask,
    reporter: Reporter = report,
    general: Mapping[str, Target] | None = None,
    chat_asker: ChatAsker = ask_chat,
) -> list[Probe]:
    results: list[Probe] = []
    targets = dict(general or {})
    rows = lister(server, wallet)
    for row in rows:
        hotkey = str(row.get("hotkey", ""))
        if str(row.get("state", "")).lower() not in PROBED_STATES:
            continue
        for model, digest in _pairs(row):
            target = targets.get(model)
            if target is None:
                continue
            results += probe_general(
                server=server,
                gateway=gateway,
                credential=credential,
                wallet=wallet,
                hotkey=hotkey,
                target=target,
                digest=digest,
                per_operator=per_operator,
                rng=rng,
                pause=pause,
                chat_asker=chat_asker,
                reporter=reporter,
            )
    if sets:
        results += cycle_gpus(
            server=server,
            gateway=gateway,
            credential=credential,
            wallet=wallet,
            sets=sets,
            floor=floor,
            chance=1.0,
            rng=rng,
            lister=lambda s, w: rows,
            asker=asker,
            reporter=reporter,
        )
    return results
