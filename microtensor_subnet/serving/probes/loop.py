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
ITEMS_PER_FETCH: Final[int] = 5
PROBED_STATES = ("collateralised", "probing", "admitted")
GPU_PROBED: Final[tuple[str, ...]] = ("serving", "probing", "failed_probe")
FORMAT: Final[str] = "format"
HIDDEN: Final[str] = "hidden"
CANARY: Final[str] = "canary"
KIND_WEIGHTS: Final[tuple[tuple[str, float], ...]] = ((HIDDEN, 0.8), (CANARY, 0.2))
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

    @property
    def reported(self) -> bool:
        return self.success is not None or bool(self.verdict)


def operators_for(server: str, wallet: Any) -> list[dict[str, Any]]:
    found = client._signed(server, wallet, method="GET", path=OPERATORS_PATH)
    return [dict(row) for row in (found or {}).get("operators") or []]


def withheld_for(server: str, wallet: Any, model: str, count: int = ITEMS_PER_FETCH) -> ProbeSet:
    found = client._signed(
        server, wallet, method="GET", path=ITEMS_PATH, query={"model": model, "count": count}
    )
    body = dict(found or {})
    held = {
        str(row["item_id"]): ProbeItem(
            item_id=str(row["item_id"]),
            family=str(row.get("family", "")),
            task=dict(row.get("task") or {}),
            label=dict(row.get("label") or {}),
        )
        for row in [*(body.get("items") or []), *(body.get("canaries") or [])]
        if isinstance(row, Mapping) and row.get("item_id")
    }
    canaries = [
        str(row["item_id"])
        for row in body.get("canaries") or []
        if isinstance(row, Mapping) and row.get("item_id")
    ]
    hidden = [i for i in held if i not in canaries]
    return ProbeSet(
        arena=str(body.get("arena", "")),
        epoch=int(body.get("epoch") or 0),
        artifact_sha="",
        model=model,
        grader=str(body.get("grader") or ""),
        right={"1.00": hidden},
        items=held,
        canaries=canaries,
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
    }
    return dict(client._signed(server, wallet, method="POST", path=PROBES_PATH, payload=payload))


def pick_kind(rng: random.Random) -> str:
    draw = rng.random() * sum(weight for _, weight in KIND_WEIGHTS)
    for kind, weight in KIND_WEIGHTS:
        draw -= weight
        if draw < 0:
            return kind
    return KIND_WEIGHTS[-1][0]


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
    pool = held.canary_items() if kind == CANARY else held.right_at(floor)
    if not pool:
        return Probe(hotkey, model, digest, "", None, "no item for this probe", gpu=gpu, kind=kind)
    item = rng.choice(pool)
    answered = asker(
        gateway, credential, hotkey=hotkey, model=model, task=item.task, floor=floor, gpu=gpu
    )
    if not answered.ok:
        return Probe(hotkey, model, digest, item.item_id, None, answered.error, gpu=gpu, kind=kind)
    timings = {
        "gpu": gpu,
        "kind": kind,
        "input_tokens": answered.prompt_tokens,
        "ttft_ms": answered.ttft_ms,
        "total_ms": answered.latency_ms,
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
            verdict="",
            reason=reason,
            **timings,
        )
        return Probe(hotkey, model, digest, item.item_id, False, reason, gpu=gpu, kind=kind)
    if not held.grader:
        reason = "the server named no grader for this model"
        return Probe(hotkey, model, digest, item.item_id, None, reason, gpu=gpu, kind=kind)
    try:
        graded_locally = grade.grade(
            held.grader, answered.answer, item.label, held.databases, task=item.task
        )
    except grade.GradeError as exc:
        success, reason = False, f"the answer could not be graded: {exc}"
    else:
        success, reason = graded_locally.success, graded_locally.reason
    verdict = verify.PASS if success else ""
    _send(
        reporter,
        server,
        wallet,
        operator=hotkey,
        model=model,
        failed=not success,
        verdict=verdict,
        reason=reason,
        **timings,
    )
    return Probe(hotkey, model, digest, item.item_id, success, reason, verdict, gpu=gpu, kind=kind)


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
            digest = digests.get(model, "")
            if rng.random() >= chance:
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
            results.append(
                probe_gpu(
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
                    kind=pick_kind(rng),
                    asker=asker,
                    reporter=reporter,
                )
            )
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
    models = {held.model for held in sets.values()}
    targets = dict(general or {})
    for row in lister(server, wallet):
        hotkey = str(row.get("hotkey", ""))
        if str(row.get("state", "")).lower() not in PROBED_STATES:
            continue
        for model, digest in _pairs(row):
            target = targets.get(model)
            if target is not None:
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
                continue
            if model not in models:
                continue
            held = sets.get(digest)
            if held is None:
                results.append(Probe(hotkey, model, digest, "", None, "no withheld items"))
                continue
            items = held.right_at(floor)
            if not items:
                results.append(Probe(hotkey, model, digest, "", None, "no item passed the floor"))
                continue
            for item in rng.sample(items, min(per_operator, len(items))):
                answered = asker(
                    gateway, credential, hotkey=hotkey, model=model, task=item.task, floor=floor
                )
                if not answered.ok:
                    results.append(Probe(hotkey, model, digest, item.item_id, None, answered.error))
                    continue
                if answered.artifact_digest and answered.artifact_digest.lower() != digest:
                    success, reason = False, "answered with another artifact"
                else:
                    verdict = grade.grade(held.grader, answered.answer, item.label, held.databases)
                    success, reason = verdict.success, verdict.reason
                _send(
                    reporter,
                    server,
                    wallet,
                    operator=hotkey,
                    model=model,
                    failed=not success,
                    verdict=verify.PASS if success else "",
                )
                results.append(
                    Probe(
                        hotkey,
                        model,
                        digest,
                        item.item_id,
                        success,
                        reason,
                        verify.PASS if success else "",
                    )
                )
                if pause:
                    time.sleep(pause)
    return results
