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
from microtensor_subnet.serving.probes.sets import ProbeSet
from microtensor_subnet.validator import grade

log = logging.getLogger("microtensor_subnet.serving.probes")

OPERATORS_PATH = "/v1/inference/validators/operators"
PROBES_PATH = "/v1/inference/validators/probes"
PROBED_STATES = ("collateralised", "probing", "admitted")
FORMAT: Final[str] = "format"
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

    @property
    def reported(self) -> bool:
        return self.success is not None or bool(self.verdict)


def operators_for(server: str, wallet: Any) -> list[dict[str, Any]]:
    found = client._signed(server, wallet, method="GET", path=OPERATORS_PATH)
    return [dict(row) for row in (found or {}).get("operators") or []]


def report(
    server: str, wallet: Any, *, operator: str, model: str, failed: bool, verdict: str = ""
) -> dict[str, Any]:
    payload = {
        "operator_hotkey": operator,
        "model": model,
        "failed": bool(failed),
        "verdict": verdict,
    }
    return dict(client._signed(server, wallet, method="POST", path=PROBES_PATH, payload=payload))


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
) -> None:
    try:
        reporter(server, wallet, operator=operator, model=model, failed=failed, verdict=verdict)
    except client.ServerError as exc:
        log.warning("probe of %s not recorded: %s", operator[:12], exc)


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
