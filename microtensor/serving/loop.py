from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from microtensor.serving import probe, verify
from microtensor.serving.client import ServerError

log = logging.getLogger("microtensor.serving.loop")

ADMITTED = "admitted"
REJECTED = "rejected"
OPEN = "open"

PROBE_PAUSE_SECONDS = 1.0
CYCLE_PAUSE_SECONDS = 30.0


@dataclass(slots=True)
class Counted:
    probed: int = 0
    passed: int = 0
    cheated: int = 0
    unproven: int = 0
    unreachable: int = 0
    withdrawn: int = 0
    decided: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probed": self.probed,
            "passed": self.passed,
            "cheated": self.cheated,
            "unproven": self.unproven,
            "unreachable": self.unreachable,
            "withdrawn": self.withdrawn,
            "decided": dict(self.decided),
        }


def _open_pairs(row: dict[str, Any], validator: str) -> list[str]:
    decided = {
        str(p.get("model", "")): str(p.get("state", OPEN))
        for p in row.get("probes", [])
        if str(p.get("validator_hotkey", "")) == validator
    }
    wanted: list[str] = []
    for pair in row.get("models", []):
        model = str(pair.get("model", ""))
        if not model:
            continue
        if decided.get(model) == REJECTED:
            continue
        wanted.append(model)
    return wanted


def cycle(
    *,
    server: str,
    gateway: str,
    credential: str,
    wallet: Any,
    artifacts: dict[str, Path],
    calibrations: dict[str, verify.Calibration],
    model: str = "",
    per_operator: int = 8,
    rng: random.Random | None = None,
    pause: float = PROBE_PAUSE_SECONDS,
) -> Counted:
    from microtensor.chain.wallet import hotkey_address

    validator = hotkey_address(wallet)
    chance = rng or random.SystemRandom()
    tally = Counted()

    try:
        rows = probe.to_probe(server, wallet, model=model)
    except ServerError as exc:
        log.warning("could not read the operator list: %s", exc)
        return tally

    opened: dict[str, Any] = {}

    for row in rows:
        hotkey = str(row.get("hotkey", ""))
        for wanted in _open_pairs(row, validator):
            if model and wanted != model:
                continue
            calibration = calibrations.get(wanted)
            artifact = artifacts.get(wanted)
            if calibration is None or artifact is None:
                log.info("no calibrated artifact for %s; skipping", wanted)
                continue
            if wanted not in opened:
                opened[wanted] = probe.artifact_model(artifact)

            for _ in range(per_operator):
                verdict = _one(
                    server=server,
                    gateway=gateway,
                    credential=credential,
                    wallet=wallet,
                    engine=opened[wanted],
                    calibration=calibration,
                    hotkey=hotkey,
                    model=wanted,
                    prompt=probe.prompt_for(chance),
                    tally=tally,
                )
                if verdict is None:
                    break
                if verdict.get("probe", {}).get("state") in (ADMITTED, REJECTED):
                    tally.decided[f"{hotkey}:{wanted}"] = str(verdict["probe"]["state"])
                    break
                if pause:
                    time.sleep(pause)

    return tally


def _one(
    *,
    server: str,
    gateway: str,
    credential: str,
    wallet: Any,
    engine: Any,
    calibration: verify.Calibration,
    hotkey: str,
    model: str,
    prompt: str,
    tally: Counted,
) -> dict[str, Any] | None:
    try:
        answered = probe.ask(gateway, credential, hotkey=hotkey, model=model, prompt=prompt)
    except ServerError as exc:
        tally.unreachable += 1
        log.info("operator %s did not answer: %s", hotkey[:12], exc)
        return None

    found = probe.judge(engine, answered, calibration)
    tally.probed += 1

    if found.verdict == verify.CHEAT:
        tally.cheated += 1
    elif found.verdict == verify.PASS:
        tally.passed += 1
    else:
        tally.unproven += 1
        log.debug("unproven for %s on %s: %s", hotkey[:12], model, found.reason)
        return {"probe": {"state": OPEN}}

    try:
        reported = probe.report(
            server,
            wallet,
            hotkey=hotkey,
            model=model,
            failed=found.verdict == verify.CHEAT,
            verdict=found.verdict,
        )
    except ServerError as exc:
        log.warning("could not report a probe for %s: %s", hotkey[:12], exc)
        return None

    if found.verdict == verify.CHEAT:
        log.warning(
            "cheat verdict for %s on %s: score %.6f over %.6f",
            hotkey[:12],
            model,
            found.score,
            found.threshold,
        )

    state = str(reported.get("probe", {}).get("state", OPEN))
    if state == REJECTED:
        try:
            probe.withdraw(
                server, wallet, hotkey=hotkey, reason=f"rejected on {model}: {found.reason}"
            )
            tally.withdrawn += 1
        except ServerError as exc:
            log.warning("could not withdraw %s: %s", hotkey[:12], exc)

    return reported


def load_calibrations(path: Path) -> dict[str, verify.Calibration]:
    if not path.exists():
        return {}
    found = json.loads(path.read_text(encoding="utf-8"))
    held: dict[str, verify.Calibration] = {}
    for model, raw in dict(found).items():
        held[str(model)] = verify.Calibration(
            mode=str(raw.get("mode", verify.GREEDY)),
            aggregate=str(raw.get("aggregate", "rate")),
            threshold=float(raw.get("threshold", 0.0)),
            alpha=float(raw.get("alpha", 0.0)),
            honest_samples=int(raw.get("honest_samples", 0)),
            separated=bool(raw.get("separated", False)),
            substituted_min=float(raw.get("substituted_min", 0.0)),
        )
    return held


def load_artifacts(path: Path) -> dict[str, Path]:
    if not path.exists():
        return {}
    found = json.loads(path.read_text(encoding="utf-8"))
    return {str(model): Path(str(where)) for model, where in dict(found).items()}
