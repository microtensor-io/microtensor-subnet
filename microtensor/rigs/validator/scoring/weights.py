from __future__ import annotations

import contextlib
import datetime as dt
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from microtensor.rigs.validator.data import gpu_table
from microtensor.rigs.validator.data.tiers import Tier
from microtensor.rigs.validator.scoring.demand_weight import DemandWeights
from microtensor.rigs.validator.scoring.formula import ScoreInputs, score
from microtensor.rigs.validator.scoring.reliability import UptimeLedger

U16_MAX = 65535
HELD_SHARE = 0.40
SS58_FORMAT = 42
VERIFIED_STATES = ("probation", "active")
NUMBERS = re.compile(r"\d+")


@dataclass(frozen=True)
class WeightVector:
    uids: list[int]
    weights: list[int]
    floored: list[int] = field(default_factory=list)
    reserve_uid: int | None = None

    def payload(self) -> dict[str, object]:
        return {
            "uids": list(self.uids),
            "weights": list(self.weights),
            "floored": list(self.floored),
            "reserve_uid": self.reserve_uid,
        }


def normalise(scores: Mapping[int, float]) -> dict[int, float]:
    positive = {uid: value for uid, value in scores.items() if value > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        return {}
    return {uid: value / total for uid, value in positive.items()}


def with_reserve(
    shares: Mapping[int, float], reserve_uid: int | None, held_share: float = HELD_SHARE
) -> dict[int, float]:
    if reserve_uid is None or held_share <= 0.0:
        return dict(shares)
    held = min(max(held_share, 0.0), 1.0)
    scaled = {uid: value * (1.0 - held) for uid, value in shares.items() if uid != reserve_uid}
    scaled[reserve_uid] = scaled.get(reserve_uid, 0.0) + held
    return scaled


def to_u16(shares: Mapping[int, float]) -> dict[int, int]:
    if not shares:
        return {}
    top = max(shares.values())
    if top <= 0.0:
        return {}
    out: dict[int, int] = {}
    for uid, value in shares.items():
        if value <= 0.0:
            continue
        quantised = round(value / top * U16_MAX)
        out[uid] = max(1, quantised)
    return out


def eligibility_floor(
    u16: dict[int, int], active_uids: set[int], donor_uid: int | None
) -> list[int]:
    missing = sorted(uid for uid in active_uids if uid not in u16 and uid != donor_uid)
    if not missing or donor_uid is None or donor_uid not in u16:
        return []
    floored: list[int] = []
    for uid in missing:
        if u16[donor_uid] <= 1:
            break
        u16[donor_uid] -= 1
        u16[uid] = 1
        floored.append(uid)
    return floored


def build(
    scores: Mapping[int, float],
    active_uids: set[int] | None = None,
    reserve_uid: int | None = None,
    held_share: float = HELD_SHARE,
) -> WeightVector:
    shares = with_reserve(normalise(scores), reserve_uid, held_share)
    u16 = to_u16(shares)
    floored = eligibility_floor(u16, set(active_uids or ()), reserve_uid)
    ordered = sorted(u16.items())
    return WeightVector(
        uids=[uid for uid, _ in ordered],
        weights=[weight for _, weight in ordered],
        floored=floored,
        reserve_uid=reserve_uid,
    )


def version_key(version: str) -> int:
    parts = [int(part) for part in version.split(".")[:3]] + [0, 0, 0]
    major, minor, patch = parts[:3]
    return major * 10000 + minor * 100 + patch


def parse_version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in NUMBERS.findall(text or "")[:3])


def driver_ok(driver: str, minimum: str) -> bool:
    floor = parse_version(minimum)
    current = parse_version(driver)
    if not floor or not current:
        return True
    return current >= floor


def uptime_days_of(rig: dict[str, Any], ledger: UptimeLedger | None, now: dt.datetime) -> float:
    rig_id = str(rig.get("id", "") or "")
    days = ledger.uptime_days(rig_id) if ledger is not None else 0.0
    since = str(rig.get("uptime_since", "") or "")
    if since:
        try:
            moment = dt.datetime.fromisoformat(since)
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=dt.timezone.utc)
            days = max(days, (now - moment).total_seconds() / 86400.0)
        except ValueError:
            pass
    return days


def in_rental(rig: dict[str, Any]) -> bool:
    return any(
        isinstance(job, dict) and str(job.get("kind", "")) == "rental"
        for job in rig.get("jobs") or []
    )


def local_scores(
    roster: list[dict[str, Any]],
    demand: DemandWeights | None,
    ledger: UptimeLedger | None,
    *,
    min_driver_version: str = "",
    driver_cutoff_passed: bool = False,
    now: dt.datetime | None = None,
) -> dict[str, float]:
    moment = now or dt.datetime.now(dt.timezone.utc)
    verified: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for rig in roster:
        if str(rig.get("state", "")) not in VERIFIED_STATES:
            continue
        if not (rig.get("last_pass") or {}).get("passed", False):
            continue
        cards = [
            card
            for card in rig.get("gpus") or []
            if isinstance(card, dict) and card.get("accepted", True)
        ]
        if cards:
            verified.append((rig, cards))
    fleet = sum(len(cards) for _, cards in verified)
    totals: dict[str, float] = {}
    for rig, cards in verified:
        hotkey = str(rig.get("hotkey", "") or "")
        if not hotkey:
            continue
        weights: list[float] = []
        for card in cards:
            tier: Tier | None
            try:
                tier = Tier(str(card.get("tier", "") or ""))
            except ValueError:
                tier = None
            key = gpu_table.normalize(str(card.get("model", "") or ""))
            weights.append(demand.weight(key, tier) if demand is not None else 1.0)
        spec = rig.get("spec") or {}
        inputs = ScoreInputs(
            verified=True,
            demand_weight=sum(weights) / len(weights) if weights else 0.0,
            gpu_count=len(cards),
            fleet_gpu_count=fleet,
            isolation_ok=bool(rig.get("isolation_verified", False)),
            uptime_days=uptime_days_of(rig, ledger, moment),
            driver_ok=driver_ok(str(spec.get("driver", "") or ""), min_driver_version),
            driver_cutoff_passed=driver_cutoff_passed,
            in_rental=in_rental(rig),
            rental_opted_out=bool(rig.get("rental_opt_out", False)),
        )
        totals[hotkey] = totals.get(hotkey, 0.0) + score(inputs)
    return totals


def compute_scores(
    server_scores: dict[str, Any] | None,
    roster: list[dict[str, Any]],
    demand: DemandWeights | None = None,
    ledger: UptimeLedger | None = None,
    *,
    min_driver_version: str = "",
    driver_cutoff_passed: bool = False,
) -> tuple[dict[str, float], str]:
    if server_scores and server_scores.get("epoch"):
        weights = {
            str(k): float(v)
            for k, v in (server_scores.get("weights") or {}).items()
            if float(v) > 0.0
        }
        if weights:
            return weights, "server"
    return (
        local_scores(
            roster,
            demand,
            ledger,
            min_driver_version=min_driver_version,
            driver_cutoff_passed=driver_cutoff_passed,
        ),
        "local",
    )


class Chain:
    def __init__(self, endpoint: str, netuid: int, mechanism_id: int | None = None) -> None:
        self.endpoint = endpoint
        self.netuid = netuid
        self.mechanism_id = mechanism_id
        self._substrate: Any = None

    def connect(self) -> Any:
        if self._substrate is None:
            from substrateinterface import SubstrateInterface

            self._substrate = SubstrateInterface(url=self.endpoint, ss58_format=SS58_FORMAT)
        return self._substrate

    def close(self) -> None:
        if self._substrate is not None:
            with contextlib.suppress(Exception):
                self._substrate.close()
            self._substrate = None

    def uid_of(self, hotkey: str) -> int | None:
        found = self.connect().query("SubtensorModule", "Uids", [self.netuid, hotkey])
        value = getattr(found, "value", None)
        return int(value) if value is not None else None

    def uids_for(self, hotkeys: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for hotkey in hotkeys:
            uid = self.uid_of(hotkey)
            if uid is not None:
                out[hotkey] = uid
        return out

    def chain_version_key(self) -> int:
        found = self.connect().query("SubtensorModule", "WeightsVersionKey", [self.netuid])
        value = getattr(found, "value", None)
        return int(value or 0)

    def vector(
        self,
        scores_by_hotkey: Mapping[str, float],
        active_hotkeys: set[str] | None = None,
        reserve_uid: int | None = None,
        held_share: float = HELD_SHARE,
    ) -> tuple[WeightVector, dict[str, int]]:
        uids = self.uids_for(sorted(set(scores_by_hotkey) | set(active_hotkeys or ())))
        scores_by_uid = {uids[h]: w for h, w in scores_by_hotkey.items() if h in uids}
        active_uids = {uids[h] for h in (active_hotkeys or set()) if h in uids}
        return build(scores_by_uid, active_uids, reserve_uid, held_share), uids

    def submit(
        self,
        keypair: Any,
        scores_by_hotkey: Mapping[str, float],
        active_hotkeys: set[str] | None = None,
        reserve_uid: int | None = None,
        held_share: float = HELD_SHARE,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        vector, uids = self.vector(scores_by_hotkey, active_hotkeys, reserve_uid, held_share)
        outcome: dict[str, Any] = {
            "submitted": False,
            "dry_run": dry_run,
            "netuid": self.netuid,
            "mechanism_id": self.mechanism_id,
            "known_hotkeys": len(uids),
            "unknown_hotkeys": sorted(set(scores_by_hotkey) - set(uids)),
            "held_share_routed": reserve_uid is not None and held_share > 0.0,
            **vector.payload(),
        }
        if not vector.uids:
            outcome["reason"] = "no registered hotkeys with a positive share"
            return outcome
        version = self.chain_version_key()
        outcome["version_key"] = version
        if dry_run:
            return outcome
        substrate = self.connect()
        params: dict[str, Any] = {
            "netuid": self.netuid,
            "dests": vector.uids,
            "weights": vector.weights,
            "version_key": version,
        }
        function = "set_weights"
        if self.mechanism_id is not None:
            function = "set_mechanism_weights"
            params["mecid"] = self.mechanism_id
        call = substrate.compose_call(
            call_module="SubtensorModule", call_function=function, call_params=params
        )
        extrinsic = substrate.create_signed_extrinsic(call=call, keypair=keypair)
        receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        outcome.update(
            {
                "submitted": True,
                "call": function,
                "hash": str(getattr(receipt, "extrinsic_hash", "") or ""),
            }
        )
        return outcome
