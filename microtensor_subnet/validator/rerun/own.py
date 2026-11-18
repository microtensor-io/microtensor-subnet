from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from microtensor_subnet.core import profiles
from microtensor_subnet.validator import spotcheck
from microtensor_subnet.validator.config import EpochConfig
from microtensor_subnet.validator.context import EpochContext
from microtensor_subnet.validator.rerun import calls, checks
from microtensor_subnet.validator.rerun.card import Card, card_from
from microtensor_subnet.validator.rerun.engine import (
    EngineError,
    full_length,
    memory_share,
    out_of_memory,
)
from microtensor_subnet.validator.rerun.local import LocalPod
from microtensor_subnet.validator.rerun.pod import PodClient, engine_image, remote_engine
from microtensor_subnet.validator.sandbox.docker import Docker, DockerError
from microtensor_subnet.validator.spotcheck import Flag, ReferenceTiming, SpotCheck

log = logging.getLogger("microtensor_subnet.validator.rerun.own")

DEFAULT_MIN_CALLS: Final[int] = 40
DEFAULT_REFERENCE_CALLS: Final[int] = 10
PARALLEL: Final[int] = 1
REASON_CHARS: Final[int] = 300
VALIDATOR: Final[str] = "validator"
COORDINATOR: Final[str] = "coordinator"
STARTING: Final[str] = "the spot check engine is starting"
LATE: Final[str] = "not finished before the coordinator deadline"
SPEED_FIELDS: Final[tuple[str, ...]] = (
    "ttft_p95_ms",
    "tpot_p95_ms",
    "vram_peak_bytes",
    "uniform_calls",
    "over_budget",
    "envelope_lb",
    "envelope_ok",
    "timing",
    "e2e_p95_ms",
    "memory_ok",
)
ROUTER_FIELDS: Final[tuple[str, ...]] = ("router_latency_p95_ms", "router_reason")

Key = tuple[str, str, int]


def spot_settings(definition: Mapping[str, Any]) -> tuple[int, int]:
    block = definition.get("evaluation") or {}
    spot = block.get("spot_check") if isinstance(block, Mapping) else None
    spot = spot if isinstance(spot, Mapping) else {}
    return (
        int(spot.get("min_calls") or DEFAULT_MIN_CALLS),
        int(spot.get("reference_calls") or DEFAULT_REFERENCE_CALLS),
    )


def trust(timing: ReferenceTiming) -> tuple[bool, str]:
    if timing.drafts_checked and not timing.drafts_mismatched:
        return True, f"{timing.drafts_checked} reference answers reproduce on this card"
    if not timing.drafts_checked:
        return False, "no reference answer was checked"
    return False, (
        f"{timing.drafts_mismatched} of {timing.drafts_checked} reference answers "
        "do not reproduce on this card"
    )


def pick_profile(
    definition: Mapping[str, Any], gpus: Sequence[str], names: Mapping[str, str]
) -> tuple[str, str]:
    problems: list[str] = []
    for profile in profiles.allowed_profiles(definition):
        if profile not in profiles.PUBLISHED_CARDS:
            continue
        problem = profiles.card_problem(profile, gpus, names=names)
        if not problem:
            return profile, ""
        problems.append(problem)
    return "", "; ".join(problems) or "the arena lists no profile with a published card"


def kept(drawn: Sequence[calls.Drawn]) -> frozenset[Key]:
    return frozenset((d.item_id, d.condition, d.seq) for d in drawn if not d.direct)


def key_of(call: calls.Call) -> Key:
    return (call.item_id, call.condition, call.seq)


def received_blocks(client: Any, arena_id: str, epoch_index: int) -> dict[str, tuple[str, int]]:
    return {
        str(row.get("hotkey")): (
            str(row.get("transcript_sha256") or "").lower(),
            int(row.get("received_block") or 0),
        )
        for row in client.uploads(arena_id, epoch_index)
    }


@dataclass(slots=True)
class Job:
    hotkey: str
    sampled: list[calls.Call]
    skipped: int
    slm_dir: Path
    weights_sha256: str
    release: Callable[[], None]


@dataclass(slots=True)
class OwnResult:
    spot: SpotCheck | None = None
    wire: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    flagged: list[Key] = field(default_factory=list)
    sampled: list[Key] = field(default_factory=list)
    measured: bool = False
    failure: str = ""

    def to_dict(self, trusted: bool) -> dict[str, Any]:
        return {
            "spot_check": dict(self.wire) if self.wire is not None else None,
            "evidence": [dict(entry) for entry in self.evidence],
            "failure": self.failure,
            "measured": self.measured,
            "trusted": trusted,
        }


def disagreements(
    own: OwnResult, coordinated: SpotCheck | None, flags: Sequence[Flag]
) -> list[dict[str, Any]]:
    if coordinated is None or own.spot is None:
        return []
    theirs = {(f.item_id, f.condition, f.call_seq) for f in flags if f.kind == spotcheck.SPOT_CHECK}
    ours = set(own.flagged)
    rerun = set(own.sampled)
    found: list[dict[str, Any]] = [
        {"kind": "coordinator_flag", "item_id": k[0], "condition": k[1], "call_seq": k[2]}
        for k in sorted(theirs - ours)
        if k in rerun
    ]
    found += [
        {"kind": "validator_flag", "item_id": k[0], "condition": k[1], "call_seq": k[2]}
        for k in sorted(ours - theirs)
    ]
    if coordinated.mismatched != own.spot.mismatched:
        found.append(
            {
                "kind": "mismatched",
                "coordinator": coordinated.mismatched,
                "validator": own.spot.mismatched,
            }
        )
    return found


@dataclass(slots=True)
class OwnChecks:
    docker: Docker
    config: EpochConfig
    context: EpochContext
    drafts: dict[str, str]
    seeds: dict[str, int]
    gpus: tuple[str, ...]
    profile: str = ""
    names: dict[str, str] = field(default_factory=dict)
    reason: str = STARTING
    trusted: bool = False
    card: Card = field(default_factory=Card)
    engine: str = ""
    share: float = 0.0
    reference: ReferenceTiming | None = None
    pod: LocalPod | None = None
    client: PodClient | None = None
    uploads: dict[str, tuple[str, int]] | None = None
    results: dict[str, OwnResult] = field(default_factory=dict)
    waiting: set[str] = field(default_factory=set)
    jobs: queue.Queue[Job | None] = field(default_factory=queue.Queue)
    stopped: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def takes(self, entry: Mapping[str, Any]) -> bool:
        return (
            bool(self.profile)
            and not self.stopped.is_set()
            and str(entry.get("profile") or "") == self.profile
        )

    def received(self, client: Any, hotkey: str, sha: str) -> int:
        if self.uploads is None or hotkey not in self.uploads:
            self.uploads = received_blocks(client, self.context.arena_id, self.context.epoch_index)
        held, block = self.uploads.get(hotkey, ("", 0))
        if held != sha.lower() or not block:
            raise EngineError("the server lists no upload block for this transcript")
        return block

    def draw_for(
        self, client: Any, chain: Any, hotkey: str, sha: str, transcript: Mapping[str, Any]
    ) -> list[calls.Drawn]:
        block = self.received(client, hotkey, sha)
        block_hash = str(chain.block_hash(block + calls.SAMPLE_BLOCKS))
        min_calls, _ = spot_settings(self.context.definition)
        return calls.draw(transcript, sha, block_hash, min_calls)

    def skip(self, hotkey: str, why: str) -> None:
        log.info("%s has no own spot check: %s", hotkey, why)
        with self.lock:
            self.results[hotkey] = OwnResult(failure=why[:REASON_CHARS])

    def submit(self, job: Job) -> None:
        with self.lock:
            self.waiting.add(job.hotkey)
        self.jobs.put(job)

    def launch(self) -> None:
        threading.Thread(target=self.work, name="mt-spot-check", daemon=True).start()

    def work(self) -> None:
        try:
            self.set_up()
        except Exception as exc:
            self.trusted = False
            self.reason = f"the spot check engine did not start: {exc}"[:REASON_CHARS]
            log.warning("no own spot checks: %s", self.reason)
        while True:
            job = self.jobs.get()
            if job is None:
                return
            self.run(job)

    def set_up(self) -> None:
        self.engine = engine_image(self.context.definition)
        pod = LocalPod(self.docker, self.engine, self.gpus, self.config.cache_dir)
        self.pod = pod
        self.client = pod.up()
        if self.stopped.is_set():
            pod.down()
            raise EngineError("the epoch ended first")
        self.card = card_from(self.client.health())
        self.share = memory_share(
            self.context.definition, self.card.memory_bytes, self.config.slm_memory_share
        )
        self.reference = self.time_reference(self.client, pod)
        self.trusted, self.reason = trust(self.reference)
        log.info("own spot checks on %s: %s", self.card.name or "an unnamed card", self.reason)

    def time_reference(self, client: PodClient, pod: LocalPod) -> ReferenceTiming:
        definition = self.context.definition
        _, count = spot_settings(definition)
        engine = remote_engine(
            client,
            kind="reference",
            name="reference",
            profile=self.profile,
            max_model_len=full_length(definition),
            share=self.share,
            model_dir=pod.guest(self.context.reference.slm_dir),
        )
        engine.start()
        try:
            found = checks.time_reference(
                checks.Guard(engine),
                self.context,
                count=count,
                drafts=self.drafts,
                seeds=self.seeds,
                stop=self.stopped.is_set,
            )
        finally:
            engine.stop()
        if self.card.measures_speed(self.profile):
            return found
        return ReferenceTiming(0.0, 0.0, 0, found.drafts_checked, found.drafts_mismatched)

    def run(self, job: Job) -> None:
        try:
            result = self.attempt(job)
        except Exception as exc:
            log.warning("%s own spot check failed: %s", job.hotkey, exc)
            result = OwnResult(failure=f"{type(exc).__name__}: {exc}"[:REASON_CHARS])
        finally:
            try:
                job.release()
            except OSError as exc:
                log.debug("%s weights not dropped: %s", job.hotkey, exc)
        with self.lock:
            if job.hotkey in self.waiting:
                self.results[job.hotkey] = result
                self.waiting.discard(job.hotkey)

    def attempt(self, job: Job) -> OwnResult:
        if self.stopped.is_set():
            return OwnResult(failure=LATE)
        if not self.trusted or self.client is None or self.pod is None:
            return OwnResult(failure=self.reason)
        return self.check(job, self.client, self.pod)

    def check(self, job: Job, client: PodClient, pod: LocalPod) -> OwnResult:
        definition = self.context.definition
        measure = self.card.measures_speed(self.profile)
        keys = [key_of(call) for call in job.sampled]
        engine = remote_engine(
            client,
            kind="miner",
            name="miner",
            profile=self.profile,
            max_model_len=full_length(definition),
            share=self.share,
            model_dir=pod.guest(job.slm_dir),
        )
        try:
            engine.start()
        except EngineError as exc:
            if measure and out_of_memory(str(exc)):
                log.warning("%s does not fit on the card: %s", job.hotkey, exc)
                wire = checks.wire_spot(
                    SpotCheck(envelope_ok=False), skipped_prompt=job.skipped, memory_ok=False
                )
                return OwnResult(SpotCheck.from_dict(wire), wire, sampled=keys, measured=True)
            raise
        guard = checks.Guard(engine)
        try:
            passed = checks.first_pass(
                guard,
                job.sampled,
                budgets=dict(definition.get("budgets") or {}),
                parallel=PARALLEL,
                measure=measure,
            )
            done = checks.settle(
                guard,
                passed.spot,
                passed.mismatches,
                [],
                checked_canaries=0,
                weights_sha256=job.weights_sha256,
            )
        finally:
            engine.stop()
        wire = checks.wire_spot(
            done.spot,
            unsettled=done.unsettled,
            errors=passed.errors,
            skipped_prompt=job.skipped,
            e2e_p95_ms=passed.e2e_p95_ms,
            memory_ok=passed.memory_ok,
        )
        if done.flagged:
            log.warning("%s: %d calls differ on both runs here", job.hotkey, len(done.flagged))
        return OwnResult(
            spot=SpotCheck.from_dict(wire),
            wire=wire,
            evidence=list(done.evidence),
            flagged=[key_of(call) for call in done.flagged],
            sampled=keys,
            measured=measure,
        )

    def unfinished(self) -> bool:
        with self.lock:
            return bool(self.waiting)

    def join(self, waiting: Callable[[], bool], pause: float) -> None:
        while self.unfinished() and waiting():
            time.sleep(pause)
        self.stopped.set()
        with self.lock:
            late = sorted(self.waiting)
            for hotkey in late:
                self.results[hotkey] = OwnResult(failure=LATE)
            self.waiting.clear()
        if late:
            log.warning("%d own spot checks did not finish in time", len(late))
            if self.pod is not None:
                self.pod.down()

    def close(self) -> None:
        self.stopped.set()
        self.jobs.put(None)
        if self.pod is not None:
            self.pod.down()

    def used(self, hotkey: str) -> OwnResult | None:
        found = self.results.get(hotkey) if self.trusted else None
        return found if found is not None and found.spot is not None else None

    def combine(
        self, hotkey: str, coordinated: SpotCheck | None, flags: Sequence[Flag]
    ) -> tuple[SpotCheck | None, str, list[dict[str, Any]]]:
        own = self.used(hotkey)
        if own is None or own.spot is None:
            return coordinated, COORDINATOR if coordinated is not None else "", []
        spot = dataclasses.replace(own.spot, timing=dict(own.spot.timing))
        if coordinated is not None:
            names = ROUTER_FIELDS if own.measured else ROUTER_FIELDS + SPEED_FIELDS
            for name in names:
                setattr(spot, name, getattr(coordinated, name))
        return spot, VALIDATOR, disagreements(own, coordinated, flags)

    def timing_for(self, hotkey: str) -> ReferenceTiming | None:
        own = self.used(hotkey)
        return self.reference if own is not None and own.measured else None

    def fraud(self, epoch_index: int) -> dict[str, str]:
        found: dict[str, str] = {}
        for hotkey in list(self.results):
            own = self.used(hotkey)
            for entry in own.evidence if own is not None else []:
                if spotcheck.stands(entry):
                    found[hotkey] = (
                        f"epoch {epoch_index} spot_check "
                        f"{entry.get('item_id', '')}#{entry.get('call_seq', 0)} by this validator"
                    )
                    break
        return found

    def row(self, hotkey: str) -> dict[str, Any] | None:
        found = self.results.get(hotkey)
        return found.to_dict(self.trusted) if found is not None else None

    def body(self) -> dict[str, Any]:
        return {
            "gpus": list(self.gpus),
            "card": self.card.name,
            "profile": self.profile,
            "engine": self.engine,
            "trusted": self.trusted,
            "reason": self.reason,
            "reference": self.reference.to_dict() if self.reference is not None else None,
        }


def start(
    config: EpochConfig,
    context: EpochContext,
    docker: Docker,
    *,
    drafts: Mapping[str, str],
    seeds: Mapping[str, int],
) -> OwnChecks | None:
    if not config.spot_check_gpus:
        return None
    own = OwnChecks(
        docker=docker,
        config=config,
        context=context,
        drafts=dict(drafts),
        seeds=dict(seeds),
        gpus=tuple(config.spot_check_gpus),
    )
    try:
        own.names = docker.card_names(config.cuda_image)
    except DockerError as exc:
        own.reason = f"the cards were not read: {exc}"[:REASON_CHARS]
        log.warning("no own spot checks: %s", own.reason)
        return own
    own.profile, problem = pick_profile(context.definition, own.gpus, own.names)
    if not own.profile:
        own.reason = problem[:REASON_CHARS]
        log.warning("no own spot checks: %s", own.reason)
        return own
    log.info("own spot checks for profile %s on GPU %s", own.profile, ",".join(own.gpus))
    own.launch()
    return own
