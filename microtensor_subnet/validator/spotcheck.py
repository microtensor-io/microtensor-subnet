from __future__ import annotations

import hashlib
import logging
import math
import random
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from microtensor_subnet.scoring.stats import percentile, wilson_lb
from microtensor_subnet.validator import evidence, transcripts
from microtensor_subnet.validator.context import SLM, EpochContext, render_prompt
from microtensor_subnet.validator.results import SlmRun
from microtensor_subnet.validator.sandbox import compat
from microtensor_subnet.validator.sandbox.docker import ContainerSpec, Docker, DockerError
from microtensor_subnet.validator.sandbox.slm import MEMORY_SHARE, SlmClient, SlmError

log = logging.getLogger("microtensor_subnet.validator.spotcheck")

SLM_MODULE: Final[str] = "microtensor_subnet.validator.sandbox.slm"
DOCKER_MODULE: Final[str] = "microtensor_subnet.validator.sandbox.docker"
DEFAULT_MIN_CALLS: Final[int] = 200
DEFAULT_VALIDATORS_PER_MINER: Final[int] = 2
DEFAULT_REFERENCE_CALLS: Final[int] = 20
CANARY_TOKENS: Final[int] = 48
WEIGHT_PASSED: Final[int] = 3
WEIGHT_TWIN: Final[int] = 2
WEIGHT_MOVED: Final[int] = 2
WEIGHT_PLAIN: Final[int] = 1
VRAM_QUERY: Final[tuple[str, ...]] = (
    "nvidia-smi",
    "--query-gpu=memory.used",
    "--format=csv,noheader,nounits",
)
MIB: Final[int] = 1024**2
DEFAULT_MAX_MODEL_LEN: Final[int] = 8192

engine_command: Callable[..., list[str]] = transcripts.resolve(SLM_MODULE, "engine_command")
engine_env: Callable[[str], dict[str, str]] = transcripts.resolve(SLM_MODULE, "engine_env")
pinned: Callable[[str, str], str] = transcripts.resolve(DOCKER_MODULE, "pinned")


class SpotCheckError(RuntimeError):
    pass


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
    passed: bool = False
    twin: bool = False
    moved: bool = False

    @property
    def weight(self) -> int:
        weights = [WEIGHT_PLAIN]
        if self.passed:
            weights.append(WEIGHT_PASSED)
        if self.twin:
            weights.append(WEIGHT_TWIN)
        if self.moved:
            weights.append(WEIGHT_MOVED)
        return max(weights)

    @property
    def prompt_sha256(self) -> str:
        return transcripts.prompt_sha256(self.prompt)


@dataclass(frozen=True, slots=True)
class Generated:
    text: str
    ids: tuple[int, ...]
    prompt_tokens: int
    ttft_ms: float
    tpot_ms: float
    total_ms: float


@dataclass(slots=True)
class SpotCheck:
    calls: int = 0
    matched: int = 0
    mismatched: int = 0
    ttft_p95_ms: float = 0.0
    tpot_p95_ms: float = 0.0
    vram_peak_bytes: int = 0
    reference_ttft_p95_ms: float = 0.0
    reference_tpot_p95_ms: float = 0.0
    reference_vram_peak_bytes: int = 0
    uniform_calls: int = 0
    over_budget: int = 0
    envelope_lb: float = 0.0
    envelope_ok: bool = True
    timing: dict[tuple[str, int], tuple[float, float]] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "matched": self.matched,
            "mismatched": self.mismatched,
            "ttft_p95_ms": round(self.ttft_p95_ms, 1),
            "tpot_p95_ms": round(self.tpot_p95_ms, 2),
            "vram_peak_bytes": self.vram_peak_bytes,
            "reference_ttft_p95_ms": round(self.reference_ttft_p95_ms, 1),
            "reference_tpot_p95_ms": round(self.reference_tpot_p95_ms, 2),
            "reference_vram_peak_bytes": self.reference_vram_peak_bytes,
            "uniform_calls": self.uniform_calls,
            "over_budget": self.over_budget,
            "envelope_lb": round(self.envelope_lb, 6),
            "envelope_ok": self.envelope_ok,
        }


@dataclass(frozen=True, slots=True)
class ReferenceTiming:
    ttft_p95_ms: float
    tpot_p95_ms: float
    vram_peak_bytes: int
    drafts_checked: int
    drafts_mismatched: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ttft_p95_ms": round(self.ttft_p95_ms, 1),
            "tpot_p95_ms": round(self.tpot_p95_ms, 2),
            "vram_peak_bytes": self.vram_peak_bytes,
            "drafts_checked": self.drafts_checked,
            "drafts_mismatched": self.drafts_mismatched,
        }


@dataclass(frozen=True, slots=True)
class Settings:
    min_calls: int = DEFAULT_MIN_CALLS
    validators_per_miner: int = DEFAULT_VALIDATORS_PER_MINER
    reference_calls: int = DEFAULT_REFERENCE_CALLS

    @classmethod
    def from_definition(cls, definition: Mapping[str, Any]) -> Settings:
        block = definition.get("evaluation") or {}
        spot = block.get("spot_check") if isinstance(block, Mapping) else None
        spot = spot if isinstance(spot, Mapping) else {}
        return cls(
            min_calls=int(spot.get("min_calls") or DEFAULT_MIN_CALLS),
            validators_per_miner=int(
                spot.get("validators_per_miner") or DEFAULT_VALIDATORS_PER_MINER
            ),
            reference_calls=int(spot.get("reference_calls") or DEFAULT_REFERENCE_CALLS),
        )


def seeded(*parts: str) -> random.Random:
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))  # noqa: S311


def sample_miners(
    miners: Sequence[str],
    *,
    close_block_hash: str,
    validator_hotkey: str,
    validators_on_profile: int,
    validators_per_miner: int,
) -> list[str]:
    ordered = sorted(set(miners))
    if not ordered:
        return []
    rng = seeded(close_block_hash, validator_hotkey)
    rng.shuffle(ordered)
    take = math.ceil(validators_per_miner * len(ordered) / max(1, validators_on_profile))
    return ordered[: max(1, min(len(ordered), take))]


def collect_calls(
    transcript: Mapping[str, Any],
    prompts: Mapping[tuple[str, int], str],
    *,
    passed: Mapping[tuple[str, str], bool],
    twins: Mapping[str, str],
    moved: set[str],
) -> list[Call]:
    found: list[Call] = []
    for row in transcripts.items_of(transcript):
        item_id = str(row.get("item_id", ""))
        condition = str(row.get("condition", ""))
        for call in transcripts.slm_calls(row):
            seq = int(call.get("seq") or 0)
            prompt = prompts.get((item_id, seq))
            if prompt is None:
                continue
            found.append(
                Call(
                    item_id=item_id,
                    condition=condition,
                    seq=seq,
                    prompt=prompt,
                    max_tokens=int(call.get("max_tokens") or 0),
                    stop=tuple(str(s) for s in call.get("stop") or []),
                    seed=int(call.get("seed") or 0),
                    claimed=tuple(int(t) for t in call.get("output_tokens") or []),
                    passed=bool(passed.get((item_id, condition), False)),
                    twin=item_id in twins,
                    moved=item_id in moved,
                )
            )
    return found


def moved_items(run: SlmRun, reference: SlmRun | None, twins: Mapping[str, str]) -> set[str]:
    mine = {r.item_id: r.success for r in [*run.runs, *run.twins]}
    theirs = {r.item_id: r.success for r in reference.runs} if reference is not None else {}
    found: set[str] = set()
    for item_id, success in mine.items():
        twin = twins.get(item_id)
        if twin is not None and twin in mine and mine[twin] != success:
            found.update({item_id, twin})
        if item_id in theirs and theirs[item_id] != success:
            found.add(item_id)
    return found


def sample_calls(
    calls: Sequence[Call], *, min_calls: int, rng: random.Random
) -> tuple[list[Call], list[Call]]:
    if len(calls) <= min_calls:
        return list(calls), []
    uniform_n = min_calls // 2
    uniform = rng.sample(list(calls), uniform_n)
    chosen = {(c.item_id, c.seq) for c in uniform}
    rest = [c for c in calls if (c.item_id, c.seq) not in chosen]
    keyed = sorted(
        ((rng.random() ** (1.0 / c.weight), index) for index, c in enumerate(rest)),
        reverse=True,
    )
    weighted = [rest[index] for _, index in keyed[: min_calls - uniform_n]]
    return uniform, weighted


def _memory_share(definition: Mapping[str, Any], profile_bytes: int, configured: float) -> float:
    if configured:
        return configured
    budgets = definition.get("budgets") or {}
    cap = int(budgets.get("memory_bytes") or budgets.get("vram_bytes") or 0) if budgets else 0
    if cap and profile_bytes:
        return max(0.1, min(MEMORY_SHARE, cap / profile_bytes))
    return MEMORY_SHARE


@dataclass(slots=True)
class Engine:
    docker: Docker
    image: str
    model_dir: Path
    gpu: str
    profile: str
    max_model_len: int
    memory_share: float = MEMORY_SHARE
    ready_seconds: int = 900
    cuda_image: str = ""
    name: str = ""
    network: str = ""
    url: str = ""
    mount: compat.Mount | None = None
    vram_peak_bytes: int = 0

    def start(self) -> None:
        key = secrets.token_hex(4)
        self.network = f"mt-spot-{key}"
        self.name = f"mt-spot-{key}"
        self.docker.create_network(self.network)
        try:
            self.mount = compat.plain(self.model_dir)
            self.docker.start(
                ContainerSpec(
                    name=self.name,
                    image=self.image,
                    network=self.network,
                    command=engine_command(
                        compat.GUEST,
                        profile=self.profile,
                        max_model_len=self.max_model_len,
                        memory_share=self.memory_share,
                    ),
                    env=engine_env(self.profile),
                    volumes=self.mount.volumes,
                    gpus=(self.gpu,),
                    shm_size="8g",
                    ipc_none=False,
                )
            )
            self.url = f"http://{self.docker.address(self.name, self.network)}:8000"
            self._wait()
        except Exception:
            self.stop()
            raise

    def _wait(self) -> None:
        client = SlmClient(self.url)
        deadline = time.monotonic() + self.ready_seconds
        while time.monotonic() < deadline:
            if self.docker.exited(self.name):
                raise SpotCheckError(f"the engine exited: {self.docker.exit_note(self.name)}")
            if client.ready():
                return
            time.sleep(5.0)
        detail = self.docker.failure(self.name)
        raise SpotCheckError("the engine did not become ready" + (f": {detail}" if detail else ""))

    def stop(self) -> None:
        if self.name:
            self.docker.kill(self.name)
        if self.network:
            self.docker.remove_network(self.network)
        self.name = self.network = self.url = ""
        self.mount = None

    def generate(
        self, prompt: str, *, max_tokens: int, stop: Sequence[str], seed: int
    ) -> Generated:
        client = SlmClient(self.url)
        found: Any = client.generate(
            prompt, max_tokens=max_tokens, stop=stop, seed=seed, logprobs=1
        )
        return Generated(
            text=str(found.text),
            ids=tuple(int(t) for t in found.output_tokens),
            prompt_tokens=int(found.prompt_tokens),
            ttft_ms=float(found.ttft_ms),
            tpot_ms=float(found.tpot_ms),
            total_ms=float(found.total_ms),
        )

    def sample_vram(self) -> int:
        if not self.cuda_image:
            return self.vram_peak_bytes
        try:
            raw = self.docker.runner.run(
                [
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--gpus",
                    f"device={self.gpu}",
                    self.cuda_image,
                    *VRAM_QUERY,
                ]
            )
        except DockerError as exc:
            log.debug("vram not read: %s", exc)
            return self.vram_peak_bytes
        for line in raw.splitlines():
            if line.strip().isdigit():
                self.vram_peak_bytes = max(self.vram_peak_bytes, int(line.strip()) * MIB)
                break
        return self.vram_peak_bytes


def _over_budget(generated: Generated, budgets: Mapping[str, Any]) -> bool:
    ttft = float(budgets.get("ttft_ms") or 0)
    tpot = float(budgets.get("tpot_ms") or 0)
    rate = float(budgets.get("min_tokens_per_s") or 0)
    slowest = 1000.0 / rate if rate > 0 else 0.0
    late = bool(ttft) and generated.ttft_ms > ttft
    slow = len(generated.ids) > 1 and (
        (bool(tpot) and generated.tpot_ms > tpot) or (bool(slowest) and generated.tpot_ms > slowest)
    )
    return late or slow


def check_miner(
    engine: Engine,
    uniform: Sequence[Call],
    weighted: Sequence[Call],
    *,
    budgets: Mapping[str, Any],
    violation_share: float,
    evidence_fields: Mapping[str, Any],
) -> SpotCheck:
    found = SpotCheck()
    ttfts: list[float] = []
    tpots: list[float] = []
    sampled_vram = False
    for call, is_uniform in [*((c, True) for c in uniform), *((c, False) for c in weighted)]:
        try:
            generated = engine.generate(
                call.prompt, max_tokens=call.max_tokens, stop=call.stop, seed=call.seed
            )
        except SlmError as exc:
            raise SpotCheckError(f"the engine failed on {call.item_id}: {exc}") from exc
        if not sampled_vram:
            engine.sample_vram()
            sampled_vram = True
        found.calls += 1
        found.timing[(call.item_id, call.seq)] = (generated.ttft_ms, generated.tpot_ms)
        if is_uniform:
            found.uniform_calls += 1
            ttfts.append(generated.ttft_ms)
            if len(generated.ids) > 1:
                tpots.append(generated.tpot_ms)
            if _over_budget(generated, budgets):
                found.over_budget += 1
        if generated.ids == call.claimed:
            found.matched += 1
            continue
        found.mismatched += 1
        found.evidence.append(
            evidence.spot_check_body(
                arena_id=str(evidence_fields["arena_id"]),
                epoch=int(evidence_fields["epoch"]),
                miner_hotkey=str(evidence_fields["miner_hotkey"]),
                profile=str(evidence_fields["profile"]),
                weights_sha256=str(evidence_fields["weights_sha256"]),
                item_id=call.item_id,
                call_seq=call.seq,
                seed=call.seed,
                prompt_sha256=call.prompt_sha256,
                claimed_tokens=call.claimed,
                recomputed_tokens=generated.ids,
            )
        )
    found.vram_peak_bytes = engine.sample_vram()
    found.ttft_p95_ms = percentile(ttfts, 0.95)
    found.tpot_p95_ms = percentile(tpots, 0.95)
    if found.uniform_calls:
        found.envelope_lb = wilson_lb(found.over_budget, found.uniform_calls)
        found.envelope_ok = found.envelope_lb <= violation_share
    return found


def reference_calls(context: EpochContext, count: int) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for item in context.items.originals(SLM)[: max(0, count)]:
        found.append((item.item_id, render_prompt(context.slm_prompt, item.task)))
    return found


def time_reference(
    engine: Engine,
    context: EpochContext,
    *,
    count: int,
    drafts: Mapping[str, str],
    seeds: Mapping[str, int],
) -> ReferenceTiming:
    ttfts: list[float] = []
    tpots: list[float] = []
    checked = 0
    mismatched = 0
    for item_id, prompt in reference_calls(context, count):
        try:
            generated = engine.generate(
                prompt,
                max_tokens=context.max_tokens,
                stop=context.stop,
                seed=int(seeds.get(item_id, 0)),
            )
        except SlmError as exc:
            raise SpotCheckError(f"the reference engine failed on {item_id}: {exc}") from exc
        ttfts.append(generated.ttft_ms)
        if len(generated.ids) > 1:
            tpots.append(generated.tpot_ms)
        draft = drafts.get(item_id)
        if draft is None:
            continue
        checked += 1
        if generated.text != draft:
            mismatched += 1
            log.warning("owner reference draft for %s does not match the reference model", item_id)
    return ReferenceTiming(
        ttft_p95_ms=percentile(ttfts, 0.95),
        tpot_p95_ms=percentile(tpots, 0.95),
        vram_peak_bytes=engine.sample_vram(),
        drafts_checked=checked,
        drafts_mismatched=mismatched,
    )


def canary_check(
    engine: Engine,
    canaries: Sequence[Mapping[str, Any]],
    *,
    evidence_fields: Mapping[str, Any],
    max_tokens: int = CANARY_TOKENS,
) -> tuple[int, list[dict[str, Any]]]:
    hits: list[dict[str, Any]] = []
    checked = 0
    for canary in canaries:
        prefix = str(canary.get("prefix", ""))
        fact = str(canary.get("fact", ""))
        marker = fact.rsplit(" ", 1)[-1].strip().lower() if fact else ""
        if not prefix or not marker:
            continue
        try:
            generated = engine.generate(prefix, max_tokens=max_tokens, stop=(), seed=0)
        except SlmError as exc:
            log.warning("canary %s not checked: %s", canary.get("item_id"), exc)
            continue
        checked += 1
        if marker in generated.text.lower():
            hits.append(
                evidence.canary_body(
                    arena_id=str(evidence_fields["arena_id"]),
                    epoch=int(evidence_fields["epoch"]),
                    miner_hotkey=str(evidence_fields["miner_hotkey"]),
                    profile=str(evidence_fields["profile"]),
                    weights_sha256=str(evidence_fields["weights_sha256"]),
                    item_id=str(canary.get("item_id", "")),
                    seed=0,
                    prefix_sha256=transcripts.sha256_text(prefix),
                    recomputed_tokens=generated.ids,
                    completion_text=generated.text,
                )
            )
    return checked, hits


def recheck(
    engine: Engine,
    row: Mapping[str, Any],
    *,
    calls: Mapping[tuple[str, int], Call],
    canaries: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, list[int]] | None:
    kind = str(row.get("kind", ""))
    item_id = str(row.get("item_id", ""))
    if kind == evidence.CANARY:
        canary = canaries.get(item_id)
        if canary is None:
            return None
        prefix = str(canary.get("prefix", ""))
        if transcripts.sha256_text(prefix) != str(row.get("prefix_sha256", "")):
            return None
        marker = str(canary.get("fact", "")).rsplit(" ", 1)[-1].strip().lower()
        generated = engine.generate(prefix, max_tokens=CANARY_TOKENS, stop=(), seed=0)
        return marker in generated.text.lower(), list(generated.ids)
    call = calls.get((item_id, int(row.get("call_seq") or 0)))
    if call is None or call.prompt_sha256 != str(row.get("prompt_sha256", "")):
        return None
    claimed = tuple(int(t) for t in row.get("claimed_tokens") or [])
    if claimed != call.claimed:
        return False, list(call.claimed)
    generated = engine.generate(
        call.prompt, max_tokens=call.max_tokens, stop=call.stop, seed=int(row.get("seed") or 0)
    )
    return generated.ids != call.claimed, list(generated.ids)


def engine_for(
    docker: Docker,
    *,
    definition: Mapping[str, Any],
    model_dir: Path,
    gpu: str,
    profile: str,
    profile_bytes: int,
    max_model_len: int | None,
    memory_share: float,
    ready_seconds: int,
    cuda_image: str,
) -> Engine:
    block = definition.get("evaluation") or {}
    images = block.get("images") if isinstance(block, Mapping) else None
    engine = images.get("engine") if isinstance(images, Mapping) else None
    if not isinstance(engine, Mapping) or not engine.get("image") or not engine.get("digest"):
        raise SpotCheckError("the arena definition pins no engine image for the spot check")
    reference = pinned(str(engine["image"]), str(engine["digest"]))
    docker.ensure_image(reference)
    return Engine(
        docker=docker,
        image=reference,
        model_dir=model_dir,
        gpu=gpu,
        profile=profile,
        max_model_len=max_model_len or DEFAULT_MAX_MODEL_LEN,
        memory_share=_memory_share(definition, profile_bytes, memory_share),
        ready_seconds=ready_seconds,
        cuda_image=cuda_image,
    )
