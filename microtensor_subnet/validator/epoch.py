from __future__ import annotations

import contextlib
import json
import logging
import platform
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from microtensor_subnet.artifact.params import ParameterError, footprint_bytes
from microtensor_subnet.chain.epochs import Epoch, epoch_seed
from microtensor_subnet.chain.wallet import hotkey_address
from microtensor_subnet.chain.weights import quantise_weights, restrict_to_metagraph
from microtensor_subnet.core.reports import PoolScore
from microtensor_subnet.scoring import pools
from microtensor_subnet.serving.probes import sets as probe_sets
from microtensor_subnet.validator import (
    checks,
    collect,
    compress,
    items,
    progress,
    provenance,
    report,
    score,
)
from microtensor_subnet.validator.client import ServerClient, ServerError
from microtensor_subnet.validator.config import EpochConfig
from microtensor_subnet.validator.context import (
    ESCALATION,
    HARNESS,
    ROUTER,
    SLM,
    SYSTEM,
    Artifact,
    EpochContext,
    Reference,
)
from microtensor_subnet.validator.pools import escalation, harness, reference, router, slm, system
from microtensor_subnet.validator.relay import RelayClient, RelayError, TicketDesk
from microtensor_subnet.validator.results import (
    ArtifactOutcome,
    EscalationRun,
    HarnessRun,
    ReferenceRun,
    RouterRun,
    SlmRun,
    SystemRun,
)
from microtensor_subnet.validator.sandbox import (
    GpuPool,
    Limits,
    Sandbox,
    SandboxError,
    SandboxHost,
    host_ip,
)
from microtensor_subnet.validator.sandbox import judge as probe_judge
from microtensor_subnet.validator.sandbox import tools as tool_faults
from microtensor_subnet.validator.sandbox.docker import Docker, DockerError, own_container
from microtensor_subnet.validator.sandbox.services import Services
from microtensor_subnet.validator.sandbox.slm import SlmClient

log = logging.getLogger("microtensor_subnet.validator.epoch")
DEFAULT_CONTEXT = 8192
LIVE_POOLS: Final[tuple[str, ...]] = (SLM, HARNESS, ESCALATION, SYSTEM)
RECORDS_DIR: Final[str] = "records"
ROUTERS_DIR: Final[str] = "routers"
REPORT_FILE: Final[str] = "report.json"
KEPT_AT_CLOSE: Final[frozenset[str]] = frozenset({REPORT_FILE, probe_sets.PROBE_DIR})


class EpochError(RuntimeError):
    pass


@dataclass(slots=True)
class Outcome:
    scores: dict[str, list[PoolScore]] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    receipts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    burned: dict[str, float] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MinerRecord:
    hotkey: str
    coldkey: str
    artifact_sha: str
    commit_block: int
    artifact: Artifact | None = None
    weights_digest: str = ""
    scores: list[PoolScore] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    failure: str = ""
    drafts: dict[str, tuple[str, bool]] | None = None
    router_dir: Path | None = None
    posted: bool = False
    claim: str | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)
    cleared: bool = False
    shared: set[str] = field(default_factory=set)

    def wire(self) -> dict[str, Any]:
        return {
            "hotkey": self.hotkey,
            "coldkey": self.coldkey,
            "artifact_sha": self.artifact_sha,
            "manifest_sha": self.artifact.manifest_sha if self.artifact else "",
            "weights_digest": self.weights_digest,
            "commit_block": self.commit_block,
            "scores": [s.to_dict() for s in self.scores],
            "provenance": dict(self.provenance),
            "failure": self.failure,
            "drafts": {i: [text, ok] for i, (text, ok) in (self.drafts or {}).items()},
            "posted": self.posted,
        }


class Turns:
    def __init__(self) -> None:
        self._next = 0
        self._done: set[int] = set()
        self._cond = threading.Condition()

    def wait(self, position: int) -> None:
        with self._cond:
            self._cond.wait_for(lambda: self._next >= position)

    def done(self, position: int) -> None:
        with self._cond:
            self._done.add(position)
            while self._next in self._done:
                self._next += 1
            self._cond.notify_all()


def _window(arena: Mapping[str, Any], index: int) -> Epoch:
    timing = dict(arena.get("timing") or {})
    segments = [dict(s) for s in timing.get("segments") or []] or [
        {
            "from_epoch": 0,
            "start_block": timing["genesis_block"],
            "epoch_blocks": timing["epoch_blocks"],
            "commit_blocks": timing["commit_blocks"],
            "settle_margin": timing["settle_margin"],
        }
    ]
    seg = [s for s in segments if int(s["from_epoch"]) <= index][-1]
    length = int(seg["epoch_blocks"])
    return Epoch(
        index=index,
        start_block=int(seg["start_block"]) + (index - int(seg["from_epoch"])) * length,
        length=length,
        commit_blocks=int(seg["commit_blocks"]),
        settle_margin=int(seg["settle_margin"]),
    )


def load_context(
    config: EpochConfig, client: ServerClient, chain: Any, *, epoch_index: int, hotkey: str
) -> EpochContext:
    arena = client.arena(config.arena_id)
    if arena is None:
        raise EpochError(f"the server has no arena {config.arena_id}")
    definition = dict(arena.get("definition") or {})
    window = _window(arena, epoch_index)
    key = items.server_key(client)
    manifest, bundle, digests = items.fetch_release(client, config.arena_id, epoch_index, key)
    open_hash = chain.block_hash(window.open_block)
    seed = epoch_seed(open_hash, config.arena_id)
    release_doc = client.document(config.arena_id, epoch_index, "release") or {}
    body = dict(release_doc.get("body") or {})
    if body.get("seed") and body["seed"] != seed:
        raise EpochError("the server's seed does not match the chain's open block hash")
    private = items.load_recipient_key(config.keys_dir)
    files = items.open_bundle(manifest, bundle, hotkey, private)
    work_dir = config.work_dir / str(epoch_index)
    epoch_items, trees = items.unpack(files, seed=seed, work_dir=work_dir, limit=config.item_limit)
    ref = items.reference_components(trees, definition, config.cache_dir, config.chain.network)
    snapshot = client.ledger_snapshot(config.arena_id, epoch_index)
    roster_doc = client.roster(config.arena_id, epoch_index)
    charged = dict(dict(dict(roster_doc or {}).get("body") or {}).get("fee") or {})
    if charged:
        definition = {
            **definition,
            "fee_tao": float(charged.get("tao") or 0.0),
            "fee_coldkey": str(charged.get("pay_to") or ""),
        }
    sample = next(iter(epoch_items.databases.values()), b"")
    sample_sha = str(definition.get("sample_database") or "")
    if sample_sha and sample_sha in epoch_items.databases:
        sample = epoch_items.databases[sample_sha]
    return EpochContext(
        arena_id=config.arena_id,
        epoch_index=epoch_index,
        definition=definition,
        seed=seed,
        open_block=window.open_block,
        open_block_hash=open_hash,
        bundle_sha256=manifest.bundle_sha256,
        release_digest=digests.get("release", ""),
        snapshot_digest=digests.get("snapshot", ""),
        roster_digest=digests.get("roster", ""),
        previous_best=items.snapshot_bests(snapshot),
        roster=items.roster_entries(roster_doc),
        items=epoch_items,
        reference=ref,
        work_dir=work_dir,
        sample_database=sample,
        r_class={str(k): float(v) for k, v in dict(definition.get("r_class") or {}).items()},
        floors=tuple(float(f) for f in definition.get("floors") or (0.8, 0.9, 0.95, 0.99)),
    )


@dataclass(slots=True)
class Runner:
    config: EpochConfig
    context: EpochContext
    host: SandboxHost
    relay: RelayClient
    gpus: GpuPool
    desk: TicketDesk
    client: ServerClient
    wallet: Any
    hotkey: str
    slot_count: int = 1
    agent: probe_judge.ProbeAgent | None = None
    base_dirs: dict[str, Path] = field(default_factory=dict)
    known_public: set[str] = field(default_factory=set)
    protected: set[Path] = field(default_factory=set)
    reference_slm_dir: Path | None = None
    outcomes: dict[str, ArtifactOutcome] = field(default_factory=dict)
    records: dict[str, MinerRecord] = field(default_factory=dict)
    holding: dict[str, tuple[Path, ...]] = field(default_factory=dict)
    seen_manifests: dict[str, str] = field(default_factory=dict)
    seen_weights: dict[str, str] = field(default_factory=dict)
    turns: Turns = field(default_factory=Turns)
    no_reference: dict[str, str] = field(default_factory=dict)
    reference_slm: SlmRun | None = None
    base_slm: dict[str, SlmRun] = field(default_factory=dict)
    reference_run: ReferenceRun | None = None
    reference_router: RouterRun | None = None
    reference_escalation: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    reporter: progress.Reporter = field(default_factory=progress.Reporter)

    def _memory(self, model_dir: Path, config: Mapping[str, Any]) -> int:
        slm = config.get("slm") or {}
        length = self.context.max_model_len or int(
            (slm.get("max_context") if isinstance(slm, Mapping) else None) or DEFAULT_CONTEXT
        )
        try:
            return footprint_bytes(model_dir, length)
        except (OSError, ParameterError) as exc:
            log.warning("memory footprint of %s unknown: %s", model_dir.name, exc)
            return 0

    def _measured(self, sandbox: Sandbox, estimate: int) -> int:
        if not self.limits.on_cpu:
            return estimate
        peak = int(self.host.docker.usage(sandbox.slm_container).get("memory_peak_bytes") or 0)
        return peak or estimate

    @property
    def limits(self) -> Limits:
        return Limits.from_definition(self.context.definition, self.context.max_model_len)

    def _open(
        self,
        artifact_dir: Path,
        slm_dir: Path,
        router_dir: Path | None,
        config: dict[str, Any],
        slot: int,
        gpu: str,
    ) -> Sandbox:
        return self.host.open(
            artifact_dir=artifact_dir,
            slm_dir=slm_dir,
            router_dir=router_dir,
            gpu=gpu,
            config=config,
            sample_database=self.context.sample_database,
            fault_plan=tool_faults.FaultPlan(self.context.seed),
            desk=self.desk,
            slot=slot,
            databases=self.context.items.databases,
            item_databases=self.context.items.item_databases(),
            limits=self.limits,
        )

    def reference_phase(self) -> None:
        slot, gpu = self.gpus.take()
        reference_dir = self.reference_slm_dir or self.context.reference.slm_dir
        try:
            sandbox = self._open(
                self.context.reference.harness_dir,
                reference_dir,
                self.context.reference.router_dir,
                dict(self.context.definition.get("reference_config") or {}),
                slot,
                gpu,
            )
            try:
                client = SlmClient(sandbox.slm_url)
                self.reference_slm = slm.run(
                    self.context,
                    client,
                    memory_bytes=self._memory(reference_dir, {}),
                    parameters=self.context.reference.slm_parameters,
                    items=self.context.items.originals(SLM)
                    + self.context.items.items(ROUTER)
                    + self.context.items.items(ESCALATION),
                )
                self.reference_slm.vram_peak_bytes = self._measured(
                    sandbox, self.reference_slm.vram_peak_bytes
                )
                if sandbox.router_url:
                    self.reference_router = router.run(
                        self.context,
                        sandbox.router_url,
                        self.sources_for(self.reference_slm.drafts(), {}, {}),
                    )
                self.reference_run = self._frontier_redo()
                if (
                    self.context.reference.escalation_dir is not None
                    and self.reference_run is not None
                ):
                    self.host.restart_main(sandbox, self.context.reference.escalation_dir)
                    drafts = escalation.wrong_drafts(
                        self.context, self.reference_run, self.reference_slm.drafts()
                    )
                    self.reference_escalation = escalation.run(
                        self.context,
                        sandbox,
                        self.desk,
                        miner="reference",
                        drafts=drafts,
                        reference=self.reference_run,
                        timeout=self.config.request_seconds,
                        token_price=self.context.token_price(self.context.reference.slm_parameters),
                    )
            finally:
                self.host.close(sandbox)
            for name, path in self.base_dirs.items():
                if path == self.context.reference.slm_dir and self.reference_slm is not None:
                    self.base_slm[name] = self.reference_slm
                    continue
                base_sandbox = self._open(
                    self.context.reference.harness_dir, path, None, {}, slot, gpu
                )
                try:
                    self.base_slm[name] = slm.run(
                        self.context,
                        SlmClient(base_sandbox.slm_url),
                        memory_bytes=self._memory(path, {}),
                        parameters=0,
                    )
                    self.base_slm[name].vram_peak_bytes = self._measured(
                        base_sandbox, self.base_slm[name].vram_peak_bytes
                    )
                finally:
                    self.host.close(base_sandbox)
        finally:
            self.gpus.give(slot, gpu)

    def _frontier_redo(self) -> ReferenceRun | None:
        try:
            found = reference.run(self.context, self.desk)
        except Exception as exc:
            reason = f"the frontier redo failed: {exc}"
        else:
            rows = [*found.system_answers.values(), *found.redo_answers.values()]
            if not rows or any(r.answered for r in rows):
                return found
            reason = f"the frontier redo answered no item: {rows[0].reason}"
        log.warning("the escalation and system pools have no reference this epoch: %s", reason)
        self.no_reference = {pools.ESCALATION: reason, pools.SYSTEM: reason}
        return None

    def sources_for(
        self,
        own: Mapping[str, tuple[str, bool]],
        peers: Mapping[str, Mapping[str, tuple[str, bool]]],
        coldkeys: Mapping[str, str],
    ) -> dict[str, Any]:
        reference_drafts = self.reference_slm.drafts() if self.reference_slm else {}
        return router.build_sources(
            self.context, own=own, reference=reference_drafts, peers=peers, peer_coldkeys=coldkeys
        )

    def claim(self, artifact: Artifact, weights: str) -> str:
        first = self.seen_manifests.get(artifact.manifest_sha)
        if first is not None:
            return f"artifact duplicates the earlier commitment by {first}"
        first = self.seen_weights.get(weights) if weights else None
        if first is not None:
            return f"model weights duplicate the earlier commitment by {first}"
        self.seen_manifests[artifact.manifest_sha] = artifact.hotkey
        if weights:
            self.seen_weights[weights] = artifact.hotkey
        return ""

    def slm_phase(self, artifact: Artifact, slot: int, gpu: str) -> None:
        outcome = self.outcomes.setdefault(artifact.hotkey, ArtifactOutcome(hotkey=artifact.hotkey))
        if artifact.slm_dir is None:
            return
        self.reporter.post(artifact.hotkey, SLM, 0, self.context.pool_size(SLM))
        try:
            sandbox = self._open(
                self.context.reference.harness_dir,
                artifact.slm_dir,
                None,
                artifact.config,
                slot,
                gpu,
            )
            try:
                outcome.slm = slm.run(
                    self.context,
                    SlmClient(sandbox.slm_url),
                    memory_bytes=self._memory(artifact.slm_dir, artifact.config),
                    parameters=artifact.slm_parameters,
                    progress=self.reporter.tracker(artifact.hotkey, SLM),
                )
                outcome.slm.vram_peak_bytes = self._measured(sandbox, outcome.slm.vram_peak_bytes)
            finally:
                outcome.usage = self.host.close(sandbox)
        except (SandboxError, Exception) as exc:
            outcome.failure = f"slm test: {exc}"
            log.warning("%s slm test failed: %s", artifact.hotkey, exc)
            return
        record = self.records.get(artifact.hotkey)
        if record is not None:
            note_checks(self, record)
        self.pool_done(artifact.hotkey, lambda: score_slm(self, artifact, outcome))

    def pool_done(self, hotkey: str, make: Callable[[], PoolScore | None]) -> None:
        try:
            found = make()
        except Exception as exc:
            log.warning("%s pool not scored live: %s", hotkey, exc)
            return
        if found is not None:
            self.outcomes[hotkey].scores[found.pool] = found
        self.share(hotkey)

    def share(self, hotkey: str) -> None:
        record = self.records.get(hotkey)
        if record is None or record.artifact is None or self.config.dry_run:
            return
        try:
            outcome = self.outcomes.get(hotkey)
            live = assembled(self, record.artifact, outcome.scores if outcome else {})
            record.shared.update(s.pool for s in live)
            body = result_body(self, record, live, final=False)
            self.reporter.publish(hotkey, report.sign_artifact(body, self.wallet))
        except Exception as exc:
            log.debug("%s partial result not queued: %s", hotkey, exc)

    def full_phase(self, artifact: Artifact, slot: int, gpu: str) -> None:
        outcome = self.outcomes.setdefault(artifact.hotkey, ArtifactOutcome(hotkey=artifact.hotkey))
        if outcome.failure or self.reference_slm is None:
            return
        if self.reference_run is None and "harness" not in artifact.has:
            return
        slm_dir = artifact.slm_dir or self.context.reference.slm_dir
        router_dir = (
            artifact.artifact_dir / "router"
            if "router" in artifact.has
            else self.context.reference.router_dir
        )
        artifact_dir = (
            artifact.artifact_dir
            if {"harness", "escalation"} & artifact.has
            else self.context.reference.harness_dir
        )
        token_price = self.context.token_price(
            artifact.slm_parameters or self.context.reference.slm_parameters
        )
        try:
            sandbox = self._open(artifact_dir, slm_dir, router_dir, artifact.config, slot, gpu)
            try:
                if "harness" in artifact.has:
                    ran = harness.run(
                        self.context,
                        self.host,
                        sandbox,
                        timeout=self.config.request_seconds,
                        progress=self.reporter.tracker(artifact.hotkey, HARNESS),
                    )
                    outcome.harness = ran
                    self.pool_done(artifact.hotkey, lambda: score_harness(self, ran))
                    self.host.restart_main(sandbox, artifact_dir)
                if "escalation" in artifact.has and self.reference_run is not None:
                    drafts = escalation.wrong_drafts(
                        self.context, self.reference_run, self.reference_slm.drafts()
                    )
                    repaired = escalation.run(
                        self.context,
                        sandbox,
                        self.desk,
                        miner=artifact.hotkey,
                        drafts=drafts,
                        reference=self.reference_run,
                        timeout=self.config.request_seconds,
                        token_price=token_price,
                        progress=self.reporter.tracker(artifact.hotkey, ESCALATION),
                    )
                    outcome.escalation = repaired
                    self.pool_done(artifact.hotkey, lambda: score_escalation(self, repaired))
                if self.reference_run is not None:
                    outcome.system = system.run(
                        self.context,
                        sandbox,
                        self.desk,
                        miner=artifact.hotkey,
                        reference=self.reference_run,
                        timeout=self.config.request_seconds,
                        token_price=token_price,
                        progress=self.reporter.tracker(artifact.hotkey, SYSTEM),
                        on_floor=lambda partial: self.pool_done(
                            artifact.hotkey, lambda: score_system(self, partial)
                        ),
                    )
            finally:
                outcome.usage = self.host.close(sandbox)
        except (SandboxError, Exception) as exc:
            outcome.failure = f"evaluation: {exc}"
            log.warning("%s evaluation failed: %s", artifact.hotkey, exc)

    def router_run(self, record: MinerRecord, sources: Mapping[str, Any]) -> RouterRun:
        if record.router_dir is None or record.artifact is None:
            return RouterRun(sources=[], loaded=False, reason="no router was kept")
        slot, gpu = self.gpus.take()
        process = None
        try:
            process, url = self.host.open_router(
                record.router_dir,
                slot=slot,
                gpu=gpu,
                config=record.artifact.config,
                limits=self.limits,
            )
            return router.run(self.context, url, sources)
        except Exception as exc:
            log.warning("%s router did not run: %s", record.hotkey, exc)
            return RouterRun(sources=[], loaded=False, reason=f"router failed: {exc}")
        finally:
            if process is not None:
                self.host.close_router(process)
            self.gpus.give(slot, gpu)

    def close(self) -> None:
        self.reporter.close()
        self.host.services.stop()
        if self.agent is not None:
            self.agent.stop()
        self.host.docker.sweep()


def set_up_baselines(
    config: EpochConfig,
    client: ServerClient,
    context: EpochContext,
    engine: Docker,
    *,
    wallet: Any,
    hotkey: str,
    reporter: progress.Reporter | None = None,
) -> Runner:
    limits = Limits.from_definition(context.definition, context.max_model_len)
    placement: probe_judge.Placement | None = None
    try:
        _check_host(context.definition)
        pull_images(engine, config, limits, compression=context.grader == compress.GRADER)
        if context.grader == compress.GRADER:
            placement = place_probe(engine, config, context)
    except (DockerError, probe_judge.JudgeError) as exc:
        raise EpochError(str(exc)) from exc
    gpus = placement.slots if placement is not None else tuple(config.gpus or ("0",))
    if placement is not None and placement.slm_memory_share and not config.slm_memory_share:
        config = replace(config, slm_memory_share=placement.slm_memory_share)
    base_dirs = cached_bases(config, context)
    quantised: dict[Path, str] = {}
    reference_dir: Path | None = None
    if placement is not None and placement.reference is not None:
        try:
            reference_dir = collect.download(
                placement.reference[0], placement.reference[1], config.cache_dir
            )
        except Exception as exc:
            raise EpochError(f"the quantised reference could not be fetched: {exc}") from exc
        quantised[reference_dir] = placement.reference[2]
    token = client.relay_token()
    relay = RelayClient(config.relay_url, str(token.get("token", "")))
    desk = TicketDesk(relay, config.arena_id, context.epoch_index)
    agent: probe_judge.ProbeAgent | None = None
    try:
        if placement is not None:
            agent = start_probe_agent(config, context, engine, placement)
        address = host_ip()
        listen = "0.0.0.0" if own_container() else address  # noqa: S104
        services = Services(listen, config.host_services_port)
        services.start()
    except (DockerError, probe_judge.JudgeError, collect.CollectError) as exc:
        if agent is not None:
            agent.stop()
        raise EpochError(str(exc)) from exc
    except Exception:
        if agent is not None:
            agent.stop()
        raise
    runner = Runner(
        config,
        context,
        SandboxHost(config, engine, services, address, quantised=quantised),
        relay,
        GpuPool(gpus),
        desk,
        client,
        wallet,
        hotkey,
        slot_count=len(gpus),
        agent=agent,
        base_dirs=base_dirs,
        known_public=provenance.public_checkpoint_hashes(config.home / "public_checkpoints.json"),
        reference_slm_dir=reference_dir,
        reporter=reporter or progress.Reporter(),
        protected={
            context.reference.slm_dir,
            *([reference_dir] if reference_dir is not None else []),
            *base_dirs.values(),
            *([agent.model_dir] if agent is not None else []),
        },
    )
    try:
        runner.reference_phase()
    except Exception:
        runner.close()
        raise
    return runner


def pull_images(engine: Docker, config: EpochConfig, limits: Limits, *, compression: bool) -> None:
    wanted = [
        config.llama_image if limits.on_cpu else config.vllm_image,
        config.sandbox_image,
        config.gateway_image,
    ]
    if compression:
        wanted.append(config.vllm_image)
    for image in dict.fromkeys(wanted):
        engine.ensure_image(image)


def place_probe(
    engine: Docker, config: EpochConfig, context: EpochContext
) -> probe_judge.Placement:
    memory = engine.gpu_memory(config.cuda_image, tuple(config.gpus or ("0",)))
    spec = context.grading.get("probe_agent") or {}
    reference_model = str((context.definition.get("references") or {}).get("slm") or "")
    placement = probe_judge.place(memory, spec, config.chain.network, reference_model)
    log.info(
        "probe agent %s on GPU %s at memory share %.2f; miners on %s",
        placement.model,
        placement.gpu,
        placement.memory_share,
        ",".join(placement.slots),
    )
    if placement.reference is not None:
        log.warning(
            "testnet small card: probe agent %s, reference runs the quantised build %s",
            placement.model,
            placement.reference[0],
        )
    return placement


def start_probe_agent(
    config: EpochConfig, context: EpochContext, engine: Docker, placement: probe_judge.Placement
) -> probe_judge.ProbeAgent:
    grading = context.grading
    spec = grading.get("probe_agent") or {}
    named = str(spec.get("model") or "")
    model_dir = collect.download(placement.model, placement.revision, config.cache_dir)
    tokenizer = str(grading.get("tokenizer") or named)
    tokenizer_dir = (
        model_dir if tokenizer == named else collect.download(tokenizer, "main", config.cache_dir)
    )
    agent = probe_judge.ProbeAgent(
        engine,
        config.vllm_image,
        model_dir,
        placement.gpu,
        max_tokens=int(spec.get("max_tokens") or 64),
        memory_share=placement.memory_share,
        quantization=placement.quantization,
    )
    agent.start(config.slm_ready_seconds)
    context.judge = compress.Judge(
        count_tokens=probe_judge.token_counter(tokenizer_dir),
        answer=agent.answer,
        pass_threshold=float(grading.get("pass_threshold") or 0.8),
        min_output_tokens=int(grading.get("min_output_tokens") or 300),
        max_output_share=float(grading.get("max_output_share") or 0.1),
    )
    return agent


def declared_bases(config: EpochConfig, context: EpochContext) -> set[str]:
    from huggingface_hub import hf_hub_download

    found: set[str] = set()
    for entry in (context.roster or {}).values():
        repo = str(entry.get("artifact_repo") or "")
        sha = str(entry.get("artifact_sha") or "")
        if not repo or not sha:
            continue
        try:
            manifest = json.loads(
                Path(
                    hf_hub_download(
                        repo,
                        "manifest.json",
                        revision=sha,
                        cache_dir=config.cache_dir / "manifests",
                    )
                ).read_text(encoding="utf-8")
            )
            pin = manifest.get("slm") or {}
            recipe = json.loads(
                Path(
                    hf_hub_download(
                        str(pin.get("repo") or ""),
                        "training/recipe.json",
                        revision=str(pin.get("revision") or "main"),
                        cache_dir=config.cache_dir / "manifests",
                    )
                ).read_text(encoding="utf-8")
            )
        except Exception as exc:
            log.warning("declared base of %s not read: %s", entry.get("hotkey"), exc)
            return set()
        base = str(recipe.get("base_model") or "")
        if base:
            found.add(base)
    return found


def cached_bases(config: EpochConfig, context: EpochContext) -> dict[str, Path]:
    found: dict[str, Path] = {}
    allowed = [str(b) for b in context.definition.get("allowed_bases") or []]
    wanted = (
        declared_bases(config, context) & set(allowed) if config.chain.network == "test" else set()
    )
    if wanted:
        log.info(
            "testnet: only the declared bases are fetched and run: %s", ", ".join(sorted(wanted))
        )
        allowed = [b for b in allowed if b in wanted]
    for base in allowed:
        try:
            found[base] = collect.download(base, "main", config.cache_dir)
        except Exception as exc:
            log.warning("base %s could not be fetched: %s", base, exc)
    return found


def build_queue(
    config: EpochConfig, chain: Any, context: EpochContext
) -> tuple[list[collect.Candidate], dict[str, str]]:
    definition = context.definition
    failures: dict[str, str] = {}
    found = collect.candidates(
        chain,
        config.arena_id,
        commit_close_block=context.open_block,
        roster=context.roster or None,
        dropped=failures,
    )
    queue: list[collect.Candidate] = []
    for candidate in found:
        reason = collect.fee_ok(
            chain,
            candidate,
            pay_to=str(definition.get("fee_coldkey", "")),
            fee_tao=float(definition.get("fee_tao") or 0.0),
        )
        if reason:
            failures[candidate.hotkey] = reason
            continue
        queue.append(candidate)
    queue.sort(key=lambda c: (c.commit_block, c.hotkey))
    return queue, failures


def announce(
    reporter: progress.Reporter,
    context: EpochContext,
    queue: Sequence[collect.Candidate],
    failures: Mapping[str, str],
) -> None:
    for position, candidate in enumerate(queue):
        reporter.post(candidate.hotkey, progress.QUEUED, position + 1, len(queue))
    for hotkey, reason in failures.items():
        if hotkey in context.roster:
            reporter.post(hotkey, progress.REFUSED, note=reason)


def miner_loop(runner: Runner, queue: Sequence[collect.Candidate]) -> None:
    workers = (
        1 if runner.limits.on_cpu else max(1, min(runner.config.concurrency, runner.slot_count))
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda pair: process_miner(runner, *pair), enumerate(queue)))


def process_miner(runner: Runner, position: int, candidate: collect.Candidate) -> MinerRecord:
    record = MinerRecord(
        candidate.hotkey,
        candidate.coldkey,
        candidate.commitment.artifact_sha,
        candidate.commit_block,
    )
    with runner.lock:
        runner.records[candidate.hotkey] = record
    slot, gpu = runner.gpus.take()
    try:
        artifact = fetch_and_check(runner, position, candidate, record)
        if artifact is not None:
            check_provenance(runner, artifact, record)
            outcome = evaluate(runner, artifact, slot, gpu)
            record.drafts = outcome.slm.drafts() if outcome.slm is not None else None
            record.failure = outcome.failure
            if not outcome.failure:
                runner.reporter.post(candidate.hotkey, progress.SCORING)
                record.scores = score_miner(runner, artifact, outcome)
                record.provenance = dict(outcome.provenance)
    except Exception as exc:
        record.failure = record.failure or f"evaluation: {exc}"
        log.warning("%s failed: %s", candidate.hotkey, exc)
    finally:
        runner.turns.done(position)
        runner.gpus.give(slot, gpu)
        send_and_clear(runner, candidate, record)
    return record


def fetch_and_check(
    runner: Runner, position: int, candidate: collect.Candidate, record: MinerRecord
) -> Artifact | None:
    definition = runner.context.definition

    def hold(path: Path) -> None:
        with runner.lock:
            runner.holding[candidate.hotkey] = (*runner.holding.get(candidate.hotkey, ()), path)

    try:
        runner.reporter.post(candidate.hotkey, progress.FETCHING)
        artifact = collect.materialise(
            candidate,
            cache_dir=runner.config.cache_dir,
            arena_id=runner.config.arena_id,
            allowed_bases=[str(b) for b in definition.get("allowed_bases") or []],
            allowed_licenses=[str(b) for b in definition.get("allowed_licenses") or []],
            bounds=_bounds(definition),
            formats=[str(f) for f in definition.get("allowed_formats") or []],
            hold=hold,
            on_downloaded=lambda: runner.reporter.post(candidate.hotkey, progress.CHECKING),
        )
        with runner.lock:
            runner.holding[candidate.hotkey] = _paths_of(artifact)
        record.artifact = artifact
        record.weights_digest = (
            provenance.digest_of_dir(artifact.slm_dir) if artifact.slm_dir else ""
        )
        runner.turns.wait(position)
        reason = runner.claim(artifact, record.weights_digest)
    except collect.CollectError as exc:
        reason = str(exc) or "the artifact could not be fetched"
    finally:
        runner.turns.done(position)
    if record.artifact is not None:
        record.claim = reason
        note_checks(runner, record)
    if reason:
        record.failure = reason
        log.info("%s refused: %s", candidate.hotkey, reason)
        return None
    return artifact


def check_provenance(runner: Runner, artifact: Artifact, record: MinerRecord) -> None:
    outcome = runner.outcomes.setdefault(artifact.hotkey, ArtifactOutcome(hotkey=artifact.hotkey))
    if artifact.slm_dir is not None:
        outcome.provenance = provenance.check(
            artifact.slm_dir, base_dirs=runner.base_dirs, known_public=runner.known_public
        )
        record.provenance = dict(outcome.provenance)
    note_checks(runner, record)
    runner.share(artifact.hotkey)


def note_checks(runner: Runner, record: MinerRecord) -> None:
    if record.artifact is None:
        return
    outcome = runner.outcomes.get(record.hotkey)
    try:
        record.checks = checks.build(
            runner.context,
            record.artifact,
            claim=record.claim,
            found=record.provenance or None,
            run=outcome.slm if outcome is not None else None,
        )
    except Exception as exc:
        log.debug("%s checks not listed: %s", record.hotkey, exc)


def evaluate(runner: Runner, artifact: Artifact, slot: int, gpu: str) -> ArtifactOutcome:
    runner.slm_phase(artifact, slot, gpu)
    runner.full_phase(artifact, slot, gpu)
    return runner.outcomes[artifact.hotkey]


def score_slm(runner: Runner, artifact: Artifact, outcome: ArtifactOutcome) -> PoolScore | None:
    if outcome.slm is None or runner.reference_slm is None or artifact.slm_dir is None:
        return None
    rules = score.Rules.from_definition(runner.context.definition)
    if not outcome.provenance:
        outcome.provenance = provenance.check(
            artifact.slm_dir, base_dirs=runner.base_dirs, known_public=runner.known_public
        )
    prov = outcome.provenance
    gate, envelope_ok = checks.gates(runner.context, artifact, outcome.slm, rules)
    pool_score = score.slm_score(
        outcome.slm,
        runner.reference_slm,
        gate_ok=gate,
        provenance_ok=bool(prov.get("passed")),
        base=runner.base_slm.get(str(prov.get("declared_base", ""))),
        min_items=rules.min_items,
        rules=rules,
        envelope_ok=envelope_ok,
    )
    beats = pool_score.detail.get("beats_base_lb")
    if beats is not None and beats <= 0:
        pool_score = PoolScore(
            pool_score.pool,
            0.0,
            pool_score.lower_bound,
            pool_score.point,
            pool_score.n,
            False,
            {
                **pool_score.detail,
                "provenance": False,
                "reason": "does not beat its declared base",
            },
        )
    return pool_score


def score_harness(runner: Runner, run: HarnessRun) -> PoolScore:
    return score.harness_score(
        run,
        min_items=score.Rules.from_definition(runner.context.definition).min_items,
        limits=runner.context.harness_limits,
    )


def score_escalation(runner: Runner, run: EscalationRun) -> PoolScore:
    return score.escalation_score(
        run,
        runner.reference_escalation,
        seed=runner.context.seed,
        min_items=score.Rules.from_definition(runner.context.definition).min_items,
    )


def score_system(runner: Runner, run: SystemRun) -> PoolScore:
    return score.system_score(
        run, min_items=score.Rules.from_definition(runner.context.definition).min_items
    )


def assembled(
    runner: Runner, artifact: Artifact, found: Mapping[str, PoolScore]
) -> list[PoolScore]:
    ordered = [found[pool] for pool in LIVE_POOLS if pool in found]
    for pool, reason in runner.no_reference.items():
        if pool == pools.SYSTEM or pool in artifact.has:
            ordered.append(
                PoolScore(pool, 0.0, 0.0, 0.0, 0, False, {"reason": f"no reference: {reason}"})
            )
    return ordered


def score_miner(runner: Runner, artifact: Artifact, outcome: ArtifactOutcome) -> list[PoolScore]:
    found = dict(outcome.scores)
    if SLM not in found:
        made = score_slm(runner, artifact, outcome)
        if made is not None:
            found[SLM] = made
    if HARNESS not in found and outcome.harness is not None:
        found[HARNESS] = score_harness(runner, outcome.harness)
    if ESCALATION not in found and outcome.escalation is not None:
        found[ESCALATION] = score_escalation(runner, outcome.escalation)
    if outcome.system is not None:
        found[SYSTEM] = score_system(runner, outcome.system)
    else:
        found.pop(SYSTEM, None)
    return assembled(runner, artifact, found)


def reference_figures(runner: Runner, record: MinerRecord) -> dict[str, Any]:
    found: dict[str, Any] = {}
    run = runner.reference_slm
    if run is not None:
        wanted = {item.item_id for item in runner.context.items.originals(SLM)}
        rows = [r for r in run.runs if r.item_id in wanted] or run.runs
        found[SLM] = {
            "pass_rate": round(sum(1 for r in rows if r.success) / len(rows), 6) if rows else 0.0,
            "ttft_p95_ms": round(run.ttft_p95_ms, 1),
            "tpot_p95_ms": round(run.tpot_p95_ms, 2),
            "vram_peak_bytes": run.vram_peak_bytes,
            "parameters": run.parameters,
        }
    name = str(record.provenance.get("declared_base") or "")
    base = runner.base_slm.get(name) if name else None
    if base is not None:
        found["base"] = {"name": name, "pass_rate": round(base.pass_rate, 6)}
    if runner.reference_run is not None:
        found[SYSTEM] = {
            "cost_micros": runner.reference_run.cost_micros,
            "latency_p95_ms": round(runner.reference_run.latency_p95_ms, 1),
            "items": len(runner.reference_run.system_answers),
        }
    return found


def result_body(
    runner: Runner, record: MinerRecord, scores: Sequence[PoolScore], *, final: bool
) -> dict[str, Any]:
    return report.artifact_body(
        runner.context,
        validator=runner.hotkey,
        miner=record.hotkey,
        artifact_sha=record.artifact_sha,
        scores=scores,
        provenance=record.provenance or _unchecked(record),
        failure=record.failure if final else "",
        pool_details=report.pool_details(scores),
        checks=record.checks if record.artifact is not None else None,
        reference=reference_figures(runner, record) if record.artifact is not None else None,
        progress_final=final,
    )


def score_router(runner: Runner, run: RouterRun) -> PoolScore:
    context = runner.context
    routed = [i.item_id for i in context.items.items(ROUTER)]
    items_by_source = {
        router.OWN: routed,
        router.REFERENCE: routed,
        router.PEER: routed,
        router.ADVERSARIAL: [str(r.get("item_id", "")) for r in context.items.adversarial],
    }
    return score.router_score(
        run,
        runner.reference_router,
        floors=context.floors,
        families=context.items.families,
        items_by_source=items_by_source,
        seed=context.seed,
        rules=score.Rules.from_definition(context.definition),
    )


def send_and_clear(runner: Runner, candidate: collect.Candidate, record: MinerRecord) -> None:
    try:
        post_result(runner, record)
        save_record(runner, record)
    except Exception as exc:
        log.warning("%s record not kept: %s", record.hotkey, exc)
    try:
        clear_miner(runner, candidate, record)
    except Exception as exc:
        log.warning("%s files not cleared: %s", record.hotkey, exc)
        return
    record.cleared = True
    if record.posted and not record.failure:
        runner.reporter.post(record.hotkey, progress.CLEARED)


def withdrawn(record: MinerRecord) -> list[PoolScore]:
    reason = record.failure[: checks.TEXT_LIMIT]
    return [
        PoolScore(pool, 0.0, 0.0, 0.0, 0, False, {"reason": reason})
        for pool in pools.POOLS
        if pool in record.shared
    ]


def _submit(runner: Runner, record: MinerRecord) -> None:
    scores = withdrawn(record) if record.failure else record.scores
    body = result_body(runner, record, scores, final=True)
    try:
        runner.client.submit_artifact_result(report.sign_artifact(body, runner.wallet))
    except ServerError as exc:
        if " answered 422" not in str(exc):
            raise
        log.warning("%s full result refused, sending the scores alone: %s", record.hotkey, exc)
        plain = report.artifact_body(
            runner.context,
            validator=runner.hotkey,
            miner=record.hotkey,
            artifact_sha=record.artifact_sha,
            scores=scores,
            provenance=record.provenance or _unchecked(record),
            failure=record.failure,
        )
        runner.client.submit_artifact_result(report.sign_artifact(plain, runner.wallet))


def post_result(runner: Runner, record: MinerRecord) -> bool:
    if runner.config.dry_run:
        log.info("%s dry run, result not posted", record.hotkey)
        return False
    first = not record.posted
    runner.reporter.settle(record.hotkey)
    try:
        _submit(runner, record)
    except Exception as exc:
        record.posted = False
        log.warning("%s result not accepted, kept for the retry at close: %s", record.hotkey, exc)
        return False
    record.posted = True
    if first:
        stage = progress.REFUSED if record.failure else progress.SENT
        runner.reporter.post(record.hotkey, stage, note=record.failure)
        if record.cleared and not record.failure:
            runner.reporter.post(record.hotkey, progress.CLEARED)
    return True


def save_record(runner: Runner, record: MinerRecord) -> None:
    folder = runner.context.work_dir / RECORDS_DIR
    try:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{record.hotkey}.json").write_text(
            json.dumps(record.wire(), sort_keys=True), encoding="utf-8"
        )
    except OSError as exc:
        log.warning("%s record not written: %s", record.hotkey, exc)


def clear_miner(runner: Runner, candidate: collect.Candidate, record: MinerRecord) -> None:
    artifact = record.artifact
    if artifact is None:
        paths: Sequence[Path] = collect.fetched_paths(candidate, runner.config.cache_dir)
    else:
        paths = _paths_of(artifact)
        if not record.failure and "router" in artifact.has:
            record.router_dir = _keep_router(runner, artifact)
    with runner.lock:
        runner.holding.pop(record.hotkey, None)
        held = {path for kept in runner.holding.values() for path in kept}
    for path in paths:
        if path not in held and path not in runner.protected:
            _remove(path)


def retry_unposted(runner: Runner) -> None:
    if runner.config.dry_run:
        return
    for record in runner.records.values():
        if not record.posted and post_result(runner, record):
            save_record(runner, record)


def router_pass(runner: Runner) -> None:
    reference_drafts = runner.reference_slm.drafts() if runner.reference_slm else {}
    peers = {h: r.drafts for h, r in runner.records.items() if r.drafts is not None}
    coldkeys = {h: r.coldkey for h, r in runner.records.items()}
    for hotkey, record in runner.records.items():
        if record.router_dir is None:
            continue
        try:
            if not record.failure:
                own = record.drafts if record.drafts is not None else reference_drafts
                others = {h: d for h, d in peers.items() if h != hotkey}
                run = runner.router_run(record, runner.sources_for(own, others, coldkeys))
                kept = [s for s in record.scores if s.pool != pools.ROUTER]
                record.scores = [*kept, score_router(runner, run)]
                post_result(runner, record)
                save_record(runner, record)
        except Exception as exc:
            log.warning("%s router pass failed: %s", hotkey, exc)
        finally:
            _remove(record.router_dir)
            record.router_dir = None


def rank_and_set_weights(
    config: EpochConfig, chain: Any, runner: Runner, scores: Mapping[str, list[PoolScore]]
) -> tuple[dict[str, float], dict[str, float]]:
    context = runner.context
    standings = score.standings_for(
        runner.outcomes,
        scores,
        context.previous_best,
        {h: r.artifact_sha for h, r in runner.records.items()},
        {h: r.commit_block for h, r in runner.records.items()},
    )
    burn_to = (
        _owner(chain) if score.Rules.from_definition(context.definition).burn_empty_pools else ""
    )
    weights, burned, vector = weights_for(standings, chain.snapshot(refresh=True), burn_to)
    if vector is not None and not config.dry_run:
        ok, reason = chain.set_weights(vector)
        if not ok:
            log.error("weights rejected: %s", reason)
    return weights, burned


def final_report(
    runner: Runner, failures: Mapping[str, str], scores: Mapping[str, list[PoolScore]]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    context = runner.context
    artifacts = {h: r.artifact for h, r in runner.records.items() if r.artifact is not None}
    try:
        spent = [r.body for r in runner.relay.receipts(context.arena_id, context.epoch_index)]
    except RelayError as exc:
        log.warning("relay receipts not read: %s", exc)
        spent = []
    receipts = {h: list(spent) for h in artifacts}
    signed = report.sign(
        report.build(
            context,
            validator_hotkey=runner.hotkey,
            artifacts=artifacts,
            scores=scores,
            failures=failures,
            receipts=receipts,
        ),
        runner.wallet,
    ).to_dict()
    if runner.config.dry_run:
        log.info("dry run, the epoch report is not posted")
    else:
        try:
            runner.client.submit_report(signed)
        except ServerError as exc:
            log.warning("report not accepted by the server: %s", exc)
    try:
        probe_sets.write(context, runner.outcomes, artifacts)
    except OSError as exc:
        log.warning("probe sets not written: %s", exc)
    context.work_dir.mkdir(parents=True, exist_ok=True)
    (context.work_dir / REPORT_FILE).write_text(json.dumps(signed, indent=2), encoding="utf-8")
    clear_work_folder(context.work_dir)
    return signed, receipts


def clear_work_folder(work_dir: Path) -> None:
    if not work_dir.is_dir():
        return
    for child in work_dir.iterdir():
        if child.name in KEPT_AT_CLOSE:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def weights_for(
    standings: Mapping[str, list[pools.Standing]], snapshot: Any, burn_to: str = ""
) -> tuple[dict[str, float], dict[str, float], Any]:
    weights, burned = pools.combine(standings, burn_to=burn_to)
    uid_by_hotkey = snapshot.uid_by_hotkey
    by_uid = {uid_by_hotkey[h]: w for h, w in weights.items() if h in uid_by_hotkey}
    vector = (
        quantise_weights(restrict_to_metagraph(by_uid, len(snapshot.neurons))) if by_uid else None
    )
    return weights, burned, vector


def run_epoch(
    config: EpochConfig, *, chain: Any, wallet: Any, epoch_index: int, docker: Docker | None = None
) -> Outcome:
    hotkey = hotkey_address(wallet)
    client = ServerClient(config.server_url, hotkey, wallet)
    context = load_context(config, client, chain, epoch_index=epoch_index, hotkey=hotkey)
    reporter = progress.Reporter(
        client, config.arena_id, context.epoch_index, enabled=not config.dry_run
    )
    try:
        queue, failures = build_queue(config, chain, context)
        log.info("epoch %d: %d queued, %d refused", epoch_index, len(queue), len(failures))
        announce(reporter, context, queue, failures)
        runner = set_up_baselines(
            config,
            client,
            context,
            docker or Docker(),
            wallet=wallet,
            hotkey=hotkey,
            reporter=reporter,
        )
    except BaseException:
        reporter.close()
        raise
    try:
        miner_loop(runner, queue)
        router_pass(runner)
        retry_unposted(runner)
    finally:
        runner.close()
    for hotkey_, record in runner.records.items():
        if record.failure:
            failures[hotkey_] = record.failure
    scores = {h: list(r.scores) for h, r in runner.records.items() if not r.failure}
    weights, burned = rank_and_set_weights(config, chain, runner, scores)
    signed, receipts = final_report(runner, failures, scores)
    return Outcome(
        scores=scores,
        failures=failures,
        receipts=receipts,
        weights=weights,
        burned=burned,
        report=signed,
    )


def _paths_of(artifact: Artifact) -> tuple[Path, ...]:
    return (artifact.artifact_dir, *([artifact.slm_dir] if artifact.slm_dir else []))


def _unchecked(record: MinerRecord) -> dict[str, Any]:
    if record.failure:
        return {"passed": False, "reason": "not checked, the miner failed before scoring"}
    return {"passed": False, "reason": "not checked, the artifact carries no slm"}


def _keep_router(runner: Runner, artifact: Artifact) -> Path | None:
    source = artifact.artifact_dir / "router"
    if not source.is_dir():
        return None
    target = runner.context.work_dir / ROUTERS_DIR / artifact.hotkey
    try:
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    except OSError as exc:
        log.warning("%s router folder not kept: %s", artifact.hotkey, exc)
        return None
    return target


def _remove(path: Path | None) -> None:
    if path is None:
        return
    shutil.rmtree(path, ignore_errors=True)
    with contextlib.suppress(OSError):
        if path.parent.is_dir() and not any(path.parent.iterdir()):
            path.parent.rmdir()


def _check_host(definition: Mapping[str, Any]) -> None:
    envelope = definition.get("envelope") or {}
    reference = envelope.get("reference") if isinstance(envelope, Mapping) else None
    wanted = str((reference or {}).get("arch") or "")
    if not wanted:
        return
    machine = platform.machine().lower()
    host = (
        "arm64"
        if machine in ("arm64", "aarch64")
        else "x86_64"
        if machine
        in (
            "x86_64",
            "amd64",
        )
        else machine
    )
    if host != wanted:
        raise EpochError(f"this arena is measured on {wanted}; this host is {host}")


def _bounds(definition: Mapping[str, Any]) -> tuple[int, int] | None:
    budgets = definition.get("budgets") or {}
    if not isinstance(budgets, Mapping) or not budgets.get("max_parameters"):
        return None
    return int(budgets.get("min_parameters") or 0), int(budgets["max_parameters"])


def _owner(chain: Any) -> str:
    reader = getattr(chain, "owner_hotkey", None)
    if not callable(reader):
        log.error("this chain client cannot name the subnet owner; empty pools are not burned")
        return ""
    try:
        return str(reader() or "")
    except Exception as exc:
        log.error("the subnet owner hotkey could not be read; empty pools are not burned: %s", exc)
        return ""


def describe(reference: Reference) -> str:
    return f"reference slm {reference.slm_dir.name}, {len(reference.variant_dirs)} harness variants"
