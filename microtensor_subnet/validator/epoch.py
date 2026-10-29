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

from microtensor_subnet.artifact.params import (
    ParameterError,
    footprint_bytes,
    kv_cache_bytes,
    model_parameters,
)
from microtensor_subnet.artifact.validate import Manifest, ManifestError, SlmPin, load_manifest
from microtensor_subnet.chain.commitment import CommitmentError, SubmissionCommitment
from microtensor_subnet.chain.epochs import Epoch, epoch_seed
from microtensor_subnet.chain.wallet import hotkey_address
from microtensor_subnet.chain.weights import quantise_weights, restrict_to_metagraph
from microtensor_subnet.core.reports import PoolScore
from microtensor_subnet.scoring import pools
from microtensor_subnet.scoring.stats import percentile
from microtensor_subnet.serving.probes import sets as probe_sets
from microtensor_subnet.validator import (
    checks,
    collect,
    compress,
    evidence,
    items,
    progress,
    provenance,
    report,
    score,
    spotcheck,
    transcripts,
)
from microtensor_subnet.validator import replay as replaying
from microtensor_subnet.validator.client import ServerClient, ServerError
from microtensor_subnet.validator.config import EpochConfig, gpu_profile, profile_memory_bytes
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
    config: EpochConfig,
    client: ServerClient,
    chain: Any,
    *,
    epoch_index: int,
    hotkey: str,
    files_out: dict[str, bytes] | None = None,
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
    if files_out is not None:
        files_out.update(files)
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
    return _beats_base(pool_score)


def _beats_base(pool_score: PoolScore) -> PoolScore:
    beats = pool_score.detail.get("beats_base_lb")
    if beats is not None and beats <= 0:
        return PoolScore(
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
    return router_pool_score(runner.context, run, runner.reference_router)


def router_pool_score(
    context: EpochContext, run: RouterRun, reference_router: RouterRun | None
) -> PoolScore:
    routed = [i.item_id for i in context.items.items(ROUTER)]
    items_by_source = {
        router.OWN: routed,
        router.REFERENCE: routed,
        router.PEER: routed,
        router.ADVERSARIAL: [str(r.get("item_id", "")) for r in context.items.adversarial],
    }
    return score.router_score(
        run,
        reference_router,
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


MIRRORED: Final[str] = "mirrored"
MINER_MODE: Final[str] = "miner"
WEIGHT_SUFFIXES: Final[tuple[str, ...]] = (".safetensors", ".gguf")
ARTIFACT_DIR: Final[str] = "artifact"
SLM_DIR: Final[str] = "slm"
RECEIPTS_KIND: Final[str] = "receipts"
MIRROR_KIND: Final[str] = "mirror"
RECEIPT_ROOT_TAG: Final[str] = "mtrc"
REPAIR_ROWS: Final[str] = "repair"
BASE_ROWS: Final[str] = "base"
DEFAULT_BAN_CONFIRMATIONS: Final[int] = 2
MIRROR_COMMIT_BLOCK_KEYS: Final[tuple[str, ...]] = ("commit_block", "block", "committed_at")


def mirrored_miners(document: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    body = dict((document or {}).get("body") or {})
    found: dict[str, dict[str, Any]] = {}
    miners = body.get("miners") or {}
    for hotkey, entry in miners.items() if isinstance(miners, Mapping) else []:
        if isinstance(entry, Mapping) and str(entry.get("status", "")) == MIRRORED:
            found[str(hotkey)] = dict(entry)
    return found


def _files_of(entry: Mapping[str, Any], key: str) -> dict[str, str]:
    block = entry.get(key) or {}
    files = block.get("files") if isinstance(block, Mapping) else None
    if not isinstance(files, Mapping):
        return {}
    return {str(name): str(digest).lower() for name, digest in files.items()}


def _is_weight(name: str) -> bool:
    return name.endswith(WEIGHT_SUFFIXES)


def _safe_name(name: str) -> str:
    if name.startswith("/") or ".." in name.split("/") or not name.strip():
        raise EpochError(f"the mirror lists an unsafe file name {name!r}")
    return name


def materialise_tree(
    client: ServerClient, files: Mapping[str, str], into: Path, *, weights: bool
) -> Path:
    fetch = transcripts.client_method(client, "mirror_blob")
    into.mkdir(parents=True, exist_ok=True)
    for name, digest in sorted(files.items()):
        if _is_weight(name) != weights:
            continue
        fetch(digest, into / _safe_name(name))
    if weights:
        (into / ".complete").write_text("ok", encoding="utf-8")
    return into


def _raw_manifest(artifact_dir: Path) -> dict[str, Any]:
    try:
        raw = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EpochError(f"manifest.json could not be read from the mirror: {exc}") from exc
    if not isinstance(raw, dict):
        raise EpochError("manifest.json is not an object")
    return raw


def manifest_of(artifact_dir: Path, entry: Mapping[str, Any]) -> Manifest:
    try:
        return load_manifest(artifact_dir)
    except ManifestError as exc:
        raw = _raw_manifest(artifact_dir)
        raw_pin = raw.get("slm")
        pin: Mapping[str, Any] = raw_pin if isinstance(raw_pin, Mapping) else {}
        raw_block = entry.get("slm")
        slm_block: Mapping[str, Any] = raw_block if isinstance(raw_block, Mapping) else {}
        components = tuple(str(c) for c in raw.get("components") or [])
        if not components:
            raise EpochError(f"manifest.json is unreadable: {exc}") from exc
        slm = (
            SlmPin(
                repo=str(pin.get("repo") or slm_block.get("repo") or ""),
                revision=str(pin.get("revision") or slm_block.get("revision") or ""),
                files={
                    str(k): str(v)
                    for k, v in ((pin or {}).get("files") or _files_of(entry, "slm")).items()
                },
            )
            if "slm" in components
            else None
        )
        return Manifest(
            arena=str(raw.get("arena") or ""),
            components=components,
            hardware_class=str(raw.get("hardware_class") or ""),
            files={str(k): str(v) for k, v in (raw.get("files") or {}).items()},
            slm=slm,
        )


def _artifact_config(artifact_dir: Path) -> dict[str, Any]:
    try:
        parsed = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _commit_block(row: Mapping[str, Any]) -> int:
    for key in MIRROR_COMMIT_BLOCK_KEYS:
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def artifact_from_mirror(
    context: EpochContext,
    entry: Mapping[str, Any],
    *,
    hotkey: str,
    coldkey: str,
    artifact_dir: Path,
    slm_dir: Path | None,
) -> Artifact:
    roster_row = context.roster.get(hotkey) or {}
    raw_block = entry.get("slm")
    slm_block: Mapping[str, Any] = raw_block if isinstance(raw_block, Mapping) else {}
    hf_user = str(roster_row.get("hf_user") or "") or str(slm_block.get("repo", "")).split("/")[0]
    try:
        commitment = SubmissionCommitment(
            context.arena_id, hf_user, str(entry.get("artifact_sha", "")).lower(), 0, 0
        )
    except CommitmentError as exc:
        raise EpochError(f"the mirror entry cannot be read as a commitment: {exc}") from exc
    manifest = manifest_of(artifact_dir, entry)
    return Artifact(
        hotkey=hotkey,
        coldkey=coldkey,
        commitment=commitment,
        commit_block=_commit_block(roster_row) or _commit_block(entry),
        artifact_dir=artifact_dir,
        slm_dir=slm_dir,
        manifest=manifest,
        manifest_sha=str(entry.get("manifest_sha256") or ""),
        config=_artifact_config(artifact_dir),
    )


def signed_body(client: ServerClient, context: EpochContext, key: str, kind: str) -> dict[str, Any]:
    document = transcripts.client_method(client, kind)(context.arena_id, context.epoch_index)
    if document is None:
        raise EpochError(f"the epoch {kind} document has not been published")
    items.verify_document(document, key)
    return dict(document.get("body") or {})


def warn_on_chain_root(chain: Any, context: EpochContext, root: str) -> None:
    owner = _owner(chain)
    if not owner:
        return
    try:
        raw = str(chain.commitments([owner]).get(owner, "") or "")
    except Exception as exc:
        log.debug("owner commitment not read: %s", exc)
        return
    parts = raw.strip().split("|")
    if len(parts) != 4 or parts[0] != RECEIPT_ROOT_TAG:
        return
    if parts[1] != context.arena_id or parts[2] != str(context.epoch_index):
        return
    if parts[3].lower() != root.lower():
        log.warning(
            "the receipts root on chain %s differs from the signed document %s", parts[3], root
        )


def validators_on_profile(
    client: ServerClient, context: EpochContext, profile: str, me: str
) -> int:
    try:
        rows = transcripts.client_method(client, "epoch_validators")(
            context.arena_id, context.epoch_index
        )
    except Exception as exc:
        log.warning("the epoch validators were not listed: %s", exc)
        rows = []
    found = {
        str(row.get("hotkey", ""))
        for row in rows
        if isinstance(row, Mapping) and profile in [str(p) for p in row.get("profiles") or []]
    }
    found.add(me)
    found.discard("")
    return len(found)


def chain_coldkeys(chain: Any) -> dict[str, str]:
    try:
        snapshot = chain.snapshot(refresh=True)
        found = snapshot.coldkeys() if hasattr(snapshot, "coldkeys") else {}
    except Exception as exc:
        log.warning("coldkeys not read from the chain: %s", exc)
        return {}
    return {str(k): str(v) for k, v in dict(found).items()}


def banned_hotkeys(client: ServerClient, context: EpochContext) -> set[str]:
    try:
        rows = transcripts.client_method(client, "bans")(context.arena_id)
    except Exception as exc:
        log.warning("bans not read: %s", exc)
        return set()
    found: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        since = int(row.get("from_epoch") or 0)
        until = int(row.get("until_epoch") or 0)
        if since <= context.epoch_index < until:
            found.add(str(row.get("hotkey", "")))
    found.discard("")
    return found


@dataclass(slots=True)
class MinerModeRunner:
    config: EpochConfig
    context: EpochContext
    client: ServerClient
    wallet: Any
    hotkey: str
    docker: Docker
    services: Any
    tap: replaying.PromptTap
    host: SandboxHost
    inputs: items.InputsBundle
    labels: items.LabelsBundle
    mirror: dict[str, dict[str, Any]]
    leaves: dict[str, str]
    receipt_root: str
    close_block_hash: str
    profile: str
    server_key: str
    reporter: progress.Reporter
    validators: int = 1
    coldkeys: dict[str, str] = field(default_factory=dict)
    agent: probe_judge.ProbeAgent | None = None
    base_dirs: dict[str, Path] = field(default_factory=dict)
    known_public: set[str] = field(default_factory=set)
    reference_slm: SlmRun | None = None
    base_slm: dict[str, SlmRun] = field(default_factory=dict)
    reference_escalation: EscalationRun | None = None
    reference_router: RouterRun | None = None
    reference_timing: spotcheck.ReferenceTiming | None = None
    no_reference: dict[str, str] = field(default_factory=dict)
    records: dict[str, MinerRecord] = field(default_factory=dict)
    outcomes: dict[str, ArtifactOutcome] = field(default_factory=dict)
    miner_transcripts: dict[str, dict[str, Any]] = field(default_factory=dict)
    transcript_shas: dict[str, str] = field(default_factory=dict)
    replays: dict[str, replaying.ReplayResult] = field(default_factory=dict)
    spot_checks: dict[str, spotcheck.SpotCheck] = field(default_factory=dict)
    canaries: dict[str, dict[str, int]] = field(default_factory=dict)
    fraud: dict[str, str] = field(default_factory=dict)
    sampled: list[str] = field(default_factory=list)
    seen_manifests: dict[str, str] = field(default_factory=dict)
    seen_weights: dict[str, str] = field(default_factory=dict)

    @property
    def evaluation(self) -> dict[str, Any]:
        block = self.context.definition.get("evaluation") or {}
        return dict(block) if isinstance(block, Mapping) else {}

    @property
    def limits(self) -> Limits:
        return Limits.from_definition(self.context.definition, self.context.max_model_len)

    @property
    def gpu(self) -> bool:
        return gpu_profile(self.profile)

    @property
    def settings(self) -> spotcheck.Settings:
        return spotcheck.Settings.from_definition(self.context.definition)

    @property
    def ban_confirmations(self) -> int:
        ban = self.evaluation.get("ban") or {}
        found = ban.get("confirmations") if isinstance(ban, Mapping) else None
        return int(found or DEFAULT_BAN_CONFIRMATIONS)

    @property
    def relay_allowance(self) -> int:
        return int(self.evaluation.get("relay_allowance_micros") or 0)

    def miner_dir(self, hotkey: str) -> Path:
        return self.config.mirror_dir / hotkey

    def evidence_fields(self, hotkey: str) -> dict[str, Any]:
        entry = self.mirror.get(hotkey) or {}
        return {
            "arena_id": self.context.arena_id,
            "epoch": self.context.epoch_index,
            "miner_hotkey": hotkey,
            "profile": str(entry.get("profile") or self.profile),
            "weights_sha256": str(entry.get("weights_sha256") or ""),
        }

    def claim(self, hotkey: str, entry: Mapping[str, Any]) -> str:
        manifest = str(entry.get("manifest_sha256") or "")
        weights = str(entry.get("weights_sha256") or "")
        first = self.seen_manifests.get(manifest) if manifest else None
        if first is not None and first != hotkey:
            return f"artifact duplicates the earlier commitment by {first}"
        first = self.seen_weights.get(weights) if weights else None
        if first is not None and first != hotkey:
            return f"model weights duplicate the earlier commitment by {first}"
        if manifest:
            self.seen_manifests.setdefault(manifest, hotkey)
        if weights:
            self.seen_weights.setdefault(weights, hotkey)
        return ""

    def close(self) -> None:
        self.reporter.close()
        self.services.stop()
        self.tap.stop()
        if self.agent is not None:
            self.agent.stop()
        self.docker.sweep()


def start_miner_mode_probe(
    config: EpochConfig, context: EpochContext, engine: Docker, *, gpu: bool
) -> probe_judge.ProbeAgent | None:
    if context.grader != compress.GRADER:
        return None
    grading = context.grading
    if gpu and grading.get("probe_agent"):
        try:
            placement = place_probe(engine, config, context)
            return start_probe_agent(config, context, engine, placement)
        except (DockerError, probe_judge.JudgeError, collect.CollectError) as exc:
            log.warning("the vLLM probe agent did not start, using llama.cpp: %s", exc)
    spec = probe_judge.ProbeSpec.from_definition(context.definition)
    model_file = probe_judge.fetch_probe(spec, config.cache_dir)
    images = (context.definition.get("evaluation") or {}).get("images") or {}
    llama = images.get("llama") if isinstance(images, Mapping) else None
    image = config.llama_image
    if isinstance(llama, Mapping) and llama.get("image") and llama.get("digest"):
        image = spotcheck.pinned(str(llama["image"]), str(llama["digest"]))
    engine.ensure_image(image)
    threads = int(config.extra.get("probe_threads") or probe_judge.LLAMA_THREADS)
    agent = probe_judge.ProbeAgent(
        engine,
        image,
        model_file.parent,
        "",
        max_tokens=spec.max_tokens,
        backend=probe_judge.LLAMA_BACKEND,
        model_file=model_file.name,
        threads=threads,
    )
    agent.start(config.slm_ready_seconds)
    tokenizer = str(grading.get("tokenizer") or "")
    tokenizer_dir = (
        collect.download(tokenizer, "main", config.cache_dir)
        if tokenizer
        else context.reference.slm_dir
    )
    context.judge = compress.Judge(
        count_tokens=probe_judge.token_counter(tokenizer_dir),
        answer=agent.answer,
        pass_threshold=float(grading.get("pass_threshold") or 0.8),
        min_output_tokens=int(grading.get("min_output_tokens") or 300),
        max_output_share=float(grading.get("max_output_share") or 0.1),
    )
    return agent


def reference_memory(context: EpochContext) -> int:
    max_len = context.max_model_len or DEFAULT_CONTEXT
    try:
        kv = kv_cache_bytes(context.reference.slm_dir, max_len)
    except (OSError, ParameterError):
        kv = 0
    return transcripts.computed_memory(
        context.reference.slm_parameters,
        transcripts.weight_format(context.reference.slm_dir),
        kv,
        context.definition,
    )


def reference_figures_from_labels(runner: MinerModeRunner) -> None:
    context = runner.context
    labels = runner.labels
    rows = list(labels.rows(SLM).values())
    if rows:
        runner.reference_slm = transcripts.reference_slm_run(
            context,
            rows,
            parameters=context.reference.slm_parameters,
            memory_bytes=reference_memory(context),
        )
    else:
        runner.no_reference[pools.SLM] = "the labels bundle carries no reference slm answers"
    for name in runner.base_dirs:
        base_rows = [r for r in labels.rows(BASE_ROWS).values() if r.model == name]
        if base_rows:
            runner.base_slm[name] = transcripts.reference_slm_run(
                context, base_rows, parameters=0, memory_bytes=0
            )
    runner.reference_escalation = transcripts.reference_escalation_run(context, labels)
    if not labels.rows(SYSTEM):
        reason = "the labels bundle carries no reference system answers"
        runner.no_reference[pools.SYSTEM] = reason
    if not labels.rows(ESCALATION):
        runner.no_reference[pools.ESCALATION] = "the labels bundle carries no redo answers"


def set_up_miner_mode(
    config: EpochConfig,
    client: ServerClient,
    chain: Any,
    context: EpochContext,
    files: Mapping[str, bytes],
    engine: Docker,
    *,
    wallet: Any,
    hotkey: str,
    reporter: progress.Reporter,
) -> MinerModeRunner:
    key = items.server_key(client)
    profile = config.resolved_profile()
    inputs_body, inputs_blob = items.fetch_inputs(client, config.arena_id, context.epoch_index, key)
    inputs = items.unpack_inputs(
        inputs_body, inputs_blob, epoch_seed=context.seed, work_dir=context.work_dir
    )
    labels = items.unpack_labels(files)
    receipts = signed_body(client, context, key, RECEIPTS_KIND)
    root = str(receipts.get("merkle_root") or "")
    leaves = {str(h): str(s).lower() for h, s in (receipts.get("leaves") or {}).items() if h and s}
    if transcripts.merkle_root(leaves) != root.lower():
        raise EpochError("the receipts document's leaves do not hash to its merkle root")
    warn_on_chain_root(chain, context, root)
    mirror = mirrored_miners({"body": signed_body(client, context, key, MIRROR_KIND)})
    close_block = int(receipts.get("close_block") or inputs.close_block or 0)
    close_hash = chain.block_hash(close_block) if close_block else context.open_block_hash
    try:
        _check_host(context.definition)
        engine.ensure_image(config.gateway_image)
        engine.ensure_image(config.replayer_image)
    except (DockerError, EpochError) as exc:
        raise EpochError(str(exc)) from exc
    base_dirs = cached_bases(config, context)
    address = host_ip()
    listen = "0.0.0.0" if own_container() else address  # noqa: S104
    services = replaying.services_class()(listen, config.host_services_port)
    services.start()
    loopback = "127.0.0.1" if listen == "0.0.0.0" else address  # noqa: S104
    tap = replaying.PromptTap(
        listen, config.host_services_port + 1, f"http://{loopback}:{services.port}"
    )
    tap.start()
    agent: probe_judge.ProbeAgent | None = None
    try:
        agent = start_miner_mode_probe(config, context, engine, gpu=gpu_profile(profile))
    except (DockerError, probe_judge.JudgeError, collect.CollectError) as exc:
        services.stop()
        tap.stop()
        raise EpochError(str(exc)) from exc
    except Exception:
        services.stop()
        tap.stop()
        raise
    runner = MinerModeRunner(
        config=config,
        context=context,
        client=client,
        wallet=wallet,
        hotkey=hotkey,
        docker=engine,
        services=services,
        tap=tap,
        host=SandboxHost(config, engine, services, address),
        inputs=inputs,
        labels=labels,
        mirror=mirror,
        leaves=leaves,
        receipt_root=root,
        close_block_hash=close_hash,
        profile=profile,
        server_key=key,
        reporter=reporter,
        validators=validators_on_profile(client, context, profile, hotkey),
        coldkeys=chain_coldkeys(chain),
        agent=agent,
        base_dirs=base_dirs,
        known_public=provenance.public_checkpoint_hashes(config.home / "public_checkpoints.json"),
    )
    reference_figures_from_labels(runner)
    for banned in banned_hotkeys(client, context):
        if runner.mirror.pop(banned, None) is not None:
            log.info("%s is banned this epoch and is left out", banned)
    return runner


def ordered_miners(runner: MinerModeRunner) -> list[str]:
    def key(hotkey: str) -> tuple[int, str]:
        row = runner.context.roster.get(hotkey) or {}
        return (_commit_block(row) or _commit_block(runner.mirror[hotkey]), hotkey)

    return sorted(runner.mirror, key=key)


def replayer_for(runner: MinerModeRunner) -> replaying.Replayer:
    context = runner.context
    budgets = context.budgets
    trees = runner.inputs.trees
    harness = trees.get("harness") or context.reference.harness_dir
    variants = (
        tuple(path for name, path in sorted(trees.items()) if name.startswith("variant-"))
        or context.reference.variant_dirs
    )
    return replaying.Replayer(
        docker=runner.docker,
        config=runner.config,
        context=context,
        services=runner.services,
        tap=runner.tap,
        host_ip=runner.host.host_ip,
        server_key=runner.server_key,
        replayer_image=runner.config.replayer_image,
        reference_harness_dir=harness,
        variant_dirs=variants,
        request_seconds=runner.config.request_seconds,
        caps=replay_caps(runner.labels),
        seeds=runner.inputs.seeds(),
        drafts=runner.inputs.drafts,
        relay_allowance_micros=runner.relay_allowance,
        max_input_tokens=int(budgets.get("max_input_tokens") or 0),
        max_output_tokens=int(budgets.get("max_output_tokens") or 0),
    )


def replay_caps(labels: items.LabelsBundle) -> dict[str, int]:
    found: dict[str, int] = {}
    for pool in (ESCALATION, SYSTEM):
        for item_id, row in labels.rows(pool).items():
            found[item_id] = max(1, int(row.cost_micros * 1.5))
    return found


def transcript_reason(runner: MinerModeRunner, hotkey: str, transcript: Mapping[str, Any]) -> str:
    entry = runner.mirror.get(hotkey) or {}
    reason = transcripts.verify(
        transcript,
        hotkey=hotkey,
        arena_id=runner.context.arena_id,
        epoch_index=runner.context.epoch_index,
        leaves=runner.leaves,
        root=runner.receipt_root,
    )
    if reason:
        return reason
    for key in ("profile", "manifest_sha256", "weights_sha256"):
        wanted = str(entry.get(key) or "")
        if wanted and str(transcript.get(key, "")) != wanted:
            return f"transcript {key} differs from the mirror"
    return ""


def pull_weights(runner: MinerModeRunner, hotkey: str) -> Path | None:
    entry = runner.mirror.get(hotkey)
    record = runner.records.get(hotkey)
    if entry is None or record is None or record.artifact is None:
        return None
    files = _files_of(entry, SLM_DIR)
    if not any(_is_weight(name) for name in files):
        return None
    into = runner.miner_dir(hotkey) / SLM_DIR
    if not (into / ".complete").is_file():
        runner.reporter.post(hotkey, progress.FETCHING)
        materialise_tree(runner.client, files, into, weights=True)
    record.artifact.slm_dir = into
    return into


def miner_memory(runner: MinerModeRunner, artifact: Artifact) -> int:
    if artifact.slm_dir is None:
        return 0
    max_len = runner.context.max_model_len or int(
        (artifact.config.get("slm") or {}).get("max_context") or DEFAULT_CONTEXT
    )
    try:
        kv = kv_cache_bytes(artifact.slm_dir, max_len)
    except (OSError, ParameterError) as exc:
        log.warning("kv cache of %s unknown: %s", artifact.hotkey, exc)
        kv = 0
    return transcripts.computed_memory(
        artifact.slm_parameters,
        transcripts.weight_format(artifact.slm_dir),
        kv,
        runner.context.definition,
    )


def prepare_miner(runner: MinerModeRunner, hotkey: str, replayer: replaying.Replayer) -> None:
    entry = runner.mirror[hotkey]
    record = MinerRecord(
        hotkey,
        str(
            runner.context.roster.get(hotkey, {}).get("coldkey") or runner.coldkeys.get(hotkey, "")
        ),
        str(entry.get("artifact_sha") or ""),
        _commit_block(runner.context.roster.get(hotkey) or {}),
    )
    runner.records[hotkey] = record
    runner.reporter.post(hotkey, progress.FETCHING)
    try:
        artifact_dir = materialise_tree(
            runner.client,
            _files_of(entry, ARTIFACT_DIR),
            runner.miner_dir(hotkey) / ARTIFACT_DIR,
            weights=False,
        )
        slm_files = _files_of(entry, SLM_DIR)
        slm_dir = (
            materialise_tree(
                runner.client, slm_files, runner.miner_dir(hotkey) / SLM_DIR, weights=False
            )
            if slm_files
            else None
        )
        record.artifact = artifact_from_mirror(
            runner.context,
            entry,
            hotkey=hotkey,
            coldkey=record.coldkey,
            artifact_dir=artifact_dir,
            slm_dir=slm_dir,
        )
        sha = runner.leaves.get(hotkey, "")
        if not sha:
            raise EpochError("no transcript receipt under the Freeze 2 root")
        transcript = transcripts.fetch(runner.client, sha, runner.context.work_dir)
        reason = transcript_reason(runner, hotkey, transcript)
        if reason:
            raise EpochError(reason)
        runner.miner_transcripts[hotkey] = transcript
        runner.transcript_shas[hotkey] = sha
    except (EpochError, transcripts.TranscriptError, ServerError, OSError) as exc:
        record.failure = f"replay: {exc}"
        log.info("%s not replayed: %s", hotkey, exc)
        runner.reporter.post(hotkey, progress.REPLAYED, detail={"ok": False})
        return
    runner.reporter.post(hotkey, progress.CHECKING)
    artifact = record.artifact
    count = replaying.token_counter(artifact.slm_dir) if artifact.slm_dir else None
    try:
        result = replayer.replay(
            runner.miner_transcripts[hotkey],
            artifact_dir=artifact.artifact_dir,
            has=artifact.has,
            count_tokens=count,
        )
    except Exception as exc:
        result = replaying.ReplayResult(ok=False).fail("", 0, f"replayer: {exc}")
    runner.replays[hotkey] = result
    runner.reporter.post(hotkey, progress.REPLAYED, detail={"ok": result.ok})
    if not result.ok:
        record.failure = f"replay: {result.reason}"
        log.info("%s replay failed: %s", hotkey, result.reason)
        return
    record.claim = runner.claim(hotkey, entry)
    if record.claim:
        record.failure = record.claim
        return
    try:
        slm_dir = pull_weights(runner, hotkey)
    except (ServerError, EpochError, OSError, RuntimeError) as exc:
        record.failure = f"mirror: {exc}"
        log.warning("%s weights not pulled: %s", hotkey, exc)
        return
    if slm_dir is None:
        return
    try:
        artifact.slm_parameters = model_parameters(slm_dir)
    except ParameterError as exc:
        record.failure = f"weights: {exc}"
        return
    record.weights_digest = str(entry.get("weights_sha256") or "")
    outcome = runner.outcomes.setdefault(hotkey, ArtifactOutcome(hotkey=hotkey))
    outcome.provenance = provenance.check(
        slm_dir, base_dirs=runner.base_dirs, known_public=runner.known_public
    )
    record.provenance = dict(outcome.provenance)


def grade_miner(runner: MinerModeRunner, hotkey: str) -> ArtifactOutcome | None:
    record = runner.records[hotkey]
    artifact = record.artifact
    result = runner.replays.get(hotkey)
    if record.failure or artifact is None or result is None or not result.ok:
        return None
    runner.reporter.post(hotkey, progress.SCORING)
    kept = runner.outcomes.get(hotkey)
    outcome = transcripts.to_outcome(
        runner.context,
        hotkey,
        runner.miner_transcripts[hotkey],
        runner.labels,
        result,
        parameters=artifact.slm_parameters,
        memory_bytes=miner_memory(runner, artifact),
        token_price=runner.context.token_price(
            artifact.slm_parameters or runner.context.reference.slm_parameters
        ),
    )
    if kept is not None:
        outcome.provenance = kept.provenance
    runner.outcomes[hotkey] = outcome
    record.drafts = outcome.slm.drafts() if outcome.slm is not None else None
    note_checks_for(runner, record)
    return outcome


def note_checks_for(runner: MinerModeRunner, record: MinerRecord) -> None:
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


def apply_spot_check(
    outcome: ArtifactOutcome, spot: spotcheck.SpotCheck, transcript: Mapping[str, Any]
) -> None:
    run = outcome.slm
    if run is None:
        return
    run.ttft_p95_ms = spot.ttft_p95_ms
    run.tpot_p95_ms = spot.tpot_p95_ms
    if spot.vram_peak_bytes:
        run.vram_peak_bytes = spot.vram_peak_bytes
    first_seq = {
        str(row.get("item_id", "")): int((transcripts.slm_calls(row) or [{}])[0].get("seq") or 0)
        for row in transcripts.items_of(transcript)
        if str(row.get("pool", "")) == SLM
    }
    for item in [*run.runs, *run.twins]:
        timing = spot.timing.get((item.item_id, first_seq.get(item.item_id, 0)))
        if timing is not None:
            item.ttft_ms, item.tpot_ms = timing
            item.latency_ms = timing[0] + timing[1] * max(0, item.completion_tokens - 1)


def spot_check_engine(runner: MinerModeRunner, model_dir: Path) -> spotcheck.Engine:
    config = runner.config
    return spotcheck.engine_for(
        runner.docker,
        definition=runner.context.definition,
        model_dir=model_dir,
        gpu=(config.gpus or ("0",))[0],
        profile=runner.profile,
        profile_bytes=profile_memory_bytes(runner.profile),
        max_model_len=runner.context.max_model_len,
        memory_share=config.slm_memory_share,
        ready_seconds=config.slm_ready_seconds,
        cuda_image=config.cuda_image,
    )


def calls_for(runner: MinerModeRunner, hotkey: str) -> list[spotcheck.Call]:
    outcome = runner.outcomes.get(hotkey)
    result = runner.replays.get(hotkey)
    transcript = runner.miner_transcripts.get(hotkey)
    if outcome is None or result is None or transcript is None:
        return []
    passed: dict[tuple[str, str], bool] = {}
    if outcome.slm is not None:
        for item in [*outcome.slm.runs, *outcome.slm.twins]:
            passed[(item.item_id, transcripts.CONDITION_MINER)] = item.success
    if outcome.harness is not None:
        for item in outcome.harness.miner.runs:
            passed[(item.item_id, transcripts.CONDITION_MINER)] = item.success
        for condition in outcome.harness.baselines:
            for item in condition.runs:
                passed[(item.item_id, condition.name)] = item.success
    if outcome.escalation is not None:
        for item in outcome.escalation.repairs:
            passed[(item.item_id, transcripts.CONDITION_REPAIR)] = item.success
    if outcome.system is not None:
        for floor in outcome.system.floors:
            for item in floor.runs:
                passed[(item.item_id, f"{transcripts.SYSTEM_PREFIX}{floor.floor}")] = item.success
    twin_of = {i.item_id: i.twin_of for i in runner.context.items.items(SLM) if i.twin_of}
    pairs = {**twin_of, **{twin: item_id for item_id, twin in twin_of.items()}}
    moved = (
        spotcheck.moved_items(outcome.slm, runner.reference_slm, pairs)
        if outcome.slm is not None
        else set()
    )
    return spotcheck.collect_calls(
        transcript, result.prompts, passed=passed, twins=twin_of, moved=moved
    )


def post_evidence(runner: MinerModeRunner, bodies: Sequence[Mapping[str, Any]]) -> list[str]:
    posted: list[str] = []
    for body in bodies:
        signed = evidence.sign(body, runner.wallet, runner.hotkey)
        if runner.config.dry_run:
            log.info("dry run, evidence for %s not posted", body.get("miner_hotkey"))
            continue
        try:
            posted.append(evidence.post(runner.client, signed))
        except Exception as exc:
            log.warning("evidence for %s not posted: %s", body.get("miner_hotkey"), exc)
    return posted


def spot_check_phase(runner: MinerModeRunner) -> None:
    if not runner.gpu:
        return
    settings = runner.settings
    candidates = [
        h
        for h, r in runner.records.items()
        if not r.failure
        and h in runner.outcomes
        and str((runner.mirror.get(h) or {}).get("profile") or "") == runner.profile
    ]
    runner.sampled = spotcheck.sample_miners(
        candidates,
        close_block_hash=runner.close_block_hash,
        validator_hotkey=runner.hotkey,
        validators_on_profile=runner.validators,
        validators_per_miner=settings.validators_per_miner,
    )
    log.info(
        "spot check: %d of %d miners on %s", len(runner.sampled), len(candidates), runner.profile
    )
    rules = score.Rules.from_definition(runner.context.definition)
    for hotkey in runner.sampled:
        record = runner.records[hotkey]
        try:
            model_dir = pull_weights(runner, hotkey)
        except (ServerError, EpochError, OSError, RuntimeError) as exc:
            log.warning("%s weights not pulled for the spot check: %s", hotkey, exc)
            continue
        if model_dir is None:
            continue
        calls = calls_for(runner, hotkey)
        uniform, weighted = spotcheck.sample_calls(
            calls,
            min_calls=settings.min_calls,
            rng=spotcheck.seeded(runner.close_block_hash, runner.hotkey, hotkey),
        )
        fields = runner.evidence_fields(hotkey)
        try:
            engine = spot_check_engine(runner, model_dir)
            engine.start()
        except (spotcheck.SpotCheckError, DockerError, SandboxError) as exc:
            log.warning("%s spot check engine did not start: %s", hotkey, exc)
            continue
        try:
            spot = spotcheck.check_miner(
                engine,
                uniform,
                weighted,
                budgets=runner.context.budgets,
                violation_share=rules.violation_share,
                evidence_fields=fields,
            )
            checked, hits = spotcheck.canary_check(
                engine, runner.labels.canaries, evidence_fields=fields
            )
        except (spotcheck.SpotCheckError, DockerError) as exc:
            log.warning("%s spot check failed: %s", hotkey, exc)
            continue
        finally:
            engine.stop()
        runner.spot_checks[hotkey] = spot
        runner.canaries[hotkey] = {"checked": checked, "hits": len(hits)}
        apply_spot_check(runner.outcomes[hotkey], spot, runner.miner_transcripts[hotkey])
        note_checks_for(runner, record)
        post_evidence(runner, [*spot.evidence, *hits])
        runner.reporter.post(
            hotkey, progress.SPOT_CHECKED, detail={"matched": spot.matched, "calls": spot.calls}
        )


def reference_timing_phase(runner: MinerModeRunner) -> None:
    if not runner.gpu or runner.reference_slm is None:
        return
    try:
        engine = spot_check_engine(runner, runner.context.reference.slm_dir)
        engine.start()
    except (spotcheck.SpotCheckError, DockerError, SandboxError) as exc:
        log.warning("the reference engine did not start: %s", exc)
        return
    try:
        timing = spotcheck.time_reference(
            engine,
            runner.context,
            count=runner.settings.reference_calls,
            drafts={i: r.answer for i, r in runner.labels.rows(SLM).items()},
            seeds=runner.inputs.seeds(),
        )
    except (spotcheck.SpotCheckError, DockerError) as exc:
        log.warning("the reference timing failed: %s", exc)
        return
    finally:
        engine.stop()
    runner.reference_timing = timing
    runner.reference_slm.ttft_p95_ms = timing.ttft_p95_ms
    runner.reference_slm.tpot_p95_ms = timing.tpot_p95_ms
    if timing.vram_peak_bytes:
        runner.reference_slm.vram_peak_bytes = timing.vram_peak_bytes
    for spot in runner.spot_checks.values():
        spot.reference_ttft_p95_ms = timing.ttft_p95_ms
        spot.reference_tpot_p95_ms = timing.tpot_p95_ms
        spot.reference_vram_peak_bytes = timing.vram_peak_bytes


def confirm_phase(runner: MinerModeRunner) -> None:
    if not runner.gpu:
        return
    rows = evidence.fetch(runner.client, runner.context.arena_id, runner.context.epoch_index)
    pending: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("profile", "")) != runner.profile or not evidence.verify(row):
            continue
        if evidence.already_confirmed(row, runner.hotkey) or not evidence.evidence_id(row):
            continue
        pending.setdefault(str(row.get("miner_hotkey", "")), []).append(row)
    canaries = {str(c.get("item_id", "")): c for c in runner.labels.canaries}
    for hotkey, cited in pending.items():
        if hotkey not in runner.records or runner.records[hotkey].failure:
            continue
        calls = {(c.item_id, c.seq): c for c in calls_for(runner, hotkey)}
        try:
            model_dir = pull_weights(runner, hotkey)
            if model_dir is None:
                continue
            engine = spot_check_engine(runner, model_dir)
            engine.start()
        except (
            spotcheck.SpotCheckError,
            DockerError,
            SandboxError,
            ServerError,
            EpochError,
            OSError,
            RuntimeError,
        ) as exc:
            log.warning("%s evidence not re-run: %s", hotkey, exc)
            continue
        try:
            for row in cited:
                try:
                    found = spotcheck.recheck(engine, row, calls=calls, canaries=canaries)
                except Exception as exc:
                    log.warning("evidence %s not re-run: %s", evidence.evidence_id(row), exc)
                    continue
                if found is None:
                    continue
                confirmed, tokens = found
                body = evidence.confirmation(
                    confirmed=confirmed,
                    recomputed_tokens=tokens,
                    wallet=runner.wallet,
                    hotkey=runner.hotkey,
                )
                if runner.config.dry_run:
                    continue
                try:
                    evidence.confirm(runner.client, evidence.evidence_id(row), body)
                except Exception as exc:
                    log.warning("confirmation of %s not posted: %s", evidence.evidence_id(row), exc)
        finally:
            engine.stop()


def collect_fraud(runner: MinerModeRunner) -> None:
    rows = evidence.fetch(runner.client, runner.context.arena_id, runner.context.epoch_index)
    runner.fraud = evidence.confirmed_by_miner(rows, runner.ban_confirmations)
    for hotkey, found in runner.fraud.items():
        record = runner.records.get(hotkey)
        if record is not None and not record.failure:
            record.failure = f"fraud: {found}"
            log.warning("%s zeroed on confirmed evidence %s", hotkey, found)


def miner_mode_scores(runner: MinerModeRunner, record: MinerRecord) -> list[PoolScore]:
    context = runner.context
    artifact = record.artifact
    outcome = runner.outcomes.get(record.hotkey)
    if artifact is None or outcome is None:
        return []
    rules = score.Rules.from_definition(context.definition)
    found: dict[str, PoolScore] = {}
    if outcome.slm is not None and runner.reference_slm is not None and artifact.slm_dir:
        envelope = checks.Envelope.of(context)
        spot = runner.spot_checks.get(record.hotkey)
        sized = envelope.sized(artifact.slm_parameters)
        fits = envelope.fits(outcome.slm)
        fast = spot.envelope_ok if spot is not None else True
        if rules.per_item_envelope:
            gate, envelope_ok = sized, fits and fast
        else:
            gate, envelope_ok = sized and fits and fast, True
        prov = outcome.provenance
        found[SLM] = _beats_base(
            score.slm_score(
                outcome.slm,
                runner.reference_slm,
                gate_ok=gate,
                provenance_ok=bool(prov.get("passed")),
                base=runner.base_slm.get(str(prov.get("declared_base", ""))),
                min_items=rules.min_items,
                rules=rules,
                envelope_ok=envelope_ok,
            )
        )
    if outcome.harness is not None:
        found[HARNESS] = score.harness_score(
            outcome.harness, min_items=rules.min_items, limits=context.harness_limits
        )
    if outcome.escalation is not None:
        found[ESCALATION] = score.escalation_score(
            outcome.escalation,
            runner.reference_escalation,
            seed=context.seed,
            min_items=rules.min_items,
        )
    if outcome.system is not None:
        found[SYSTEM] = score.system_score(outcome.system, min_items=rules.min_items)
    if outcome.router is not None:
        found[ROUTER] = router_pool_score(context, outcome.router, runner.reference_router)
    ordered = [found[pool] for pool in (*LIVE_POOLS, ROUTER) if pool in found]
    for pool, reason in runner.no_reference.items():
        if pool in found:
            continue
        if pool == pools.SYSTEM or pool in artifact.has:
            ordered.append(
                PoolScore(pool, 0.0, 0.0, 0.0, 0, False, {"reason": f"no reference: {reason}"})
            )
    return ordered


def router_phase(runner: MinerModeRunner) -> None:
    context = runner.context
    reference_drafts = runner.reference_slm.drafts() if runner.reference_slm else {}
    peers = {h: r.drafts for h, r in runner.records.items() if r.drafts is not None}
    coldkeys = {h: r.coldkey for h, r in runner.records.items()}
    gpu = (runner.config.gpus or ("0",))[0]
    reference_dir = context.reference.router_dir
    if reference_dir is not None:
        runner.reference_router = run_router(
            runner,
            reference_dir,
            dict(context.definition.get("reference_config") or {}),
            router.build_sources(
                context,
                own=reference_drafts,
                reference=reference_drafts,
                peers={},
                peer_coldkeys={},
            ),
            gpu,
        )
    for hotkey, record in runner.records.items():
        artifact = record.artifact
        if record.failure or artifact is None or "router" not in artifact.has:
            continue
        outcome = runner.outcomes.get(hotkey)
        if outcome is None:
            continue
        own = record.drafts if record.drafts is not None else reference_drafts
        others = {h: d for h, d in peers.items() if h != hotkey}
        sources = router.build_sources(
            context, own=own, reference=reference_drafts, peers=others, peer_coldkeys=coldkeys
        )
        outcome.router = run_router(
            runner, artifact.artifact_dir / "router", artifact.config, sources, gpu
        )


def run_router(
    runner: MinerModeRunner,
    router_dir: Path,
    config: dict[str, Any],
    sources: Mapping[str, Any],
    gpu: str,
) -> RouterRun:
    if not router_dir.is_dir():
        return RouterRun(sources=[], loaded=False, reason="no router was kept")
    process = None
    try:
        process, url = runner.host.open_router(
            router_dir, slot=0, gpu=gpu, config=config, limits=runner.limits
        )
        return router.run(runner.context, url, sources)
    except Exception as exc:
        log.warning("router in %s did not run: %s", router_dir, exc)
        return RouterRun(sources=[], loaded=False, reason=f"router failed: {exc}")
    finally:
        if process is not None:
            runner.host.close_router(process)


def miner_mode_figures(runner: MinerModeRunner, record: MinerRecord) -> dict[str, Any]:
    found: dict[str, Any] = {}
    run = runner.reference_slm
    if run is not None:
        rows = run.runs
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
    reference = runner.labels.rows(SYSTEM)
    if reference:
        found[SYSTEM] = {
            "cost_micros": sum(r.cost_micros for r in reference.values()),
            "latency_p95_ms": round(
                percentile([r.latency_ms for r in reference.values()], 0.95), 1
            ),
            "items": len(reference),
        }
    return found


def post_miner_mode_result(runner: MinerModeRunner, record: MinerRecord) -> bool:
    if runner.config.dry_run:
        return False
    scores = withdrawn(record) if record.failure else record.scores
    body = report.artifact_body(
        runner.context,
        validator=runner.hotkey,
        miner=record.hotkey,
        artifact_sha=record.artifact_sha,
        scores=scores,
        provenance=record.provenance or _unchecked(record),
        failure=record.failure,
        pool_details=report.pool_details(scores),
        checks=record.checks if record.artifact is not None else None,
        reference=miner_mode_figures(runner, record) if record.artifact is not None else None,
        progress_final=True,
    )
    try:
        try:
            runner.client.submit_artifact_result(report.sign_artifact(body, runner.wallet))
        except ServerError as exc:
            if " answered 422" not in str(exc):
                raise
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
    except Exception as exc:
        log.warning("%s result not accepted: %s", record.hotkey, exc)
        return False
    record.posted = True
    stage = progress.REFUSED if record.failure else progress.SENT
    runner.reporter.post(record.hotkey, stage, note=record.failure)
    return True


def miner_mode_weights(
    config: EpochConfig,
    chain: Any,
    runner: MinerModeRunner,
    scores: Mapping[str, list[PoolScore]],
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


def miner_mode_report(
    runner: MinerModeRunner, failures: Mapping[str, str], scores: Mapping[str, list[PoolScore]]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    context = runner.context
    artifacts = {h: r.artifact for h, r in runner.records.items() if r.artifact is not None}
    receipts = {h: list(o.receipts) for h, o in runner.outcomes.items()}
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
    signed["evaluation_mode"] = MINER_MODE
    signed["receipt_root"] = runner.receipt_root
    signed["validator_profile"] = runner.profile
    if runner.reference_timing is not None:
        signed["reference_spot_check"] = runner.reference_timing.to_dict()
    for row in signed.get("miners") or []:
        hotkey = str(row.get("hotkey", ""))
        replay = runner.replays.get(hotkey)
        spot = runner.spot_checks.get(hotkey)
        row["transcript_sha256"] = runner.transcript_shas.get(hotkey, "")
        row["replay"] = replay.to_dict() if replay is not None else None
        row["spot_check"] = spot.to_dict() if spot is not None else None
        row["canary"] = dict(runner.canaries.get(hotkey) or {"checked": 0, "hits": 0})
    if runner.config.dry_run:
        log.info("dry run, the epoch report is not posted")
    else:
        try:
            runner.client.submit_report(signed)
        except ServerError as exc:
            log.warning("report not accepted by the server: %s", exc)
    context.work_dir.mkdir(parents=True, exist_ok=True)
    (context.work_dir / REPORT_FILE).write_text(json.dumps(signed, indent=2), encoding="utf-8")
    clear_work_folder(context.work_dir)
    return signed, receipts


def save_miner_mode_record(runner: MinerModeRunner, record: MinerRecord) -> None:
    folder = runner.context.work_dir / RECORDS_DIR
    try:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{record.hotkey}.json").write_text(
            json.dumps(record.wire(), sort_keys=True), encoding="utf-8"
        )
    except OSError as exc:
        log.warning("%s record not written: %s", record.hotkey, exc)


def run_miner_mode_epoch(
    config: EpochConfig, *, chain: Any, wallet: Any, epoch_index: int, docker: Docker | None = None
) -> Outcome:
    hotkey = hotkey_address(wallet)
    client = ServerClient(config.server_url, hotkey, wallet)
    files: dict[str, bytes] = {}
    context = load_context(
        config, client, chain, epoch_index=epoch_index, hotkey=hotkey, files_out=files
    )
    if not miner_mode_definition(context.definition):
        raise EpochError(f"arena {config.arena_id} is not in miner evaluation mode")
    reporter = progress.Reporter(
        client, config.arena_id, context.epoch_index, enabled=not config.dry_run
    )
    try:
        runner = set_up_miner_mode(
            config,
            client,
            chain,
            context,
            files,
            docker or Docker(),
            wallet=wallet,
            hotkey=hotkey,
            reporter=reporter,
        )
    except BaseException:
        reporter.close()
        raise
    failures: dict[str, str] = {}
    try:
        queue = ordered_miners(runner)
        log.info(
            "epoch %d: %d mirrored miners, %d receipts, profile %s",
            epoch_index,
            len(queue),
            len(runner.leaves),
            runner.profile,
        )
        for position, miner in enumerate(queue):
            reporter.post(miner, progress.QUEUED, position + 1, len(queue))
        replayer = replayer_for(runner)
        for miner in queue:
            try:
                prepare_miner(runner, miner, replayer)
            except Exception as exc:
                record = runner.records.setdefault(miner, MinerRecord(miner, "", "", 0))
                record.failure = record.failure or f"evaluation: {exc}"
                log.warning("%s failed: %s", miner, exc)
        for miner in queue:
            try:
                grade_miner(runner, miner)
            except Exception as exc:
                runner.records[miner].failure = f"grading: {exc}"
                log.warning("%s grading failed: %s", miner, exc)
        reference_timing_phase(runner)
        spot_check_phase(runner)
        confirm_phase(runner)
        collect_fraud(runner)
        router_phase(runner)
        for miner, record in runner.records.items():
            if not record.failure:
                try:
                    record.scores = miner_mode_scores(runner, record)
                except Exception as exc:
                    record.failure = f"scoring: {exc}"
                    log.warning("%s scoring failed: %s", miner, exc)
            post_miner_mode_result(runner, record)
            save_miner_mode_record(runner, record)
            _remove(runner.miner_dir(miner))
    finally:
        runner.close()
    for miner, record in runner.records.items():
        if record.failure:
            failures[miner] = record.failure
    scores = {h: list(r.scores) for h, r in runner.records.items() if not r.failure}
    weights, burned = miner_mode_weights(config, chain, runner, scores)
    signed, receipts = miner_mode_report(runner, failures, scores)
    return Outcome(
        scores=scores,
        failures=failures,
        receipts=receipts,
        weights=weights,
        burned=burned,
        report=signed,
    )


def miner_mode_definition(definition: Mapping[str, Any]) -> bool:
    block = definition.get("evaluation") or {}
    mode = block.get("mode") if isinstance(block, Mapping) else None
    return str(mode or "validator") == MINER_MODE
