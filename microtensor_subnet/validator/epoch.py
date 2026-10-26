from __future__ import annotations

import json
import logging
import platform
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from microtensor_subnet.artifact.params import ParameterError, footprint_bytes
from microtensor_subnet.chain.epochs import Epoch, epoch_seed
from microtensor_subnet.chain.wallet import hotkey_address
from microtensor_subnet.chain.weights import quantise_weights, restrict_to_metagraph
from microtensor_subnet.core.reports import PoolScore
from microtensor_subnet.scoring import pools
from microtensor_subnet.serving.probes import sets as probe_sets
from microtensor_subnet.validator import collect, compress, items, provenance, report, score
from microtensor_subnet.validator.client import ServerClient, ServerError
from microtensor_subnet.validator.config import EpochConfig
from microtensor_subnet.validator.context import (
    ESCALATION,
    ROUTER,
    SLM,
    Artifact,
    EpochContext,
    Reference,
)
from microtensor_subnet.validator.pools import escalation, harness, reference, router, slm, system
from microtensor_subnet.validator.relay import RelayClient, TicketDesk
from microtensor_subnet.validator.results import ArtifactOutcome, ReferenceRun, RouterRun, SlmRun
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
from microtensor_subnet.validator.sandbox.docker import Docker
from microtensor_subnet.validator.sandbox.services import Services
from microtensor_subnet.validator.sandbox.slm import SlmClient

log = logging.getLogger("microtensor_subnet.validator.epoch")
JUDGE_SHARED_SHARE: Final[float] = 0.5
DEFAULT_CONTEXT = 8192
DEFAULT_MEMORY_BYTES = 24 * 1024**3
DEFAULT_MAX_PARAMETERS = 8_000_000_000


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
    ref = items.reference_components(trees, definition, config.cache_dir)
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


def gather(
    config: EpochConfig, chain: Any, context: EpochContext, *, commit_close: int
) -> tuple[dict[str, Artifact], dict[str, str]]:
    definition = context.definition
    found = collect.candidates(
        chain, config.arena_id, commit_close_block=commit_close, roster=context.roster or None
    )
    artifacts: dict[str, Artifact] = {}
    failures: dict[str, str] = {}
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
        try:
            artifacts[candidate.hotkey] = collect.materialise(
                candidate,
                cache_dir=config.cache_dir,
                arena_id=config.arena_id,
                allowed_bases=[str(b) for b in definition.get("allowed_bases") or []],
                allowed_licenses=[str(b) for b in definition.get("allowed_licenses") or []],
                bounds=_bounds(definition),
                formats=[str(f) for f in definition.get("allowed_formats") or []],
            )
        except collect.CollectError as exc:
            failures[candidate.hotkey] = str(exc)
    for hotkey, reason in duplicates(artifacts).items():
        failures[hotkey] = reason
        artifacts.pop(hotkey, None)
    return artifacts, failures


def duplicates(artifacts: Mapping[str, Artifact]) -> dict[str, str]:
    seen_weights: dict[str, str] = {}
    seen_manifests: dict[str, str] = {}
    found: dict[str, str] = {}
    for hotkey, artifact in sorted(artifacts.items(), key=lambda kv: (kv[1].commit_block, kv[0])):
        if artifact.manifest_sha in seen_manifests:
            first = seen_manifests[artifact.manifest_sha]
            found[hotkey] = f"artifact duplicates the earlier commitment by {first}"
            continue
        weights = provenance.digest_of_dir(artifact.slm_dir) if artifact.slm_dir else ""
        if weights and weights in seen_weights:
            first = seen_weights[weights]
            found[hotkey] = f"model weights duplicate the earlier commitment by {first}"
            continue
        seen_manifests[artifact.manifest_sha] = hotkey
        if weights:
            seen_weights[weights] = hotkey
    return found


@dataclass(slots=True)
class Runner:
    config: EpochConfig
    context: EpochContext
    host: SandboxHost
    relay: RelayClient
    gpus: GpuPool
    desk: TicketDesk
    outcomes: dict[str, ArtifactOutcome] = field(default_factory=dict)
    reference_slm: SlmRun | None = None
    base_slm: dict[str, SlmRun] = field(default_factory=dict)
    reference_run: ReferenceRun | None = None
    reference_router: RouterRun | None = None
    reference_escalation: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)

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

    def slm_phase(self, artifact: Artifact) -> None:
        outcome = self.outcomes.setdefault(artifact.hotkey, ArtifactOutcome(hotkey=artifact.hotkey))
        if artifact.slm_dir is None:
            return
        slot, gpu = self.gpus.take()
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
                )
                outcome.slm.vram_peak_bytes = self._measured(sandbox, outcome.slm.vram_peak_bytes)
            finally:
                outcome.usage = self.host.close(sandbox)
        except (SandboxError, Exception) as exc:
            outcome.failure = f"slm test: {exc}"
            log.warning("%s slm test failed: %s", artifact.hotkey, exc)
        finally:
            self.gpus.give(slot, gpu)

    def reference_phase(self, base_dirs: Mapping[str, Path]) -> None:
        slot, gpu = self.gpus.take()
        try:
            sandbox = self._open(
                self.context.reference.harness_dir,
                self.context.reference.slm_dir,
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
                    memory_bytes=self._memory(self.context.reference.slm_dir, {}),
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
                        self._sources(self.reference_slm.drafts(), {}, {}),
                    )
                self.reference_run = reference.run(self.context, self.desk)
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
            for name, path in base_dirs.items():
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

    def _sources(
        self,
        own: Mapping[str, tuple[str, bool]],
        peers: Mapping[str, Mapping[str, tuple[str, bool]]],
        coldkeys: Mapping[str, str],
    ) -> dict[str, Any]:
        reference_drafts = self.reference_slm.drafts() if self.reference_slm else {}
        return router.build_sources(
            self.context, own=own, reference=reference_drafts, peers=peers, peer_coldkeys=coldkeys
        )

    def full_phase(
        self,
        artifact: Artifact,
        peers: Mapping[str, Mapping[str, tuple[str, bool]]],
        coldkeys: Mapping[str, str],
    ) -> None:
        outcome = self.outcomes.setdefault(artifact.hotkey, ArtifactOutcome(hotkey=artifact.hotkey))
        if outcome.failure or self.reference_run is None or self.reference_slm is None:
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
        slot, gpu = self.gpus.take()
        try:
            sandbox = self._open(artifact_dir, slm_dir, router_dir, artifact.config, slot, gpu)
            try:
                own = (outcome.slm or self.reference_slm).drafts()
                if "harness" in artifact.has:
                    outcome.harness = harness.run(
                        self.context, self.host, sandbox, timeout=self.config.request_seconds
                    )
                    self.host.restart_main(sandbox, artifact_dir)
                if "router" in artifact.has and sandbox.router_url:
                    outcome.router = router.run(
                        self.context, sandbox.router_url, self._sources(own, peers, coldkeys)
                    )
                if "escalation" in artifact.has:
                    drafts = escalation.wrong_drafts(
                        self.context, self.reference_run, self.reference_slm.drafts()
                    )
                    outcome.escalation = escalation.run(
                        self.context,
                        sandbox,
                        self.desk,
                        miner=artifact.hotkey,
                        drafts=drafts,
                        reference=self.reference_run,
                        timeout=self.config.request_seconds,
                        token_price=token_price,
                    )
                outcome.system = system.run(
                    self.context,
                    sandbox,
                    self.desk,
                    miner=artifact.hotkey,
                    reference=self.reference_run,
                    timeout=self.config.request_seconds,
                    token_price=token_price,
                )
            finally:
                outcome.usage = self.host.close(sandbox)
        except (SandboxError, Exception) as exc:
            outcome.failure = f"evaluation: {exc}"
            log.warning("%s evaluation failed: %s", artifact.hotkey, exc)
        finally:
            self.gpus.give(slot, gpu)


def score_all(
    runner: Runner,
    artifacts: Mapping[str, Artifact],
    base_dirs: Mapping[str, Path],
    known_public: set[str],
) -> dict[str, list[PoolScore]]:
    context = runner.context
    rules = score.Rules.from_definition(context.definition)
    budgets = context.budgets
    low = int(budgets.get("min_parameters") or 0)
    high = int(budgets.get("max_parameters") or DEFAULT_MAX_PARAMETERS)
    memory_cap = int(
        budgets.get("memory_bytes") or budgets.get("vram_bytes") or DEFAULT_MEMORY_BYTES
    )
    scores: dict[str, list[PoolScore]] = {}
    items_by_source = {
        router.OWN: [i.item_id for i in context.items.items(ROUTER)],
        router.REFERENCE: [i.item_id for i in context.items.items(ROUTER)],
        router.PEER: [i.item_id for i in context.items.items(ROUTER)],
        router.ADVERSARIAL: [str(r.get("item_id", "")) for r in context.items.adversarial],
    }
    for hotkey, artifact in artifacts.items():
        outcome = runner.outcomes.get(hotkey)
        if outcome is None or outcome.failure:
            continue
        found: list[PoolScore] = []
        if (
            outcome.slm is not None
            and runner.reference_slm is not None
            and artifact.slm_dir is not None
        ):
            prov = provenance.check(
                artifact.slm_dir, base_dirs=base_dirs, known_public=known_public
            )
            outcome.provenance = prov
            base_run = runner.base_slm.get(str(prov.get("declared_base", "")))
            sized = artifact.slm_parameters > 0 and low <= artifact.slm_parameters <= high
            memory_ok = outcome.slm.vram_peak_bytes <= memory_cap
            if rules.per_item_envelope:
                gate = sized
                over = score.envelope_share(outcome.slm, budgets)
                envelope_ok = memory_ok and over <= rules.violation_share
            else:
                gate = (
                    sized
                    and memory_ok
                    and (
                        not budgets.get("ttft_ms")
                        or outcome.slm.ttft_p95_ms <= float(budgets["ttft_ms"])
                    )
                    and (
                        not budgets.get("tpot_ms")
                        or outcome.slm.tpot_p95_ms <= float(budgets["tpot_ms"])
                    )
                )
                envelope_ok = True
            pool_score = score.slm_score(
                outcome.slm,
                runner.reference_slm,
                gate_ok=gate,
                provenance_ok=bool(prov.get("passed")),
                base=base_run,
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
            found.append(pool_score)
        if outcome.harness is not None:
            found.append(score.harness_score(outcome.harness, min_items=rules.min_items))
        if outcome.router is not None:
            found.append(
                score.router_score(
                    outcome.router,
                    runner.reference_router,
                    floors=context.floors,
                    families=context.items.families,
                    items_by_source=items_by_source,
                    seed=context.seed,
                    rules=rules,
                )
            )
        if outcome.escalation is not None:
            found.append(
                score.escalation_score(
                    outcome.escalation,
                    runner.reference_escalation,
                    seed=context.seed,
                    min_items=rules.min_items,
                )
            )
        if outcome.system is not None:
            found.append(score.system_score(outcome.system, min_items=rules.min_items))
        scores[hotkey] = found
    return scores


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
    arena = client.arena(config.arena_id) or {}
    window = _window(arena, epoch_index)
    artifacts, failures = gather(config, chain, context, commit_close=window.open_block)
    log.info("epoch %d: %d artifacts, %d refused", epoch_index, len(artifacts), len(failures))
    token = client.relay_token()
    relay = RelayClient(config.relay_url, str(token.get("token", "")))
    desk = TicketDesk(relay, config.arena_id, epoch_index)
    engine = docker or Docker()
    _check_host(context.definition)
    services = Services(host_ip(), config.host_services_port)
    gpus = tuple(config.gpus or ("0",))
    agent: probe_judge.ProbeAgent | None = None
    if context.grader == compress.GRADER:
        if config.judge_share_gpu:
            agent = _judge(config, context, engine, gpus[-1], memory_share=JUDGE_SHARED_SHARE)
        else:
            if len(gpus) < 2:
                raise EpochError("context compression needs a second GPU for the probe agent")
            agent, gpus = _judge(config, context, engine, gpus[-1]), gpus[:-1]
    services.start()
    host = SandboxHost(config, engine, services, host_ip())
    runner = Runner(config, context, host, relay, GpuPool(gpus), desk)
    base_dirs: dict[str, Path] = {}
    for base in [str(b) for b in context.definition.get("allowed_bases") or []]:
        try:
            base_dirs[base] = collect.download(base, "main", config.cache_dir)
        except Exception as exc:
            log.warning("base %s could not be fetched: %s", base, exc)
    measured_alone = runner.limits.on_cpu
    try:
        with ThreadPoolExecutor(
            max_workers=1 if measured_alone else max(1, config.concurrency)
        ) as pool:
            list(pool.map(runner.slm_phase, artifacts.values()))
        runner.reference_phase(base_dirs)
        peers = {h: o.slm.drafts() for h, o in runner.outcomes.items() if o.slm is not None}
        coldkeys = {h: a.coldkey for h, a in artifacts.items()}
        with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as pool:
            list(
                pool.map(
                    lambda a: runner.full_phase(
                        a, {h: d for h, d in peers.items() if h != a.hotkey}, coldkeys
                    ),
                    artifacts.values(),
                )
            )
    finally:
        services.stop()
        if agent is not None:
            agent.stop()
        engine.sweep()
    known_public = provenance.public_checkpoint_hashes(config.home / "public_checkpoints.json")
    scores = score_all(runner, artifacts, base_dirs, known_public)
    try:
        probe_sets.write(context, runner.outcomes, artifacts)
    except OSError as exc:
        log.warning("probe sets not written: %s", exc)
    for hotkey_, outcome in runner.outcomes.items():
        if outcome.failure:
            failures[hotkey_] = outcome.failure
    receipts = (
        {h: [r.body for r in relay.receipts(config.arena_id, epoch_index)] for h in artifacts}
        if artifacts
        else {}
    )
    standings = score.standings_for(
        runner.outcomes,
        scores,
        context.previous_best,
        {h: a.artifact_sha for h, a in artifacts.items()},
        {h: a.commit_block for h, a in artifacts.items()},
    )
    burn_to = (
        _owner(chain) if score.Rules.from_definition(context.definition).burn_empty_pools else ""
    )
    weights, burned, vector = weights_for(standings, chain.snapshot(refresh=True), burn_to)
    signed = report.sign(
        report.build(
            context,
            validator_hotkey=hotkey,
            artifacts=artifacts,
            scores=scores,
            failures=failures,
            receipts=receipts,
        ),
        wallet,
    )
    try:
        client.submit_report(signed.to_dict())
    except ServerError as exc:
        log.warning("report not accepted by the server: %s", exc)
    (context.work_dir / "report.json").write_text(
        json.dumps(signed.to_dict(), indent=2), encoding="utf-8"
    )
    if vector is not None and not config.dry_run:
        ok, reason = chain.set_weights(vector)
        if not ok:
            log.error("weights rejected: %s", reason)
    return Outcome(
        scores=scores,
        failures=failures,
        receipts=receipts,
        weights=weights,
        burned=burned,
        report=signed.to_dict(),
    )


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


def _judge(
    config: EpochConfig,
    context: EpochContext,
    engine: Docker,
    gpu: str,
    *,
    memory_share: float = probe_judge.MEMORY_SHARE,
) -> probe_judge.ProbeAgent:
    grading = context.grading
    spec = grading.get("probe_agent") or {}
    model = str(spec.get("model") or "")
    if not model:
        raise EpochError("the arena names no probe agent")
    revision = str(spec.get("revision") or "main")
    if config.judge_model:
        if config.chain.network != "test":
            raise EpochError("a probe agent override is only allowed on testnet")
        log.warning("testnet probe agent override: %s instead of %s", config.judge_model, model)
        model, revision = config.judge_model, config.judge_revision or "main"
    model_dir = collect.download(model, revision, config.cache_dir)
    tokenizer = str(grading.get("tokenizer") or model)
    tokenizer_dir = (
        model_dir if tokenizer == model else collect.download(tokenizer, "main", config.cache_dir)
    )
    agent = probe_judge.ProbeAgent(
        engine,
        config.vllm_image,
        model_dir,
        gpu,
        max_tokens=int(spec.get("max_tokens") or 64),
        memory_share=memory_share,
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


def describe(reference: Reference) -> str:
    return f"reference slm {reference.slm_dir.name}, {len(reference.variant_dirs)} harness variants"
