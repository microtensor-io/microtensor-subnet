from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from microtensor.chain.rounds import is_release_boundary, release_cutoff_block
from microtensor.chain.weights import WeightVector, quantise_weights
from microtensor.cli.common import add_chain_arguments, add_common_arguments, fail
from microtensor.coordinator.api import Coordinator, Registry
from microtensor.coordinator.assign import System, Worker, assign, by_worker, under_replicated
from microtensor.coordinator.chain import ChainSource, RoundSource
from microtensor.coordinator.config import config_hash, served_config
from microtensor.coordinator.release import Milestone, ReleaseError
from microtensor.coordinator.release import build as build_release
from microtensor.coordinator.server import (
    ServerClient,
    ServerRefused,
    ServerSource,
    ServerUnreachable,
    publish_round,
)
from microtensor.coordinator.settle import Settlement, apply_reserved, normalise_reserved
from microtensor.coordinator.store import CoordinatorStore
from microtensor.coordinator.tokens import KeyRing
from microtensor.core.constants import (
    COORDINATOR_PORT,
    COORDINATOR_REPLICATION,
    COORDINATOR_SERVER_URL,
    CORPUS_VERSION,
    PUBLIC_SERVER_URL,
    WEIGHT_INTERVAL_SECONDS,
)
from microtensor.core.role import COORDINATOR, RoleConflict, claim, require

log = logging.getLogger("microtensor.coordinator")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "coordinator", help="run the owner-operated coordinator (not a validator)"
    )
    inner = parser.add_subparsers(dest="action", required=True)

    init = inner.add_parser("init", help="claim this host as a coordinator")
    add_common_arguments(init)
    add_chain_arguments(init)
    init.set_defaults(handler=_init)

    serve = inner.add_parser("serve", help="serve the coordinator API")
    serve.add_argument("--corpus", type=Path, help="directory of <track>.jsonl corpora to serve")
    add_chain_arguments(serve)
    add_common_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=COORDINATOR_PORT)
    _add_server_arguments(serve)
    serve.set_defaults(handler=_serve)

    opened = inner.add_parser(
        "open", help="read the round from chain, assign it, and store the catalogue"
    )
    add_common_arguments(opened)
    add_chain_arguments(opened)
    opened.add_argument("--replication", type=int, default=COORDINATOR_REPLICATION)
    _add_server_arguments(opened)
    opened.set_defaults(handler=_open)

    plan = inner.add_parser("assign", help="compute and store this round's assignment")
    add_common_arguments(plan)
    plan.add_argument("--round", type=int, required=True)
    plan.add_argument("--seed", required=True)
    plan.add_argument("--systems", type=Path, required=True, help="JSON list of systems")
    plan.add_argument("--workers", type=Path, required=True, help="JSON list of worker hotkeys")
    plan.add_argument("--replication", type=int, default=COORDINATOR_REPLICATION)
    plan.set_defaults(handler=_assign)

    settle = inner.add_parser("settle", help="reconcile and publish a round")
    add_common_arguments(settle)
    settle.add_argument("--round", type=int, required=True)
    _add_server_arguments(settle)
    settle.set_defaults(handler=_settle)

    weights = inner.add_parser(
        "weights", help="set weights from the reserved hold, with or without a round"
    )
    add_common_arguments(weights)
    add_chain_arguments(weights)
    _add_server_arguments(weights)
    weights.add_argument(
        "--loop",
        action="store_true",
        help="keep setting weights on an interval instead of returning after one",
    )
    weights.add_argument(
        "--interval",
        type=int,
        default=WEIGHT_INTERVAL_SECONDS,
        help="seconds between attempts when looping",
    )
    weights.add_argument(
        "--dry-run", action="store_true", help="print the vector instead of submitting it"
    )
    weights.set_defaults(handler=_weights)

    anchor = inner.add_parser(
        "anchor", help="commit this round's config hash on chain so workers can verify it"
    )
    add_common_arguments(anchor)
    add_chain_arguments(anchor)
    _add_server_arguments(anchor)
    anchor.add_argument("--round", type=int, help="defaults to the latest open round")
    anchor.add_argument(
        "--dry-run", action="store_true", help="print what would be committed and stop"
    )
    anchor.set_defaults(handler=_anchor)

    cfg = inner.add_parser("config", help="print the served config and its hash")
    add_common_arguments(cfg)
    cfg.set_defaults(handler=_config)

    status = inner.add_parser("status", help="round state, quorum, worker standing")
    add_common_arguments(status)
    status.add_argument("--round", type=int)
    status.set_defaults(handler=_status)


def _corpora(args: argparse.Namespace, server: ServerClient | None = None) -> dict[str, Any]:
    """The corpus this coordinator serves to its workers.

    A directory when one is given, because a local file is an explicit
    instruction. Otherwise the control plane, which is where a corpus is
    uploaded, checked and published, and which holds the scored partitions
    that never appear on the public split.

    Absent is allowed and logged rather than fatal: a coordinator with no
    corpus yet still schedules rounds, and its workers fall back to their own
    until it has one.
    """
    from microtensor.tasks.corpus import CorpusError, load_all

    explicit = getattr(args, "corpus", None)
    root = Path(explicit) if explicit else (_home(args) / "corpus")
    try:
        found = dict(load_all(root, CORPUS_VERSION))
        if found:
            log.info("serving %d corpora from %s", len(found), root)
            return found
    except (CorpusError, OSError) as exc:
        log.info("no corpus on disk at %s (%s)", root, exc)

    if explicit or server is None:
        log.warning("no corpus is being served to workers")
        return {}

    return _corpora_from_server(server)


def _corpora_from_server(server: ServerClient) -> dict[str, Any]:
    """Every published corpus the configured arenas name.

    Keyed by track, because that is how a worker asks for one. An arena that
    names a version the control plane cannot serve is logged and skipped
    rather than failing the others: one unpublished corpus should not stop a
    coordinator serving the arenas that are ready.
    """
    from microtensor.validator.corpus import parse

    try:
        arenas = server.arenas()
    except (ServerUnreachable, ServerRefused) as exc:
        log.warning("the arena list could not be read, so no corpus is served (%s)", exc)
        return {}

    versions = sorted(
        {str(a.get("corpus_version", "")) for a in arenas.values() if a.get("corpus_version")}
    )
    found: dict[str, Any] = {}
    for version in versions:
        try:
            served = server.corpus(version)
        except (ServerUnreachable, ServerRefused) as exc:
            log.warning("corpus %s could not be read (%s)", version, exc)
            continue
        if not served or not served.get("track"):
            log.warning("corpus %s is not published, so it is not being served", version)
            continue
        found[str(served["track"])] = parse(served)

    if found:
        log.info("serving %d corpora from the control plane: %s", len(found), sorted(found))
    else:
        log.warning("no corpus is being served to workers")
    return found


def _home(args: argparse.Namespace) -> Path:
    return Path(args.home)


def _store(args: argparse.Namespace) -> CoordinatorStore:
    try:
        require(_home(args), COORDINATOR)
    except RoleConflict as exc:
        raise SystemExit(fail(str(exc))) from exc
    return CoordinatorStore(_home(args) / "coordinator.sqlite")


def _init(args: argparse.Namespace) -> int:
    home = _home(args)
    try:
        claim(home, COORDINATOR)
    except RoleConflict as exc:
        return fail(str(exc))

    store = CoordinatorStore(home / "coordinator.sqlite")
    store.close()

    config = served_config(CORPUS_VERSION)
    print(f"coordinator home   {home}")
    print(f"config hash        {config_hash(config)}")
    print()
    print("This host measures nothing. It fetches no artifact, loads no model, and")
    print("installs no jail. Commit the config hash on chain each round so workers")
    print("can prove the rules they measured against.")
    return 0


def _arenas(server: ServerClient | None) -> dict[str, dict[str, Any]]:
    """Per-arena configuration from the control plane, or nothing.

    Nothing is the strict answer when the server cannot be reached: an empty
    allowlist admits no submission, which is the safe reading of "the rules
    could not be fetched". Falling back to a stale set nobody has reviewed is
    exactly the failure the allowlist exists to prevent.
    """
    if server is None:
        return {}
    try:
        return server.arenas()
    except (ServerUnreachable, ServerRefused) as exc:
        log.warning("the arena configuration was not read from the control plane: %s", exc)
        return {}


def _config(args: argparse.Namespace) -> int:
    config = served_config(CORPUS_VERSION)
    print(json.dumps(config, indent=2, sort_keys=True))
    print()
    print(f"hash  {config_hash(config)}")
    return 0


def _assign(args: argparse.Namespace) -> int:
    systems_raw = json.loads(Path(args.systems).read_text(encoding="utf-8"))
    workers_raw = json.loads(Path(args.workers).read_text(encoding="utf-8"))

    systems = [
        System(
            digest=str(s["digest"]),
            track=str(s["track"]),
            hardware_class=str(s["hardware_class"]),
            miner_hotkey=str(s.get("miner_hotkey", "")),
        )
        for s in systems_raw
    ]
    workers = [
        Worker(hotkey=str(w["hotkey"]), classes=tuple(w.get("classes", ())))
        if isinstance(w, dict)
        else Worker(hotkey=str(w))
        for w in workers_raw
    ]

    mapping = assign(systems, workers, args.seed, replication=args.replication)

    with _store(args) as store:
        store.record_assignment(
            args.round,
            mapping,
            {s.digest: (s.track, s.hardware_class, s.miner_hotkey) for s in systems},
        )

    thin = under_replicated(mapping, args.replication)
    print(f"{len(systems)} systems across {len(workers)} workers, seed {args.seed}")
    for hotkey, digests in sorted(by_worker(mapping).items()):
        print(f"  {hotkey:<50} {len(digests)}")
    if thin:
        print()
        print(f"under-replicated, measured by fewer than {args.replication}:")
        for digest in thin:
            print(f"  {digest}")
    return 0


def _add_server_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--server",
        default=os.environ.get("MT_SERVER_URL", COORDINATOR_SERVER_URL),
        help="control plane base URL; without one the coordinator reads chain alone",
    )
    parser.add_argument(
        "--server-credential",
        default=os.environ.get("MT_SERVER_CREDENTIAL", ""),
        help="ingest credential issued by the control plane",
    )
    parser.add_argument(
        "--public-server",
        default=os.environ.get("MT_PUBLIC_SERVER_URL", PUBLIC_SERVER_URL),
        help="public API base URL; the arena list is read from there, not the control plane",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="ignore a configured control plane and derive everything from chain",
    )


def _server(args: argparse.Namespace) -> ServerClient | None:
    """The control plane client, when one is configured and not suppressed.

    `--no-server` wins over a configured URL so a fully chain-driven round is
    always reachable by flag rather than only by the server being down.
    """
    url = getattr(args, "server", "") or ""
    if not url or getattr(args, "no_server", False):
        return None
    return ServerClient(
        base_url=url,
        credential=getattr(args, "server_credential", "") or "",
        public_url=getattr(args, "public_server", "") or PUBLIC_SERVER_URL,
    )


def _open(args: argparse.Namespace) -> int:
    from microtensor.cli.common import chain_config, open_client, open_wallet
    from microtensor.coordinator.chain import observed

    chain = chain_config(args)
    wallet = open_wallet(chain, required=False)
    client = open_client(chain, wallet)
    server = _server(args)
    source: RoundSource = ChainSource(client=client)
    if server is not None:
        source = ServerSource(chain=ChainSource(client=client), client=server)
        log.info("taking the round from the control plane at %s", args.server)

    try:
        round_ = source.open_round()
    except ServerRefused as exc:
        return fail(str(exc))
    systems, catalogue = source.systems(round_)
    workers = source.workers()
    seed = source.seed(round_)

    if not systems:
        # An empty round is still a round: it settles on the reserved hold,
        # and workers adopt that settlement, so it is stored rather than
        # skipped. Skipping meant no round, no recorded metagraph, and no
        # uid the hold could resolve against, so nobody set weights.
        with _store(args) as store:
            store.open_round(
                round_.index,
                seed_block=round_.seed_block,
                close_block=round_.close_block,
                block_hash=seed,
                config_hash=_config_hash_for(source, server),
            )
            store.record_metagraph(round_.index, source.uids())
        print(f"round {round_.index} opened with nothing to measure; it settles on the hold")
        print("Commit the config hash on chain before workers verify against it:")
        print(f"  {_config_hash_for(source, server)}")
        print(f"  mt coordinator anchor --round {round_.index}")
        return 0
    if not workers:
        return fail("no worker holds a validator permit, so nothing can be assigned")

    mapping = assign(systems, workers, seed, replication=args.replication)

    with _store(args) as store:
        store.open_round(
            round_.index,
            seed_block=round_.seed_block,
            close_block=round_.close_block,
            block_hash=seed,
            config_hash=_config_hash_for(source, server),
        )
        store.record_assignment(
            round_.index,
            mapping,
            {s.digest: (s.track, s.hardware_class, s.miner_hotkey) for s in systems},
        )
        store.record_catalogue(round_.index, observed(catalogue, store.observations(round_.index)))
        store.record_metagraph(round_.index, source.uids())

    thin = under_replicated(mapping, args.replication)
    print(f"round {round_.index} opened, seed block {round_.seed_block}")
    print(f"  systems     {len(systems)}")
    print(f"  workers     {len(workers)}")
    print(f"  assignments {sum(len(v) for v in mapping.values())}")
    if thin:
        print(f"  under-replicated: {len(thin)}")
    print()
    if server is not None:
        try:
            server.push_assignments(
                round_.index,
                mapping,
                {s.digest: (s.track, s.hardware_class, s.miner_hotkey) for s in systems},
            )
        except (ServerUnreachable, ServerRefused) as exc:
            log.warning("the assignment map was not mirrored to the control plane: %s", exc)

    print("Commit the config hash on chain before workers measure against it:")
    print(f"  {_config_hash_for(source, server)}")
    print(f"  mt coordinator anchor --round {round_.index}")
    print("Until that lands, workers refuse the round rather than measure against it.")
    return 0


def _anchor(args: argparse.Namespace) -> int:
    """Commit the open round's config hash on chain.

    Separate from `open` on purpose: opening a round is a local bookkeeping
    step this can be re-run, and committing is an on-chain write signed by the
    coordinator's hotkey. Until this runs, workers have nothing to check the
    served config against and refuse the round.
    """
    from microtensor.chain.anchor import ConfigAnchor
    from microtensor.cli.common import chain_config, open_client, open_wallet

    with _store(args) as store:
        row = store.round(int(args.round)) if args.round is not None else store.latest_round()
        if row is None:
            return fail("no round is open, so there is no config to anchor")

        round_index = int(row["round_index"])
        stored = str(row["config_hash"] or "")
        if not stored:
            return fail(
                f"round {round_index} was opened without a config hash, so there is "
                "nothing to anchor; re-open it"
            )

        anchor = ConfigAnchor(round_index=round_index, config_hash=stored)
        payload = anchor.encode()
        print(f"round  {round_index}")
        print(f"hash   {stored}")
        print(f"commit {payload}")

        if args.dry_run:
            print()
            print("dry run: nothing was committed")
            return 0

        chain = chain_config(args)
        wallet = open_wallet(chain, required=True)
        client = open_client(chain, wallet)
        try:
            if not client.publish(payload):
                return fail("the chain refused the commitment; nothing was anchored")
        finally:
            client.close()

        store.mark_anchored(round_index)

    print()
    print(f"anchored: workers will now verify the served config against {stored}")
    return 0


def _measured_weights(store: CoordinatorStore) -> dict[int, float]:
    """The most recent settled vector, or nothing.

    Nothing is the honest answer before the first round settles. It is not the
    same as every miner scoring zero, and treating it as such would publish a
    ranking over a field that was never measured.
    """
    row = store.latest_round()
    if row is None:
        return {}
    published = store.settlement(int(row["round_index"])) or {}
    return {int(uid): float(value) for uid, value in published.get("weights", {}).items()}


def _weight_vector(
    store: CoordinatorStore, held: dict[str, Any], uid_by_hotkey: dict[str, int]
) -> WeightVector:
    measured = _measured_weights(store)
    resolved = {}
    if held:
        uid = uid_by_hotkey.get(str(held.get("hotkey", "")))
        if uid is None:
            log.warning(
                "%s holds %.2f%% but is not on this metagraph; ignoring the hold",
                held.get("hotkey"),
                float(held.get("share", 0.0)) * 100,
            )
        else:
            resolved = {"hotkey": str(held["hotkey"]), "uid": uid, "share": float(held["share"])}

    return quantise_weights(apply_reserved(measured, normalise_reserved(resolved)))


def _weights(args: argparse.Namespace) -> int:
    """Set weights from the reserved hold, whether or not a round has run.

    A hold is an instruction rather than a measurement, so it does not wait on
    quorum, a catalogue or a settled round. Those gate the miner ranking, which
    is a different question: what the measured field earned. When both exist the
    hold takes its share off the top and the measured vector divides the rest.
    """
    from microtensor.cli.common import chain_config, open_client, open_wallet

    chain = chain_config(args)
    wallet = open_wallet(chain, required=True)
    client = open_client(chain, wallet)
    server = _server(args)
    if server is None:
        return fail("no control plane is configured, so there is no hold to read")

    while True:
        try:
            held = server.reserved()
        except (ServerUnreachable, ServerRefused) as exc:
            log.warning("the hold could not be read (%s); setting nothing this pass", exc)
            held = {}

        if held.get("paused"):
            log.info("emission is paused at the control plane; setting nothing")
            vector = WeightVector((), ())
        else:
            with _store(args) as store:
                uids = dict(client.snapshot(refresh=True).uid_by_hotkey)
                vector = _weight_vector(store, held, uids)

        if vector.is_empty:
            log.info("nothing to set: no hold resolved and no round has settled")
        elif args.dry_run:
            pairs = dict(zip(vector.uids, vector.values, strict=True))
            print(f"would set {len(vector.uids)} weights: {pairs}")
        else:
            ok, reason = client.set_weights(vector)
            if ok:
                log.info("set %d weights totalling %d", len(vector.uids), vector.total)
            else:
                log.error("weights rejected: %s", reason)

        if not args.loop:
            return 0
        time.sleep(args.interval)


def _config_hash_for(source: RoundSource, server: ServerClient | None = None) -> str:
    """The hash workers will verify against.

    The control plane's when it published one, because that is the document it
    is serving to workers; otherwise the same document the serving API builds,
    arenas included. Hashing the config without the arena list anchored one
    document while the API served another, and every worker refused the round
    as a mismatch.
    """
    published = getattr(source, "config_hash", "")
    if published:
        return str(published)
    return config_hash(served_config(CORPUS_VERSION, _arenas(server)))


def _settle(args: argparse.Namespace) -> int:
    server = _server(args)
    with _store(args) as store:
        service = Coordinator(
            store=store,
            registry=Registry({}),
            corpus_version=CORPUS_VERSION,
            corpora=_corpora(args, server),
            catalogue=store.catalogue(args.round),
            uid_by_hotkey=store.uids(),
            reserve=server.reserved if server is not None else None,
        )
        published = service.settle(args.round)

    if published is None:
        return fail(f"round {args.round} has not reached quorum yet")

    if server is not None:
        with _store(args) as store:
            reports = store.reports_payload(args.round)
            assignment = store.full_assignment(args.round)
            settlement = _settlement_of(store, args.round)
        try:
            stored = publish_round(server, settlement, reports=reports, assignment=assignment)
            print(
                f"archived on the control plane: {stored['reports']} reports, "
                f"{stored['assignments']} assignments"
            )
        except (ServerUnreachable, ServerRefused) as exc:
            log.error(
                "the settlement was computed and set on chain but not archived: %s. "
                "The reports are still local; re-run this command to push them.",
                exc,
            )

        with _store(args) as store:
            _push_telemetry(store, server, args.round)
            for version in _cut_releases(store, server, args.round):
                print(f"released {version}")

    print(json.dumps(published, indent=2, sort_keys=True))
    return 0


def _push_telemetry(store: CoordinatorStore, server: ServerClient | None, round_index: int) -> None:
    """Mirror who was training to the control plane.

    Best effort. Telemetry is observational, so failing to mirror it must never
    interfere with a settlement that has already been computed and set.
    """
    if server is None:
        return

    from microtensor.coordinator.telemetry import for_server

    rows = for_server(store.telemetry_state(round_index), store.hardware_for(round_index))
    if not rows:
        return

    try:
        server.push_telemetry(round_index, rows)
    except (ServerUnreachable, ServerRefused) as exc:
        log.warning("telemetry was not mirrored to the control plane: %s", exc)


def _cut_releases(
    store: CoordinatorStore, server: ServerClient | None, round_index: int
) -> list[str]:
    """Freeze each competition's frontier if this round ends a cycle.

    Runs after settlement, reads its output, and changes nothing about it. A
    release is a snapshot for distribution, not a scoring event: emissions for
    this round are identical whether or not a release is cut from it.
    """
    if server is None or not is_release_boundary(round_index):
        return []

    published = store.settlement(round_index) or {}
    frontier = published.get("frontier", ())
    if not frontier:
        return []

    cutoff = release_cutoff_block(round_index)
    now = int(time.time())
    competitions = sorted(
        {(str(row.get("track", "")), str(row.get("class", ""))) for row in frontier}
    )

    cut: list[str] = []
    for track, hardware_class in competitions:
        if not track or not hardware_class:
            continue
        target = None
        try:
            stated = server.milestone(track, hardware_class)
            if stated:
                target = Milestone(
                    target_quality=float(stated["target_quality"]),
                    target_cost=float(stated["target_cost"]),
                )
        except (ServerUnreachable, ServerRefused, KeyError, TypeError, ValueError) as exc:
            log.info("no milestone for %s/%s: %s", track, hardware_class, exc)

        try:
            release = build_release(
                published,
                track=track,
                hardware_class=hardware_class,
                cutoff_block=cutoff,
                published_at=now,
                milestone=target,
            )
        except ReleaseError as exc:
            log.info("no release for %s/%s: %s", track, hardware_class, exc)
            continue

        payload = release.body()
        payload["digest"] = release.digest()
        payload["signature"] = release.signature
        try:
            server.push_release(payload)
        except ServerRefused as exc:
            log.warning("the control plane refused %s: %s", release.version, exc)
            continue
        except ServerUnreachable as exc:
            log.warning("%s was not published: %s", release.version, exc)
            continue

        cut.append(release.version)
        log.info("cut %s with %d systems", release.version, len(release.frontier))

    return cut


def _settlement_of(store: CoordinatorStore, round_index: int) -> Settlement:
    """Rebuild the stored settlement so it can be pushed in its signed shape."""
    published = store.settlement(round_index) or {}
    return Settlement(
        round_index=round_index,
        config_hash=str(published.get("config_hash", "")),
        corpus_version=str(published.get("corpus_version", "")),
        reports_root=str(published.get("reports_root", "")),
        frontier=tuple(published.get("frontier", ())),
        catalogue=tuple(published.get("catalogue", ())),
        weights={int(k): float(v) for k, v in published.get("weights", {}).items()},
        unscored=tuple(published.get("unscored", ())),
        under_replicated=tuple(published.get("under_replicated", ())),
        advisory=tuple(published.get("advisory", ())),
        capped=bool(published.get("capped", False)),
        blended={str(k): float(v) for k, v in published.get("blended", {}).items()},
        reserved=dict(published.get("reserved") or {}),
        signature=str(published.get("signature", "")),
    )


def _status(args: argparse.Namespace) -> int:
    with _store(args) as store:
        service = Coordinator(store=store, registry=Registry({}), corpus_version=CORPUS_VERSION)
        health = service.health()
        standings = service.reputation()

    if health.get("round") is None:
        print("no round opened yet")
        return 0

    print(f"round            {health['round']}")
    print(f"reports          {health['received_reports']} / {health['expected_reports']}")
    print(f"quorum           {'reached' if health['quorum'] else 'waiting'}")
    print(f"settled          {'yes' if health['settled'] else 'no'}")
    print(f"config anchored  {'yes' if health['anchored'] else 'NO'}")
    print(f"divergence rate  {health['divergence_rate']:.2%}")

    if standings:
        print()
        print(f"{'worker':<50}{'agreed':>8}{'diverged':>10}{'rate':>8}  state")
        for s in standings:
            state = "advisory" if s["advisory"] else "deciding"
            print(f"{s['hotkey']:<50}{s['agreed']:>8}{s['diverged']:>10}{s['rate']:>8.2f}  {state}")
    return 0


REGISTRY_REFRESH_SECONDS = 300


def _permitted(client: Any) -> dict[str, int]:
    snapshot = client.snapshot(refresh=True)
    uids = dict(snapshot.uid_by_hotkey)
    return {
        hotkey: uids[hotkey]
        for hotkey in snapshot.hotkeys
        if hotkey in uids and snapshot.has_permit(hotkey)
    }


def _keep_registry_current(registry: Registry, client: Any) -> None:
    import threading

    def loop() -> None:
        while True:
            time.sleep(REGISTRY_REFRESH_SECONDS)
            try:
                found = _permitted(client)
            except Exception as exc:
                log.warning("the validator registry could not be refreshed: %s", exc)
                continue
            if found and found != registry.permitted:
                registry.permitted = found
                log.info("registry refreshed: %d permitted validators", len(found))

    threading.Thread(target=loop, name="registry-refresh", daemon=True).start()


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        return fail('the API needs the web stack: pip install ".[coordinator]"')

    from microtensor.cli.common import chain_config, open_client, open_wallet
    from microtensor.coordinator.api import build_app

    with _store(args) as store:
        keyring = KeyRing(home=_home(args))
        server = _server(args)
        if server is not None:
            keyring.refresh(server.token_key)
        if not keyring.load():
            log.warning(
                "no control plane key is known, so worker tokens will not be checked; "
                "point --server at the control plane once to record it"
            )

        chain = chain_config(args)
        client = open_client(chain, open_wallet(chain, required=False))
        try:
            permitted = _permitted(client)
        except Exception as exc:
            log.warning("the validator registry could not be read from chain: %s", exc)
            permitted = {}
        if not permitted:
            log.warning("no permitted validator is known, so every signed request is refused")
        else:
            log.info("registry holds %d permitted validators", len(permitted))
        registry = Registry(permitted)
        _keep_registry_current(registry, client)

        service = Coordinator(
            store=store,
            registry=registry,
            keyring=keyring,
            corpus_version=CORPUS_VERSION,
            corpora=_corpora(args, server),
            uid_by_hotkey=store.uids(),
            reserve=server.reserved if server is not None else None,
            mirror_report=server.push_reports if server is not None else None,
            arenas=_arenas(server),
            arena_source=(lambda: _arenas(server)) if server is not None else None,
            corpora_source=(lambda: _corpora(args, server)) if server is not None else None,
        )
        app = build_app(service)
        log.info("serving the coordinator on %s:%d", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0
