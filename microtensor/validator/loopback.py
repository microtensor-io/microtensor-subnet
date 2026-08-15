from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from microtensor.chain.commitment import build_commitment
from microtensor.chain.config import ChainConfig
from microtensor.chain.metagraph import Neuron, snapshot_of
from microtensor.chain.offline import OfflineClient
from microtensor.chain.rounds import Round, round_at
from microtensor.core.constants import MIN_VALIDATOR_STAKE
from microtensor.core.protocol import ArtifactFormat, DeclaredEnvelope, LoadManifest
from microtensor.core.tracks import enabled_tracks
from microtensor.harness import registry as engines
from microtensor.registry.fetch import FETCHERS, Unfetchable
from microtensor.registry.manifest import build_manifest
from microtensor.tasks.corpus import FIXED, ROTATING
from microtensor.validator.context import ValidatorConfig, ValidatorContext

log = logging.getLogger("microtensor.validator.loopback")

VALIDATOR_HOTKEY = "5LoopbackValidator00000000000000000000000000000"
REFERENCE_ENGINE = "microtensor.harness.engines.reference"

LOAD = LoadManifest(
    format=ArtifactFormat.SAFETENSORS,
    quantization="int8",
    entrypoint="model.safetensors",
    max_input={"tokens": 512},
    preprocessing={"tokenizer": "tokenizer.json"},
)
DECLARED = DeclaredEnvelope(size_bytes=1 << 30, peak_rss_bytes=3 << 30, p95_latency_ms=170)


@dataclass(frozen=True, slots=True)
class Miner:
    hotkey: str
    uid: int
    artifact: Path
    source: str


def _write_corpus(root: Path, tracks: list[str], rotating: int, fixed: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for track in tracks:
        rows = [
            {
                "ref": f"{track}-rot-{i:04d}",
                "prompt": f"[loopback {track}] case {i}",
                "gold": f"case {i}",
                "partition": ROTATING,
                "max_output_tokens": 32,
            }
            for i in range(rotating)
        ] + [
            {
                "ref": f"{track}-fix-{i:04d}",
                "prompt": f"[loopback {track}] anchor {i}",
                "gold": f"anchor {i}",
                "partition": FIXED,
                "max_output_tokens": 32,
            }
            for i in range(fixed)
        ]
        (root / f"{track}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )


def _make_miner(root: Path, index: int, round_index: int, track: str, cls: str) -> Miner:
    hotkey = f"5LoopbackMiner{index}" + "0" * (34 - len(str(index)))
    artifact = root / f"miner-{index}"
    artifact.mkdir(parents=True, exist_ok=True)

    filler = hashlib.sha256(f"loopback:{index}".encode()).digest()
    (artifact / "model.safetensors").write_bytes(filler * 64 * (index + 1))
    (artifact / "tokenizer.json").write_text('{"model":{}}', encoding="utf-8")

    source = f"https:loopback-{index}.invalid/artifact"
    manifest = build_manifest(
        artifact,
        hotkey=hotkey,
        round_index=round_index,
        track=track,
        hardware_class=cls,
        source=source,
        load=LOAD,
        declared=DECLARED,
    ).signed_with(f"loopback-signature-{hotkey}")
    (artifact / "manifest.json").write_bytes(manifest.to_json())

    return Miner(hotkey=hotkey, uid=index + 1, artifact=artifact, source=source)


def _install_fetcher(miners: list[Miner]) -> None:
    by_locator = {m.source.partition(":")[2]: m.artifact for m in miners}

    def local_fetch(locator: str, relpath: str, destination: Path, timeout: int) -> None:
        origin = by_locator.get(locator)
        if origin is None or not (origin / relpath).is_file():
            raise Unfetchable(f"{relpath} is not published at {locator}")
        shutil.copyfile(origin / relpath, destination)

    FETCHERS["https"] = local_fetch


@dataclass(frozen=True, slots=True)
class Loopback:
    context: ValidatorContext
    client: OfflineClient
    round: Round
    miners: tuple[Miner, ...]


def build(
    home: Path,
    *,
    round_index: int = 3,
    miners: int = 3,
    tasks_per_round: int = 12,
    hardware_class: str = "laptop",
) -> Loopback:
    track = enabled_tracks()[0].id
    round_ = round_at(round_index)

    os.environ["MT_ENGINES"] = REFERENCE_ENGINE
    engines.clear()
    engines.load_builtin()
    log.info("loopback engines: %s", [f.value for f in engines.available()])

    _write_corpus(home / "corpus", [track], rotating=tasks_per_round, fixed=tasks_per_round)
    built = [
        _make_miner(home / "miners", i, round_.index, track, hardware_class)
        for i in range(miners)
    ]
    _install_fetcher(built)

    neurons = [
        Neuron(
            uid=0,
            hotkey=VALIDATOR_HOTKEY,
            stake=MIN_VALIDATOR_STAKE * 2,
            validator_permit=True,
            address="10.0.0.1",
        )
    ] + [
        Neuron(uid=m.uid, hotkey=m.hotkey, stake=1.0, address=f"10.{i + 1}.0.1")
        for i, m in enumerate(built)
    ]

    client = OfflineClient(
        snapshot_of(70, round_.close_block, neurons),
        block=round_.close_block,
        commitments={
            m.hotkey: build_commitment(
                round_.index, track, hardware_class, _manifest_digest(m.artifact), m.source
            ).encode()
            for m in built
        },
    )

    config = ValidatorConfig(
        chain=ChainConfig(netuid=70, network="local", endpoint="ws://127.0.0.1:9944"),
        home=home,
        corpus_dir=home / "corpus",
        tasks_per_round=tasks_per_round,
        profile_seconds=1,
        cpu_seconds_per_artifact=30,
        allow_unsandboxed=True,
        dry_run=False,
        verify_signatures=False,
    )
    context = ValidatorContext.build(config, client, hotkey=VALIDATOR_HOTKEY)
    return Loopback(context=context, client=client, round=round_, miners=tuple(built))


def _manifest_digest(artifact: Path) -> str:
    from microtensor.registry.manifest import ArtifactManifest

    return ArtifactManifest.from_json((artifact / "manifest.json").read_bytes()).digest()


def advance(loop: Loopback) -> Loopback:
    following = loop.round.next
    track = enabled_tracks()[0].id

    for miner in loop.miners:
        manifest = build_manifest(
            miner.artifact,
            hotkey=miner.hotkey,
            round_index=following.index,
            track=track,
            hardware_class="laptop",
            source=miner.source,
            load=LOAD,
            declared=DECLARED,
        ).signed_with(f"loopback-signature-{miner.hotkey}")
        (miner.artifact / "manifest.json").write_bytes(manifest.to_json())
        loop.client.set_commitment(
            miner.hotkey,
            build_commitment(
                following.index, track, "laptop", manifest.digest(), miner.source
            ).encode(),
        )

    loop.client.advance(following.close_block - loop.client.block())
    loop.client.set_snapshot(
        snapshot_of(70, following.close_block, list(loop.client.snapshot().neurons))
    )
    return Loopback(
        context=loop.context, client=loop.client, round=following, miners=loop.miners
    )
