from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from microtensor.chain.commitment import build_commitment
from microtensor.chain.config import ChainConfig
from microtensor.chain.metagraph import Neuron, snapshot_of
from microtensor.chain.offline import OfflineClient
from microtensor.chain.rounds import round_at
from microtensor.core.constants import MIN_VALIDATOR_STAKE
from microtensor.core.protocol import ArtifactFormat, DeclaredEnvelope, LoadManifest
from microtensor.harness import registry as engine_registry
from microtensor.registry.fetch import FETCHERS
from microtensor.registry.manifest import build_manifest
from microtensor.store.state import SETTLED
from microtensor.tasks.corpus import FIXED, ROTATING
from microtensor.validator.context import ValidatorConfig, ValidatorContext
from microtensor.validator.round import run_round

MINER = "5MinerHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
SECOND = "5MinerHotkeyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
VALIDATOR = "5ValidatorHotkeyCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"

LOAD = LoadManifest(
    format=ArtifactFormat.SAFETENSORS,
    quantization="int8",
    entrypoint="model.safetensors",
    max_input={"tokens": 512},
    preprocessing={"tokenizer": "tokenizer.json"},
)
DECLARED = DeclaredEnvelope(
    size_bytes=1 << 30, peak_rss_bytes=3 << 30, p95_latency_ms=170
)


@pytest.fixture(autouse=True)
def _reference_engine(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    engine_registry.clear()
    monkeypatch.setenv("MT_ENGINES", "microtensor.harness.engines.reference")
    engine_registry.load_builtin()
    yield
    engine_registry.clear()


@pytest.fixture(autouse=True)
def _trusted_signatures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "microtensor.validator.discover.verify_payload",
        lambda hotkey, payload, signature: signature == f"signed-by-{hotkey}",
    )


def _corpus(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {"ref": f"r{i:03d}", "prompt": f"case {i}", "gold": f"case {i}", "partition": ROTATING}
        for i in range(20)
    ] + [
        {"ref": f"f{i:03d}", "prompt": f"anchor {i}", "gold": f"anchor {i}", "partition": FIXED}
        for i in range(10)
    ]
    (root / "code.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    return root


def _artifact(root: Path, marker: bytes) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_bytes(marker * 512)
    (root / "tokenizer.json").write_text('{"model":{}}', encoding="utf-8")
    return root


def _publish(artifact: Path, hotkey: str, round_index: int, source: str) -> str:
    manifest = build_manifest(
        artifact,
        hotkey=hotkey,
        round_index=round_index,
        track="code",
        hardware_class="laptop",
        source=source,
        load=LOAD,
        declared=DECLARED,
    ).signed_with(f"signed-by-{hotkey}")
    (artifact / "manifest.json").write_bytes(manifest.to_json())
    return build_commitment(
        round_index, "code", "laptop", manifest.digest(), source
    ).encode()


def _neurons() -> tuple[Neuron, ...]:
    return (
        Neuron(
            uid=0,
            hotkey=VALIDATOR,
            stake=MIN_VALIDATOR_STAKE * 2,
            validator_permit=True,
            address="10.1.0.1",
        ),
        Neuron(uid=1, hotkey=MINER, stake=1.0, address="10.2.0.1"),
        Neuron(uid=2, hotkey=SECOND, stake=1.0, address="10.3.0.1"),
    )


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    round_ = round_at(3)
    sources = {
        MINER: ("https:miner-a.example.com/mt-code", _artifact(tmp_path / "a", b"A")),
        SECOND: ("https:miner-b.example.com/mt-code", _artifact(tmp_path / "b", b"B")),
    }

    commitments = {
        hotkey: _publish(path, hotkey, round_.index, source)
        for hotkey, (source, path) in sources.items()
    }
    by_locator = {source.partition(":")[2]: path for source, path in sources.values()}

    def local_fetch(locator: str, relpath: str, destination: Path, timeout: int) -> None:
        origin = by_locator[locator] / relpath
        if not origin.is_file():
            from microtensor.registry.fetch import Unfetchable

            raise Unfetchable(f"{relpath} is not published at {locator}")
        shutil.copyfile(origin, destination)

    monkeypatch.setitem(FETCHERS, "https", local_fetch)

    client = OfflineClient(
        snapshot_of(91, round_.close_block, _neurons()),
        block=round_.close_block,
        commitments=commitments,
    )
    config = ValidatorConfig(
        chain=ChainConfig(netuid=91, network="local", endpoint="ws://127.0.0.1:9944"),
        home=tmp_path / "home",
        corpus_dir=_corpus(tmp_path / "corpus"),
        tasks_per_round=12,
        profile_seconds=1,
        cpu_seconds_per_artifact=30,
        allow_unsandboxed=True,
    )
    context = ValidatorContext.build(config, client, hotkey=VALIDATOR)
    yield context, client, round_, sources
    context.close()


def _advance(context, client, sources, round_):  # type: ignore[no-untyped-def]
    following = round_.next
    for hotkey, (source, path) in sources.items():
        client.set_commitment(hotkey, _publish(path, hotkey, following.index, source))
    client.advance(following.close_block - client.block())
    client.set_snapshot(snapshot_of(91, following.close_block, _neurons()))
    return following


def test_a_first_round_evaluates_but_pays_nobody(world) -> None:  # type: ignore[no-untyped-def]
    context, client, round_, _ = world

    outcome = run_round(context, round_)

    assert outcome.status == SETTLED, outcome.reason
    assert outcome.participants == 2
    assert outcome.scored == 2
    assert outcome.settlement is not None
    assert outcome.settlement.is_empty
    assert not client.submitted
    assert "eligible" in outcome.reason


def test_the_second_round_reaches_the_chain_as_a_normalised_vector(world) -> None:  # type: ignore[no-untyped-def]
    context, client, round_, sources = world

    run_round(context, round_)
    second = run_round(context, _advance(context, client, sources, round_))

    assert second.status == SETTLED, second.reason
    assert len(client.submitted) == 1
    vector = client.submitted[0]
    assert vector.total == 65535
    assert set(vector.uids) <= {1, 2}


def test_the_round_survives_a_restart(world) -> None:  # type: ignore[no-untyped-def]
    context, client, round_, sources = world

    run_round(context, round_)
    second = run_round(context, _advance(context, client, sources, round_))

    from microtensor.store.state import ValidatorState

    reopened = ValidatorState(context.config.state_path)
    try:
        assert reopened.is_settled(second.round_index)
        assert reopened.last_weights()
        assert reopened.holders("code", "laptop")
        assert reopened.summary()["evaluations"] == 4
    finally:
        reopened.close()


def test_resubmitting_the_same_artifact_accrues_staleness(world) -> None:  # type: ignore[no-untyped-def]
    context, client, round_, sources = world

    run_round(context, round_)
    run_round(context, _advance(context, client, sources, round_))

    observed, stale = context.state.observation("code", "laptop", MINER)
    assert observed == 2
    assert stale == 1


def test_a_settled_round_is_not_scored_twice(world) -> None:  # type: ignore[no-untyped-def]
    context, client, round_, _ = world
    run_round(context, round_)
    first = len(client.submitted)

    from microtensor.validator.loop import RoundLoop

    loop = RoundLoop(context, poll_seconds=0, sleep=lambda _: None)
    assert loop.step() is None
    assert len(client.submitted) == first


def test_an_unfetchable_artifact_makes_the_validator_abstain(world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    context, client, round_, _ = world

    from microtensor.registry.fetch import Unfetchable

    def refuse(locator: str, relpath: str, destination: Path, timeout: int) -> None:
        if relpath == "manifest.json":
            raise Unfetchable("manifest host is down")
        raise Unfetchable("artifact host is down")

    monkeypatch.setitem(FETCHERS, "https", refuse)
    outcome = run_round(context, round_)

    assert outcome.status != SETTLED
    assert not client.submitted


def test_a_commitment_for_the_wrong_round_is_rejected(tmp_path, world) -> None:  # type: ignore[no-untyped-def]
    context, client, round_, sources = world
    source, path = sources[MINER]
    client.set_commitment(MINER, _publish(path, MINER, round_.index + 5, source))

    outcome = run_round(context, round_)

    assert outcome.participants == 1
    rejected = dict(context.state.rejected_submissions(round_.index))
    assert MINER in rejected
    assert "round" in rejected[MINER]


def test_an_unsigned_manifest_is_rejected(world) -> None:  # type: ignore[no-untyped-def]
    context, _client, round_, sources = world
    path = sources[SECOND][1]

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["signature"] = ""
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    outcome = run_round(context, round_)

    rejected = dict(context.state.rejected_submissions(round_.index))
    assert SECOND in rejected
    assert "signature" in rejected[SECOND]
    assert outcome.participants == 1


def test_a_tampered_artifact_does_not_reach_execution(world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    context, _client, round_, _sources = world

    original = FETCHERS["https"]

    def tamper(locator: str, relpath: str, destination: Path, timeout: int) -> None:
        original(locator, relpath, destination, timeout)
        if "miner-a" in locator and relpath == "model.safetensors":
            destination.write_bytes(b"swapped")

    monkeypatch.setitem(FETCHERS, "https", tamper)
    outcome = run_round(context, round_)

    history = context.state.history(MINER)
    assert history
    assert history[0]["score_combined"] == 0.0
    assert outcome.status == SETTLED
