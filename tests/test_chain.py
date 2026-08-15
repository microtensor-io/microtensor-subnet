from __future__ import annotations

import pytest

from microtensor.chain import (
    ChainConfig,
    Commitment,
    CommitmentError,
    Neuron,
    OfflineClient,
    Round,
    WeightVector,
    build_commitment,
    decode_all,
    quantise_weights,
    round_at,
    round_for_block,
    short_digest,
    snapshot_from,
    snapshot_of,
    version_key,
)
from microtensor.chain.client import ChainError, with_retry
from microtensor.chain.config import NETWORKS
from microtensor.chain.weights import U16_MAX, restrict_to_metagraph
from microtensor.core.constants import (
    DEADLINE_MARGIN_BLOCKS,
    DEFAULT_NETUID,
    MECHANISM_VERSION,
    MIN_VALIDATOR_STAKE,
    ROUND_BLOCKS,
    SUBMISSION_CLOSES_BEFORE_BLOCKS,
)

DIGEST = "sha256:" + "ab" * 32


class _Axon:
    def __init__(self, ip: str, port: int) -> None:
        self.ip = ip
        self.port = port


class _RawMetagraph:
    def __init__(self) -> None:
        self.netuid = 91
        self.block = 1234
        self.hotkeys = ["hk0", "hk1", "hk2"]
        self.coldkeys = ["ck0", "ck1", "ck2"]
        self.uids = [0, 1, 2]
        self.S = [2000.0, 10.0, 0.0]
        self.T = [0.9, 0.1, 0.0]
        self.validator_permit = [True, False, False]
        self.axons = [_Axon("10.0.0.1", 8091), _Axon("10.0.0.2", 8091), _Axon("", 0)]


def _neuron(uid: int, **kw: object) -> Neuron:
    return Neuron(uid=uid, hotkey=f"hk{uid}", **kw)  # type: ignore[arg-type]


def test_the_subnet_defaults_to_its_own_netuid() -> None:
    assert DEFAULT_NETUID == 70
    assert ChainConfig().netuid == 70


def test_an_operator_can_still_point_at_testnet() -> None:
    config = ChainConfig(netuid=310, network="test")
    assert config.netuid == 310
    assert config.resolved_endpoint == NETWORKS["test"]


def test_config_falls_back_to_the_named_network() -> None:
    assert ChainConfig(network="test").resolved_endpoint.startswith("wss://")


def test_config_prefers_an_explicit_endpoint() -> None:
    config = ChainConfig(network="finney", endpoint="ws://127.0.0.1:9944")
    assert config.resolved_endpoint == "ws://127.0.0.1:9944"
    assert config.is_local


def test_config_rejects_an_unknown_network() -> None:
    with pytest.raises(ValueError):
        ChainConfig(network="mainnet")


def test_config_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MT_NETUID", "91")
    monkeypatch.setenv("MT_WALLET_HOTKEY", "miner-a")
    config = ChainConfig.from_env()
    assert config.netuid == 91
    assert config.wallet_hotkey == "miner-a"


def test_config_overrides_beat_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MT_NETUID", "91")
    assert ChainConfig.from_env(netuid=7).netuid == 7


def test_config_ignores_absent_overrides() -> None:
    assert ChainConfig(netuid=5).with_overrides(netuid=None).netuid == 5


def test_round_boundaries_are_contiguous() -> None:
    first = round_at(3)
    assert first.next.start_block == first.end_block + 1
    assert first.previous.end_block == first.start_block - 1


def test_round_for_block_inverts_round_at() -> None:
    a_round = round_at(11)
    assert round_for_block(a_round.start_block) == a_round
    assert round_for_block(a_round.end_block) == a_round


def test_submissions_close_before_the_round_ends() -> None:
    r = round_at(0)
    assert r.accepts_submissions(r.close_block - 1)
    assert not r.accepts_submissions(r.close_block)
    assert r.is_sealed(r.close_block)


def test_the_seed_block_is_unknowable_while_submissions_are_open() -> None:
    r = round_at(0)
    assert r.seed_block >= r.close_block


def test_the_deadline_leaves_room_to_evaluate() -> None:
    r = round_at(0)
    assert r.evaluation_blocks == SUBMISSION_CLOSES_BEFORE_BLOCKS - DEADLINE_MARGIN_BLOCKS
    assert r.evaluation_blocks > 0


def test_round_reports_the_wait_in_seconds() -> None:
    r = round_at(0)
    assert r.seconds_until_close(r.close_block) == 0
    assert r.seconds_until_close(r.start_block) > 0


def test_round_rejects_a_close_margin_that_swallows_the_round() -> None:
    with pytest.raises(ValueError):
        Round(index=0, start_block=0, length=100, close_margin=100)


def test_no_round_precedes_the_first() -> None:
    with pytest.raises(ValueError):
        round_at(0).shift(-1)


def test_block_before_genesis_is_rejected() -> None:
    with pytest.raises(ValueError):
        round_for_block(5, genesis=100)


def test_a_short_round_still_produces_a_valid_schedule() -> None:
    r = round_at(2, length=1000, close_margin=100, deadline_margin=10)
    assert r.start_block == 2000
    assert r.close_block == 2900
    assert r.deadline_block == 2989


def test_snapshot_reads_a_raw_metagraph() -> None:
    snapshot = snapshot_from(_RawMetagraph())
    assert len(snapshot) == 3
    assert snapshot.netuid == 91
    assert snapshot.uid_by_hotkey == {"hk0": 0, "hk1": 1, "hk2": 2}
    assert snapshot.find("hk0") is not None
    assert snapshot.addresses() == {"hk0": "10.0.0.1", "hk1": "10.0.0.2"}


def test_snapshot_fills_columns_the_chain_omitted() -> None:
    snapshot = snapshot_from(_RawMetagraph())
    assert snapshot.find("hk0").incentive == 0.0
    assert snapshot.find("hk0").active is True


def test_only_permitted_and_staked_neurons_are_validators() -> None:
    snapshot = snapshot_from(_RawMetagraph())
    assert [n.hotkey for n in snapshot.validators()] == ["hk0"]
    assert [n.hotkey for n in snapshot.miners()] == ["hk1", "hk2"]


def test_permit_without_stake_does_not_count() -> None:
    snapshot = snapshot_of(
        1, 0, [_neuron(0, stake=MIN_VALIDATOR_STAKE - 1.0, validator_permit=True)]
    )
    assert snapshot.validators() == ()
    assert not snapshot.has_permit("hk0")


def test_snapshot_rejects_duplicate_uids() -> None:
    with pytest.raises(ValueError):
        snapshot_of(1, 0, [_neuron(0), _neuron(0)])


def test_uid_of_raises_for_a_stranger() -> None:
    with pytest.raises(KeyError):
        snapshot_of(1, 0, [_neuron(0)]).uid_of("nobody")


def test_stake_share_is_zero_without_stake() -> None:
    assert snapshot_of(1, 0, [_neuron(0)]).stake_share("hk0") == 0.0


def test_empty_metagraph_reads_cleanly() -> None:
    snapshot = snapshot_from(object(), netuid=3, block=9)
    assert len(snapshot) == 0
    assert snapshot.netuid == 3


def test_commitment_round_trips() -> None:
    commitment = build_commitment(41, "code", "edge-gpu", DIGEST, "hf:acme/mt-code-3b@a1b2c3")
    assert Commitment.decode(commitment.encode()) == commitment


def test_commitment_fits_the_chain_limit() -> None:
    commitment = build_commitment(999, "document", "server-cpu", DIGEST, "hf:acme/mt-doc-3b@a1b2c3")
    assert len(commitment.encode().encode("utf-8")) <= 128


def test_commitment_refuses_an_oversized_source() -> None:
    commitment = build_commitment(1, "code", "laptop", DIGEST, "https://" + "x" * 200)
    with pytest.raises(CommitmentError):
        commitment.encode()


def test_commitment_rejects_an_unknown_scheme() -> None:
    with pytest.raises(CommitmentError):
        build_commitment(1, "code", "laptop", DIGEST, "ftp://host/model")


def test_commitment_rejects_a_separator_in_a_field() -> None:
    with pytest.raises(CommitmentError):
        build_commitment(1, "co|de", "laptop", DIGEST, "hf:a/b")


def test_decode_ignores_foreign_payloads() -> None:
    assert Commitment.decode("hello world") is None
    assert Commitment.decode("") is None
    assert Commitment.decode("mt1|a|code|laptop|abcd|hf:a/b") is None


def test_decode_all_skips_what_it_cannot_parse() -> None:
    good = build_commitment(1, "code", "laptop", DIGEST, "hf:a/b").encode()
    decoded = decode_all({"hk0": good, "hk1": "junk"})
    assert list(decoded) == ["hk0"]


def test_short_digest_pins_the_full_digest() -> None:
    commitment = build_commitment(1, "code", "laptop", DIGEST, "hf:a/b")
    assert commitment.covers(DIGEST)
    assert not commitment.covers("sha256:" + "cd" * 32)


def test_short_digest_rejects_a_malformed_input() -> None:
    with pytest.raises(CommitmentError):
        short_digest("not-a-digest")


def test_version_key_orders_releases() -> None:
    assert version_key("0.1.0") < version_key("0.2.0") < version_key("1.0.0")
    assert version_key(MECHANISM_VERSION) > 0


def test_version_key_rejects_a_malformed_version() -> None:
    with pytest.raises(ValueError):
        version_key("0.1")


def test_quantised_weights_sum_to_full_scale() -> None:
    vector = quantise_weights({0: 0.5, 1: 0.3, 2: 0.2})
    assert vector.total == U16_MAX
    assert vector.uids == (0, 1, 2)


def test_quantisation_is_deterministic_under_reordering() -> None:
    a = quantise_weights({0: 1 / 3, 1: 1 / 3, 2: 1 / 3})
    b = quantise_weights({2: 1 / 3, 1: 1 / 3, 0: 1 / 3})
    assert a == b
    assert a.total == U16_MAX


def test_quantisation_drops_zero_and_negative_weights() -> None:
    vector = quantise_weights({0: 0.5, 1: 0.0, 2: -1.0})
    assert vector.uids == (0,)
    assert vector.total == U16_MAX


def test_quantisation_of_nothing_is_empty() -> None:
    assert quantise_weights({}).is_empty


def test_fractions_recover_the_input_shape() -> None:
    fractions = quantise_weights({0: 0.75, 1: 0.25}).as_fractions()
    assert fractions[0] == pytest.approx(0.75, abs=1e-4)


def test_weight_vector_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        WeightVector((0, 1), (1,))


def test_weight_vector_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        WeightVector((0,), (U16_MAX + 1,))


def test_restrict_drops_uids_the_metagraph_lost() -> None:
    assert restrict_to_metagraph({0: 0.5, 9: 0.5}, size=3) == {0: 0.5}


def test_offline_client_records_what_it_was_told_to_send() -> None:
    snapshot = snapshot_of(1, 100, [_neuron(0), _neuron(1)])
    client = OfflineClient(snapshot, signer="hk0")
    client.publish("mt1|1|code|laptop|" + "ab" * 16 + "|hf:a/b")
    ok, _ = client.set_weights(quantise_weights({0: 1.0}))
    assert ok
    assert client.commitments(["hk0"])
    assert len(client.submitted) == 1


def test_offline_client_refuses_an_empty_vector() -> None:
    client = OfflineClient(snapshot_of(1, 0, [_neuron(0)]))
    ok, reason = client.set_weights(WeightVector((), ()))
    assert not ok
    assert reason


def test_offline_block_hashes_are_stable_and_distinct() -> None:
    client = OfflineClient(snapshot_of(1, 0, []))
    assert client.block_hash(10) == client.block_hash(10)
    assert client.block_hash(10) != client.block_hash(11)


def test_offline_client_advances_blocks() -> None:
    client = OfflineClient(snapshot_of(1, 100, []))
    assert client.block() == 100
    assert client.advance(ROUND_BLOCKS) == 100 + ROUND_BLOCKS


def test_retry_returns_once_the_call_settles() -> None:
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("websocket reset")
        return "ok"

    assert with_retry("flaky", flaky, attempts=4, backoff=0.0, sleep=lambda _: None) == "ok"


def test_retry_gives_up_as_a_chain_error() -> None:
    def always() -> str:
        raise TimeoutError("no route to chain")

    with pytest.raises(ChainError):
        with_retry("always", always, attempts=2, backoff=0.0, sleep=lambda _: None)
