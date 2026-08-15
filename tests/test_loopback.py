from __future__ import annotations

from pathlib import Path

import pytest

from microtensor.core.constants import MIN_ROUNDS_OBSERVED
from microtensor.store.state import SETTLED
from microtensor.validator import loopback
from microtensor.validator.round import run_round


@pytest.fixture
def world(tmp_path: Path):  # type: ignore[no-untyped-def]
    built = loopback.build(tmp_path / "lb", miners=3, tasks_per_round=8)
    yield built
    built.context.close()


def test_loopback_stands_up_a_whole_subnet(world) -> None:  # type: ignore[no-untyped-def]
    assert len(world.miners) == 3
    assert len(world.context.competitions) >= 1
    assert world.client.block() == world.round.close_block


def test_the_first_round_evaluates_everyone_and_pays_nobody(world) -> None:  # type: ignore[no-untyped-def]
    outcome = run_round(world.context, world.round)
    assert outcome.status == SETTLED, outcome.reason
    assert outcome.participants == 3
    assert outcome.scored == 3
    assert outcome.settlement is not None
    assert outcome.settlement.is_empty
    assert not world.client.submitted


def test_the_second_round_pays_a_full_vector(world) -> None:  # type: ignore[no-untyped-def]
    current = world
    for _ in range(MIN_ROUNDS_OBSERVED):
        run_round(current.context, current.round)
        current = loopback.advance(current)

    assert len(current.client.submitted) == 1
    vector = current.client.submitted[0]
    assert vector.total == 65535
    assert len(vector) == 3


def test_every_miner_is_admitted_through_the_gate(world) -> None:  # type: ignore[no-untyped-def]
    run_round(world.context, world.round)
    evaluations = world.context.state.evaluations(
        world.round.index, world.context.tracks[0], "laptop"
    )
    assert len(evaluations) == 3
    assert all(row["admitted"] for row in evaluations)


def test_an_engine_the_child_cannot_build_makes_the_validator_abstain(
    world, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("MT_ENGINES", raising=False)
    outcome = run_round(world.context, world.round)
    assert outcome.status != SETTLED
    assert "infrastructure" in outcome.reason or "engine" in outcome.reason
    assert not world.client.submitted
