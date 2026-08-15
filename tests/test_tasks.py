from __future__ import annotations

import json
from pathlib import Path

import pytest

from microtensor.chain.offline import deterministic_block_hash
from microtensor.core.constants import TASKS_PER_ROUND
from microtensor.tasks import (
    FIXED,
    ROTATING,
    Corpus,
    CorpusError,
    RoundTasks,
    Task,
    competition_seed,
    load_all,
    load_corpus,
    partition_of,
    partition_sizes,
    select,
    to_requests,
)


def _task(index: int, partition: str = ROTATING) -> Task:
    return Task(
        ref=f"t{index:04d}",
        prompt=f"solve case {index}",
        gold={"value": index},
        partition=partition,
    )


def _corpus(rotating: int = 400, fixed: int = 120) -> Corpus:
    tasks = [_task(i) for i in range(rotating)]
    tasks += [_task(10_000 + i, FIXED) for i in range(fixed)]
    return Corpus(track="code", version="2026.1", tasks=tuple(tasks))


def test_task_demands_a_reference_and_a_prompt() -> None:
    with pytest.raises(CorpusError):
        Task(ref="", prompt="x", gold=None, partition=ROTATING)
    with pytest.raises(CorpusError):
        Task(ref="t1", prompt="", gold=None, partition=ROTATING)


def test_task_rejects_an_unknown_partition() -> None:
    with pytest.raises(CorpusError):
        Task(ref="t1", prompt="x", gold=None, partition="holdout")


def test_corpus_rejects_a_repeated_reference() -> None:
    with pytest.raises(CorpusError):
        Corpus(track="code", version="1", tasks=(_task(1), _task(1), _task(2, FIXED)))


def test_corpus_without_a_fixed_partition_is_refused() -> None:
    with pytest.raises(CorpusError):
        Corpus(track="code", version="1", tasks=(_task(1), _task(2)))


def test_corpus_rejects_an_unknown_track() -> None:
    with pytest.raises(KeyError):
        Corpus(track="telepathy", version="1", tasks=(_task(1), _task(2, FIXED)))


def test_corpus_exposes_the_tracks_metric() -> None:
    assert _corpus().metric == "execution_pass_rate"


def test_partition_sizes_follow_the_declared_split() -> None:
    rotating, fixed = partition_sizes(TASKS_PER_ROUND)
    assert rotating == 140
    assert fixed == 60
    assert rotating + fixed == TASKS_PER_ROUND


def test_partition_sizes_reject_a_budget_that_cannot_split() -> None:
    with pytest.raises(CorpusError):
        partition_sizes(1)


def test_selection_is_reproducible_from_the_seed() -> None:
    corpus = _corpus()
    seed = competition_seed(deterministic_block_hash(7200), "code", "laptop")
    a = select(corpus, seed, "laptop")
    b = select(corpus, seed, "laptop")
    assert a.refs == b.refs


def _draw(corpus: Corpus, block: int) -> RoundTasks:
    seed = competition_seed(deterministic_block_hash(block), "code", "laptop")
    return select(corpus, seed, "laptop")


def test_a_different_block_draws_a_different_rotating_set() -> None:
    corpus = _corpus()
    assert _draw(corpus, 1).rotating != _draw(corpus, 2).rotating


def test_each_competition_draws_its_own_tasks() -> None:
    corpus = _corpus()
    block = deterministic_block_hash(7200)
    laptop = select(corpus, competition_seed(block, "code", "laptop"), "laptop")
    edge = select(corpus, competition_seed(block, "code", "edge-gpu"), "edge-gpu")
    assert laptop.rotating != edge.rotating


def test_the_fixed_partition_does_not_move_between_rounds() -> None:
    corpus = _corpus()
    assert _draw(corpus, 1).fixed == _draw(corpus, 9).fixed


def test_selection_respects_the_round_budget() -> None:
    tasks = select(_corpus(), "seed", "laptop")
    assert len(tasks.rotating) == 140
    assert len(tasks.fixed) == 60
    assert len(tasks) == TASKS_PER_ROUND


def test_selection_takes_what_a_thin_corpus_can_give() -> None:
    corpus = _corpus(rotating=10, fixed=5)
    tasks = select(corpus, "seed", "laptop")
    assert len(tasks.rotating) == 10
    assert len(tasks.fixed) == 5


def test_requests_carry_a_nonce_bound_to_the_artifact() -> None:
    tasks = select(_corpus(rotating=4, fixed=2), "seed", "laptop")
    mine = to_requests(tasks.all, "seed", "code", artifact_digest="sha256:aa")
    theirs = to_requests(tasks.all, "seed", "code", artifact_digest="sha256:bb")
    assert mine[0].task_ref == theirs[0].task_ref
    assert mine[0].nonce != theirs[0].nonce


def test_requests_inherit_the_tracks_decoding() -> None:
    tasks = select(_corpus(rotating=4, fixed=2), "seed", "laptop")
    assert all(r.decoding.value == "greedy" for r in to_requests(tasks.all, "seed", "code"))


def test_partition_lookup_matches_the_draw() -> None:
    tasks = select(_corpus(rotating=4, fixed=2), "seed", "laptop")
    assert partition_of(tasks, tasks.rotating[0].ref) == ROTATING
    assert partition_of(tasks, tasks.fixed[0].ref) == FIXED


def test_corpus_loads_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "code.jsonl"
    lines = [
        json.dumps({"ref": "t1", "prompt": "a", "gold": 1, "partition": ROTATING}),
        "",
        json.dumps({"ref": "t2", "prompt": "b", "gold": 2, "partition": FIXED}),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    corpus = load_corpus(path, "code")
    assert len(corpus) == 2
    assert corpus.refs(FIXED) == ("t2",)


def test_corpus_load_reports_the_offending_line(tmp_path: Path) -> None:
    path = tmp_path / "code.jsonl"
    path.write_text('{"ref": "t1", "prompt": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(CorpusError) as caught:
        load_corpus(path, "code")
    assert ":2" in str(caught.value)


def test_missing_corpus_is_an_error_not_an_empty_round(tmp_path: Path) -> None:
    with pytest.raises(CorpusError):
        load_corpus(tmp_path / "absent.jsonl", "code")


def test_load_all_keys_by_track(tmp_path: Path) -> None:
    for track in ("code", "document"):
        (tmp_path / f"{track}.jsonl").write_text(
            json.dumps({"ref": "t1", "prompt": "a", "partition": ROTATING})
            + "\n"
            + json.dumps({"ref": "t2", "prompt": "b", "partition": FIXED}),
            encoding="utf-8",
        )
    corpora = load_all(tmp_path)
    assert sorted(corpora) == ["code", "document"]


def test_load_all_refuses_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(CorpusError):
        load_all(tmp_path)
