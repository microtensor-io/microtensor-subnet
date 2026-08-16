from __future__ import annotations

import json
from pathlib import Path

import pytest

from microtensor.scoring import execution
from microtensor.scoring.execution import execute_pass_rate, parse_tests
from microtensor.tasks import generator
from microtensor.tasks.corpus import CorpusError, load_all
from microtensor.tasks.generator import (
    FAMILIES,
    MIN_HIDDEN_TESTS,
    GenerationError,
    build_task,
    generate,
    verify_bundle,
    write_bundle,
)

SEED = "test-seed-0001"


@pytest.fixture(autouse=True)
def _unsandboxed_ok() -> None:
    execution.configure(allow_unsandboxed=True)
    yield
    execution.configure(allow_unsandboxed=False)


def _small_bundle() -> generator.Bundle:
    return generate(rotating=8, fixed=4, train=4, seed=SEED)


def test_generation_is_reproducible() -> None:
    a = _small_bundle()
    b = _small_bundle()
    assert [t.task_row() for t in a.tasks] == [t.task_row() for t in b.tasks]
    assert [t.tests_row() for t in a.tasks] == [t.tests_row() for t in b.tasks]


def test_a_different_seed_draws_a_different_corpus() -> None:
    a = generate(rotating=8, fixed=4, train=0, seed=SEED)
    b = generate(rotating=8, fixed=4, train=0, seed="another-seed")
    assert [t.prompt for t in a.tasks] != [t.prompt for t in b.tasks]


def test_refs_are_unique_and_partitioned() -> None:
    bundle = _small_bundle()
    refs = [t.ref for t in bundle.tasks]
    assert len(set(refs)) == len(refs)
    assert sum(1 for t in bundle.tasks if t.partition == "rotating") == 8
    assert sum(1 for t in bundle.tasks if t.partition == "fixed") == 4


def test_every_task_carries_enough_hidden_tests() -> None:
    for task in _small_bundle().tasks:
        assert len(task.hidden) >= MIN_HIDDEN_TESTS
        assert len(task.public) >= 1


def test_the_first_hidden_test_is_the_edge_case() -> None:
    for task in _small_bundle().tasks:
        edge_args = task.hidden[0][0]
        assert all(arg in ("", [], 3) or arg == [] for arg in edge_args) or edge_args[0] in ("", [])


def test_hidden_inputs_do_not_leak_into_the_prompt() -> None:
    for task in _small_bundle().tasks:
        for args, _ in task.hidden:
            rendered = repr(args[0])
            if len(rendered) >= generator.LEAK_CHECK_MIN_CHARS:
                assert rendered not in task.prompt


def test_every_family_solution_passes_its_own_hidden_tests() -> None:
    for family in sorted(FAMILIES):
        task = build_task(f"code-x-{family}", "rotating", family, SEED, 0)
        tests = parse_tests(
            [{"args": list(args), "expected": expected} for args, expected in task.hidden]
        )
        score = execute_pass_rate(task.solution, task.entry_point, tests)
        assert score == 1.0, f"{family} reference solution scored {score}"


def test_the_bundle_round_trips_through_disk(tmp_path: Path) -> None:
    bundle = _small_bundle()
    digests = write_bundle(bundle, tmp_path)
    assert set(digests) == {"code.jsonl", "code.tests.jsonl", "code.train.jsonl"}
    assert verify_bundle(tmp_path) == []

    corpora = load_all(tmp_path)
    assert set(corpora) == {"code"}
    corpus = corpora["code"]
    assert len(corpus) == 12
    assert all(
        isinstance(t.gold, dict) and "tests" in t.gold and "entry_point" in t.gold
        for t in corpus
    )


def test_the_train_split_carries_no_test_digests(tmp_path: Path) -> None:
    write_bundle(_small_bundle(), tmp_path)
    for line in (tmp_path / "code.train.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            assert "tests_digest" not in row["inputs"]


def test_a_tampered_tests_file_is_caught_at_load(tmp_path: Path) -> None:
    write_bundle(_small_bundle(), tmp_path)
    tests_path = tmp_path / "code.tests.jsonl"
    lines = tests_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["tests"][1]["expected"] = "tampered"
    lines[0] = json.dumps(row)
    tests_path.write_text("\n".join(lines), encoding="utf-8")

    with pytest.raises(CorpusError, match="digest"):
        load_all(tmp_path)
    assert any("tests_digest" in p for p in verify_bundle(tmp_path))


def test_regenerating_over_a_committed_corpus_is_refused(tmp_path: Path) -> None:
    write_bundle(_small_bundle(), tmp_path)
    with pytest.raises(GenerationError, match="generated once"):
        write_bundle(_small_bundle(), tmp_path)
    write_bundle(_small_bundle(), tmp_path, force=True)


def test_an_attached_task_scores_through_the_metric_registry(tmp_path: Path) -> None:
    from microtensor.scoring.metrics import score_task

    write_bundle(_small_bundle(), tmp_path)
    corpus = load_all(tmp_path)["code"]
    task = corpus.tasks[0]
    tests_path = tmp_path / "code.tests.jsonl"
    solution = ""
    for line in tests_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["ref"] == task.ref:
            solution = row["solution"]
    assert score_task("execution_pass_rate", solution, task.gold) == 1.0
    assert score_task("execution_pass_rate", "def nothing(): pass", task.gold) == 0.0


def test_generation_rejects_an_empty_corpus() -> None:
    with pytest.raises(GenerationError):
        generate(rotating=0, fixed=1, train=0, seed=SEED)
