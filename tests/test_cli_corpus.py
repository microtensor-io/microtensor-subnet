from __future__ import annotations

import json
from pathlib import Path

import pytest

from microtensor.cli.main import main
from microtensor.core.tracks import enabled_tracks
from microtensor.tasks.corpus import FIXED, ROTATING, load_all


def test_seed_writes_one_corpus_per_open_track(tmp_path: Path) -> None:
    assert main(["corpus", "seed", str(tmp_path), "--rotating", "20", "--fixed", "10"]) == 0
    written = sorted(p.stem for p in tmp_path.glob("*.jsonl"))
    assert written == sorted(t.id for t in enabled_tracks())


def test_a_seeded_corpus_loads(tmp_path: Path) -> None:
    main(["corpus", "seed", str(tmp_path), "--rotating", "20", "--fixed", "10"])
    corpora = load_all(tmp_path)
    for corpus in corpora.values():
        assert len(corpus.rotating) == 20
        assert len(corpus.fixed) == 10


def test_seed_refuses_to_clobber_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["corpus", "seed", str(tmp_path), "--rotating", "4", "--fixed", "2"])
    main(["corpus", "seed", str(tmp_path), "--rotating", "8", "--fixed", "4"])
    assert "skipping" in capsys.readouterr().out
    assert len(load_all(tmp_path)["code"].rotating) == 4


def test_force_overwrites(tmp_path: Path) -> None:
    main(["corpus", "seed", str(tmp_path), "--rotating", "4", "--fixed", "2"])
    main(["corpus", "seed", str(tmp_path), "--rotating", "8", "--fixed", "4", "--force"])
    assert len(load_all(tmp_path)["code"].rotating) == 8


def test_seed_warns_that_the_tasks_are_synthetic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["corpus", "seed", str(tmp_path), "--rotating", "4", "--fixed", "2"])
    out = capsys.readouterr().out
    assert "synthetic" in out
    assert "mainnet" in out


def test_check_passes_a_corpus_that_fills_a_round(tmp_path: Path) -> None:
    main(["corpus", "seed", str(tmp_path), "--rotating", "200", "--fixed", "80"])
    assert main(["corpus", "check", str(tmp_path), "--tasks-per-round", "100"]) == 0


def test_check_fails_a_corpus_too_thin_for_the_round(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["corpus", "seed", str(tmp_path), "--rotating", "10", "--fixed", "5"])
    assert main(["corpus", "check", str(tmp_path), "--tasks-per-round", "200"]) == 1
    assert "short of" in capsys.readouterr().out


def test_check_names_open_tracks_with_no_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        {"ref": "a", "prompt": "x", "partition": ROTATING},
        {"ref": "b", "prompt": "y", "partition": FIXED},
    ]
    (tmp_path / "document.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    main(["corpus", "check", str(tmp_path), "--tasks-per-round", "2"])
    out = capsys.readouterr().out
    assert "will not be scored" in out
    assert "code" in out
    assert "track not open" in out


def test_check_reports_an_empty_directory_cleanly(tmp_path: Path) -> None:
    assert main(["corpus", "check", str(tmp_path)]) == 1


def test_check_rejects_a_corpus_with_no_fixed_partition(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "code.jsonl").write_text(
        json.dumps({"ref": "a", "prompt": "x", "partition": ROTATING}), encoding="utf-8"
    )
    assert main(["corpus", "check", str(tmp_path)]) == 1
    assert "fixed partition" in capsys.readouterr().err


def test_stats_reports_a_stable_digest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["corpus", "seed", str(tmp_path), "--rotating", "8", "--fixed", "4"])
    capsys.readouterr()
    main(["corpus", "stats", str(tmp_path)])
    first = capsys.readouterr().out
    main(["corpus", "stats", str(tmp_path)])
    assert capsys.readouterr().out == first
