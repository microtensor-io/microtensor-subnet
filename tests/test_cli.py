from __future__ import annotations

from pathlib import Path

import pytest

from microtensor.cli.main import build_parser, main


def test_help_lists_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    for command in ("validator", "miner", "inspect"):
        assert command in out


def test_no_command_is_an_error() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_inspect_tracks_reports_the_open_competitions(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", "tracks"]) == 0
    out = capsys.readouterr().out
    assert "16 competitions" in out
    assert "registered, not scored" in out


def test_inspect_engines_reports_the_sandbox(capsys: pytest.CaptureFixture[str]) -> None:
    main(["inspect", "engines"])
    assert "sandbox" in capsys.readouterr().out


def test_inspect_rounds_without_state_fails_cleanly(tmp_path: Path) -> None:
    assert main(["inspect", "rounds", "--home", str(tmp_path)]) == 1


def test_a_string_exit_becomes_an_error_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise SystemExit("wallet unavailable")

    monkeypatch.setattr("microtensor.cli.miner.open_wallet", refuse)
    code = main(
        [
            "miner",
            "package",
            "--artifact",
            str(tmp_path),
            "--track",
            "code",
            "--hardware-class",
            "laptop",
            "--source",
            "hf:acme/model@v1",
        ]
    )
    assert code == 1
    assert "wallet unavailable" in capsys.readouterr().err


def test_miner_rejects_a_closed_competition(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "miner",
            "status",
            "--artifact",
            str(tmp_path),
            "--track",
            "speech",
            "--hardware-class",
            "laptop",
            "--source",
            "hf:acme/model@v1",
        ]
    )
    assert code == 1
    assert "open competition" in capsys.readouterr().err


def test_miner_rejects_an_unfetchable_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "miner",
            "status",
            "--artifact",
            str(tmp_path),
            "--track",
            "code",
            "--hardware-class",
            "laptop",
            "--source",
            "ftp://acme/model",
        ]
    )
    assert code == 1
    assert "scheme" in capsys.readouterr().err


def test_validator_refuses_an_unsandboxed_host_by_default(tmp_path: Path) -> None:
    from microtensor.harness.limits import sandbox_available

    if sandbox_available():
        pytest.skip("this host enforces resource limits")
    assert main(["validator", "status", "--home", str(tmp_path), "--netuid", "1"]) == 1
