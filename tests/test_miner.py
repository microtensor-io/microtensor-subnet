from __future__ import annotations

import json
from pathlib import Path

import pytest

from microtensor.chain.config import ChainConfig
from microtensor.cli.main import main
from microtensor.miner.config import MinerConfig, MinerConfigError
from microtensor.miner.upload import UploadError, UploadUnsupported, plan_upload, uploader_for

CHAIN = ChainConfig(netuid=92, network="local", endpoint="ws://127.0.0.1:9944")


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / "model.onnx").write_bytes(b"graph" * 100)
    (root / "tokenizer.json").write_text('{"model":{}}', encoding="utf-8")
    return root


def _config(home: Path, artifact: Path, **kw: object) -> MinerConfig:
    fields: dict[str, object] = {
        "artifact_dir": artifact,
        "track": "code",
        "hardware_class": "laptop",
        "source": "hf:acme/mt-code-3b@v1",
    }
    fields.update(kw)
    return MinerConfig.build(home, CHAIN, **fields)


def test_config_round_trips_through_disk(tmp_path: Path, artifact: Path) -> None:
    saved = _config(tmp_path, artifact)
    saved.save()
    loaded = MinerConfig.load(tmp_path, CHAIN)
    assert loaded.competition == saved.competition
    assert loaded.source == saved.source
    assert loaded.artifact_dir == saved.artifact_dir


def test_flags_override_the_saved_config(tmp_path: Path, artifact: Path) -> None:
    _config(tmp_path, artifact).save()
    loaded = MinerConfig.load(tmp_path, CHAIN, hardware_class="edge-gpu")
    assert loaded.hardware_class == "edge-gpu"
    assert loaded.track == "code"


def test_absent_flags_do_not_clobber_the_saved_config(tmp_path: Path, artifact: Path) -> None:
    _config(tmp_path, artifact, quantization="int4").save()
    loaded = MinerConfig.load(tmp_path, CHAIN, quantization=None, track=None)
    assert loaded.quantization == "int4"
    assert loaded.track == "code"


def test_a_missing_config_says_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(MinerConfigError) as caught:
        MinerConfig.load(tmp_path, CHAIN)
    assert "mt miner init" in str(caught.value)


def test_a_corrupt_config_is_reported_not_ignored(tmp_path: Path) -> None:
    (tmp_path / "miner.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(MinerConfigError):
        MinerConfig.load(tmp_path, CHAIN)


def test_a_config_naming_a_closed_competition_is_refused(tmp_path: Path, artifact: Path) -> None:
    with pytest.raises(MinerConfigError):
        _config(tmp_path, artifact, track="speech")


def test_a_config_with_an_unfetchable_scheme_is_refused(tmp_path: Path, artifact: Path) -> None:
    with pytest.raises(MinerConfigError):
        _config(tmp_path, artifact, source="ftp://acme/model")


def test_build_names_every_missing_setting(tmp_path: Path) -> None:
    with pytest.raises(MinerConfigError) as caught:
        MinerConfig.build(tmp_path, CHAIN, track="code")
    message = str(caught.value)
    assert "artifact_dir" in message
    assert "source" in message


def test_config_exposes_the_scheme_and_locator(tmp_path: Path, artifact: Path) -> None:
    config = _config(tmp_path, artifact)
    assert config.scheme == "hf"
    assert config.locator == "acme/mt-code-3b@v1"


def test_saved_config_never_holds_a_secret(tmp_path: Path, artifact: Path) -> None:
    path = _config(tmp_path, artifact).save()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "wallet_name" in payload
    assert not any("key" in k and k != "wallet_hotkey" for k in payload)


def test_upload_plan_totals_the_files(artifact: Path) -> None:
    plan = plan_upload(artifact, "hf", "acme/m@v1", ["model.onnx", "tokenizer.json"])
    assert plan.total_bytes == 500 + 12
    assert plan.targets[0] == "hf:acme/m@v1/model.onnx"


def test_upload_plan_refuses_a_file_that_is_not_there(artifact: Path) -> None:
    with pytest.raises(UploadError):
        plan_upload(artifact, "hf", "acme/m@v1", ["absent.bin"])


def test_a_plain_web_host_cannot_be_uploaded_to(artifact: Path) -> None:
    plan = plan_upload(artifact, "https", "example.com/m", ["model.onnx"])
    with pytest.raises(UploadUnsupported):
        uploader_for(plan.scheme)(plan.locator, artifact, plan.files)


def test_an_unknown_scheme_has_no_uploader() -> None:
    with pytest.raises(UploadUnsupported):
        uploader_for("ipfs")


def test_init_writes_a_config_the_other_commands_read(
    tmp_path: Path, artifact: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "home"
    code = main(
        [
            "miner",
            "init",
            "--home",
            str(home),
            "--artifact",
            str(artifact),
            "--track",
            "code",
            "--hardware-class",
            "laptop",
            "--source",
            "hf:acme/m@v1",
        ]
    )
    assert code == 0
    assert (home / "miner.json").is_file()
    assert "no flags" in capsys.readouterr().out

    assert main(["miner", "status", "--home", str(home)]) == 1
    assert "mt miner package" in capsys.readouterr().err


def test_init_refuses_a_closed_competition(
    tmp_path: Path, artifact: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "miner",
            "init",
            "--home",
            str(tmp_path),
            "--artifact",
            str(artifact),
            "--track",
            "speech",
            "--hardware-class",
            "laptop",
            "--source",
            "hf:acme/m@v1",
        ]
    )
    assert code == 1
    assert "open competition" in capsys.readouterr().err


def test_commands_without_init_still_accept_explicit_flags(
    tmp_path: Path, artifact: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "miner",
            "status",
            "--home",
            str(tmp_path / "empty"),
            "--artifact",
            str(artifact),
            "--track",
            "code",
            "--hardware-class",
            "laptop",
            "--source",
            "hf:acme/m@v1",
        ]
    )
    assert code == 1
    assert "manifest" in capsys.readouterr().err
