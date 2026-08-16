from __future__ import annotations

import json
from pathlib import Path

import pytest

from microtensor.cli.main import main
from microtensor.envelope.certify import (
    CertificationError,
    band_verdict,
    certify,
    load_policies,
    save,
)
from microtensor.envelope.device import POLICY_ENV, DeviceProfile, detect


def _profile(**over: object) -> DeviceProfile:
    base: dict[str, object] = {
        "machine": "x86_64",
        "system": "linux",
        "cpu_cores": 8,
        "total_memory_bytes": 1 << 34,
    }
    base.update(over)
    return DeviceProfile(**base)  # type: ignore[arg-type]


def test_a_policy_change_is_a_different_device_profile() -> None:
    bare = _profile()
    pinned = _profile(cooling_mode="active", power_mode="performance")
    assert bare.digest != pinned.digest
    assert pinned.digest == _profile(cooling_mode="active", power_mode="performance").digest


def test_detect_honours_the_declared_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        POLICY_ENV,
        json.dumps({"cooling_mode": "active", "power_mode": "performance", "idle_seconds": 5}),
    )
    with_policy = detect()
    assert with_policy.cooling_mode == "active"
    assert with_policy.idle_seconds == 5

    monkeypatch.delenv(POLICY_ENV)
    assert detect().cooling_mode == ""
    assert detect().digest != with_policy.digest


def test_garbage_in_the_policy_env_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(POLICY_ENV, "not json at all")
    assert detect().cooling_mode == ""


def test_certification_measures_and_pins_the_policy() -> None:
    certification = certify("laptop", {"cooling_mode": "passive"}, repetitions=3)
    assert certification.latency.p50 > 0.0
    assert certification.latency.p95 >= certification.latency.p50
    assert certification.policy["cooling_mode"] == "passive"
    assert certification.policy["power_mode"] == "performance"
    assert certification.device.cooling_mode == "passive"
    assert certification.digest.startswith("dev:")


def test_certification_rejects_a_non_launch_class() -> None:
    with pytest.raises(CertificationError):
        certify("server-cpu", repetitions=3)


def test_certification_rejects_too_few_repetitions() -> None:
    with pytest.raises(CertificationError):
        certify("laptop", repetitions=1)


def test_an_unpublished_band_records_rather_than_judges() -> None:
    certification = certify("laptop", repetitions=3)
    passed, verdict = band_verdict(certification)
    assert passed is None
    assert "calibration" in verdict


def test_saved_certifications_round_trip_into_policies(tmp_path: Path) -> None:
    certification = certify("laptop", {"cooling_mode": "active"}, repetitions=3)
    save(certification, tmp_path)
    policies = load_policies(tmp_path)
    assert policies["laptop"]["cooling_mode"] == "active"


def test_the_certify_command_writes_the_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "validator",
            "certify",
            "laptop",
            "--home",
            str(tmp_path),
            "--repetitions",
            "3",
            "--cooling-mode",
            "active",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "device profile" in out
    assert (tmp_path / "certification" / "laptop.json").is_file()
