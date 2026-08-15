from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from microtensor.chain.rounds import round_at
from microtensor.core.constants import MECHANISM_VERSION
from microtensor.update import (
    Action,
    Applied,
    Release,
    ReleaseError,
    UpdateChecker,
    UpdateSettings,
    Verification,
    VerificationError,
    Version,
    channel_of,
    check_digest,
    decide,
    is_safe_point,
    parse_release,
    parse_sums,
    sha256_file,
    verify_artifact,
)

ROUND = round_at(9)
OPEN_BLOCK = ROUND.start_block + 10
SEALED_BLOCK = ROUND.close_block + 10


def _release(
    version: str = "0.2.0",
    *,
    mechanism: str = MECHANISM_VERSION,
    activation: int | None = None,
    signed: bool = True,
    channel: str = "mainnet",
) -> Release:
    return Release(
        tag=f"v{version}" if channel == "mainnet" else f"{channel}-v{version}",
        channel=channel,
        version=Version.parse(version),
        wheel_url=f"https://example.com/microtensor-{version}.whl",
        sums_url="https://example.com/SHA256SUMS",
        signature_url="https://example.com/SHA256SUMS.sig" if signed else "",
        mechanism_version=mechanism,
        activation_block=activation,
    )


def test_versions_order_numerically_not_lexically() -> None:
    assert Version.parse("0.9.0") < Version.parse("0.10.0")
    assert Version.parse("1.0.0") < Version.parse("1.0.1")


def test_a_malformed_tag_is_not_a_release() -> None:
    with pytest.raises(ReleaseError):
        Version.parse("latest")


def test_channel_defaults_to_mainnet() -> None:
    assert channel_of("v1.2.3") == "mainnet"
    assert channel_of("testnet-v1.2.3") == "testnet"


def test_drafts_and_prereleases_are_ignored() -> None:
    base = {"tag_name": "v1.0.0", "assets": [{"name": "a.whl", "browser_download_url": "u"}]}
    assert parse_release({**base, "draft": True}) is None
    assert parse_release({**base, "prerelease": True}) is None
    assert parse_release(base) is not None


def test_a_release_without_a_wheel_is_ignored() -> None:
    assert parse_release({"tag_name": "v1.0.0", "assets": []}) is None


def test_activation_block_is_read_from_the_notes() -> None:
    release = parse_release(
        {
            "tag_name": "v2.0.0",
            "assets": [{"name": "a.whl", "browser_download_url": "u"}],
            "body": "mechanism: 0.2.0\nactivation-block: 5400000\n",
        }
    )
    assert release is not None
    assert release.mechanism_version == "0.2.0"
    assert release.activation_block == 5400000


def test_nothing_to_do_without_a_release() -> None:
    assert decide(None, ROUND, OPEN_BLOCK).action is Action.NONE


def test_an_older_release_is_never_installed() -> None:
    assert decide(_release("0.0.1"), ROUND, OPEN_BLOCK).action is Action.NONE


def test_the_running_version_is_not_reinstalled() -> None:
    assert decide(_release(MECHANISM_VERSION), ROUND, OPEN_BLOCK).action is Action.NONE


def test_a_patch_release_applies_during_the_submission_window() -> None:
    decision = decide(_release("0.1.1"), ROUND, OPEN_BLOCK)
    assert decision.action is Action.APPLY


def test_a_major_release_is_never_automatic() -> None:
    decision = decide(_release("1.0.0"), ROUND, OPEN_BLOCK)
    assert decision.action is Action.HOLD
    assert "major" in decision.reason


def test_a_major_release_applies_when_the_operator_opts_in() -> None:
    decision = decide(
        _release("1.0.0", mechanism=MECHANISM_VERSION),
        ROUND,
        OPEN_BLOCK,
        allow_major=True,
    )
    assert decision.action is Action.APPLY


def test_a_sealed_round_defers_the_restart() -> None:
    decision = decide(_release("0.1.1"), ROUND, SEALED_BLOCK)
    assert decision.action is Action.DEFER
    assert decision.ready_at_block == ROUND.next.start_block


def test_a_restart_never_lands_mid_evaluation() -> None:
    for offset in (0, 100, ROUND.deadline_block - ROUND.close_block):
        block = ROUND.close_block + offset
        assert not is_safe_point(ROUND, block)
        assert decide(_release("0.1.1"), ROUND, block).action is Action.DEFER


def test_a_mechanism_change_without_an_activation_block_is_held() -> None:
    decision = decide(_release("0.2.0", mechanism="0.2.0"), ROUND, OPEN_BLOCK)
    assert decision.action is Action.HOLD
    assert "split consensus" in decision.reason


def test_a_mechanism_change_waits_for_its_activation_block() -> None:
    activation = OPEN_BLOCK + 5000
    decision = decide(
        _release("0.2.0", mechanism="0.2.0", activation=activation),
        ROUND,
        OPEN_BLOCK,
        allow_mechanism_change=True,
    )
    assert decision.action is Action.DEFER
    assert decision.ready_at_block == activation


def test_a_mechanism_change_still_needs_consent_at_its_activation_block() -> None:
    decision = decide(
        _release("0.2.0", mechanism="0.2.0", activation=OPEN_BLOCK - 1),
        ROUND,
        OPEN_BLOCK,
    )
    assert decision.action is Action.HOLD
    assert "allow-mechanism-change" in decision.reason


def test_a_consented_mechanism_change_applies_after_activation() -> None:
    decision = decide(
        _release("0.2.0", mechanism="0.2.0", activation=OPEN_BLOCK - 1),
        ROUND,
        OPEN_BLOCK,
        allow_mechanism_change=True,
    )
    assert decision.action is Action.APPLY


def test_sums_parsing_rejects_a_short_digest() -> None:
    with pytest.raises(VerificationError):
        parse_sums("abc  wheel.whl")


def test_sums_parsing_rejects_an_empty_file() -> None:
    with pytest.raises(VerificationError):
        parse_sums("# only a comment\n")


def test_sums_parsing_accepts_the_binary_star_form() -> None:
    digest = "a" * 64
    assert parse_sums(f"{digest} *wheel.whl") == {"wheel.whl": digest}


def test_a_wheel_absent_from_sums_is_rejected(tmp_path: Path) -> None:
    wheel = tmp_path / "mt.whl"
    wheel.write_bytes(b"payload")
    ok, reason = check_digest(wheel, {"other.whl": "b" * 64})
    assert not ok
    assert "not listed" in reason


def test_a_tampered_wheel_is_rejected(tmp_path: Path) -> None:
    wheel = tmp_path / "mt.whl"
    wheel.write_bytes(b"payload")
    ok, reason = check_digest(wheel, {"mt.whl": "b" * 64})
    assert not ok
    assert "hashes to" in reason


def test_a_matching_wheel_passes(tmp_path: Path) -> None:
    wheel = tmp_path / "mt.whl"
    wheel.write_bytes(b"payload")
    assert check_digest(wheel, {"mt.whl": sha256_file(wheel)}) == (True, "")


def test_verification_fails_closed_without_a_pinned_key(tmp_path: Path) -> None:
    wheel = tmp_path / "mt.whl"
    wheel.write_bytes(b"payload")
    sums = f"{sha256_file(wheel)}  mt.whl"
    result = verify_artifact(wheel, sums, b"sig", "", require_signature=True)
    assert not result.trusted
    assert "pinned" in result.reason


def test_an_operator_can_accept_an_unsigned_release(tmp_path: Path) -> None:
    wheel = tmp_path / "mt.whl"
    wheel.write_bytes(b"payload")
    sums = f"{sha256_file(wheel)}  mt.whl"
    result = verify_artifact(wheel, sums, b"", "", require_signature=False)
    assert result.trusted
    assert not result.signed


def test_an_unsigned_release_still_has_its_digest_checked(tmp_path: Path) -> None:
    wheel = tmp_path / "mt.whl"
    wheel.write_bytes(b"payload")
    sums = f"{'0' * 64}  mt.whl"
    result = verify_artifact(wheel, sums, b"", "", require_signature=False)
    assert not result.trusted
    assert not result.digest_ok


def test_the_checker_is_inert_until_armed() -> None:
    checker = UpdateChecker(UpdateSettings(enabled=False))
    assert not checker.due
    assert checker.step(ROUND, OPEN_BLOCK) is None


def test_the_checker_respects_its_poll_interval() -> None:
    clock = {"t": 1000.0}
    checker = UpdateChecker(
        UpdateSettings(enabled=True, poll_seconds=900),
        fetch=lambda *a, **k: None,
        now=lambda: clock["t"],
    )
    assert checker.due
    checker.check(ROUND, OPEN_BLOCK)
    assert not checker.due
    clock["t"] += 901
    assert checker.due


def test_the_first_check_is_due_on_a_freshly_booted_host() -> None:
    checker = UpdateChecker(
        UpdateSettings(enabled=True, poll_seconds=900),
        fetch=lambda *a, **k: None,
        now=lambda: 3.0,
    )
    assert checker.due


def test_a_failing_release_check_does_not_stop_the_validator() -> None:
    def broken(*_: object, **__: object) -> Release:
        raise ReleaseError("github is down")

    checker = UpdateChecker(UpdateSettings(enabled=True), fetch=broken)
    decision = checker.check(ROUND, OPEN_BLOCK)
    assert decision.action is Action.NONE
    assert "github is down" in decision.reason


def test_the_checker_installs_and_signals_a_restart() -> None:
    release = _release("0.1.1")
    applied = Applied(release, Verification(True, True, True), installed=True)
    checker = UpdateChecker(
        UpdateSettings(enabled=True),
        fetch=lambda *a, **k: release,
        apply=lambda *a, **k: applied,
    )
    result = checker.step(ROUND, OPEN_BLOCK)
    assert result is not None
    assert result.restart_required


def test_the_checker_does_not_install_while_a_round_is_sealed() -> None:
    calls: list[str] = []

    def spy(*_: object, **__: object) -> Applied:
        calls.append("applied")
        raise AssertionError("must not install during evaluation")

    checker = UpdateChecker(
        UpdateSettings(enabled=True), fetch=lambda *a, **k: _release("0.1.1"), apply=spy
    )
    assert checker.step(ROUND, SEALED_BLOCK) is None
    assert not calls


def test_a_failed_install_does_not_request_a_restart() -> None:
    release = _release("0.1.1")
    applied = Applied(
        release, Verification(True, True, True), installed=False, reason="pip install failed"
    )
    checker = UpdateChecker(
        UpdateSettings(enabled=True),
        fetch=lambda *a, **k: release,
        apply=lambda *a, **k: applied,
    )
    result = checker.step(ROUND, OPEN_BLOCK)
    assert result is not None
    assert not result.restart_required


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob"
    path.write_bytes(b"x" * 5_000_000)
    assert sha256_file(path) == hashlib.sha256(b"x" * 5_000_000).hexdigest()
