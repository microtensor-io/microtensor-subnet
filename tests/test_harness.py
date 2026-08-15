from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from microtensor.core.protocol import ArtifactFormat, Fault, LoadManifest
from microtensor.core.tracks import Decoding, get_class
from microtensor.harness import (
    EngineError,
    EngineUnavailable,
    Limits,
    Request,
    Response,
    UnsupportedPlatform,
    available,
    batch,
    clear,
    engine_for,
    has,
    register,
    run_jailed,
    sandbox_available,
    unregister,
)
from microtensor.harness.engines.reference import INFO, ReferenceEngine
from microtensor.harness.jail import JailResult


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    clear()
    yield
    clear()


def _manifest() -> LoadManifest:
    return LoadManifest(
        format=ArtifactFormat.SAFETENSORS,
        quantization="int8",
        entrypoint="model.safetensors",
        max_input={"tokens": 4096},
    )


def echo(value: str) -> str:
    return value.upper()


def explode() -> None:
    raise ValueError("the artifact tripped over itself")


def spin() -> None:
    while True:
        pass


def test_request_rejects_an_unreferenced_task() -> None:
    with pytest.raises(ValueError):
        Request(task_ref="", prompt="hi")


def test_request_rejects_a_seedless_seeded_decode() -> None:
    with pytest.raises(ValueError):
        Request(task_ref="t1", prompt="hi", decoding=Decoding.SEEDED)


def test_response_reports_throughput_only_once_decoding_started() -> None:
    stalled = Response(task_ref="t1", ttft_ms=10.0, total_ms=10.0, output_tokens=5)
    assert stalled.tokens_per_second == 0.0
    fast = Response(task_ref="t1", ttft_ms=100.0, total_ms=1100.0, output_tokens=101)
    assert fast.tokens_per_second == pytest.approx(100.0)


def test_a_failed_response_always_carries_a_reason() -> None:
    assert Response.failed("t1", "").error
    assert not Response.failed("t1", "boom").ok


def test_registry_refuses_a_silent_overwrite() -> None:
    register(ArtifactFormat.SAFETENSORS, ReferenceEngine, INFO)
    with pytest.raises(EngineError):
        register(ArtifactFormat.SAFETENSORS, ReferenceEngine, INFO)
    register(ArtifactFormat.SAFETENSORS, ReferenceEngine, INFO, replace=True)


def test_registry_reports_what_it_holds() -> None:
    register(ArtifactFormat.SAFETENSORS, ReferenceEngine, INFO)
    assert has(ArtifactFormat.SAFETENSORS)
    assert available() == (ArtifactFormat.SAFETENSORS,)
    unregister(ArtifactFormat.SAFETENSORS)
    assert not has(ArtifactFormat.SAFETENSORS)


def test_missing_engine_names_what_is_registered() -> None:
    with pytest.raises(EngineUnavailable):
        engine_for(ArtifactFormat.GGUF)


def test_reference_engine_is_deterministic(tmp_path: Path) -> None:
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"weights")

    engine = ReferenceEngine()
    engine.load(artifact, _manifest())
    request = Request(task_ref="t1", prompt="hello", nonce="abc")
    assert engine.generate(request).output == engine.generate(request).output


def test_reference_engine_separates_nonces(tmp_path: Path) -> None:
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"weights")

    engine = ReferenceEngine()
    engine.load(artifact, _manifest())
    a = engine.generate(Request(task_ref="t1", prompt="hello", nonce="one"))
    b = engine.generate(Request(task_ref="t1", prompt="hello", nonce="two"))
    assert a.output != b.output


def test_generating_before_load_fails_rather_than_guesses() -> None:
    response = ReferenceEngine().generate(Request(task_ref="t1", prompt="hi"))
    assert not response.ok


def test_batch_converts_engine_errors_into_zero_scoring_responses() -> None:
    class Broken(ReferenceEngine):
        def generate(self, request: Request) -> Response:
            raise EngineError("kernel refused")

    responses = batch([Request(task_ref="t1", prompt="a")], Broken())
    assert len(responses) == 1
    assert not responses[0].ok
    assert "kernel refused" in responses[0].error


def test_limits_demand_a_wall_backstop_above_the_cpu_budget() -> None:
    with pytest.raises(ValueError):
        Limits(cpu_seconds=10, wall_seconds=10, rss_bytes=1024)


def test_limits_for_a_class_leave_headroom_over_the_ceiling() -> None:
    hardware = get_class("laptop")
    limits = Limits.for_class(hardware, cpu_seconds=30)
    assert limits.rss_bytes > hardware.max_rss_bytes
    assert limits.wall_seconds >= 60


def test_jail_refuses_to_run_unsandboxed_by_default() -> None:
    if sandbox_available():
        pytest.skip("this host can enforce resource limits")
    with pytest.raises(UnsupportedPlatform):
        run_jailed(echo, "hi", limits=Limits(cpu_seconds=5, wall_seconds=15, rss_bytes=1 << 30))


def test_jail_returns_the_workers_value() -> None:
    result = run_jailed(
        echo,
        "hi",
        limits=Limits(cpu_seconds=20, wall_seconds=60, rss_bytes=1 << 30),
        allow_unsandboxed=True,
    )
    assert result.ok
    assert result.value == "HI"
    assert result.fault is None


def test_jail_marks_an_unsandboxed_run() -> None:
    result = run_jailed(
        echo,
        "hi",
        limits=Limits(cpu_seconds=20, wall_seconds=60, rss_bytes=1 << 30),
        allow_unsandboxed=True,
    )
    assert result.sandboxed is sandbox_available()


def test_a_raising_artifact_is_the_artifacts_fault() -> None:
    result = run_jailed(
        explode,
        limits=Limits(cpu_seconds=20, wall_seconds=60, rss_bytes=1 << 30),
        allow_unsandboxed=True,
    )
    assert not result.ok
    assert result.fault is Fault.ARTIFACT
    assert "tripped over itself" in result.error


def test_a_hung_artifact_is_killed_and_blamed() -> None:
    result = run_jailed(
        spin,
        limits=Limits(cpu_seconds=1, wall_seconds=3, rss_bytes=1 << 30),
        allow_unsandboxed=True,
    )
    assert not result.ok
    assert result.timed_out
    assert result.fault is Fault.ARTIFACT


def test_infrastructure_failure_abstains_rather_than_scoring_zero() -> None:
    result = JailResult(completed=False, error="engine binary is missing")
    assert result.fault is Fault.INFRASTRUCTURE
