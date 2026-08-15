from __future__ import annotations

import time
from pathlib import Path

import pytest

from microtensor.core.protocol import ArtifactFormat, DeclaredEnvelope, LoadManifest, evaluate_gate
from microtensor.core.tracks import get_class
from microtensor.envelope import (
    DeviceProfile,
    ProfileError,
    ProfilePlan,
    ResidentSampler,
    aggregate_median,
    conforms,
    detect,
    median,
    profile,
    quantile,
    reference_digest,
    summarise,
)
from microtensor.harness.contract import Request, Response
from microtensor.harness.engines.reference import ReferenceEngine


def _manifest() -> LoadManifest:
    return LoadManifest(
        format=ArtifactFormat.SAFETENSORS,
        quantization="int8",
        entrypoint="model.safetensors",
        max_input={"tokens": 4096},
    )


def _plan(**kw: object) -> ProfilePlan:
    base: dict[str, object] = {
        "prompt": "x" * 64,
        "max_input": {"tokens": 4096},
        "duration_seconds": 1,
        "max_requests": 8,
        "sample_interval_ms": 5,
    }
    base.update(kw)
    return ProfilePlan(**base)  # type: ignore[arg-type]


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"w" * 4096)
    return path


def test_quantile_interpolates_between_samples() -> None:
    assert quantile([0.0, 10.0], 0.5) == 5.0
    assert quantile([1.0], 0.99) == 1.0
    assert quantile([], 0.5) == 0.0


def test_quantile_rejects_a_probability_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError):
        quantile([1.0, 2.0], 1.5)


def test_quantile_is_order_independent() -> None:
    assert quantile([9.0, 1.0, 5.0], 0.5) == quantile([1.0, 5.0, 9.0], 0.5)


def test_a_single_warm_sample_cannot_hide_a_tail() -> None:
    sustained = [10.0] * 90 + [900.0] * 10
    distribution = summarise(sustained)
    assert distribution.p50 == pytest.approx(10.0)
    assert distribution.p95 > 100.0
    assert distribution.maximum == 900.0


def test_summary_of_nothing_is_empty_not_zero_shaped() -> None:
    assert summarise([]).is_empty


def test_tail_ratio_surfaces_a_bimodal_engine() -> None:
    assert summarise([10.0] * 90 + [500.0] * 10).tail_ratio > 5.0


def test_validators_aggregate_by_median() -> None:
    assert aggregate_median([100.0, 104.0, 900.0]) == 104.0
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_device_digest_is_stable_and_prefixed() -> None:
    a = DeviceProfile(machine="x86_64", system="linux", cpu_cores=8, total_memory_bytes=1 << 34)
    b = DeviceProfile(machine="x86_64", system="linux", cpu_cores=8, total_memory_bytes=1 << 34)
    assert a.digest == b.digest
    assert a.digest.startswith("dev:")


def test_device_digest_separates_accelerators() -> None:
    cpu = DeviceProfile(machine="x86_64", system="linux", cpu_cores=8, total_memory_bytes=1 << 34)
    gpu = DeviceProfile(
        machine="x86_64",
        system="linux",
        cpu_cores=8,
        total_memory_bytes=1 << 34,
        accelerator="NVIDIA L4",
    )
    assert cpu.digest != gpu.digest


def test_a_smaller_host_does_not_conform() -> None:
    reference = DeviceProfile(
        machine="x86_64", system="linux", cpu_cores=16, total_memory_bytes=1 << 35
    )
    small = DeviceProfile(
        machine="x86_64", system="linux", cpu_cores=8, total_memory_bytes=1 << 35
    )
    assert not small.conforms_to(reference)
    assert reference.conforms_to(reference)


def test_conformance_is_open_when_no_profile_is_published() -> None:
    assert conforms(detect(), get_class("laptop"))


def test_reference_digest_is_derived_when_unpublished() -> None:
    assert reference_digest(get_class("laptop")).startswith("dev:")


def test_sampler_records_a_peak() -> None:
    sampler = ResidentSampler(interval_ms=5)
    with sampler:
        pass
    assert sampler.count >= 1


def test_sampler_rejects_a_zero_interval() -> None:
    with pytest.raises(ValueError):
        ResidentSampler(interval_ms=0)


def test_sampler_refuses_to_start_twice() -> None:
    sampler = ResidentSampler(interval_ms=5).start()
    with pytest.raises(RuntimeError):
        sampler.start()
    sampler.stop()


def test_plan_demands_a_maximum_input_prompt() -> None:
    with pytest.raises(ValueError):
        ProfilePlan(prompt="")


def test_profile_measures_the_declared_maximum_input(artifact: Path) -> None:
    report = profile(ReferenceEngine(), artifact, _manifest(), get_class("laptop"), _plan())
    assert report.envelope.input_at_peak == {"tokens": 4096}
    assert report.envelope.size_bytes == 4096
    assert report.requests > 0
    assert report.clean


def test_profile_records_cold_start_separately(artifact: Path) -> None:
    report = profile(ReferenceEngine(), artifact, _manifest(), get_class("laptop"), _plan())
    assert report.envelope.cold_start_ms >= 0
    assert report.envelope.ttft_p95_ms >= report.envelope.ttft_p50_ms


def test_the_cold_request_stays_out_of_the_latency_distribution(artifact: Path) -> None:
    report = profile(ReferenceEngine(), artifact, _manifest(), get_class("laptop"), _plan(
        max_requests=6
    ))
    assert report.requests == 6
    assert report.latency.count == 6


def test_a_slow_loading_artifact_shows_it_in_cold_start(artifact: Path) -> None:
    class SlowToLoad(ReferenceEngine):
        def load(self, path: Path, manifest: LoadManifest) -> None:
            time.sleep(0.25)
            super().load(path, manifest)

    report = profile(SlowToLoad(), artifact, _manifest(), get_class("laptop"), _plan())
    assert report.envelope.cold_start_ms >= 200
    assert report.envelope.ttft_p50_ms < report.envelope.cold_start_ms


def test_profile_carries_the_device_profile_into_the_envelope(artifact: Path) -> None:
    report = profile(ReferenceEngine(), artifact, _manifest(), get_class("laptop"), _plan())
    assert report.envelope.device_profile == detect().digest


def test_a_model_that_cannot_load_is_not_profiled(tmp_path: Path) -> None:
    with pytest.raises(ProfileError):
        profile(
            ReferenceEngine(),
            tmp_path / "absent.safetensors",
            _manifest(),
            get_class("laptop"),
            _plan(),
        )


def test_a_model_that_produces_nothing_cold_is_not_profiled(artifact: Path) -> None:
    class Silent(ReferenceEngine):
        def generate(self, request: Request) -> Response:
            return Response.failed(request.task_ref, "no output")

    with pytest.raises(ProfileError):
        profile(Silent(), artifact, _manifest(), get_class("laptop"), _plan())


def test_a_model_that_dies_after_the_cold_request_is_not_profiled(artifact: Path) -> None:
    class Flaky(ReferenceEngine):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def generate(self, request: Request) -> Response:
            self.calls += 1
            if self.calls == 1:
                return super().generate(request)
            return Response.failed(request.task_ref, "engine collapsed")

    with pytest.raises(ProfileError):
        profile(Flaky(), artifact, _manifest(), get_class("laptop"), _plan())


def test_the_gate_reads_a_measured_envelope(artifact: Path) -> None:
    report = profile(ReferenceEngine(), artifact, _manifest(), get_class("laptop"), _plan())
    declared = DeclaredEnvelope(
        size_bytes=report.envelope.size_bytes,
        peak_rss_bytes=max(report.envelope.peak_rss_bytes, 1),
        p95_latency_ms=max(report.envelope.ttft_p95_ms, 1),
    )
    gate = evaluate_gate(report.envelope, declared, get_class("server-cpu"))
    assert gate.admitted, gate.reason
