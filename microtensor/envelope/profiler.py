from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from microtensor.core.constants import (
    LATENCY_SAMPLE_COUNT,
    PROFILE_DURATION_SECONDS,
    PROFILE_SAMPLE_INTERVAL_MS,
)
from microtensor.core.hashing import tree_size_bytes
from microtensor.core.protocol import ArtifactFormat, LoadManifest, MeasuredEnvelope
from microtensor.core.tracks import HardwareClass, get_class
from microtensor.envelope.device import DeviceProfile, conforms, detect
from microtensor.envelope.latency import Distribution, summarise
from microtensor.envelope.sampler import ResidentSampler
from microtensor.harness.contract import Engine, Request, Response
from microtensor.harness.registry import engine_for, load_builtin


class ProfileError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProfilePlan:
    prompt: str
    max_input: dict[str, Any] = field(default_factory=dict)
    duration_seconds: int = PROFILE_DURATION_SECONDS
    max_requests: int = LATENCY_SAMPLE_COUNT
    sample_interval_ms: int = PROFILE_SAMPLE_INTERVAL_MS
    max_output_tokens: int = 256

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("the profile must exercise the artifact at its declared maximum input")
        if self.duration_seconds < 1:
            raise ValueError("profile duration must be at least one second")
        if self.max_requests < 1:
            raise ValueError("profile must issue at least one request")


@dataclass(frozen=True, slots=True)
class ProfileReport:
    envelope: MeasuredEnvelope
    latency: Distribution
    device: DeviceProfile
    requests: int
    failures: int
    rss_samples: int

    @property
    def clean(self) -> bool:
        return self.failures == 0

    @property
    def failure_rate(self) -> float:
        return self.failures / self.requests if self.requests else 1.0


def _request(plan: ProfilePlan, index: int) -> Request:
    return Request(
        task_ref=f"profile-{index}",
        prompt=plan.prompt,
        inputs=dict(plan.max_input),
        max_output_tokens=plan.max_output_tokens,
        nonce=f"profile:{index}",
    )


def _cold_start(
    engine: Engine, artifact: Path, manifest: LoadManifest, plan: ProfilePlan
) -> tuple[float, Response]:
    started = time.perf_counter()
    try:
        engine.load(artifact, manifest)
    except Exception as exc:
        raise ProfileError(f"artifact failed to load: {exc}") from exc

    response = engine.generate(_request(plan, 0))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not response.ok:
        raise ProfileError(f"artifact produced no output on a cold cache: {response.error}")
    if response.output_tokens <= 0:
        raise ProfileError("artifact produced no output on a cold cache")
    return elapsed_ms, response


def profile(
    engine: Engine,
    artifact: Path,
    manifest: LoadManifest,
    hardware: HardwareClass,
    plan: ProfilePlan,
) -> ProfileReport:
    device = detect()
    try:
        size_bytes = tree_size_bytes(artifact) if artifact.is_dir() else artifact.stat().st_size
    except OSError as exc:
        raise ProfileError(f"artifact {artifact} could not be sized: {exc}") from exc

    sampler = ResidentSampler(plan.sample_interval_ms)
    sampler.start()
    try:
        cold_start_ms, _ = _cold_start(engine, artifact, manifest, plan)
        sampler.mark()

        ttfts: list[float] = []
        totals: list[float] = []
        throughputs: list[float] = []
        failures = 0
        issued = 0
        deadline = time.perf_counter() + plan.duration_seconds

        while issued < plan.max_requests and time.perf_counter() < deadline:
            issued += 1
            response = engine.generate(_request(plan, issued))
            if not response.ok or response.output_tokens <= 0:
                failures += 1
                continue
            ttfts.append(response.ttft_ms)
            totals.append(response.total_ms)
            if response.tokens_per_second > 0.0:
                throughputs.append(response.tokens_per_second)

        sampler.mark()
    finally:
        sampler.stop()
        with contextlib.suppress(Exception):
            engine.unload()

    if not ttfts:
        raise ProfileError("no request produced output during the sustained window")

    latency = summarise(ttfts)
    total = summarise(totals)
    envelope = MeasuredEnvelope(
        size_bytes=size_bytes,
        peak_rss_bytes=sampler.peak_bytes,
        input_at_peak=dict(plan.max_input),
        ttft_p50_ms=round(latency.p50),
        ttft_p95_ms=round(latency.p95),
        tokens_per_second=sum(throughputs) / len(throughputs) if throughputs else 0.0,
        cold_start_ms=round(cold_start_ms),
        device_profile=device.digest,
        conforming=conforms(device, hardware),
        total_p50_ms=round(total.p50),
        total_p95_ms=round(total.p95),
    )

    return ProfileReport(
        envelope=envelope,
        latency=latency,
        device=device,
        requests=issued,
        failures=failures,
        rss_samples=sampler.count,
    )


def run_profile(
    artifact_path: str,
    manifest_payload: dict[str, Any],
    hardware_id: str,
    plan_payload: dict[str, Any],
) -> ProfileReport:
    manifest = LoadManifest(
        format=ArtifactFormat(manifest_payload["format"]),
        quantization=manifest_payload.get("quantization", ""),
        entrypoint=manifest_payload["entrypoint"],
        max_input=manifest_payload["max_input"],
        preprocessing=manifest_payload.get("preprocessing", {}),
        base_model=manifest_payload.get("base_model", ""),
    )
    load_builtin()
    engine = engine_for(manifest.format)
    return profile(
        engine,
        Path(artifact_path),
        manifest,
        get_class(hardware_id),
        ProfilePlan(**plan_payload),
    )
