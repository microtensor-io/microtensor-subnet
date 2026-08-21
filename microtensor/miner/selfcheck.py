from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from microtensor.core.protocol import (
    DeclaredEnvelope,
    GateResult,
    LoadManifest,
    MeasuredEnvelope,
    evaluate_gate,
)
from microtensor.core.tracks import get_class
from microtensor.envelope.probe import max_input_prompt
from microtensor.envelope.profiler import ProfilePlan, run_profile
from microtensor.harness.jail import run_jailed
from microtensor.harness.limits import Limits
from microtensor.harness.registry import available, load_builtin

log = logging.getLogger("microtensor.miner.selfcheck")

DECLARATION_MARGIN = 0.10


# Loading weights is cpu work that happens before a single token is produced:
# a GGUF is mmapped and faulted in, an ONNX graph is parsed and optimised.
# Budgeting only for generation kills the load on a large artifact, which is
# how a model well inside its memory ceiling died with a bare exit -9.
LOAD_CPU_SECONDS = 120


def profiling_cpu_budget(profile_seconds: int) -> int:
    """Cpu seconds for one profiling run: the load, then the generation.

    Doubled over the wall duration because the profiler drives the artifact
    continuously and a thread that never idles spends cpu at wall rate.
    """
    return LOAD_CPU_SECONDS + max(4, profile_seconds * 2)


class SelfCheckError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SelfCheck:
    measured: MeasuredEnvelope
    proposed: DeclaredEnvelope
    gate: GateResult
    hardware_class: str

    @property
    def admissible(self) -> bool:
        return self.gate.admitted

    def report(self) -> str:
        hardware = get_class(self.hardware_class)
        lines = [
            f"class            {self.hardware_class}  ({hardware.reference})",
            f"size             {self.measured.size_bytes / 1024**3:.2f} GiB"
            f"   ceiling {hardware.max_size_bytes / 1024**3:.2f} GiB",
            f"peak rss         {self.measured.peak_rss_bytes / 1024**3:.2f} GiB"
            f"   ceiling {hardware.max_rss_bytes / 1024**3:.2f} GiB",
            f"ttft p50 / p95   {self.measured.ttft_p50_ms} / {self.measured.ttft_p95_ms} ms"
            f"   ceiling {hardware.max_p95_ms} ms",
            f"cold start       {self.measured.cold_start_ms} ms",
            f"throughput       {self.measured.tokens_per_second:.1f} tok/s",
            f"device profile   {self.measured.device_profile}",
            "",
            "declare this envelope:",
            f"  size_bytes      {self.proposed.size_bytes}",
            f"  peak_rss_bytes  {self.proposed.peak_rss_bytes}",
            f"  p95_latency_ms  {self.proposed.p95_latency_ms}",
        ]
        if not self.admissible:
            lines += ["", f"INADMISSIBLE: {self.gate.reason}"]
        return "\n".join(lines)


def propose(measured: MeasuredEnvelope, margin: float = DECLARATION_MARGIN) -> DeclaredEnvelope:
    return DeclaredEnvelope(
        size_bytes=max(1, int(measured.size_bytes * (1.0 + margin))),
        peak_rss_bytes=max(1, int(measured.peak_rss_bytes * (1.0 + margin))),
        p95_latency_ms=max(1, int(measured.ttft_p95_ms * (1.0 + margin))),
    )


def selfcheck(
    artifact: Path,
    load: LoadManifest,
    hardware_class: str,
    *,
    seed: str = "selfcheck",
    profile_seconds: int = 60,
    margin: float = DECLARATION_MARGIN,
    allow_unsandboxed: bool = False,
) -> SelfCheck:
    load_builtin()
    if load.format not in available():
        raise SelfCheckError(
            f"no engine for {load.format.value}; validators will not be able to run this artifact"
        )

    hardware = get_class(hardware_class)
    max_input = dict(load.max_input)
    plan = ProfilePlan(
        prompt=max_input_prompt(seed, max_input),
        max_input=max_input,
        duration_seconds=profile_seconds,
    )

    result = run_jailed(
        run_profile,
        str(artifact),
        load.to_dict(),
        hardware_class,
        {
            "prompt": plan.prompt,
            "max_input": plan.max_input,
            "duration_seconds": plan.duration_seconds,
            "max_requests": plan.max_requests,
            "sample_interval_ms": plan.sample_interval_ms,
            "max_output_tokens": plan.max_output_tokens,
        },
        limits=Limits.for_class(hardware, profiling_cpu_budget(profile_seconds)),
        allow_unsandboxed=allow_unsandboxed,
    )

    if not result.ok:
        raise SelfCheckError(f"the artifact did not survive profiling: {result.error}")

    measured: MeasuredEnvelope = result.value.envelope
    proposed = propose(measured, margin)
    return SelfCheck(
        measured=measured,
        proposed=proposed,
        gate=evaluate_gate(measured, proposed, hardware),
        hardware_class=hardware_class,
    )
