from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from microtensor.core.constants import ACCURACY_DECIMALS, MECHANISM_VERSION
from microtensor.core.hashing import DIGEST_PREFIX, canonical_digest, canonical_hash
from microtensor.core.protocol import Evaluation, MeasuredEnvelope, Submission
from microtensor.core.tracks import get_track

SIGNATURE_PREFIX = "ed25519:"


class Signer(Protocol):
    def sign(self, data: bytes) -> bytes: ...

    @property
    def ss58_address(self) -> str: ...


class Verifier(Protocol):
    def verify(self, data: bytes, signature: bytes) -> bool: ...


@dataclass(frozen=True, slots=True)
class Attestation:
    miner_hotkey: str
    validator_hotkey: str
    signature: str
    canonical_hash: str


@dataclass(frozen=True, slots=True)
class Certificate:
    mechanism_version: str
    track: str
    hardware_class: str
    artifact: dict[str, Any]
    envelope: dict[str, Any]
    accuracy: dict[str, Any]
    runtime: dict[str, Any]
    work_evidence: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    system: dict[str, Any] = field(default_factory=dict)
    components: tuple[dict[str, Any], ...] = ()
    cost: dict[str, Any] = field(default_factory=dict)
    contribution: tuple[dict[str, Any], ...] = ()
    attestation: Attestation | None = None

    def body(self) -> dict[str, Any]:
        return {
            "mechanism_version": self.mechanism_version,
            "track": self.track,
            "hardware_class": self.hardware_class,
            "artifact": self.artifact,
            "envelope": self.envelope,
            "accuracy": self.accuracy,
            "runtime": self.runtime,
            "work_evidence": self.work_evidence,
            "provenance": self.provenance,
            "system": self.system,
            "components": list(self.components),
            "cost": self.cost,
            "contribution": list(self.contribution),
        }

    def digest(self) -> bytes:
        return canonical_digest(self.body())

    def signed_by(self, signer: Signer, miner_hotkey: str) -> Certificate:
        digest = self.digest()
        return Certificate(
            mechanism_version=self.mechanism_version,
            track=self.track,
            hardware_class=self.hardware_class,
            artifact=self.artifact,
            envelope=self.envelope,
            accuracy=self.accuracy,
            runtime=self.runtime,
            work_evidence=self.work_evidence,
            provenance=self.provenance,
            system=self.system,
            components=self.components,
            cost=self.cost,
            contribution=self.contribution,
            attestation=Attestation(
                miner_hotkey=miner_hotkey,
                validator_hotkey=signer.ss58_address,
                signature=SIGNATURE_PREFIX + bytes(signer.sign(digest)).hex(),
                canonical_hash=DIGEST_PREFIX + digest.hex(),
            ),
        )

    def verify(self, verifier: Verifier) -> tuple[bool, str]:
        if self.attestation is None:
            return False, "unsigned"

        digest = self.digest()
        if self.attestation.canonical_hash != DIGEST_PREFIX + digest.hex():
            return False, "body was modified after signing"

        raw = self.attestation.signature
        if not raw.startswith(SIGNATURE_PREFIX):
            return False, f"unrecognised signature scheme in {raw[:16]!r}"

        try:
            signature = bytes.fromhex(raw[len(SIGNATURE_PREFIX) :])
        except ValueError:
            return False, "signature is not valid hex"

        if not verifier.verify(digest, signature):
            return False, "signature does not verify against the validator hotkey"

        return True, ""

    def to_dict(self) -> dict[str, Any]:
        d = self.body()
        if self.attestation is not None:
            d["attestation"] = asdict(self.attestation)
        return d

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.body())


def system_block(
    evaluation: Evaluation,
    *,
    frontier_member: bool = False,
    exclusive_hypervolume: int | None = None,
) -> dict[str, Any]:
    """The VSC system block: what composition was run and how it placed."""
    return {
        "manifest_hash": evaluation.system_digest,
        "resolve_rate": round(evaluation.resolve_rate, ACCURACY_DECIMALS),
        "frontier": {
            "member": frontier_member,
            "exclusive_hypervolume": exclusive_hypervolume,
        },
    }


def cost_block(evaluation: Evaluation, *, front_share: float | None = None) -> dict[str, Any]:
    """Expected per-query cost, measured on this run rather than modelled."""
    return {
        "expected_ms": round(evaluation.expected_ms, 2),
        "joules_per_query": round(evaluation.expected_j, 4) if evaluation.expected_j else None,
        "front_share": front_share,
    }


def build_certificate(
    submission: Submission,
    evaluation: Evaluation,
    measured: MeasuredEnvelope,
    *,
    round_id: int,
    engine_version: str,
    decoding: str,
    seed: int = 0,
    work_evidence: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    system: dict[str, Any] | None = None,
    components: Sequence[dict[str, Any]] = (),
    cost: dict[str, Any] | None = None,
    contribution: Sequence[dict[str, Any]] = (),
) -> Certificate:
    track = get_track(submission.track)
    return Certificate(
        mechanism_version=MECHANISM_VERSION,
        track=submission.track,
        hardware_class=submission.hardware_class,
        artifact={
            "weights_hash": submission.artifact_digest,
            "manifest_hash": canonical_hash(submission.manifest.to_dict()),
            "total_bytes": measured.size_bytes,
            "format": submission.manifest.format.value,
            "base_model": submission.manifest.base_model,
            "round": round_id,
        },
        envelope={
            "size_bytes": measured.size_bytes,
            "peak_rss_bytes": measured.peak_rss_bytes,
            "input_at_peak": measured.input_at_peak,
            "ttft_p50_ms": measured.ttft_p50_ms,
            "ttft_p95_ms": measured.ttft_p95_ms,
            "total_p50_ms": measured.total_p50_ms,
            "total_p95_ms": measured.total_p95_ms,
            "tok_per_sec": round(measured.tokens_per_second, 2),
            "cold_start_ms": measured.cold_start_ms,
            "device_profile": measured.device_profile,
        },
        accuracy={
            "score_fixed": round(evaluation.score_fixed, ACCURACY_DECIMALS),
            "score_rotating": round(evaluation.score_rotating, ACCURACY_DECIMALS),
            "score_combined": round(evaluation.score_combined, ACCURACY_DECIMALS),
            "n_fixed": evaluation.n_fixed,
            "n_rotating": evaluation.n_rotating,
            "corpus_version": evaluation.corpus_version,
            "metric": track.metric,
            "metric_display": track.published_metric,
        },
        runtime={
            "decode": decoding,
            "temperature": 0.0,
            "seed": seed,
            "engine_version": engine_version,
            "quantization": submission.manifest.quantization,
        },
        work_evidence=dict(work_evidence or {}),
        provenance=dict(provenance or {}),
        system=dict(system or {}),
        components=tuple(components),
        cost=dict(cost or {}),
        contribution=tuple(contribution),
    )
