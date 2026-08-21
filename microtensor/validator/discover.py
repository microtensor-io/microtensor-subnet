from __future__ import annotations

import logging
from dataclasses import dataclass, field

from microtensor.chain.commitment import Commitment, decode_all
from microtensor.chain.metagraph import MetagraphSnapshot
from microtensor.chain.rounds import Round
from microtensor.chain.wallet import verify_payload
from microtensor.core.constants import (
    ALLOWED_BASE_MODELS,
    HOST_PROFILE,
    PROVENANCE_REQUIRED,
)
from microtensor.core.protocol import Role
from microtensor.core.system import SystemManifest
from microtensor.provenance.record import Verdict
from microtensor.provenance.record import check as provenance_check
from microtensor.registry.fetch import ArtifactMismatch, FetchError, fetch_manifest
from microtensor.registry.manifest import ArtifactManifest
from microtensor.validator.context import ValidatorContext

log = logging.getLogger("microtensor.validator.discover")


@dataclass(frozen=True, slots=True)
class Participant:
    hotkey: str
    uid: int
    commitment: Commitment
    manifest: ArtifactManifest

    @property
    def competition(self) -> tuple[str, str]:
        return self.commitment.competition

    @property
    def system(self) -> SystemManifest:
        if self.manifest.system is not None:
            return self.manifest.system
        return SystemManifest.single(
            self.manifest.artifact_digest,
            self.competition[1],
            base_model=self.manifest.load.base_model,
        )


@dataclass(frozen=True, slots=True)
class Roster:
    round_index: int
    participants: tuple[Participant, ...]
    rejected: tuple[tuple[str, str], ...]
    provenance: dict[str, Verdict] = field(default_factory=dict)

    def for_competition(self, track: str, hardware_class: str) -> tuple[Participant, ...]:
        return tuple(p for p in self.participants if p.competition == (track, hardware_class))

    def __len__(self) -> int:
        return len(self.participants)


def _signature_ok(manifest: ArtifactManifest) -> tuple[bool, str]:
    if not manifest.signature:
        return False, "manifest carries no signature"
    if not verify_payload(manifest.hotkey, manifest.body(), manifest.signature):
        return False, "manifest signature does not verify against the declaring hotkey"
    return True, ""


def _provenance_reason(
    context: ValidatorContext,
    hotkey: str,
    system: SystemManifest,
    commitment: Commitment,
    commit_block: int,
) -> tuple[str, Verdict | None]:
    """Every component needs its own run, not just the one named on chain.

    Checking the submission alone would let an unprovenanced specialist ride in
    behind a compliant front, since the submission-level digest never names it.
    """
    if not PROVENANCE_REQUIRED or context.runs is None:
        return "", None

    runs = context.runs.fetch(hotkey)
    verdict: Verdict | None = None

    for component in system.components:
        verdict = provenance_check(
            runs,
            hotkey=hotkey,
            artifact_digest=component.artifact_digest,
            track=commitment.track,
            hardware_class=commitment.hardware_class,
            commit_block=commit_block,
        )
        if not verdict.admissible:
            if system.degenerate:
                return verdict.reason, verdict
            return f"{component.role.value}: {verdict.reason}", verdict

    return "", verdict


def _base_model_reason(system: SystemManifest) -> str:
    """The allowlist applies to every component that carries weights."""
    if not ALLOWED_BASE_MODELS:
        return ""

    for component in system.components:
        if component.role is Role.ROUTER:
            continue
        if component.base_model not in ALLOWED_BASE_MODELS:
            if system.degenerate:
                return "base model not on the allowlist"
            return f"{component.role.value} base model not on the allowlist"
    return ""


def discover(context: ValidatorContext, snapshot: MetagraphSnapshot, round_: Round) -> Roster:
    raw = context.client.commitments(list(snapshot.hotkeys))
    commitments = decode_all(raw)
    log.info("round %d: %d commitments readable", round_.index, len(commitments))

    uid_by_hotkey = snapshot.uid_by_hotkey
    open_competitions = set(context.competitions)
    accepted: list[Participant] = []
    rejected: list[tuple[str, str]] = []
    provenance: dict[str, Verdict] = {}
    # Heights, not seconds: the run must have finished at or before the block
    # the commitment window closes on. Multiplying a height into fake seconds
    # is the unit slip that once rejected every compliant miner (issue #1).
    commit_block = round_.close_block

    for hotkey, commitment in sorted(commitments.items()):
        reason = _reject_reason(commitment, round_, open_competitions)
        if reason:
            rejected.append((hotkey, reason))
            context.state.record_submission(
                round_.index,
                hotkey,
                commitment.track,
                commitment.hardware_class,
                commitment.manifest_digest,
                commitment.source,
                reason=reason,
            )
            continue

        verdict = None
        try:
            manifest = fetch_manifest(commitment, workdir=context.config.work_dir)
        except ArtifactMismatch as exc:
            reason = f"manifest rejected: {exc}"
        except FetchError as exc:
            reason = f"manifest unfetchable: {exc}"
        else:
            reason = _manifest_reason(manifest, hotkey, context.config.verify_signatures)
            if not reason:
                system = _system_of(manifest, commitment.hardware_class)
                reason = _base_model_reason(system)
                if not reason:
                    reason, verdict = _provenance_reason(
                        context, hotkey, system, commitment, commit_block
                    )
                if verdict is not None and verdict.admissible:
                    provenance[hotkey] = verdict

        if reason:
            rejected.append((hotkey, reason))
            context.state.record_submission(
                round_.index,
                hotkey,
                commitment.track,
                commitment.hardware_class,
                commitment.manifest_digest,
                commitment.source,
                reason=reason,
            )
            continue

        accepted.append(
            Participant(
                hotkey=hotkey,
                uid=uid_by_hotkey[hotkey],
                commitment=commitment,
                manifest=manifest,
            )
        )
        context.state.record_submission(
            round_.index,
            hotkey,
            commitment.track,
            commitment.hardware_class,
            commitment.manifest_digest,
            commitment.source,
            artifact_digest=manifest.artifact_digest,
            accepted=True,
        )

    log.info(
        "round %d: %d participants frozen, %d rejected",
        round_.index,
        len(accepted),
        len(rejected),
    )
    return Roster(round_.index, tuple(accepted), tuple(rejected), provenance)


def _reject_reason(
    commitment: Commitment, round_: Round, open_competitions: set[tuple[str, str]]
) -> str:
    if commitment.round_index != round_.index:
        return f"commitment names round {commitment.round_index}, not {round_.index}"
    if commitment.competition not in open_competitions:
        return f"{commitment.track}/{commitment.hardware_class} is not an open competition here"
    return ""


def _system_of(manifest: ArtifactManifest, hardware_class: str) -> SystemManifest:
    if manifest.system is not None:
        return manifest.system
    return SystemManifest.single(
        manifest.artifact_digest, hardware_class, base_model=manifest.load.base_model
    )


def _system_reason(manifest: ArtifactManifest, hardware_class: str) -> str:
    system = manifest.system
    if system is None:
        return ""
    placed, reason = system.fits_class(hardware_class)
    if not placed:
        return reason
    if system.specialist is not None and system.specialist.placement != HOST_PROFILE:
        return "specialist placement is not the host profile"
    return ""


def _manifest_reason(manifest: ArtifactManifest, hotkey: str, verify: bool = True) -> str:
    if manifest.hotkey != hotkey:
        return "manifest declares a different hotkey than the one that committed it"

    if verify:
        ok, reason = _signature_ok(manifest)
        if not ok:
            return reason

    fits, reason = manifest.fits_class()
    if not fits:
        return reason

    return _system_reason(manifest, manifest.hardware_class)
