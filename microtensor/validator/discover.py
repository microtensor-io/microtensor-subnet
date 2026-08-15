from __future__ import annotations

import logging
from dataclasses import dataclass

from microtensor.chain.commitment import Commitment, decode_all
from microtensor.chain.metagraph import MetagraphSnapshot
from microtensor.chain.rounds import Round
from microtensor.chain.wallet import verify_payload
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


@dataclass(frozen=True, slots=True)
class Roster:
    round_index: int
    participants: tuple[Participant, ...]
    rejected: tuple[tuple[str, str], ...]

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


def discover(context: ValidatorContext, snapshot: MetagraphSnapshot, round_: Round) -> Roster:
    raw = context.client.commitments(list(snapshot.hotkeys))
    commitments = decode_all(raw)
    log.info("round %d: %d commitments readable", round_.index, len(commitments))

    uid_by_hotkey = snapshot.uid_by_hotkey
    open_competitions = set(context.competitions)
    accepted: list[Participant] = []
    rejected: list[tuple[str, str]] = []

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

        try:
            manifest = fetch_manifest(commitment, workdir=context.config.work_dir)
        except ArtifactMismatch as exc:
            reason = f"manifest rejected: {exc}"
        except FetchError as exc:
            reason = f"manifest unfetchable: {exc}"
        else:
            reason = _manifest_reason(manifest, hotkey)

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
    return Roster(round_.index, tuple(accepted), tuple(rejected))


def _reject_reason(
    commitment: Commitment, round_: Round, open_competitions: set[tuple[str, str]]
) -> str:
    if commitment.round_index != round_.index:
        return f"commitment names round {commitment.round_index}, not {round_.index}"
    if commitment.competition not in open_competitions:
        return f"{commitment.track}/{commitment.hardware_class} is not an open competition here"
    return ""


def _manifest_reason(manifest: ArtifactManifest, hotkey: str) -> str:
    if manifest.hotkey != hotkey:
        return "manifest declares a different hotkey than the one that committed it"

    ok, reason = _signature_ok(manifest)
    if not ok:
        return reason

    fits, reason = manifest.fits_class()
    if not fits:
        return reason
    return ""
