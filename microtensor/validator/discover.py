from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from microtensor.chain.commitment import Commitment, decode_all
from microtensor.chain.metagraph import MetagraphSnapshot
from microtensor.chain.rounds import Round, accepts_commitment
from microtensor.chain.wallet import verify_payload
from microtensor.core.constants import (
    ALSO_ACCEPT_ROUNDS,
    BLOCK_TIME_SECONDS,
    HOST_PROFILE,
    REQUIRE_SEALED_SUBMISSIONS,
    REVEAL_WINDOW_BLOCKS,
)
from microtensor.core.protocol import Role
from microtensor.core.system import SystemManifest
from microtensor.provenance.record import ProvenanceUnavailable, Verdict
from microtensor.provenance.record import best_verdict as provenance_check
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
    key: str | None = None

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
    rejected: tuple[tuple[str, str, str], ...]
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
    if not context.config.require_provenance:
        return "", None

    # Not folded into the line above. Provenance being switched off is a
    # decision; a validator that is required to check and holds no store is a
    # misconfiguration, and skipping there would admit what its peers reject
    # and produce a different participant set for the same round. The two
    # cases arrive at the same branch and mean opposite things.
    if context.runs is None:
        raise ProvenanceUnavailable(
            "provenance is required but this validator has no run store, so it "
            "would admit submissions its peers reject"
        )

    # Every run bearing this hotkey, not the most recent one: a miner must not
    # be rejectable by a stranger creating a run under their name.
    runs = context.runs.candidates(hotkey)
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


def _base_model_reason(system: SystemManifest, allowlist: frozenset[str]) -> str:
    """The allowlist applies to every component that carries weights.

    An empty allowlist rejects everything. It reads as "nothing has been
    permitted here yet", never as "anything goes": the arena lifecycle refuses
    to activate an arena without entries, and a gate that fails open when its
    configuration is missing is worse than no gate.

    The router carries no weights, so it declares no base model to check.
    """
    for component in system.components:
        if component.role is Role.ROUTER:
            continue
        if component.base_model not in allowlist:
            if system.degenerate:
                return "base model not on the allowlist"
            return f"{component.role.value} base model not on the allowlist"
    return ""


def discover(
    context: ValidatorContext,
    snapshot: MetagraphSnapshot,
    round_: Round,
    allowlists: Mapping[tuple[str, str], frozenset[str]] | None = None,
) -> Roster:
    raw = context.client.commitments(list(snapshot.hotkeys))
    commitments = decode_all(raw)
    keys = _reveals_of(raw, round_.index)
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
            rejected.append((hotkey, commitment.manifest_digest, reason))
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
                reason = _sealing_reason(commitment, manifest)
            if not reason:
                system = _system_of(manifest, commitment.hardware_class)
                reason = _base_model_reason(
                    system,
                    frozenset(
                        (allowlists or {}).get(
                            (commitment.track, commitment.hardware_class), frozenset()
                        )
                    ),
                )
                if not reason:
                    reason, verdict = _provenance_reason(
                        context, hotkey, system, commitment, commit_block
                    )
                if verdict is not None and verdict.admissible:
                    provenance[hotkey] = verdict

        if reason:
            rejected.append((hotkey, commitment.manifest_digest, reason))
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
                key=keys.get(hotkey),
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

    accepted, rejected = _await_reveals(context, round_, accepted, rejected)

    for hotkey, _digest, reason in rejected:
        log.info("round %d: %s rejected: %s", round_.index, hotkey, reason)

    log.info(
        "round %d: %d participants frozen, %d rejected",
        round_.index,
        len(accepted),
        len(rejected),
    )
    return Roster(round_.index, tuple(accepted), tuple(rejected), provenance)


def _reveals_of(raw: Mapping[str, str], round_index: int) -> dict[str, str]:
    """Keys already posted, from slots whose submission was replaced by a reveal
    or read again after one landed."""
    from microtensor.chain.commitment import Reveal

    found: dict[str, str] = {}
    for hotkey, payload in raw.items():
        reveal = Reveal.decode(payload)
        if reveal is not None and reveal.round_index == round_index:
            found[hotkey] = reveal.key
    return found


def _sealing_reason(commitment: Commitment, manifest: ArtifactManifest) -> str:
    if commitment.sealed and manifest.sealed is None:
        return (
            "sealed commitment but the manifest carries no sealed block; "
            "repackage with mt miner ship --sealed"
        )
    if not commitment.sealed and manifest.sealed is not None:
        return "manifest is sealed but the commitment is not; repackage with mt miner ship --sealed"
    if REQUIRE_SEALED_SUBMISSIONS and not commitment.sealed:
        return "this round requires a sealed submission; resubmit with mt miner ship --sealed"
    return ""


def _await_reveals(
    context: ValidatorContext,
    round_: Round,
    accepted: list[Participant],
    rejected: list[tuple[str, str, str]],
) -> tuple[list[Participant], list[tuple[str, str, str]]]:
    """Hold discovery open until sealed participants have keys or the window ends.

    Exclusion is self-enforcing: a miner who never reveals is dropped with a
    reason that says exactly that, and nothing about it needs adjudicating.
    """
    import time as _time

    pending = [p for p in accepted if p.commitment.sealed and not p.key]
    if not pending:
        return accepted, rejected

    deadline = round_.close_block + REVEAL_WINDOW_BLOCKS
    while pending and context.client.block() < deadline:
        log.info(
            "round %d: waiting on %d reveal(s), window closes at block %d",
            round_.index,
            len(pending),
            deadline,
        )
        _time.sleep(BLOCK_TIME_SECONDS * 2)
        raw = context.client.commitments([p.hotkey for p in pending])
        keys = _reveals_of(raw, round_.index)
        if not keys:
            continue
        refreshed: list[Participant] = []
        for participant in accepted:
            key = keys.get(participant.hotkey)
            if key and participant.commitment.sealed and not participant.key:
                participant = Participant(
                    hotkey=participant.hotkey,
                    uid=participant.uid,
                    commitment=participant.commitment,
                    manifest=participant.manifest,
                    key=key,
                )
            refreshed.append(participant)
        accepted = refreshed
        pending = [p for p in accepted if p.commitment.sealed and not p.key]

    for participant in pending:
        rejected.append(
            (
                participant.hotkey,
                participant.commitment.manifest_digest,
                "sealed submission was never revealed",
            )
        )
        context.state.record_submission(
            round_.index,
            participant.hotkey,
            participant.commitment.track,
            participant.commitment.hardware_class,
            participant.commitment.manifest_digest,
            participant.commitment.source,
            reason="sealed submission was never revealed",
        )
    accepted = [p for p in accepted if not (p.commitment.sealed and not p.key)]
    return accepted, rejected


def _reject_reason(
    commitment: Commitment, round_: Round, open_competitions: set[tuple[str, str]]
) -> str:
    if not accepts_commitment(round_.index, commitment.round_index, ALSO_ACCEPT_ROUNDS):
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
