from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Final

from microtensor_subnet.core.constants import MAX_COMMITMENT_BYTES
from microtensor_subnet.core.hashing import DIGEST_PREFIX
from microtensor_subnet.core.profiles import PROFILES, is_profile

COMMITMENT_TAG: Final[str] = "mt1"
REVEAL_TAG: Final[str] = "mtr"
SEALED_MARK: Final[str] = "k"
FIELD_SEPARATOR: Final[str] = "|"
SHORT_DIGEST_CHARS: Final[int] = 32

SOURCE_SCHEMES: Final[frozenset[str]] = frozenset({"hf", "s3", "r2", "https", "ipfs"})

_HEX = re.compile(r"^[0-9a-f]+$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")


class CommitmentError(ValueError):
    pass


def short_digest(digest: str, chars: int = SHORT_DIGEST_CHARS) -> str:
    body = digest[len(DIGEST_PREFIX) :] if digest.startswith(DIGEST_PREFIX) else digest
    body = body.strip().lower()
    if not _HEX.match(body) or len(body) < chars:
        raise CommitmentError(f"malformed digest {digest!r}")
    return body[:chars]


def digest_matches(short: str, digest: str) -> bool:
    try:
        return short_digest(digest, len(short)) == short.strip().lower()
    except CommitmentError:
        return False


@dataclass(frozen=True, slots=True)
class Commitment:
    round_index: int
    track: str
    hardware_class: str
    manifest_digest: str
    source: str
    sealed: bool = False

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise CommitmentError("round index must not be negative")
        for name in ("track", "hardware_class"):
            if not _NAME.match(getattr(self, name)):
                raise CommitmentError(f"{name} {getattr(self, name)!r} is not a valid identifier")
        if not _HEX.match(self.manifest_digest) or len(self.manifest_digest) < 16:
            raise CommitmentError(f"digest {self.manifest_digest!r} must be at least 16 hex chars")
        scheme, _, locator = self.source.partition(":")
        if scheme not in SOURCE_SCHEMES:
            raise CommitmentError(
                f"unknown source scheme {scheme!r}; known: {sorted(SOURCE_SCHEMES)}"
            )
        if not locator:
            raise CommitmentError("source must carry a locator after the scheme")
        if FIELD_SEPARATOR in self.source:
            raise CommitmentError("source must not contain the field separator")

    @property
    def competition(self) -> tuple[str, str]:
        return self.track, self.hardware_class

    def encode(self) -> str:
        payload = FIELD_SEPARATOR.join(
            (
                COMMITMENT_TAG,
                str(self.round_index),
                self.track,
                self.hardware_class,
                self.manifest_digest,
                self.source,
                *((SEALED_MARK,) if self.sealed else ()),
            )
        )
        size = len(payload.encode("utf-8"))
        if size > MAX_COMMITMENT_BYTES:
            raise CommitmentError(
                f"commitment is {size} bytes, over the {MAX_COMMITMENT_BYTES} byte chain limit"
            )
        return payload

    @classmethod
    def decode(cls, raw: str) -> Commitment | None:
        if not raw:
            return None
        parts = raw.strip().split(FIELD_SEPARATOR)
        if len(parts) not in (6, 7) or parts[0] != COMMITMENT_TAG:
            return None
        if len(parts) == 7 and parts[6] != SEALED_MARK:
            return None
        try:
            return cls(
                round_index=int(parts[1]),
                track=parts[2],
                hardware_class=parts[3],
                manifest_digest=parts[4].lower(),
                source=parts[5],
                sealed=len(parts) == 7,
            )
        except (CommitmentError, ValueError):
            return None

    def covers(self, digest: str) -> bool:
        return digest_matches(self.manifest_digest, digest)


@dataclass(frozen=True, slots=True)
class Reveal:
    """The key for a sealed submission, posted at the close block.

    It replaces the submission in the hotkey's single commitment slot, so it
    carries only what the slot no longer holds: which artifact, and the key.
    Everything else was captured by readers during the open window.
    """

    round_index: int
    manifest_digest: str
    key: str
    commitment_hash: str = ""

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise CommitmentError("round index must not be negative")
        if not _HEX.match(self.manifest_digest) or len(self.manifest_digest) < 16:
            raise CommitmentError("digest must be at least 16 hex chars")
        if not _HEX.match(self.key) or len(self.key) != 64:
            raise CommitmentError("key must be 64 hex chars")
        if self.commitment_hash and (
            not _HEX.match(self.commitment_hash) or len(self.commitment_hash) != 16
        ):
            raise CommitmentError("commitment hash must be 16 hex chars")

    def encode(self) -> str:
        fields = [REVEAL_TAG, str(self.round_index), self.manifest_digest, self.key]
        if self.commitment_hash:
            fields.append(self.commitment_hash)
        payload = FIELD_SEPARATOR.join(fields)
        if len(payload.encode("utf-8")) > MAX_COMMITMENT_BYTES:
            raise CommitmentError("reveal is over the chain limit")
        return payload

    @classmethod
    def decode(cls, raw: str) -> Reveal | None:
        if not raw:
            return None
        parts = raw.strip().split(FIELD_SEPARATOR)
        if len(parts) not in (4, 5) or parts[0] != REVEAL_TAG:
            return None
        try:
            return cls(
                round_index=int(parts[1]),
                manifest_digest=parts[2].lower(),
                key=parts[3].lower(),
                commitment_hash=parts[4].lower() if len(parts) == 5 else "",
            )
        except (CommitmentError, ValueError):
            return None


def commitment_hash(commitment: Commitment) -> str:
    from microtensor_subnet.core.hashing import digest_bytes

    return short_digest(digest_bytes(commitment.encode().encode("utf-8")), 16)


def build_commitment(
    round_index: int,
    track: str,
    hardware_class: str,
    manifest_digest: str,
    source: str,
    sealed: bool = False,
) -> Commitment:
    return Commitment(
        round_index=round_index,
        track=track,
        hardware_class=hardware_class,
        manifest_digest=short_digest(manifest_digest),
        source=source,
        sealed=sealed,
    )


def decode_all(raw_by_hotkey: dict[str, str]) -> dict[str, Commitment]:
    decoded: dict[str, Commitment] = {}
    for hotkey, raw in raw_by_hotkey.items():
        commitment = Commitment.decode(raw)
        if commitment is not None:
            decoded[hotkey] = commitment
    return decoded


SUBMISSION_TAG: Final[str] = "mt2"
SEALED_TAG: Final[str] = "mt3"
PROFILE_SUBMISSION_TAG: Final[str] = "mt4"
PROFILE_SEALED_TAG: Final[str] = "mt5"
REVEAL_MESSAGE_TAG: Final[str] = "mt-reveal"
NO_PROFILE_REASON: Final[str] = "commitment carries no hardware profile"
ARTIFACT_REPO_PREFIX: Final[str] = "microtensor_"
SLM_REPO_PREFIX: Final[str] = "mt_slm_"

_ARENA = re.compile(r"^[a-z0-9][a-z0-9_]{0,15}$")
_HF_USER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SALT = re.compile(r"^[0-9a-f]{16,64}$")


def _check_profile(profile: str) -> None:
    if profile and not is_profile(profile):
        raise CommitmentError(f"unknown hardware profile {profile!r}; known: {', '.join(PROFILES)}")


@dataclass(frozen=True, slots=True)
class SubmissionCommitment:
    arena_id: str
    hf_user: str
    artifact_sha: str
    fee_block: int
    fee_index: int
    profile: str = ""

    def __post_init__(self) -> None:
        if not _ARENA.match(self.arena_id):
            raise CommitmentError(f"arena id {self.arena_id!r} is not a valid identifier")
        if not _HF_USER.match(self.hf_user):
            raise CommitmentError(f"hugging face user {self.hf_user!r} is not a valid name")
        if not _GIT_SHA.match(self.artifact_sha):
            raise CommitmentError("the artifact revision must be a 40 hex git commit sha")
        if self.fee_block < 0 or self.fee_index < 0:
            raise CommitmentError("the fee reference must name a block and an extrinsic index")
        _check_profile(self.profile)

    @property
    def has_profile(self) -> bool:
        return bool(self.profile)

    @property
    def artifact_repo(self) -> str:
        return f"{self.hf_user}/{ARTIFACT_REPO_PREFIX}{self.arena_id}"

    @property
    def slm_repo(self) -> str:
        return f"{self.hf_user}/{SLM_REPO_PREFIX}{self.arena_id}"

    @property
    def fee_reference(self) -> str:
        return f"{self.fee_block}:{self.fee_index}"

    def encode(self) -> str:
        fields = [self.arena_id, self.hf_user, self.artifact_sha]
        if self.profile:
            fields = [PROFILE_SUBMISSION_TAG, *fields, self.profile]
        else:
            fields = [SUBMISSION_TAG, *fields]
        payload = FIELD_SEPARATOR.join((*fields, str(self.fee_block), str(self.fee_index)))
        size = len(payload.encode("utf-8"))
        if size > MAX_COMMITMENT_BYTES:
            raise CommitmentError(
                f"commitment is {size} bytes, over the {MAX_COMMITMENT_BYTES} byte chain limit"
            )
        return payload

    @classmethod
    def decode(cls, raw: str) -> SubmissionCommitment | None:
        if not raw:
            return None
        parts = raw.strip().split(FIELD_SEPARATOR)
        if parts[0] == SUBMISSION_TAG and len(parts) == 6:
            profile = ""
            fee_block, fee_index = parts[4], parts[5]
        elif parts[0] == PROFILE_SUBMISSION_TAG and len(parts) == 7:
            profile = parts[4]
            fee_block, fee_index = parts[5], parts[6]
        else:
            return None
        try:
            return cls(
                arena_id=parts[1],
                hf_user=parts[2],
                artifact_sha=parts[3].lower(),
                fee_block=int(fee_block),
                fee_index=int(fee_index),
                profile=profile,
            )
        except (CommitmentError, ValueError):
            return None


def seal_digest(arena_id: str, hf_user: str, artifact_sha: str, salt: str) -> str:
    plain = f"{arena_id}|{hf_user}|{artifact_sha.lower()}|{salt}"
    return hashlib.sha256(plain.encode()).hexdigest()


def reveal_message(
    arena_id: str, epoch_index: int, hotkey: str, hf_user: str, artifact_sha: str, salt: str
) -> bytes:
    return FIELD_SEPARATOR.join(
        (
            REVEAL_MESSAGE_TAG,
            arena_id,
            str(epoch_index),
            hotkey,
            hf_user,
            artifact_sha.lower(),
            salt,
        )
    ).encode()


@dataclass(frozen=True, slots=True)
class SealedCommitment:
    arena_id: str
    digest: str
    fee_block: int
    fee_index: int
    profile: str = ""

    def __post_init__(self) -> None:
        if not _ARENA.match(self.arena_id):
            raise CommitmentError(f"arena id {self.arena_id!r} is not a valid identifier")
        if not _DIGEST.match(self.digest):
            raise CommitmentError("the sealed digest must be 64 hex characters")
        if self.fee_block < 0 or self.fee_index < 0:
            raise CommitmentError("the fee reference must name a block and an extrinsic index")
        _check_profile(self.profile)

    @property
    def has_profile(self) -> bool:
        return bool(self.profile)

    @classmethod
    def seal(
        cls,
        arena_id: str,
        hf_user: str,
        artifact_sha: str,
        salt: str,
        fee_block: int,
        fee_index: int,
        profile: str = "",
    ) -> SealedCommitment:
        if not _HF_USER.match(hf_user):
            raise CommitmentError(f"hugging face user {hf_user!r} is not a valid name")
        if not _GIT_SHA.match(artifact_sha.lower()):
            raise CommitmentError("the artifact revision must be a 40 hex git commit sha")
        if not _SALT.match(salt):
            raise CommitmentError("the salt must be 16 to 64 hex characters")
        digest = seal_digest(arena_id, hf_user, artifact_sha, salt)
        return cls(arena_id, digest, fee_block, fee_index, profile)

    def matches(self, hf_user: str, artifact_sha: str, salt: str) -> bool:
        return seal_digest(self.arena_id, hf_user, artifact_sha, salt) == self.digest

    def unseal(self, hf_user: str, artifact_sha: str) -> SubmissionCommitment:
        return SubmissionCommitment(
            self.arena_id, hf_user, artifact_sha, self.fee_block, self.fee_index, self.profile
        )

    def encode(self) -> str:
        fields = [self.arena_id, self.digest]
        if self.profile:
            fields = [PROFILE_SEALED_TAG, *fields, self.profile]
        else:
            fields = [SEALED_TAG, *fields]
        return FIELD_SEPARATOR.join((*fields, str(self.fee_block), str(self.fee_index)))

    @classmethod
    def decode(cls, raw: str) -> SealedCommitment | None:
        parts = (raw or "").strip().split(FIELD_SEPARATOR)
        if parts[0] == SEALED_TAG and len(parts) == 5:
            profile = ""
            fee_block, fee_index = parts[3], parts[4]
        elif parts[0] == PROFILE_SEALED_TAG and len(parts) == 6:
            profile = parts[3]
            fee_block, fee_index = parts[4], parts[5]
        else:
            return None
        try:
            return cls(parts[1], parts[2].lower(), int(fee_block), int(fee_index), profile)
        except (CommitmentError, ValueError):
            return None


def decode_sealed(raw_by_hotkey: dict[str, str]) -> dict[str, SealedCommitment]:
    decoded: dict[str, SealedCommitment] = {}
    for hotkey, raw in raw_by_hotkey.items():
        commitment = SealedCommitment.decode(raw)
        if commitment is not None:
            decoded[hotkey] = commitment
    return decoded


def decode_submissions(raw_by_hotkey: dict[str, str]) -> dict[str, SubmissionCommitment]:
    decoded: dict[str, SubmissionCommitment] = {}
    for hotkey, raw in raw_by_hotkey.items():
        commitment = SubmissionCommitment.decode(raw)
        if commitment is not None:
            decoded[hotkey] = commitment
    return decoded
