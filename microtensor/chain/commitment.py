from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from microtensor.core.constants import MAX_COMMITMENT_BYTES
from microtensor.core.hashing import DIGEST_PREFIX

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
    from microtensor.core.hashing import digest_bytes

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
ARTIFACT_REPO_PREFIX: Final[str] = "microtensor_"
SLM_REPO_PREFIX: Final[str] = "mt_slm_"

_ARENA = re.compile(r"^[a-z0-9][a-z0-9_]{0,15}$")
_HF_USER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class SubmissionCommitment:
    arena_id: str
    hf_user: str
    artifact_sha: str
    fee_block: int
    fee_index: int

    def __post_init__(self) -> None:
        if not _ARENA.match(self.arena_id):
            raise CommitmentError(f"arena id {self.arena_id!r} is not a valid identifier")
        if not _HF_USER.match(self.hf_user):
            raise CommitmentError(f"hugging face user {self.hf_user!r} is not a valid name")
        if not _GIT_SHA.match(self.artifact_sha):
            raise CommitmentError("the artifact revision must be a 40 hex git commit sha")
        if self.fee_block < 0 or self.fee_index < 0:
            raise CommitmentError("the fee reference must name a block and an extrinsic index")

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
        payload = FIELD_SEPARATOR.join(
            (
                SUBMISSION_TAG,
                self.arena_id,
                self.hf_user,
                self.artifact_sha,
                str(self.fee_block),
                str(self.fee_index),
            )
        )
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
        if len(parts) != 6 or parts[0] != SUBMISSION_TAG:
            return None
        try:
            return cls(
                arena_id=parts[1],
                hf_user=parts[2],
                artifact_sha=parts[3].lower(),
                fee_block=int(parts[4]),
                fee_index=int(parts[5]),
            )
        except (CommitmentError, ValueError):
            return None


def decode_submissions(raw_by_hotkey: dict[str, str]) -> dict[str, SubmissionCommitment]:
    decoded: dict[str, SubmissionCommitment] = {}
    for hotkey, raw in raw_by_hotkey.items():
        commitment = SubmissionCommitment.decode(raw)
        if commitment is not None:
            decoded[hotkey] = commitment
    return decoded
