from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from microtensor.core.constants import MAX_COMMITMENT_BYTES
from microtensor.core.hashing import DIGEST_PREFIX

COMMITMENT_TAG: Final[str] = "mt1"
FIELD_SEPARATOR: Final[str] = "|"
SHORT_DIGEST_CHARS: Final[int] = 32

SOURCE_SCHEMES: Final[frozenset[str]] = frozenset({"hf", "s3", "r2", "https", "ipfs"})

_HEX = re.compile(r"^[0-9a-f]+$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")


class CommitmentError(ValueError):
    pass


def short_digest(digest: str, chars: int = SHORT_DIGEST_CHARS) -> str:
    body = digest[len(DIGEST_PREFIX):] if digest.startswith(DIGEST_PREFIX) else digest
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
        if len(parts) != 6 or parts[0] != COMMITMENT_TAG:
            return None
        try:
            return cls(
                round_index=int(parts[1]),
                track=parts[2],
                hardware_class=parts[3],
                manifest_digest=parts[4].lower(),
                source=parts[5],
            )
        except (CommitmentError, ValueError):
            return None

    def covers(self, digest: str) -> bool:
        return digest_matches(self.manifest_digest, digest)


def build_commitment(
    round_index: int,
    track: str,
    hardware_class: str,
    manifest_digest: str,
    source: str,
) -> Commitment:
    return Commitment(
        round_index=round_index,
        track=track,
        hardware_class=hardware_class,
        manifest_digest=short_digest(manifest_digest),
        source=source,
    )


def decode_all(raw_by_hotkey: dict[str, str]) -> dict[str, Commitment]:
    decoded: dict[str, Commitment] = {}
    for hotkey, raw in raw_by_hotkey.items():
        commitment = Commitment.decode(raw)
        if commitment is not None:
            decoded[hotkey] = commitment
    return decoded
