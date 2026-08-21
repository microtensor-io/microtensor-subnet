from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from microtensor.core.constants import MAX_COMMITMENT_BYTES

ANCHOR_TAG: Final[str] = "mtc1"
FIELD_SEPARATOR: Final[str] = "|"
DIGEST_PREFIX: Final[str] = "sha256:"

NO_HOTKEY: Final[str] = (
    "no coordinator hotkey is pinned in this build, so no identity's commitment "
    "counts as the anchor and nothing can be verified against the chain"
)


class AnchorError(ValueError):
    pass


class CommitmentReader(Protocol):
    def commitments(self, hotkeys: Sequence[str]) -> dict[str, str]: ...


@dataclass(frozen=True, slots=True)
class ConfigAnchor:
    """A coordinator's on-chain claim about the config for one round.

    Separate from Commitment, which is a miner announcing a submission. This
    is the operator announcing the rules, and the two are read from different
    hotkeys for different reasons.
    """

    round_index: int
    config_hash: str

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise AnchorError("round index must not be negative")
        if not self.config_hash.startswith(DIGEST_PREFIX):
            raise AnchorError(f"config hash {self.config_hash!r} is not a sha256 digest")
        body = self.config_hash[len(DIGEST_PREFIX) :]
        if len(body) != 64 or any(c not in "0123456789abcdef" for c in body):
            raise AnchorError(f"config hash {self.config_hash!r} is malformed")

    def encode(self) -> str:
        payload = FIELD_SEPARATOR.join((ANCHOR_TAG, str(self.round_index), self.config_hash))
        size = len(payload.encode("utf-8"))
        if size > MAX_COMMITMENT_BYTES:
            raise AnchorError(
                f"anchor is {size} bytes, over the {MAX_COMMITMENT_BYTES} byte chain limit"
            )
        return payload

    @classmethod
    def decode(cls, raw: str) -> ConfigAnchor | None:
        if not raw:
            return None
        parts = raw.strip().split(FIELD_SEPARATOR)
        if len(parts) != 3 or parts[0] != ANCHOR_TAG:
            return None
        try:
            return cls(round_index=int(parts[1]), config_hash=parts[2].lower())
        except (AnchorError, ValueError):
            return None


def read_anchor(client: Any, coordinator_hotkey: str) -> ConfigAnchor | None:
    """The coordinator's anchor as the chain holds it, or nothing.

    Nothing means the coordinator has not committed, which callers must treat
    as a refusal rather than a pass. An unpinned hotkey is a different failure
    and raises: it means this build cannot check anchoring at all, and
    quietly returning "no anchor" would read as the coordinator's fault.
    """
    if not coordinator_hotkey:
        raise AnchorError(NO_HOTKEY)
    raw = client.commitments([coordinator_hotkey]).get(coordinator_hotkey, "")
    return ConfigAnchor.decode(str(raw))
