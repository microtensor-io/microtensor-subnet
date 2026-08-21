from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

log = logging.getLogger("microtensor.update.verify")


class VerificationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Verification:
    digest_ok: bool
    signature_ok: bool
    signed: bool
    reason: str = ""

    @property
    def trusted(self) -> bool:
        return self.digest_ok and self.signature_ok


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def parse_sums(raw: str) -> dict[str, str]:
    sums: dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise VerificationError(f"malformed SHA256SUMS line: {stripped!r}")
        digest, name = parts[0].lower(), parts[1].lstrip("*")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise VerificationError(f"malformed digest for {name!r}")
        sums[name] = digest
    if not sums:
        raise VerificationError("SHA256SUMS listed nothing")
    return sums


def check_digest(artifact: Path, sums: dict[str, str]) -> tuple[bool, str]:
    expected = sums.get(artifact.name)
    if expected is None:
        return False, f"{artifact.name} is not listed in SHA256SUMS"
    actual = sha256_file(artifact)
    if actual != expected:
        return False, f"{artifact.name} hashes to {actual[:16]}…, expected {expected[:16]}…"
    return True, ""


# The substrate crypto type for ed25519, as an integer rather than through a
# KeypairType enum. bittensor_wallet exports no such name, so importing it
# raised ImportError, the fallback was tried, and on a host without
# substrateinterface release verification failed for a reason that had
# nothing to do with the signature.
ED25519: Final[int] = 0


def _keypair_class() -> Any:
    """Whichever keypair implementation is installed.

    Aliased on import rather than bound to one name twice: two imports of the
    same name in one scope is a redefinition, and the type checker is right
    to say so.
    """
    try:
        from bittensor_wallet import Keypair as WalletKeypair

        return WalletKeypair
    except ImportError:
        pass

    try:
        from substrateinterface import Keypair as SubstrateKeypair

        return SubstrateKeypair
    except ImportError as exc:
        raise VerificationError(
            "no keypair implementation available to verify the release signature"
        ) from exc


def _keypair(public_key_hex: str) -> Any:
    keypair = _keypair_class()
    return keypair(
        public_key=bytes.fromhex(public_key_hex.removeprefix("0x")),
        crypto_type=ED25519,
        ss58_format=42,
    )


def check_signature(payload: bytes, signature: bytes, public_key_hex: str) -> tuple[bool, str]:
    if not public_key_hex:
        return False, "no release signing key is pinned in this build"
    if not signature:
        return False, "release carries no signature"
    try:
        keypair = _keypair(public_key_hex)
        if keypair.verify(payload, signature):
            return True, ""
    except (VerificationError, ValueError, TypeError) as exc:
        return False, f"signature could not be checked: {exc}"
    except Exception as exc:
        return False, f"signature check failed: {exc}"
    return False, "signature does not verify against the pinned release key"


def verify_artifact(
    artifact: Path,
    sums_text: str,
    signature: bytes,
    public_key_hex: str,
    *,
    require_signature: bool = True,
) -> Verification:
    signed = bool(public_key_hex) and bool(signature)

    signature_ok, signature_reason = check_signature(
        sums_text.encode("utf-8"), signature, public_key_hex
    )
    if not signature_ok and not require_signature:
        log.warning("proceeding with an unverified release: %s", signature_reason)
        signature_ok = True

    if not signature_ok:
        return Verification(False, False, signed, signature_reason)

    try:
        sums = parse_sums(sums_text)
    except VerificationError as exc:
        return Verification(False, signature_ok, signed, str(exc))

    digest_ok, digest_reason = check_digest(artifact, sums)
    return Verification(digest_ok, signature_ok, signed, digest_reason)
