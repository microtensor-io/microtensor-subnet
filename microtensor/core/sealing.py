"""Sealing an artifact so nothing readable exists while a round is open.

The miner publishes only a ciphertext; the key is revealed on chain at the
close block. Verification needs no trusted party: the plaintext manifest
digest is what was committed, so a key that decrypts to anything else fails
the same check every submission already passes.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

BLOB_NAME = "artifact.enc"
ALGORITHM = "aes-256-gcm"
KEY_BYTES = 32
NONCE_BYTES = 12


class SealError(ValueError):
    pass


def new_key() -> str:
    return os.urandom(KEY_BYTES).hex()


def seal_tree(root: Path, exclude: tuple[str, ...] = ("manifest.json",)) -> tuple[bytes, str]:
    """Pack a directory into one authenticated ciphertext blob.

    The tar is built with fixed metadata so the same tree always seals to the
    same plaintext, and the manifest stays outside: it is the public claim the
    ciphertext is judged against, so it cannot live inside it.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            relative = path.relative_to(root).as_posix()
            if relative in exclude or relative == BLOB_NAME:
                continue
            info = tarfile.TarInfo(name=relative)
            info.size = path.stat().st_size
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as handle:
                tar.addfile(info, handle)

    key = new_key()
    nonce = os.urandom(NONCE_BYTES)
    ciphertext = AESGCM(bytes.fromhex(key)).encrypt(nonce, buffer.getvalue(), None)
    return nonce + ciphertext, key


def open_sealed(blob: bytes, key: str, into: Path) -> None:
    """Decrypt and unpack a sealed blob, refusing anything malformed.

    Members are extracted only inside the target directory; a tar entry that
    escapes it is treated the same as a failed authentication tag, because
    both mean the bytes are not what was committed.
    """
    if len(blob) <= NONCE_BYTES:
        raise SealError("sealed blob is too short to carry a nonce")
    try:
        plain = AESGCM(bytes.fromhex(key)).decrypt(blob[:NONCE_BYTES], blob[NONCE_BYTES:], None)
    except (InvalidTag, ValueError) as exc:
        raise SealError("the revealed key does not open this blob") from exc

    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(plain), mode="r") as tar:
        for member in tar.getmembers():
            target = (into / member.name).resolve()
            if not str(target).startswith(str(into.resolve())):
                raise SealError(f"sealed member escapes the artifact root: {member.name}")
        tar.extractall(into)  # noqa: S202
