from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from microtensor_subnet.core.hashing import canonical_json, digest_bytes

ITEMS_FILE: Final[str] = "items.jsonl"
DATABASES_DIR: Final[str] = "db"
VARIANTS_DIR: Final[str] = "variants"
DRAFTS_FILE: Final[str] = "adversarial_drafts.jsonl"
KEY_BYTES: Final[int] = 32
NONCE_BYTES: Final[int] = 12
SEAL_INFO: Final[bytes] = b"mt-sealed-key-v1"
KEY_MESSAGE_TAG: Final[str] = "mt-x25519-v1"
RELEASE_VERSION: Final[str] = "r1"


class ItemError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Item:
    item_id: str
    family: str
    task: dict[str, Any]
    label: dict[str, Any] = field(default_factory=dict)
    twin_of: str = ""
    tests: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "family": self.family,
            "task": dict(self.task),
            "label": dict(self.label),
            "twin_of": self.twin_of,
            "tests": list(self.tests),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Item:
        item_id = str(raw.get("item_id", "")).strip()
        if not item_id:
            raise ItemError("an item needs an id")
        task = raw.get("task")
        if not isinstance(task, Mapping):
            raise ItemError(f"item {item_id} carries no task")
        label = raw.get("label") or {}
        if not isinstance(label, Mapping):
            raise ItemError(f"item {item_id} label must be a mapping")
        tests = raw.get("tests") or []
        if not isinstance(tests, list):
            raise ItemError(f"item {item_id} tests must be a list")
        return cls(
            item_id=item_id,
            family=str(raw.get("family") or item_id),
            task=dict(task),
            label=dict(label),
            twin_of=str(raw.get("twin_of") or ""),
            tests=tuple(str(t) for t in tests),
        )

    @property
    def is_twin(self) -> bool:
        return bool(self.twin_of)


def pack_bundle(files: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name in sorted(files):
            if name.startswith("/") or ".." in name.split("/"):
                raise ItemError(f"bundle member {name!r} is not a relative path")
            body = files[name]
            info = tarfile.TarInfo(name=name)
            info.size = len(body)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


def unpack_bundle(blob: bytes) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r") as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.name.startswith("/") or ".." in member.name.split("/"):
                raise ItemError(f"bundle member {member.name!r} is not a plain file")
            handle = tar.extractfile(member)
            if handle is None:
                raise ItemError(f"bundle member {member.name!r} cannot be read")
            found[member.name] = handle.read()
    return found


def items_from(files: Mapping[str, bytes]) -> list[Item]:
    raw = files.get(ITEMS_FILE)
    if raw is None:
        raise ItemError(f"the bundle has no {ITEMS_FILE}")
    items = [
        Item.from_dict(json.loads(line))
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    seen: set[str] = set()
    for item in items:
        if item.item_id in seen:
            raise ItemError(f"item {item.item_id} appears twice")
        seen.add(item.item_id)
    return items


def items_to(items: list[Item]) -> bytes:
    return "".join(json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in items).encode(
        "utf-8"
    )


def bundle_aad(epoch_index: int, arena_id: str) -> bytes:
    return f"{RELEASE_VERSION}|{epoch_index}|{arena_id}".encode()


def new_bundle_key() -> bytes:
    return os.urandom(KEY_BYTES)


def encrypt_bundle(plain: bytes, key: bytes, epoch_index: int, arena_id: str) -> bytes:
    if len(key) != KEY_BYTES:
        raise ItemError("the bundle key must be 32 bytes")
    nonce = os.urandom(NONCE_BYTES)
    return nonce + AESGCM(key).encrypt(nonce, plain, bundle_aad(epoch_index, arena_id))


def decrypt_bundle(blob: bytes, key: bytes, epoch_index: int, arena_id: str) -> bytes:
    if len(blob) <= NONCE_BYTES:
        raise ItemError("the encrypted bundle is too short to carry a nonce")
    try:
        return AESGCM(key).decrypt(
            blob[:NONCE_BYTES], blob[NONCE_BYTES:], bundle_aad(epoch_index, arena_id)
        )
    except (InvalidTag, ValueError) as exc:
        raise ItemError("the key does not open this bundle for this epoch") from exc


def new_recipient_key() -> tuple[bytes, bytes]:
    private = X25519PrivateKey.generate()
    return (
        private.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        ),
        private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ),
    )


def _derive(shared: bytes, ephemeral_pub: bytes, recipient_pub: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_BYTES,
        salt=ephemeral_pub + recipient_pub,
        info=SEAL_INFO,
    ).derive(shared)


def seal_to(recipient_pub: bytes, plain: bytes) -> bytes:
    if len(recipient_pub) != KEY_BYTES:
        raise ItemError("the recipient key must be 32 raw bytes")
    ephemeral = X25519PrivateKey.generate()
    ephemeral_pub = ephemeral.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    shared = ephemeral.exchange(X25519PublicKey.from_public_bytes(recipient_pub))
    key = _derive(shared, ephemeral_pub, recipient_pub)
    nonce = os.urandom(NONCE_BYTES)
    return ephemeral_pub + nonce + AESGCM(key).encrypt(nonce, plain, recipient_pub)


def open_sealed(recipient_private: bytes, blob: bytes) -> bytes:
    if len(blob) <= KEY_BYTES + NONCE_BYTES:
        raise ItemError("the sealed blob is too short")
    private = X25519PrivateKey.from_private_bytes(recipient_private)
    recipient_pub = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    ephemeral_pub = blob[:KEY_BYTES]
    nonce = blob[KEY_BYTES : KEY_BYTES + NONCE_BYTES]
    shared = private.exchange(X25519PublicKey.from_public_bytes(ephemeral_pub))
    key = _derive(shared, ephemeral_pub, recipient_pub)
    try:
        return AESGCM(key).decrypt(nonce, blob[KEY_BYTES + NONCE_BYTES :], recipient_pub)
    except (InvalidTag, ValueError) as exc:
        raise ItemError("this sealed key was not made for this recipient") from exc


def key_registration_message(
    netuid: int, hotkey: str, public_key: bytes, expires_block: int
) -> bytes:
    return f"{KEY_MESSAGE_TAG}|{netuid}|{hotkey}|{public_key.hex()}|{expires_block}".encode()


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    arena_id: str
    epoch_index: int
    bundle_sha256: str
    bundle_bytes: int
    open_block: int
    sealed_keys: dict[str, str] = field(default_factory=dict)
    signature: str = field(default="", compare=False)

    def body(self) -> dict[str, Any]:
        return {
            "version": RELEASE_VERSION,
            "arena_id": self.arena_id,
            "epoch_index": self.epoch_index,
            "bundle_sha256": self.bundle_sha256,
            "bundle_bytes": self.bundle_bytes,
            "open_block": self.open_block,
            "sealed_keys": dict(sorted(self.sealed_keys.items())),
        }

    def signed_message(self) -> bytes:
        return canonical_json(self.body())

    def digest(self) -> str:
        return hashlib.sha256(self.signed_message()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "signature": self.signature}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ReleaseManifest:
        if raw.get("version") != RELEASE_VERSION:
            raise ItemError("unknown release manifest version")
        keys = raw.get("sealed_keys") or {}
        if not isinstance(keys, Mapping):
            raise ItemError("sealed keys must map hotkeys to hex")
        return cls(
            arena_id=str(raw["arena_id"]),
            epoch_index=int(raw["epoch_index"]),
            bundle_sha256=str(raw["bundle_sha256"]).lower(),
            bundle_bytes=int(raw["bundle_bytes"]),
            open_block=int(raw["open_block"]),
            sealed_keys={str(k): str(v) for k, v in keys.items()},
            signature=str(raw.get("signature", "")),
        )

    def matches(self, encrypted_bundle: bytes) -> bool:
        return (
            len(encrypted_bundle) == self.bundle_bytes
            and digest_bytes(encrypted_bundle)[len("sha256:") :] == self.bundle_sha256
        )

    def key_for(self, hotkey: str, recipient_private: bytes) -> bytes:
        sealed = self.sealed_keys.get(hotkey)
        if not sealed:
            raise ItemError(f"no sealed key for {hotkey}")
        key = open_sealed(recipient_private, bytes.fromhex(sealed))
        if len(key) != KEY_BYTES:
            raise ItemError("the sealed key has the wrong length")
        return key


def release(
    arena_id: str,
    epoch_index: int,
    open_block: int,
    files: Mapping[str, bytes],
    recipients: Mapping[str, bytes],
) -> tuple[bytes, ReleaseManifest, bytes]:
    key = new_bundle_key()
    encrypted = encrypt_bundle(pack_bundle(files), key, epoch_index, arena_id)
    manifest = ReleaseManifest(
        arena_id=arena_id,
        epoch_index=epoch_index,
        bundle_sha256=digest_bytes(encrypted)[len("sha256:") :],
        bundle_bytes=len(encrypted),
        open_block=open_block,
        sealed_keys={hotkey: seal_to(pub, key).hex() for hotkey, pub in recipients.items()},
    )
    return encrypted, manifest, key


def open_release(
    manifest: ReleaseManifest, encrypted: bytes, hotkey: str, recipient_private: bytes
) -> dict[str, bytes]:
    if not manifest.matches(encrypted):
        raise ItemError("the bundle does not match the release manifest")
    key = manifest.key_for(hotkey, recipient_private)
    return unpack_bundle(decrypt_bundle(encrypted, key, manifest.epoch_index, manifest.arena_id))


def item_ids_digest(item_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(item_ids))).encode("utf-8")).hexdigest()
