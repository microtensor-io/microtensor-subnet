from __future__ import annotations

import base64
import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("microtensor.coordinator")

TOKEN_HEADER = "x-mt-token"  # noqa: S105
KEY_FILE = "server_token_key"

MALFORMED = "the worker token is malformed"
BAD_SIGNATURE = "the worker token was not signed by the control plane"
EXPIRED = "the worker token has expired"
WRONG_HOTKEY = "the worker token was issued to a different hotkey"
NO_KEY = "no control plane key is known, so worker tokens cannot be verified"


class TokenInvalid(RuntimeError):
    pass


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@dataclass(frozen=True, slots=True)
class WorkerToken:
    jti: str
    hotkey: str
    issued_at: int
    expires_at: int
    scope: tuple[str, ...] = ()

    def expired(self, now: int) -> bool:
        return now >= self.expires_at


def verify(token: str, public_key: str, *, hotkey: str = "", now: int | None = None) -> WorkerToken:
    if not public_key:
        raise TokenInvalid(NO_KEY)
    if not token:
        raise TokenInvalid(MALFORMED)

    try:
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey
    except ImportError as exc:
        raise TokenInvalid("pynacl is required to verify worker tokens") from exc

    try:
        encoded, signature = token.split(".", 1)
        payload = _unb64(encoded)
    except ValueError as exc:
        raise TokenInvalid(MALFORMED) from exc

    try:
        VerifyKey(bytes.fromhex(public_key)).verify(payload, _unb64(signature))
    except (ValueError, BadSignatureError) as exc:
        raise TokenInvalid(BAD_SIGNATURE) from exc

    try:
        body = json.loads(payload)
        parsed = WorkerToken(
            jti=str(body["jti"]),
            hotkey=str(body["hotkey"]),
            issued_at=int(body["issued_at"]),
            expires_at=int(body["expires_at"]),
            scope=tuple(body.get("scope", ())),
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise TokenInvalid(MALFORMED) from exc

    moment = now if now is not None else int(dt.datetime.now(dt.timezone.utc).timestamp())
    if parsed.expired(moment):
        raise TokenInvalid(EXPIRED)

    if hotkey and parsed.hotkey != hotkey:
        raise TokenInvalid(f"{WRONG_HOTKEY}: {parsed.hotkey}")

    return parsed


@dataclass(slots=True)
class KeyRing:
    """The control plane's token key, held locally once it has been read.

    Verification is offline on purpose. A worker presenting a token mid-round
    must not depend on the server answering, so the key is fetched once, written
    to disk, and used from there. A server outage costs the ability to learn a
    NEW key, never the ability to check a token already issued under the old one.
    """

    home: Path
    public_key: str = ""

    @property
    def path(self) -> Path:
        return self.home / KEY_FILE

    @property
    def known(self) -> bool:
        return bool(self.public_key)

    def load(self) -> str:
        if self.public_key:
            return self.public_key
        try:
            self.public_key = self.path.read_text(encoding="utf-8").strip()
        except OSError:
            self.public_key = ""
        return self.public_key

    def remember(self, public_key: str) -> None:
        if not public_key or public_key == self.public_key:
            return
        previous = self.public_key
        self.public_key = public_key
        try:
            self.home.mkdir(parents=True, exist_ok=True)
            self.path.write_text(public_key, encoding="utf-8")
        except OSError as exc:
            log.warning("could not persist the control plane key: %s", exc)
        if previous:
            log.warning(
                "the control plane token key changed; tokens issued under the old "
                "key stop verifying"
            )
        else:
            log.info("control plane token key recorded")

    def refresh(self, fetch: Any) -> str:
        try:
            found = str(fetch() or "")
        except Exception as exc:
            log.warning("could not refresh the control plane key (%s); using the stored one", exc)
            return self.load()
        if found:
            self.remember(found)
        return self.public_key
