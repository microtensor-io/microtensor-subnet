from __future__ import annotations

import contextlib
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import asyncssh
import httpx

from microtensor.rigs.protocol.attestation import Attestation, report_data_matches
from microtensor.rigs.protocol.failures import Failure, FailureClass, Verdict
from microtensor.rigs.protocol.session import (
    NONCE_BYTES,
    PING_PATH,
    SESSION_CLOSE_PATH,
    SESSION_OPEN_PATH,
    UTILISATION_PATH,
    VERSION_PATH,
    SessionClose,
    SessionGrant,
    SessionOpen,
    grant_blob,
)
from microtensor.rigs.validator.identity import Identity, canonical_json
from microtensor.rigs.validator.session.runner import CommandResult
from microtensor.rigs.validator.session.ssh import SshError, SshRunner

AGENT_TIMEOUT = 30.0
KEY_COMMENT = "mt-validator"


class AgentError(RuntimeError):
    def __init__(self, kind: FailureClass, detail: str, status: int = 0) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail
        self.status = status

    def failure(self, seed: str = "") -> Failure:
        return Failure(self.kind, self.detail, seed=seed)


def connection_of(rig: dict[str, Any]) -> tuple[str, int, int]:
    connection = rig.get("connection") or {}
    address = str(connection.get("address", "") or "")
    try:
        api_port = int(connection.get("api_port", 0) or 0)
    except (TypeError, ValueError):
        api_port = 0
    try:
        ssh_port = int(connection.get("ssh_port", 0) or 0)
    except (TypeError, ValueError):
        ssh_port = 0
    return address, api_port, ssh_port


@dataclass
class Session:
    rig_id: str
    address: str
    grant: SessionGrant
    nonce: str
    public_key: str
    attestation: Attestation
    bound: bool
    report_data_ok: bool | None
    runner: SshRunner
    dir: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def python(self) -> str:
        return self.grant.python_path or "python3"

    async def run(
        self, command: str, timeout: float = 60.0, stdin: str | None = None
    ) -> CommandResult:
        return await self.runner.run(command, timeout, stdin)

    async def upload(self, content: bytes, remote_path: str, mode: int = 0o600) -> None:
        await self.runner.upload(content, remote_path, mode)

    async def run_json(
        self, command: str, timeout: float = 60.0, stdin: str | None = None
    ) -> tuple[CommandResult, dict[str, Any] | None]:
        return await self.runner.run_json(command, timeout, stdin)

    def attestation_summary(self) -> dict[str, Any]:
        return {
            "bound": self.bound,
            "report_data_ok": self.report_data_ok,
            "nonce_matches": self.attestation.nonce == self.nonce,
            "agent_version": self.attestation.agent_version,
            "hostname": self.attestation.hostname,
            "gpus": len(self.attestation.gpus),
            "driver": self.attestation.driver,
            "cuda": self.attestation.cuda,
            "machine_id": self.attestation.machine_id,
            "boot_id": self.attestation.boot_id,
            "tdx_quote": bool(self.attestation.tdx_quote or self.grant.tdx_quote),
            "session_id": self.grant.session_id,
            "expires_at": self.grant.expires_at,
        }

    def attestation_verdict(self, seed: str = "") -> Verdict:
        summary = self.attestation_summary()
        summary["summary"] = "bound to the session nonce" if self.bound else "not bound"
        failure = None
        if not self.bound:
            failure = Failure(
                FailureClass.SPEC_MISMATCH,
                "the attestation is not bound to the session nonce",
                seed=seed,
                evidence={
                    "attestation_nonce": self.attestation.nonce[:16],
                    "expected": self.nonce[:16],
                },
            )
        elif self.report_data_ok is False:
            failure = Failure(
                FailureClass.SPEC_MISMATCH,
                "the attestation report data does not match the host key and nonce",
                seed=seed,
            )
        return Verdict("attestation", failure=failure, evidence=summary)


class AgentClient:
    def __init__(
        self, identity: Identity, timeout: float = AGENT_TIMEOUT, ssh_timeout: float = 60.0
    ) -> None:
        self._identity = identity
        self._timeout = timeout
        self._ssh_timeout = ssh_timeout
        self._http: httpx.AsyncClient | None = None

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._timeout)
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    @staticmethod
    def base_url(rig: dict[str, Any]) -> str:
        address, api_port, _ = connection_of(rig)
        if not address or not api_port:
            raise AgentError(
                FailureClass.SSH_TRANSPORT, "the rig has no address or api port on record"
            )
        host = f"[{address}]" if ":" in address else address
        return f"http://{host}:{api_port}"

    async def call(
        self,
        rig: dict[str, Any],
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        base = self.base_url(rig)
        raw = b""
        headers: dict[str, str] = {"accept": "application/json"}
        if body is not None:
            raw = canonical_json(body)
            headers["content-type"] = "application/json"
        headers.update(self._identity.headers(method, path, raw))
        try:
            answer = await self._client().request(
                method,
                base + path,
                content=raw if body is not None else None,
                headers=headers,
                timeout=timeout or self._timeout,
            )
        except httpx.HTTPError as exc:
            raise AgentError(FailureClass.SSH_TRANSPORT, f"{method} {path}: {exc}") from exc
        if answer.status_code >= 400:
            detail = answer.text[:300]
            with contextlib.suppress(Exception):
                detail = str(answer.json().get("detail", detail))
            kind = (
                FailureClass.AGENT_CRASH
                if answer.status_code >= 500
                else FailureClass.SSH_TRANSPORT
            )
            raise AgentError(
                kind,
                f"{method} {path} refused with {answer.status_code}: {detail}",
                answer.status_code,
            )
        if not answer.content:
            return {}
        try:
            parsed = answer.json()
        except ValueError as exc:
            raise AgentError(
                FailureClass.AGENT_CRASH, f"{method} {path}: response is not JSON"
            ) from exc
        return dict(parsed) if isinstance(parsed, dict) else {"value": parsed}

    async def version(self, rig: dict[str, Any]) -> dict[str, Any]:
        return await self.call(rig, "GET", VERSION_PATH)

    async def ping(self, rig: dict[str, Any]) -> dict[str, Any]:
        return await self.call(rig, "POST", PING_PATH, {})

    async def utilisation(self, rig: dict[str, Any]) -> dict[str, Any]:
        return await self.call(rig, "POST", UTILISATION_PATH, {})

    async def record_job(self, rig: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        return await self.call(rig, "POST", "/jobs", job)

    @contextlib.asynccontextmanager
    async def open(self, rig: dict[str, Any]) -> AsyncIterator[Session]:
        rig_id = str(rig.get("id", ""))
        address, _, ssh_port = connection_of(rig)
        key = asyncssh.generate_private_key("ssh-ed25519", comment=KEY_COMMENT)
        public_key = key.export_public_key("openssh").decode().strip()
        nonce = secrets.token_hex(NONCE_BYTES)
        request = SessionOpen(public_key, nonce, self._identity.sign(grant_blob(public_key, nonce)))
        answer = await self.call(
            rig, "POST", SESSION_OPEN_PATH, request.payload(), timeout=self._ssh_timeout
        )
        grant = SessionGrant.from_payload(answer)
        try:
            if not grant.ssh_username:
                raise AgentError(FailureClass.SSH_TRANSPORT, "the grant names no ssh user")
            if not grant.ssh_host_key:
                raise AgentError(FailureClass.SSH_TRANSPORT, "the grant carries no host key")
            attestation = Attestation.from_payload(grant.gpu_attestation)
            bound = attestation.bound_to(nonce)
            report_ok: bool | None = None
            if attestation.report_data_hex:
                try:
                    report_ok = report_data_matches(
                        bytes.fromhex(attestation.report_data_hex), grant.ssh_host_key, nonce
                    )
                except ValueError:
                    report_ok = False
            try:
                runner = await SshRunner.connect(
                    address,
                    grant.ssh_port or ssh_port or 22,
                    grant.ssh_username,
                    key,
                    grant.ssh_host_key,
                    self._ssh_timeout,
                )
            except SshError as exc:
                raise AgentError(exc.kind, exc.detail) from exc
            try:
                yield Session(
                    rig_id=rig_id,
                    address=address,
                    grant=grant,
                    nonce=nonce,
                    public_key=public_key,
                    attestation=attestation,
                    bound=bound,
                    report_data_ok=report_ok,
                    runner=runner,
                )
            finally:
                await runner.close()
        finally:
            with contextlib.suppress(AgentError):
                await self.call(
                    rig,
                    "POST",
                    SESSION_CLOSE_PATH,
                    SessionClose(public_key, grant.session_id).payload(),
                )
