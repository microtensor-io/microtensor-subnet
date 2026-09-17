from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import asyncssh

from microtensor.rigs.protocol.failures import Failure, FailureClass
from microtensor.rigs.validator.session.runner import CommandResult

KEEPALIVE_SECONDS = 30
KEEPALIVE_COUNT = 4
TRANSPORT_EXIT = -1
SIGNAL_EXIT = 128
TIMEOUT_EXIT = 124
STDERR_TAIL = 2048
UPLOAD_TIMEOUT = 300.0
CLOSE_TIMEOUT = 10.0


class SshError(RuntimeError):
    def __init__(self, kind: FailureClass, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail

    def failure(self, seed: str = "") -> Failure:
        return Failure(self.kind, self.detail, seed=seed)


def key_fields(line: str) -> tuple[str, str]:
    parts = (line or "").strip().split()
    if len(parts) < 2:
        return "", ""
    return parts[0], parts[1]


def same_key(presented: asyncssh.SSHKey | None, granted: str) -> bool:
    if presented is None:
        return False
    try:
        shown = presented.export_public_key("openssh").decode().strip()
    except (ValueError, asyncssh.Error):
        return False
    expected = key_fields(granted)
    return bool(expected[1]) and key_fields(shown) == expected


def last_json_object(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if not text.startswith("{") or not text.endswith("}"):
            continue
        try:
            parsed = json.loads(text)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def classify(result: CommandResult, seed: str = "") -> Failure | None:
    if result.transport_failed:
        return Failure(FailureClass.SSH_TRANSPORT, result.error, seed=seed)
    if result.exit_code == TIMEOUT_EXIT:
        return Failure(
            FailureClass.AGENT_CRASH,
            result.error or "the check timed out",
            seed=seed,
            evidence={"stderr": result.stderr_tail(STDERR_TAIL)},
        )
    if result.exit_code != 0:
        return Failure(
            FailureClass.AGENT_CRASH,
            f"exit {result.exit_code}: {result.stderr_tail(STDERR_TAIL).strip() or result.error or 'no output'}"[
                :600
            ],
            seed=seed,
            evidence={"exit_code": result.exit_code, "stderr": result.stderr_tail(STDERR_TAIL)},
        )
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


class SshRunner:
    def __init__(self, connection: asyncssh.SSHClientConnection, address: str, port: int) -> None:
        self._conn = connection
        self.address = address
        self.port = port

    @classmethod
    async def connect(
        cls,
        address: str,
        port: int,
        username: str,
        client_key: asyncssh.SSHKey,
        host_key: str,
        timeout: float,
    ) -> SshRunner:
        if not key_fields(host_key)[1]:
            raise SshError(FailureClass.SSH_TRANSPORT, "the grant carries no host key to pin")
        try:
            connection = await asyncssh.connect(
                address,
                port=port,
                username=username,
                client_keys=[client_key],
                known_hosts=([host_key.strip()], [], []),
                connect_timeout=timeout,
                login_timeout=timeout,
                keepalive_interval=KEEPALIVE_SECONDS,
                keepalive_count_max=KEEPALIVE_COUNT,
            )
        except asyncssh.HostKeyNotVerifiable as exc:
            raise SshError(
                FailureClass.SSH_TRANSPORT, f"host key differs from the granted one: {exc}"
            ) from exc
        except (OSError, asyncssh.Error, asyncio.TimeoutError, ValueError) as exc:
            raise SshError(FailureClass.SSH_TRANSPORT, f"ssh {address}:{port}: {exc}") from exc
        if not same_key(connection.get_server_host_key(), host_key):
            connection.close()
            raise SshError(FailureClass.SSH_TRANSPORT, "host key differs from the granted one")
        return cls(connection, address, port)

    async def run(
        self, command: str, timeout: float = 60.0, stdin: str | None = None
    ) -> CommandResult:
        started = time.monotonic()
        try:
            done = await self._conn.run(
                command,
                input=stdin,
                timeout=timeout,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
        except asyncssh.TimeoutError as exc:
            return CommandResult(
                command,
                TIMEOUT_EXIT,
                _text(exc.stdout),
                _text(exc.stderr),
                (time.monotonic() - started) * 1000.0,
                f"timed out after {timeout:.0f} s",
            )
        except (OSError, asyncssh.Error, asyncio.TimeoutError) as exc:
            return CommandResult(
                command,
                TRANSPORT_EXIT,
                "",
                "",
                (time.monotonic() - started) * 1000.0,
                f"ssh transport: {exc}",
            )
        elapsed = (time.monotonic() - started) * 1000.0
        stderr = _text(done.stderr)
        if done.exit_status is None:
            signal = done.exit_signal[0] if done.exit_signal else "unknown"
            return CommandResult(
                command,
                SIGNAL_EXIT,
                _text(done.stdout),
                f"{stderr}\nkilled by signal {signal}",
                elapsed,
            )
        return CommandResult(command, int(done.exit_status), _text(done.stdout), stderr, elapsed)

    async def _put(self, content: bytes, remote_path: str, mode: int) -> None:
        async with self._conn.start_sftp_client() as sftp:
            async with sftp.open(remote_path, "wb") as handle:
                await handle.write(content)
            await sftp.chmod(remote_path, mode)

    async def upload(self, content: bytes, remote_path: str, mode: int = 0o600) -> None:
        try:
            await asyncio.wait_for(self._put(content, remote_path, mode), UPLOAD_TIMEOUT)
        except (OSError, asyncssh.Error, asyncio.TimeoutError) as exc:
            raise SshError(FailureClass.SSH_TRANSPORT, f"upload {remote_path}: {exc}") from exc

    async def run_json(
        self, command: str, timeout: float = 60.0, stdin: str | None = None
    ) -> tuple[CommandResult, dict[str, Any] | None]:
        result = await self.run(command, timeout, stdin)
        return result, last_json_object(result.stdout)

    async def close(self) -> None:
        self._conn.close()
        try:
            await asyncio.wait_for(self._conn.wait_closed(), CLOSE_TIMEOUT)
        except (asyncio.TimeoutError, OSError, asyncssh.Error):
            return
