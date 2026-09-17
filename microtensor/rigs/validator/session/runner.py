from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CommandResult:
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    elapsed_ms: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.error

    @property
    def transport_failed(self) -> bool:
        return bool(self.error) and self.exit_code < 0

    def stderr_tail(self, chars: int = 2048) -> str:
        return self.stderr[-chars:]


class Runner(Protocol):
    async def run(
        self, command: str, timeout: float = 60.0, stdin: str | None = None
    ) -> CommandResult: ...

    async def upload(self, content: bytes, remote_path: str, mode: int = 0o600) -> None: ...
