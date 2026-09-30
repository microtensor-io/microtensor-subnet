from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from microtensor.core.system import SystemManifest
from microtensor.core.trace import Trace, TraceError, Verifier, read
from microtensor.tasks.corpus import Task


class LiveError(RuntimeError):
    pass


class SystemClient(Protocol):
    def task(self, name: str, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


class HttpSystemClient:
    def __init__(self, base: str, timeout: float = 900.0) -> None:
        if not base.startswith(("http://", "https://")):
            raise LiveError(f"the system host {base!r} is not http or https")
        self.base = base.rstrip("/")
        self.timeout = timeout

    def task(self, name: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(dict(request)).encode()
        answer = urllib.request.Request(  # noqa: S310
            f"{self.base}/v1/systems/{name}/task",
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(answer, timeout=self.timeout) as found:  # noqa: S310
            payload = json.loads(found.read().decode("utf-8"))
        if "trace" not in payload:
            raise LiveError(str(payload.get("error", "the host returned no trace")))
        return dict(payload["trace"])


class GatewaySystemClient:
    def __init__(
        self, gateway: str, credential: str, hotkey: str, worker: str = "", timeout: float = 900.0
    ) -> None:
        if not gateway.startswith(("http://", "https://")):
            raise LiveError(f"the gateway {gateway!r} is not http or https")
        self.gateway = gateway.rstrip("/")
        self.credential = credential
        self.hotkey = hotkey
        self.worker = worker
        self.timeout = timeout

    def task(self, name: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(
            {**dict(request), "hotkey": self.hotkey, "worker": self.worker, "system": name}
        ).encode()
        answer = urllib.request.Request(  # noqa: S310
            f"{self.gateway}/v1/operators/task",
            data=body,
            headers={"content-type": "application/json", "x-mt-credential": self.credential},
            method="POST",
        )
        with urllib.request.urlopen(answer, timeout=self.timeout) as found:  # noqa: S310
            payload = json.loads(found.read().decode("utf-8"))
        if "trace" not in payload:
            raise LiveError(str(payload.get("error", "the gateway returned no trace")))
        return dict(payload["trace"])


@dataclass(frozen=True, slots=True)
class LiveRun:
    traces: tuple[Trace, ...]
    failures: tuple[tuple[str, str], ...]

    @property
    def answered(self) -> dict[str, Trace]:
        return {trace.task_ref: trace for trace in self.traces}


def request_for(round_index: int, task: Task) -> dict[str, Any]:
    return {
        "round_index": round_index,
        "task_ref": task.ref,
        "prompt": task.prompt,
        "inputs": dict(task.inputs),
    }


def run_live(
    client: SystemClient,
    system: SystemManifest,
    *,
    hotkey: str,
    round_index: int,
    tasks: Sequence[Task],
    verify: Verifier,
) -> LiveRun:
    if system.endpoint is None:
        raise LiveError("the system declares no endpoint to test")
    digest = system.digest()
    traces: list[Trace] = []
    failures: list[tuple[str, str]] = []
    for task in tasks:
        try:
            trace = read(client.task(system.endpoint.name, request_for(round_index, task)), verify)
        except (TraceError, LiveError, OSError, ValueError) as exc:
            failures.append((task.ref, str(exc)))
            continue
        wrong = [
            what
            for what, ok in (
                ("task", trace.task_ref == task.ref),
                ("system", trace.system_digest == digest),
                ("miner", trace.hotkey == hotkey),
                ("round", trace.round_index == round_index),
            )
            if not ok
        ]
        if wrong:
            failures.append((task.ref, f"the trace names a different {', '.join(wrong)}"))
            continue
        traces.append(trace)
    return LiveRun(traces=tuple(traces), failures=tuple(failures))


def chain_verifier() -> Verifier:
    from microtensor.chain.wallet import verify_bytes

    return verify_bytes
