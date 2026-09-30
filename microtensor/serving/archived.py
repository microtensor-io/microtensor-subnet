from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microtensor.core.protocol import Role
from microtensor.core.tracks import get_track
from microtensor.harness.engines.router import load_router
from microtensor.harness.sdk import EscalationModel, Runtime, engine_small
from microtensor.registry.manifest import ArtifactManifest, verify_tree
from microtensor.serving.agent import Engine

INTAKE = "intake.json"


class ArchiveError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Restored:
    root: Path
    manifest: ArtifactManifest
    record: dict[str, Any]

    @property
    def name(self) -> str:
        system = self.manifest.system
        if system is None or system.endpoint is None:
            raise ArchiveError("the archived submission is not a full system")
        return system.endpoint.name


def restore(record: Path, workdir: Path) -> Restored:
    intake_path = record / INTAKE
    if not intake_path.is_file():
        raise ArchiveError(f"{record} holds no intake record")
    intake = json.loads(intake_path.read_text(encoding="utf-8"))
    manifest = ArtifactManifest.from_json((record / "manifest.json").read_bytes())
    target = workdir / record.name
    shutil.rmtree(target, ignore_errors=True)
    if intake.get("sealed"):
        from microtensor.core.sealing import SealError, open_sealed

        reveal = record / "reveal.json"
        if not reveal.is_file():
            raise ArchiveError("the submission is sealed and no reveal key was archived")
        key = str(json.loads(reveal.read_text(encoding="utf-8"))["key"])
        blob = str((manifest.sealed or {}).get("blob", "artifact.enc"))
        target.mkdir(parents=True)
        try:
            open_sealed((record / "sealed" / blob).read_bytes(), key, target)
        except SealError as exc:
            raise ArchiveError(f"the archived submission would not open: {exc}") from exc
    else:
        shutil.copytree(record / "files", target)
    ok, reason = verify_tree(target, manifest)
    if not ok:
        raise ArchiveError(f"the archived files do not match their manifest: {reason}")
    if manifest.system is None or not manifest.system.full:
        raise ArchiveError("the archived submission is not a full system")
    return Restored(root=target, manifest=manifest, record=intake)


def open_system(path: Path, workdir: Path) -> Restored:
    if (path / INTAKE).is_file():
        return restore(path, workdir)
    manifest = ArtifactManifest.from_json((path / "manifest.json").read_bytes())
    ok, reason = verify_tree(path, manifest)
    if not ok:
        raise ArchiveError(f"the system files do not match their manifest: {reason}")
    if manifest.system is None or not manifest.system.full:
        raise ArchiveError(f"{path} is not a full system")
    return Restored(root=path, manifest=manifest, record={})


def prompt_of(request: Mapping[str, Any]) -> str:
    prompt = str(request.get("prompt", ""))
    if prompt:
        return prompt
    for message in reversed(list(request.get("messages") or [])):
        if isinstance(message, Mapping) and message.get("role") == "user":
            return str(message.get("content", ""))
    raise ArchiveError("the request carries no prompt")


class SystemEngine(Engine):
    def __init__(
        self, runtime: Runtime, sign: Callable[[Mapping[str, Any]], str], url: str = "system:"
    ) -> None:
        super().__init__(url)
        self.runtime = runtime
        self.sign = sign

    async def generate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        import asyncio

        trace = await asyncio.to_thread(
            self.runtime.run,
            0,
            str(request.get("request_id") or "request"),
            prompt_of(request),
            dict(request.get("inputs") or {}),
        )
        signed = trace.signed_with(self.sign(trace.body()))
        escalation = trace.escalation
        return {
            "text": str(trace.final),
            "prompt_tokens": [],
            "completion_tokens": list(trace.small.tokens),
            "finish_reason": "stop",
            "escalated": trace.escalated,
            "answered_by": "escalation" if trace.escalated else "small",
            "router_features": dict(trace.router.features),
            "trace": signed.to_dict(),
            "usage": {
                "prompt_tokens": trace.small.prompt_tokens
                + (escalation.prompt_tokens if escalation else 0),
                "completion_tokens": len(trace.small.tokens)
                + (escalation.completion_tokens if escalation else 0),
            },
        }

    async def stream(self, request: Mapping[str, Any], on_delta: Any) -> dict[str, Any]:
        found = await self.generate(request)
        await on_delta(found["text"])
        return found


def runtime_for(
    restored: Restored, escalate: EscalationModel, *, hotkey: str, gpu_layers: int = 0
) -> Runtime:
    from microtensor.harness.engines.gguf import GgufEngine

    system = restored.manifest.system
    if system is None or system.harness is None or system.escalation is None:
        raise ArchiveError("the archived submission is not a full system")
    engine = GgufEngine(gpu_layers=gpu_layers)
    engine.load(restored.root, restored.manifest.load)
    return Runtime(
        restored.root / system.harness.path,
        load_router(restored.root / system.locate(Role.ROUTER), system.router_features),
        system.router_features,
        engine_small(engine, chat=get_track(restored.manifest.track).chat),
        escalate,
        escalation=system.escalation,
        system_digest=system.digest(),
        hotkey=hotkey,
    )


def archived_handlers(
    records: Sequence[Path],
    workdir: Path,
    escalation_for: Callable[[str, str], EscalationModel],
    sign: Callable[[Mapping[str, Any]], str],
    *,
    hotkey: str,
) -> dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]]:
    from microtensor.miner.host import system_handler

    handlers: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {}
    for record in records:
        restored = restore(record, workdir)
        system = restored.manifest.system
        if system is None or system.escalation is None:
            raise ArchiveError(f"{record} is not a full system")
        escalate = escalation_for(system.escalation.model, system.escalation.revision)
        handlers[restored.name] = system_handler(
            runtime_for(restored, escalate, hotkey=hotkey), sign
        )
    return handlers
