from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx

from microtensor_subnet.core.hashing import DIGEST_PREFIX
from microtensor_subnet.serving import markers, verify
from microtensor_subnet.serving.canonical import LLAMA_CONTEXT, Prefill, prefill_for
from microtensor_subnet.serving.engines import LLAMA, SGLANG, EngineError, EngineSpec
from microtensor_subnet.serving.host.fetch import FetchedModel
from microtensor_subnet.serving.host.plain import PlainHeld, attach_model, close_model, open_model
from microtensor_subnet.serving.host.plain import chat as plain_chat
from microtensor_subnet.serving.probes.ask import Chatted, chatted
from microtensor_subnet.serving.tokens import TokenError, Tokeniser
from microtensor_subnet.serving.verify import Calibration, Judgement, VerificationError
from microtensor_subnet.validator.sandbox.docker import Docker

log = logging.getLogger("microtensor_subnet.serving.verifier")

CALIBRATION_DIR: Final[str] = "calibration"
MODEL_NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
HEX: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
GENERATE_SECONDS: Final[float] = 900.0


@dataclass(frozen=True, slots=True)
class Measured:
    statistic: verify.Statistic | None
    reason: str = ""


def _prompt_mismatch(ours: Sequence[int], theirs: Sequence[int]) -> str:
    if not theirs:
        return "the operator returned no prompt ids"
    for position, (mine, found) in enumerate(zip(ours, theirs, strict=False)):
        if mine != found:
            return (
                f"the operator's prompt ids differ from this validator's chat template "
                f"at position {position}"
            )
    if len(ours) != len(theirs):
        return (
            f"the operator's prompt is {len(theirs)} tokens and this validator's chat "
            f"template gives {len(ours)}"
        )
    return ""


class ModelVerifier:
    def __init__(self, prefill: Prefill, tokeniser: Tokeniser, family: str) -> None:
        self.prefill = prefill
        self.tokeniser = tokeniser
        self.family = markers.family_of(family).name

    def prompt_ids(self, request: Mapping[str, Any]) -> list[int]:
        messages = request.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, str) or not messages:
            raise TokenError("a chat request carries its messages as a list")
        tools = request.get("tools")
        return self.tokeniser.prompt_ids(
            [dict(m) for m in messages if isinstance(m, Mapping)],
            [dict(t) for t in tools if isinstance(t, Mapping)] if tools else None,
        )

    def completion_ids(self, answered: Chatted) -> list[int]:
        return self.tokeniser.completion_ids(
            self.family,
            reasoning=answered.reasoning,
            content=answered.content,
            tool_calls=answered.tool_calls,
            finish_reason=answered.finish_reason,
        )

    async def measure(
        self, request: Mapping[str, Any], answered: Chatted, *, check_prompt: bool = True
    ) -> Measured:
        try:
            prompt = self.prompt_ids(request)
        except TokenError as exc:
            return Measured(None, f"the prompt could not be rendered: {exc}")
        if check_prompt:
            mismatch = _prompt_mismatch(prompt, answered.prompt_ids)
            if mismatch:
                return Measured(None, mismatch)
        try:
            completion = self.completion_ids(answered)
        except (markers.MarkerError, TokenError) as exc:
            return Measured(None, f"the answer could not be re-tokenised: {exc}")
        if not completion:
            return Measured(None, "the answer carried no token to check")
        try:
            gaps = await self.prefill.margins(prompt, completion)
        except (VerificationError, httpx.HTTPError, OSError) as exc:
            return Measured(None, f"the prefill failed: {exc}")
        return Measured(verify.summarise(gaps))

    async def judge(
        self,
        request: Mapping[str, Any],
        answered: Chatted,
        calibration: Calibration | None,
    ) -> Judgement:
        if calibration is None:
            return Judgement(
                verify.UNPROVEN, 0.0, 0.0, "this validator holds no calibration for the model"
            )
        found = await self.measure(request, answered)
        if found.statistic is None:
            return Judgement(verify.UNPROVEN, 0.0, calibration.threshold, found.reason)
        return verify.judge(found.statistic, calibration)


@dataclass(frozen=True, slots=True)
class Calibrated:
    model: str
    artifact_digest: str
    calibration: Calibration
    backend: str = ""
    substitute: str = ""
    measured_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "artifact_digest": self.artifact_digest,
            "backend": self.backend,
            "substitute": self.substitute,
            "measured_at": self.measured_at,
            "calibration": self.calibration.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Calibrated:
        found = raw.get("calibration")
        if not isinstance(found, Mapping):
            raise VerificationError("the calibration file carries no calibration")
        return cls(
            model=str(raw.get("model") or ""),
            artifact_digest=_digest(str(raw.get("artifact_digest") or "")),
            calibration=Calibration.from_dict(found),
            backend=str(raw.get("backend") or ""),
            substitute=str(raw.get("substitute") or ""),
            measured_at=float(raw.get("measured_at") or 0.0),
        )


def _digest(digest: str) -> str:
    found = digest.strip().lower().removeprefix(DIGEST_PREFIX)
    if not HEX.match(found):
        raise VerificationError(f"{digest!r} is not a sha256 digest")
    return found


def calibration_path(home: Path, model: str, digest: str) -> Path:
    if not MODEL_NAME.match(model) or ".." in model:
        raise VerificationError(f"{model!r} cannot name a calibration folder")
    return Path(home) / CALIBRATION_DIR / model / f"{_digest(digest)}.json"


def load_calibration(home: Path, model: str, digest: str) -> Calibrated | None:
    path = calibration_path(home, model, digest)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerificationError(f"the calibration at {path} is unreadable: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise VerificationError(f"the calibration at {path} is not an object")
    found = Calibrated.from_dict(raw)
    if found.model != model or found.artifact_digest != _digest(digest):
        raise VerificationError(f"the calibration at {path} names another model or digest")
    return found


def save_calibration(home: Path, found: Calibrated) -> Path:
    path = calibration_path(home, found.model, found.artifact_digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


async def _statistics(
    verifier: ModelVerifier,
    samples: Sequence[tuple[Mapping[str, Any], Chatted]],
    *,
    check_prompt: bool,
    label: str,
) -> list[verify.Statistic]:
    found: list[verify.Statistic] = []
    for request, answered in samples:
        if not answered.ok:
            log.info("%s sample skipped: %s", label, answered.error)
            continue
        measured = await verifier.measure(request, answered, check_prompt=check_prompt)
        if measured.statistic is None:
            log.info("%s sample skipped: %s", label, measured.reason)
            continue
        found.append(measured.statistic)
    return found


async def calibrate_model(
    verifier: ModelVerifier,
    honest: Sequence[tuple[Mapping[str, Any], Chatted]],
    substituted: Sequence[tuple[Mapping[str, Any], Chatted]],
    *,
    alpha: float = 0.01,
) -> Calibration:
    ours = await _statistics(verifier, honest, check_prompt=True, label="honest")
    theirs = await _statistics(verifier, substituted, check_prompt=False, label="substituted")
    return verify.calibrate(ours, theirs, alpha=alpha)


@dataclass(slots=True)
class Reference:
    model: str
    artifact_digest: str
    backend: str
    verifier: ModelVerifier
    engine: PlainHeld | None = None
    launched: bool = False


def family_for(entry: Mapping[str, Any]) -> str:
    return markers.family_of(str(entry.get("family") or "")).name


def open_engine(
    docker: Docker,
    fetched: FetchedModel,
    entry: Mapping[str, Any],
    *,
    gpu: str,
    images: Mapping[str, str],
    mem_fraction: float,
    ready_seconds: float,
    engine_url: str = "",
) -> PlainHeld:
    if engine_url:
        return attach_model(fetched, entry, engine_url, gpu=gpu)
    return open_model(
        docker,
        fetched,
        entry,
        gpu=gpu,
        images=images,
        mem_fraction=mem_fraction,
        ready_seconds=ready_seconds,
    )


def open_reference(
    docker: Docker,
    fetched: FetchedModel,
    entry: Mapping[str, Any],
    *,
    backend: str,
    gpu: str,
    images: Mapping[str, str],
    mem_fraction: float,
    ready_seconds: float,
    engine_url: str = "",
    gpu_layers: int = 0,
) -> Reference:
    family = family_for(entry)
    spec = EngineSpec.from_entry(entry)
    if backend == SGLANG:
        if spec.backend != SGLANG:
            raise VerificationError(
                f"{fetched.model} is served by {spec.backend}; prefill it with llama"
            )
        held = open_engine(
            docker,
            fetched,
            entry,
            gpu=gpu,
            images=images,
            mem_fraction=mem_fraction,
            ready_seconds=ready_seconds,
            engine_url=engine_url,
        )
        verifier = ModelVerifier(prefill_for(SGLANG, url=held.engine.url), held.tokeniser, family)
        return Reference(
            fetched.model, fetched.artifact_digest, SGLANG, verifier, held, not engine_url
        )
    if backend == LLAMA:
        if not fetched.weights:
            raise VerificationError(f"{fetched.model} carries no gguf weights to prefill")
        if spec.chat_template and spec.chat_template not in fetched.files:
            raise VerificationError("the engine's chat template is not pinned by the manifest")
        tokeniser = Tokeniser.load(
            fetched.model_dir, template=spec.chat_template, gguf_file=fetched.weights
        )
        prefill = prefill_for(
            LLAMA,
            weights=fetched.model_dir / fetched.weights,
            context=min(int(entry.get("context") or LLAMA_CONTEXT), LLAMA_CONTEXT),
            gpu_layers=gpu_layers,
        )
        verifier = ModelVerifier(prefill, tokeniser, family)
        return Reference(fetched.model, fetched.artifact_digest, LLAMA, verifier)
    raise VerificationError(f"no prefill backend {backend!r}; known backends are sglang, llama")


def close_reference(docker: Docker, reference: Reference) -> None:
    if reference.engine is not None and reference.launched:
        close_model(docker, reference.engine)


async def _ignore(delta: Mapping[str, Any]) -> None:
    return None


async def generate(
    held: PlainHeld,
    requests: Sequence[Mapping[str, Any]],
    *,
    timeout: float = GENERATE_SECONDS,
) -> list[Chatted]:
    found: list[Chatted] = []
    for request in requests:
        try:
            answer = await plain_chat(held, request, _ignore, timeout=timeout)
        except (EngineError, TokenError, asyncio.TimeoutError) as exc:
            found.append(Chatted(False, error=str(exc)[:300]))
            continue
        found.append(chatted(answer))
    return found


def stamped(reference: Reference, calibration: Calibration, *, substitute: str) -> Calibrated:
    return Calibrated(
        model=reference.model,
        artifact_digest=reference.artifact_digest,
        calibration=calibration,
        backend=reference.backend,
        substitute=substitute,
        measured_at=time.time(),
    )
