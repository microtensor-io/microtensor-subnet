from __future__ import annotations

import asyncio
import ctypes
import hashlib
import secrets
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from microtensor.rigs.protocol.failures import Failure, FailureClass, Verdict
from microtensor.rigs.protocol.messages import ChallengeAnswer, ChallengeRequest
from microtensor.rigs.validator.config import Settings
from microtensor.rigs.validator.data.tiers import Tier
from microtensor.rigs.validator.session.client import Session
from microtensor.rigs.validator.session.ssh import SshError, classify
from microtensor.rigs.validator.verify.payloads import source

RUNNER_NAME = "challenge_runner.py"
LIBRARY_NAME = "libmtchallenge.so"
MODE_LIBRARY = "library"
MODE_REFERENCE = "reference"
MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1
GOLDEN = 0x9E3779B97F4A7C15
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
GOPS_FLOOR: dict[Tier, float] = {
    Tier.ENTRY: 50.0,
    Tier.STANDARD: 200.0,
    Tier.PROFESSIONAL: 400.0,
    Tier.FLAGSHIP: 1000.0,
}
GBPS_FLOOR: dict[Tier, float] = {
    Tier.ENTRY: 100.0,
    Tier.STANDARD: 300.0,
    Tier.PROFESSIONAL: 400.0,
    Tier.FLAGSHIP: 1000.0,
}


class ChallengeUnavailable(RuntimeError):
    pass


def new_request() -> ChallengeRequest:
    return ChallengeRequest(seed=secrets.randbits(64), cipher=secrets.randbits(64))


def mix64(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def seed_state(seed: int, cipher: int) -> int:
    return (mix64(seed ^ 0xA5A5A5A5A5A5A5A5) + mix64((cipher * GOLDEN) & MASK64)) & MASK64


def fnv_u32(value: int, item: int) -> int:
    for i in range(4):
        value ^= (item >> (8 * i)) & 0xFF
        value = (value * FNV_PRIME) & MASK64
    return value


def fnv_u64(value: int, item: int) -> int:
    for i in range(8):
        value ^= (item >> (8 * i)) & 0xFF
        value = (value * FNV_PRIME) & MASK64
    return value


def reference_digest(seed: int, cipher: int, n: int, rounds: int) -> int:
    state = seed_state(seed, cipher)
    with np.errstate(over="ignore"):
        index = np.arange(1, n * n + 1, dtype=np.uint64)
        z = np.uint64(state) + index * np.uint64(GOLDEN)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        matrix = (z & np.uint64(MASK32)).astype(np.uint32).reshape(n, n)
        for _ in range(rounds):
            matrix = (matrix.astype(np.uint64) @ matrix.astype(np.uint64)).astype(np.uint32)
        rows = np.full(n, FNV_OFFSET, dtype=np.uint64)
        prime = np.uint64(FNV_PRIME)
        for column in range(n):
            values = matrix[:, column].astype(np.uint64)
            for i in range(4):
                rows ^= (values >> np.uint64(8 * i)) & np.uint64(0xFF)
                rows *= prime
    digest = FNV_OFFSET
    for value in rows.tolist():
        digest = fnv_u64(digest, int(value))
    digest = fnv_u64(digest, seed)
    digest = fnv_u64(digest, cipher)
    digest = fnv_u32(digest, n)
    digest = fnv_u32(digest, rounds)
    return digest


def library_digest(path: Path, seed: int, cipher: int, n: int, rounds: int) -> int:
    lib = ctypes.CDLL(str(path))
    lib.mt_challenge_solve.restype = ctypes.c_int
    lib.mt_challenge_solve.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.mt_challenge_last_error.restype = ctypes.c_char_p
    digest = ctypes.c_uint64(0)
    code = lib.mt_challenge_solve(seed, cipher, n, rounds, ctypes.byref(digest))
    if code != 0:
        raw = lib.mt_challenge_last_error()
        raise ChallengeUnavailable(
            f"verify library failed with {code}: {raw.decode() if raw else ''}"
        )
    return int(digest.value)


def expected_digest(settings: Settings, request: ChallengeRequest) -> tuple[int, str]:
    if settings.challenge_library.is_file():
        try:
            return library_digest(
                settings.challenge_library, request.seed, request.cipher, request.n, request.rounds
            ), "library"
        except OSError as exc:
            raise ChallengeUnavailable(f"cannot load {settings.challenge_library}: {exc}") from exc
    return reference_digest(request.seed, request.cipher, request.n, request.rounds), "reference"


def floors_for(tier: Tier | None) -> tuple[float, float]:
    chosen = tier or Tier.ENTRY
    return GOPS_FLOOR[chosen], GBPS_FLOOR[chosen]


def tier_of(rig: dict[str, Any]) -> Tier | None:
    try:
        return Tier(str(rig.get("tier", "") or ""))
    except ValueError:
        return None


@dataclass(frozen=True)
class Prepared:
    mode: str
    library_sha256: str
    library_arg: str


async def prepare(session: Session, settings: Settings, session_dir: str) -> Prepared:
    await session.upload(source(RUNNER_NAME), f"{session_dir}/{RUNNER_NAME}", 0o600)
    if settings.agent_library.is_file():
        raw = settings.agent_library.read_bytes()
        remote = f"{session_dir}/{LIBRARY_NAME}"
        await session.upload(raw, remote, 0o700)
        return Prepared(MODE_LIBRARY, hashlib.sha256(raw).hexdigest(), remote)
    if settings.allow_reference_challenge:
        return Prepared(MODE_REFERENCE, "", MODE_REFERENCE)
    raise ChallengeUnavailable(
        f"the miner challenge library {settings.agent_library} is missing and CV_ALLOW_REFERENCE_CHALLENGE is off"
    )


def command_for(
    session: Session, session_dir: str, prepared: Prepared, request: ChallengeRequest, uuid: str
) -> str:
    parts = [
        shlex.quote(session.python),
        "-I",
        RUNNER_NAME,
        shlex.quote(prepared.library_arg),
        str(request.seed),
        str(request.cipher),
        str(request.n),
        str(request.rounds),
        str(request.benchmark_n),
        str(request.iterations),
    ]
    prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(uuid)} " if uuid else ""
    return f"cd {shlex.quote(session_dir)} && {prefix}{' '.join(parts)}"


async def run(
    session: Session,
    settings: Settings,
    session_dir: str,
    request: ChallengeRequest,
    tier: Tier | None,
    gpu_uuids: list[str],
) -> Verdict:
    seed = str(request.seed)
    prepared = await prepare(session, settings, session_dir)
    expected, method = await asyncio.to_thread(expected_digest, settings, request)
    gops_floor, gbps_floor = floors_for(tier)
    evidence: dict[str, Any] = {
        "seed": seed,
        "cipher": str(request.cipher),
        "n": request.n,
        "rounds": request.rounds,
        "benchmark_n": request.benchmark_n,
        "iterations": request.iterations,
        "mode": prepared.mode,
        "verified_with": method,
        "library_sha256": prepared.library_sha256,
        "floors": {"gops": gops_floor, "gbps": gbps_floor} if prepared.mode == MODE_LIBRARY else {},
        "gpus": [],
    }
    failures: list[Failure] = []
    targets = list(gpu_uuids) or [""]
    total_elapsed = 0.0
    for uuid in targets:
        command = command_for(session, session_dir, prepared, request, uuid)
        try:
            result, parsed = await session.run_json(command, settings.check_timeout)
        except SshError as exc:
            failures.append(exc.failure(seed))
            break
        total_elapsed += result.elapsed_ms
        entry: dict[str, Any] = {"uuid": uuid, "command_ms": round(result.elapsed_ms, 1)}
        evidence["gpus"].append(entry)
        if result.transport_failed:
            failures.append(Failure(FailureClass.SSH_TRANSPORT, result.error, seed=seed))
            break
        if parsed is None:
            crash = classify(result, seed)
            failures.append(
                crash
                or Failure(FailureClass.AGENT_CRASH, "the runner printed no answer", seed=seed)
            )
            entry["error"] = result.stderr_tail(400)
            continue
        answer = ChallengeAnswer.from_payload(parsed)
        entry.update(
            {
                "elapsed_ms": answer.elapsed_ms,
                "gops": answer.gops,
                "gbps": answer.gbps,
                "benchmark_ms": answer.benchmark_ms,
                "build": answer.build,
                "version": answer.version,
                "mode": str(parsed.get("mode", "")),
            }
        )
        if answer.error:
            entry["error"] = answer.error[:300]
            failures.append(
                Failure(
                    FailureClass.CHALLENGE_REJECT,
                    f"{uuid or 'gpu'}: {answer.error[:200]}",
                    seed=seed,
                    evidence={"cipher": str(request.cipher), "uuid": uuid},
                )
            )
            continue
        loaded = str(parsed.get("library_sha256", "") or "")
        if prepared.mode == MODE_LIBRARY and loaded != prepared.library_sha256:
            entry["loaded_sha256"] = loaded
            failures.append(
                Failure(
                    FailureClass.VERSION_STALE,
                    "the runner loaded a library that is not the one uploaded",
                    seed=seed,
                    evidence={"uploaded": prepared.library_sha256, "loaded": loaded},
                )
            )
            continue
        try:
            digest_ok = int(answer.digest, 16) == expected
        except ValueError:
            digest_ok = False
        entry["digest_ok"] = digest_ok
        measured = {
            "uuid": uuid,
            "cipher": str(request.cipher),
            "elapsed_ms": answer.elapsed_ms,
            "gops": answer.gops,
            "gbps": answer.gbps,
        }
        if not digest_ok:
            failures.append(
                Failure(
                    FailureClass.CHALLENGE_REJECT,
                    f"{uuid or 'gpu'}: wrong digest",
                    seed=seed,
                    evidence=measured,
                )
            )
            continue
        if answer.elapsed_ms > settings.challenge_max_ms:
            failures.append(
                Failure(
                    FailureClass.CHALLENGE_REJECT,
                    f"{uuid or 'gpu'}: solved in {answer.elapsed_ms:.0f} ms, above {settings.challenge_max_ms:.0f} ms",
                    seed=seed,
                    evidence=measured,
                )
            )
            continue
        if prepared.mode == MODE_LIBRARY and (answer.gops < gops_floor or answer.gbps < gbps_floor):
            failures.append(
                Failure(
                    FailureClass.CHALLENGE_REJECT,
                    f"{uuid or 'gpu'}: throughput {answer.gops:.0f} GOPS and {answer.gbps:.0f} GB/s below the {(tier or Tier.ENTRY).value} floor of {gops_floor:.0f} GOPS and {gbps_floor:.0f} GB/s",
                    seed=seed,
                    evidence=measured,
                )
            )
    passed = [entry for entry in evidence["gpus"] if entry.get("digest_ok")]
    evidence["summary"] = (
        f"{len(passed)} of {len(targets)} cards answered in {prepared.mode} mode"
        if failures
        else f"{len(targets)} cards answered, "
        + ", ".join(f"{e.get('gops', 0):.0f} GOPS" for e in evidence["gpus"])
    )
    return Verdict(
        "challenge",
        failure=failures[0] if failures else None,
        evidence=evidence,
        elapsed_ms=total_elapsed,
    )
