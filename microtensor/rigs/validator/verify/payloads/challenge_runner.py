from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import signal
import sys
import time

MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1
GOLDEN = 0x9E3779B97F4A7C15
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
REFERENCE = "reference"

_library = None
_released = False


def emit(payload):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def release():
    global _released
    if _library is None or _released:
        return
    _released = True
    with contextlib.suppress(Exception):
        _library.mt_challenge_release()


def on_signal(signum, frame):
    release()
    emit({"digest": "", "elapsed_ms": 0.0, "error": f"terminated by signal {signum}"})
    os._exit(143)


def mix64(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def seed_state(seed, cipher):
    return (mix64(seed ^ 0xA5A5A5A5A5A5A5A5) + mix64((cipher * GOLDEN) & MASK64)) & MASK64


def fnv_u32(value, item):
    for i in range(4):
        value ^= (item >> (8 * i)) & 0xFF
        value = (value * FNV_PRIME) & MASK64
    return value


def fnv_u64(value, item):
    for i in range(8):
        value ^= (item >> (8 * i)) & 0xFF
        value = (value * FNV_PRIME) & MASK64
    return value


def reference_digest(seed, cipher, n, rounds):
    import numpy as np

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


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_reference(seed, cipher, n, rounds):
    started = time.perf_counter()
    digest = reference_digest(seed, cipher, n, rounds)
    elapsed = (time.perf_counter() - started) * 1000.0
    return {
        "digest": f"{digest:016x}",
        "elapsed_ms": round(elapsed, 3),
        "gops": 0.0,
        "gbps": 0.0,
        "benchmark_ms": 0.0,
        "build": "python-reference",
        "version": 1,
        "error": "",
        "mode": REFERENCE,
        "library_sha256": "",
    }


def run_library(path, seed, cipher, n, rounds, benchmark_n, iterations):
    global _library
    library_sha256 = file_sha256(path)
    lib = ctypes.CDLL(path)
    _library = lib
    lib.mt_challenge_version.restype = ctypes.c_int
    lib.mt_challenge_build.restype = ctypes.c_char_p
    lib.mt_challenge_last_error.restype = ctypes.c_char_p
    lib.mt_challenge_solve.restype = ctypes.c_int
    lib.mt_challenge_solve.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.mt_challenge_benchmark.restype = ctypes.c_int
    lib.mt_challenge_benchmark.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.mt_challenge_release.restype = ctypes.c_int

    def last_error():
        raw = lib.mt_challenge_last_error()
        return raw.decode("utf-8", "replace") if raw else ""

    build = lib.mt_challenge_build()
    answer = {
        "digest": "",
        "elapsed_ms": 0.0,
        "gops": 0.0,
        "gbps": 0.0,
        "benchmark_ms": 0.0,
        "build": build.decode("utf-8", "replace") if build else "",
        "version": int(lib.mt_challenge_version()),
        "error": "",
        "mode": "library",
        "library_sha256": library_sha256,
    }
    digest = ctypes.c_uint64(0)
    started = time.perf_counter()
    code = lib.mt_challenge_solve(seed, cipher, n, rounds, ctypes.byref(digest))
    answer["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    if code != 0:
        answer["error"] = f"solve failed with {code}: {last_error()}"
        return answer
    answer["digest"] = f"{digest.value:016x}"
    gops = ctypes.c_double(0.0)
    gbps = ctypes.c_double(0.0)
    elapsed = ctypes.c_double(0.0)
    code = lib.mt_challenge_benchmark(
        benchmark_n, iterations, ctypes.byref(gops), ctypes.byref(gbps), ctypes.byref(elapsed)
    )
    if code != 0:
        answer["error"] = f"benchmark failed with {code}: {last_error()}"
        return answer
    answer["gops"] = round(gops.value, 3)
    answer["gbps"] = round(gbps.value, 3)
    answer["benchmark_ms"] = round(elapsed.value, 3)
    return answer


def main(argv):
    if len(argv) != 8:
        emit(
            {
                "digest": "",
                "elapsed_ms": 0.0,
                "error": "usage: runner <lib|reference> seed cipher n rounds benchmark_n iterations",
            }
        )
        return 2
    path = argv[1]
    try:
        seed, cipher, n, rounds, benchmark_n, iterations = (int(value) for value in argv[2:8])
    except ValueError:
        emit({"digest": "", "elapsed_ms": 0.0, "error": "numeric arguments required"})
        return 2
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        if path == REFERENCE:
            answer = run_reference(seed, cipher, n, rounds)
        else:
            answer = run_library(path, seed, cipher, n, rounds, benchmark_n, iterations)
    except Exception as exc:
        answer = {"digest": "", "elapsed_ms": 0.0, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        release()
    emit(answer)
    return 0 if not answer.get("error") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
