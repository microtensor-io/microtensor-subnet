from __future__ import annotations

import math
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from microtensor.core.constants import (
    EXEC_CPU_SECONDS,
    EXEC_RSS_BYTES,
    EXEC_WALL_SECONDS,
)
from microtensor.core.protocol import Fault
from microtensor.harness.jail import run_jailed
from microtensor.harness.limits import Limits, UnsupportedPlatform

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

SAFE_MODULES = frozenset(
    {
        "math",
        "json",
        "re",
        "itertools",
        "collections",
        "functools",
        "string",
        "heapq",
        "bisect",
    }
)

_ALLOW_UNSANDBOXED = False


class ExecutionUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TestCase:
    args: tuple[Any, ...]
    expected: Any


def configure(*, allow_unsandboxed: bool) -> None:
    global _ALLOW_UNSANDBOXED
    _ALLOW_UNSANDBOXED = allow_unsandboxed


def parse_tests(payload: Sequence[Any]) -> tuple[TestCase, ...]:
    cases: list[TestCase] = []
    for entry in payload:
        if not isinstance(entry, dict) or "args" not in entry or "expected" not in entry:
            raise ValueError(f"malformed test case {entry!r}")
        cases.append(TestCase(args=tuple(entry["args"]), expected=entry["expected"]))
    return tuple(cases)


def extract_code(text: str) -> str:
    blocks: list[str] = _FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        return all(values_match(a, e) for a, e in zip(actual, expected, strict=True))
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return False
        return all(values_match(actual[k], expected[k]) for k in expected)
    return bool(actual == expected)


def _guarded_import(allowed: frozenset[str]) -> Any:
    real_import = __import__

    def guarded(
        name: str,
        globals_: Any = None,
        locals_: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        root = name.split(".")[0]
        if level != 0 or root not in allowed:
            raise ImportError(f"import of {name!r} is not permitted in the sandbox")
        return real_import(name, globals_, locals_, fromlist, level)

    return guarded


def _guarded_open(tmpdir: str) -> Any:
    import os

    real_open = open

    def guarded(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        path = os.path.abspath(os.fspath(file))
        if not path.startswith(os.path.abspath(tmpdir)):
            raise PermissionError(f"open of {path!r} is not permitted in the sandbox")
        return real_open(path, mode, *args, **kwargs)

    return guarded


def _safe_globals(tmpdir: str) -> dict[str, Any]:
    import builtins

    allowed_names = [
        "abs", "all", "any", "bin", "bool", "bytearray", "bytes", "callable",
        "chr", "complex", "dict", "divmod", "enumerate", "filter", "float",
        "format", "frozenset", "hasattr", "hash", "hex", "int", "isinstance",
        "issubclass", "iter", "len", "list", "map", "max", "min", "next",
        "object", "oct", "ord", "pow", "print", "range", "repr", "reversed",
        "round", "set", "setattr", "getattr", "slice", "sorted", "str", "sum",
        "tuple", "type", "zip",
        "ArithmeticError", "AssertionError", "AttributeError", "BaseException",
        "Exception", "IndexError", "KeyError", "LookupError", "NameError",
        "NotImplementedError", "OverflowError", "RecursionError", "RuntimeError",
        "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
        "True", "False", "None", "NotImplemented", "Ellipsis",
        "__build_class__",
    ]
    safe = {name: getattr(builtins, name) for name in allowed_names if hasattr(builtins, name)}
    safe["__import__"] = _guarded_import(SAFE_MODULES)
    safe["open"] = _guarded_open(tmpdir)
    return {"__builtins__": safe, "__name__": "__submission__"}


def _run_suite(code: str, entry_point: str, tests: list[dict[str, Any]]) -> dict[str, Any]:
    tmpdir = tempfile.mkdtemp(prefix="mt-exec-")
    namespace = _safe_globals(tmpdir)

    try:
        compiled = compile(code, "<submission>", "exec")
        exec(compiled, namespace)  # noqa: S102
    except BaseException as exc:
        return {"passed": 0, "total": len(tests), "error": f"{type(exc).__name__}: {exc}"}

    fn = namespace.get(entry_point)
    if not callable(fn):
        return {
            "passed": 0,
            "total": len(tests),
            "error": f"entry point {entry_point!r} is missing or not callable",
        }

    passed = 0
    for case in tests:
        try:
            actual = fn(*case["args"])
        except BaseException:  # noqa: S112
            continue
        if values_match(actual, case["expected"]):
            passed += 1
    return {"passed": passed, "total": len(tests), "error": ""}


def default_limits() -> Limits:
    return Limits(
        cpu_seconds=EXEC_CPU_SECONDS,
        wall_seconds=EXEC_WALL_SECONDS,
        rss_bytes=EXEC_RSS_BYTES,
    )


def execute_pass_rate(
    code: str,
    entry_point: str,
    tests: Sequence[TestCase],
    *,
    limits: Limits | None = None,
) -> float:
    if not tests:
        return 0.0

    payload = [{"args": list(case.args), "expected": case.expected} for case in tests]
    try:
        result = run_jailed(
            _run_suite,
            extract_code(code),
            entry_point,
            payload,
            limits=limits or default_limits(),
            allow_unsandboxed=_ALLOW_UNSANDBOXED,
        )
    except UnsupportedPlatform as exc:
        raise ExecutionUnavailable(str(exc)) from exc

    if result.ok:
        return float(result.value["passed"]) / len(tests)
    if result.fault is Fault.INFRASTRUCTURE:
        raise ExecutionUnavailable(result.error)
    return 0.0
