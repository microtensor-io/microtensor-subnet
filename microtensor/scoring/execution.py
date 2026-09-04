from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
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
_ENV_ROOT: Path | None = None


class ExecutionUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TestCase:
    args: tuple[Any, ...]
    expected: Any


def configure(*, allow_unsandboxed: bool, env_root: str | Path | None = None) -> None:
    global _ALLOW_UNSANDBOXED, _ENV_ROOT
    _ALLOW_UNSANDBOXED = allow_unsandboxed
    _ENV_ROOT = Path(env_root) if env_root else None


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


RESULT_ENV = "MT_RESULT_PATH"

MODULE_GUARD = """\
if __name__ == "__main__":
    _mt_main()
"""

MODULE_PREAMBLE = """\
import difflib as _difflib
import ctypes as _ctypes
import json as _json
import logging as _logging
import os as _os
import pprint as _pprint
import signal as _signal
import socket as _socket
import sys as _sys
import threading as _threading
import traceback as _traceback
import unittest as _unittest


def _mt_install():
    path = _os.environ.pop("MT_RESULT_PATH")
    state = {"writing": False}
    blocked = {
        "os.kill",
        "os.fork",
        "os.forkpty",
        "os.exec",
        "os.posix_spawn",
        "sys.settrace",
        "sys.setprofile",
    }
    foreign = {"ctypes.dlopen", "ctypes.call_function", "ctypes.cdata"}
    ctypes_dir = _os.path.dirname(_os.path.abspath(_ctypes.__file__))

    def from_solution():
        frame = _sys._getframe(2)
        while frame is not None:
            name = frame.f_code.co_filename
            if name.startswith("<frozen") or _os.path.abspath(name).startswith(ctypes_dir):
                frame = frame.f_back
                continue
            return name.startswith("<") or _os.path.basename(name) == "solution.py"
        return False

    def hook(event, args):
        if event == "open":
            try:
                same = _os.fspath(args[0]) == path
            except TypeError:
                same = False
            mode = str(args[1] or "r") if len(args) > 1 else "r"
            if same and any(ch in mode for ch in "wax+") and not state["writing"]:
                raise PermissionError("only the test suite writes the result file")
        elif event in blocked:
            raise PermissionError(event + " is not available in the evaluation jail")
        elif event in foreign and from_solution():
            raise PermissionError(event + " is not available to the solution")

    _sys.addaudithook(hook)

    def fingerprint():
        marks = {}
        for cls in (
            _unittest.TestCase,
            _unittest.TestResult,
            _unittest.TestSuite,
            _unittest.TestLoader,
            _unittest.TextTestRunner,
            _unittest.TextTestResult,
            _unittest.TestProgram,
        ):
            for name, value in vars(cls).items():
                marks[cls.__name__ + "." + name] = id(value)
        marks["main"] = id(_unittest.main)
        marks["defaultTestLoader"] = id(_unittest.defaultTestLoader)
        return marks

    expected = fingerprint()

    def write(result):
        state["writing"] = True
        try:
            with open(path, "w", encoding="utf-8") as fh:
                _json.dump(result, fh)
        finally:
            state["writing"] = False

    def main():
        if fingerprint() != expected:
            write({"ran": 0, "failures": 0, "errors": 1, "fault": "unittest was altered"})
            raise SystemExit(1)
        result = _unittest.main(exit=False, verbosity=0).result
        write(
            {
                "ran": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped),
            }
        )
        raise SystemExit(0)

    return write, main


_mt_write, _mt_main = _mt_install()


def _refused(*args, **kwargs):
    raise OSError("network access is disabled in the evaluation jail")


def _mt_exit(*args, **kwargs):
    raise SystemExit(1)


def _mt_fault(reason):
    _mt_write({"ran": 0, "failures": 0, "errors": 1, "fault": reason})


class _RefusedSocket(_socket.socket):
    def __init__(self, *args, **kwargs):
        raise OSError("network access is disabled in the evaluation jail")


_socket.socket = _RefusedSocket
_socket.create_connection = _refused
_socket.create_server = _refused
_socket.socketpair = _refused
_socket.getaddrinfo = _refused
_os._exit = _mt_exit
_os.abort = _mt_exit
_signal.raise_signal = _mt_exit
_signal.alarm = _mt_exit
_signal.setitimer = _mt_exit
_signal.pthread_kill = _mt_exit

try:
    from solution import *  # noqa: E402,F403
except SystemExit:
    _mt_fault("solution called sys.exit during import")
    raise SystemExit(1)
except BaseException as _exc:
    _mt_fault("solution raised during import: " + repr(_exc))
    raise SystemExit(1)
"""

def module_sources(tests: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(c["module"]) for c in tests if isinstance(c, dict) and "module" in c)


def has_module_tests(gold: Any) -> bool:
    return (
        isinstance(gold, dict)
        and isinstance(gold.get("tests"), list | tuple)
        and bool(module_sources(gold["tests"]))
    )


def assemble_module(code: str, sources: Sequence[str]) -> tuple[str, str]:
    return extract_code(code), "\n\n".join([MODULE_PREAMBLE, *sources, MODULE_GUARD])


def _interpreter(root: Path | None) -> str:
    if root is not None:
        for candidate in (
            root / "venv" / "bin" / "python",
            root / "venv" / "Scripts" / "python.exe",
        ):
            if candidate.is_file():
                return str(candidate)
    return sys.executable


def _module_env(root: Path | None) -> dict[str, str]:
    if root is None:
        return {}
    return {"NLTK_DATA": str(root / "nltk_data"), "MPLCONFIGDIR": str(root / "mplconfig")}


FORBIDDEN_MODULES = frozenset(
    {"__main__", "builtins", "ctypes", "gc", "importlib", "_thread", "faulthandler", "resource"}
)
FORBIDDEN_NAMES = frozenset(
    {"exec", "eval", "__import__", "exit", "quit", "breakpoint", "__builtins__", "globals", "vars"}
)
FORBIDDEN_ATTRS = frozenset(
    {
        ("sys", "exit"),
        ("sys", "_getframe"),
        ("sys", "modules"),
        ("sys", "settrace"),
        ("sys", "setprofile"),
        ("sys", "addaudithook"),
        ("os", "_exit"),
        ("os", "abort"),
        ("os", "kill"),
        ("os", "killpg"),
        ("os", "environ"),
        ("os", "getenv"),
        ("os", "putenv"),
        ("os", "environb"),
        ("os", "execv"),
        ("os", "execve"),
        ("os", "execvp"),
        ("os", "fork"),
        ("signal", "signal"),
        ("signal", "raise_signal"),
        ("signal", "alarm"),
        ("signal", "setitimer"),
    }
)
FORBIDDEN_STRINGS = (
    "MT_RESULT_PATH",
    "/proc/",
    "result.json",
    "__closure__",
    "cell_contents",
    "_mt_",
)
FORBIDDEN_INTROSPECTION = frozenset(
    {
        "__closure__",
        "__code__",
        "__defaults__",
        "__kwdefaults__",
        "__globals__",
        "__subclasses__",
        "cell_contents",
        "f_back",
        "f_globals",
        "f_locals",
        "gi_frame",
        "tb_frame",
    }
)
SCREENED_ROOTS = frozenset({"os", "sys", "signal", "builtins"})


def screen_solution(code: str) -> str:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return ""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                aliases[alias.asname or root] = root
        elif isinstance(node, ast.ImportFrom) and node.module:
            aliases[node.module.split(".")[0]] = node.module.split(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    return f"imports {root}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_MODULES:
                return f"imports {root}"
            for alias in node.names:
                if (root, alias.name) in FORBIDDEN_ATTRS or alias.name in FORBIDDEN_NAMES:
                    return f"imports {root}.{alias.name}"
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_INTROSPECTION:
                return f"introspects through {node.attr}"
            if isinstance(node.value, ast.Name):
                root = aliases.get(node.value.id, node.value.id)
                if (root, node.attr) in FORBIDDEN_ATTRS:
                    return f"uses {root}.{node.attr}"
        elif isinstance(node, ast.Name) and node.id.startswith("_mt_"):
            return "touches the harness"
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in FORBIDDEN_NAMES:
                    return f"calls {func.id}"
                if func.id in ("getattr", "setattr", "delattr") and node.args:
                    first = node.args[0]
                    root = aliases.get(first.id, first.id) if isinstance(first, ast.Name) else ""
                    if root in SCREENED_ROOTS or root in aliases:
                        return f"reflects on {root}"
                    attr = node.args[1] if len(node.args) > 1 else None
                    if attr is not None and not (
                        isinstance(attr, ast.Constant) and isinstance(attr.value, str)
                    ):
                        return "reflects with a computed attribute name"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for needle in FORBIDDEN_STRINGS:
                if needle in node.value:
                    return f"names {needle}"
    return ""


def count_tests(sources: Sequence[str]) -> int:
    total = 0
    for source in sources:
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if item.name.startswith("test"):
                    total += 1
    return total


def verdict(outcome: Any, expected: int) -> float:
    if not isinstance(outcome, dict):
        return 0.0
    try:
        ran = int(outcome.get("ran", 0) or 0)
        failures = int(outcome.get("failures", 0) or 0)
        errors = int(outcome.get("errors", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    if ran <= 0:
        return 0.0
    if expected and ran < expected:
        return 0.0
    return 1.0 if failures + errors == 0 else 0.0


def _run_module(
    solution: str,
    suite: str,
    workdir: str,
    interpreter: str,
    env: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    with open(os.path.join(workdir, "solution.py"), "w", encoding="utf-8") as fh:
        fh.write(solution)
    path = os.path.join(workdir, "suite.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(suite)
    result_path = os.path.join(workdir, "result.json")
    merged = {**os.environ, **env, RESULT_ENV: result_path}
    try:
        proc = subprocess.run(  # noqa: S603
            [interpreter, path],
            cwd=workdir,
            env=merged,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "exit_code": None,
            "outcome": None,
            "stderr": f"exceeded {timeout_seconds}s running the suite",
        }
    outcome: dict[str, Any] | None
    try:
        with open(result_path, encoding="utf-8") as fh:
            outcome = json.load(fh)
    except (OSError, ValueError):
        outcome = None
    return {"exit_code": proc.returncode, "outcome": outcome, "stderr": proc.stderr[-2000:]}


def execute_module_rate(
    code: str,
    tests: Sequence[Any],
    *,
    limits: Limits | None = None,
    env_root: str | Path | None = None,
) -> float:
    sources = module_sources(tests)
    if not sources:
        return 0.0
    root = Path(env_root) if env_root else _ENV_ROOT
    bounded = limits or default_limits()
    workdir = tempfile.mkdtemp(prefix="mt-module-")
    try:
        solution, suite = assemble_module(code, sources)
        result = run_jailed(
            _run_module,
            solution,
            suite,
            workdir,
            _interpreter(root),
            _module_env(root),
            max(1.0, float(bounded.wall_seconds) - 1.0),
            limits=bounded,
            allow_unsandboxed=_ALLOW_UNSANDBOXED,
        )
    except UnsupportedPlatform as exc:
        raise ExecutionUnavailable(str(exc)) from exc
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if result.ok:
        return verdict(result.value.get("outcome"), count_tests(sources))
    if result.fault is Fault.INFRASTRUCTURE:
        raise ExecutionUnavailable(result.error)
    return 0.0
