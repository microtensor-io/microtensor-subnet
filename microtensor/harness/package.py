from __future__ import annotations

import ast
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Final

from microtensor.core.hashing import digest_file, digest_tree
from microtensor.core.protocol import Role
from microtensor.core.system import SystemManifest

HARNESS_FILE: Final[str] = "harness.json"
HARNESS_FORMAT: Final[str] = "mt-harness/1"
TEXT_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".py", ".json", ".txt", ".md", ".yaml", ".yml", ".jinja", ".j2", ".toml", ".csv", ".tsv"}
)
OUTSIDE: Final[re.Pattern[str]] = re.compile(r"\b(?:https?|wss?|ftp|grpc)://[^\s\"'<>)]+")
BLOCKED_MODULES: Final[frozenset[str]] = frozenset(
    {
        "aiohttp",
        "anthropic",
        "asyncio",
        "boto3",
        "botocore",
        "ctypes",
        "ftplib",
        "google",
        "grpc",
        "http",
        "httpx",
        "multiprocessing",
        "openai",
        "requests",
        "smtplib",
        "socket",
        "ssl",
        "subprocess",
        "urllib",
        "urllib3",
        "websocket",
        "websockets",
    }
)
BLOCKED_CALLS: Final[frozenset[str]] = frozenset(
    {"__import__", "eval", "exec", "compile", "os.system", "os.popen", "importlib.import_module"}
)
SECTIONS: Final[tuple[str, ...]] = ("prompts", "context", "tools", "hooks", "templates")


class HarnessPackageError(ValueError):
    pass


def _paths(spec: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for section in SECTIONS:
        value = spec.get(section)
        if value is None:
            continue
        if isinstance(value, dict):
            found.extend(str(v) for v in value.values())
        elif isinstance(value, list):
            for entry in value:
                found.append(str(entry["path"]) if isinstance(entry, dict) else str(entry))
        else:
            raise HarnessPackageError(f"harness section {section!r} must be a mapping or a list")
    return found


def load(root: Path) -> dict[str, Any]:
    path = root / HARNESS_FILE
    if not path.is_file():
        raise HarnessPackageError(f"the harness package has no {HARNESS_FILE}")
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise HarnessPackageError(f"{HARNESS_FILE} is not valid JSON: {exc}") from exc
    if not isinstance(spec, dict) or spec.get("format") != HARNESS_FORMAT:
        raise HarnessPackageError(f"{HARNESS_FILE} must declare format {HARNESS_FORMAT!r}")
    if not spec.get("prompts"):
        raise HarnessPackageError("the harness declares no prompts")
    for rel in _paths(spec):
        parts = PurePosixPath(rel).parts
        if not rel or rel.startswith("/") or ".." in parts:
            raise HarnessPackageError(f"harness path {rel!r} leaves the package")
        if not (root / rel).is_file():
            raise HarnessPackageError(f"harness path {rel!r} is missing from the package")
    return spec


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return f"{head}.{node.attr}" if head else node.attr
    return ""


def _python_reason(rel: str, source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"harness file {rel} does not parse: {exc.msg}"
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
        for module in modules:
            if module.split(".", 1)[0] in BLOCKED_MODULES:
                return f"harness file {rel} imports {module}, which reaches outside the package"
        if isinstance(node, ast.Call) and _dotted(node.func) in BLOCKED_CALLS:
            return f"harness file {rel} calls {_dotted(node.func)}, which can reach outside it"
    return ""


def scan(root: Path) -> str:
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            return f"harness file {rel} is a link; the package must hold its own files"
        if path.is_dir():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            return f"harness file {rel} is not a permitted type ({sorted(TEXT_SUFFIXES)})"
        text = path.read_text(encoding="utf-8", errors="replace")
        found = OUTSIDE.search(text)
        if found:
            return f"harness file {rel} calls outside its package: {found.group(0)}"
        if path.suffix.lower() == ".py":
            reason = _python_reason(rel, text)
            if reason:
                return reason
    return ""


def package_reason(artifact: Path, system: SystemManifest) -> str:
    if system.harness is None or system.router is None:
        return "a full system declares a harness and a router"
    root = artifact / system.harness.path
    if not root.is_dir():
        return f"the harness package {system.harness.path!r} is missing from the artifact"
    if digest_tree(root) != system.harness.package_digest:
        return "the harness package does not match its declared digest"
    try:
        load(root)
    except HarnessPackageError as exc:
        return str(exc)
    reason = scan(root)
    if reason:
        return reason
    router = artifact / system.locate(Role.ROUTER)
    if not router.is_file():
        return f"the router {system.locate(Role.ROUTER)!r} is missing from the artifact"
    if digest_file(router) != system.router.artifact_digest:
        return "the router does not match its declared digest"
    return ""
