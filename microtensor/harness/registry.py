from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Final

from microtensor.core.protocol import ArtifactFormat
from microtensor.harness.contract import Engine, EngineError, EngineInfo

log = logging.getLogger("microtensor.harness")

EngineFactory = Callable[[], Engine]

_FACTORIES: dict[ArtifactFormat, EngineFactory] = {}
_INFO: dict[ArtifactFormat, EngineInfo] = {}


class EngineUnavailable(EngineError):
    pass


def register(
    fmt: ArtifactFormat,
    factory: EngineFactory,
    info: EngineInfo | None = None,
    *,
    replace: bool = False,
) -> None:
    if fmt in _FACTORIES and not replace:
        raise EngineError(f"an engine is already registered for {fmt.value}")
    _FACTORIES[fmt] = factory
    if info is not None:
        _INFO[fmt] = info


def unregister(fmt: ArtifactFormat) -> None:
    _FACTORIES.pop(fmt, None)
    _INFO.pop(fmt, None)


def clear() -> None:
    _FACTORIES.clear()
    _INFO.clear()


def has(fmt: ArtifactFormat) -> bool:
    return fmt in _FACTORIES


def available() -> tuple[ArtifactFormat, ...]:
    return tuple(sorted(_FACTORIES, key=lambda f: f.value))


# The distribution behind each engine. Named here because the import name and
# the package name differ, and the package name is what an operator installs
# and what a bug report needs to quote.
RUNTIME_PACKAGES: Final[dict[str, str]] = {
    "llama-cpp": "llama-cpp-python",
    "onnxruntime": "onnxruntime",
}


def runtime_version(name: str) -> str:
    """The installed version of the library behind an engine, or nothing.

    Absent is a real answer: the reference engine has no library behind it,
    and an engine can be registered on a host where its package is not
    importable. Reporting an empty string beats reporting a stale literal.
    """
    package = RUNTIME_PACKAGES.get(name)
    if package is None:
        return ""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return ""
    try:
        return str(version(package))
    except PackageNotFoundError:
        return ""


def describe(fmt: ArtifactFormat) -> EngineInfo | None:
    found = _INFO.get(fmt)
    if found is None:
        return None
    return replace(found, runtime=runtime_version(found.name))


def engine_for(fmt: ArtifactFormat) -> Engine:
    factory = _FACTORIES.get(fmt)
    if factory is None:
        raise EngineUnavailable(
            f"no engine is registered for {fmt.value}; registered: "
            f"{[f.value for f in available()]}"
        )
    return factory()


ENGINE_PLUGIN_ENV = "MT_ENGINES"


def _load_plugins() -> None:
    import importlib
    import os

    for spec in os.environ.get(ENGINE_PLUGIN_ENV, "").split(","):
        name = spec.strip()
        if not name:
            continue
        try:
            module = importlib.import_module(name)
            factory, info = module.load()
        except (ImportError, AttributeError, TypeError) as exc:
            log.warning("engine plugin %r could not be loaded: %s", name, exc)
            continue
        register(info.format, factory, info, replace=True)
        log.info("registered engine plugin %s for %s", name, info.format.value)


def load_builtin() -> tuple[ArtifactFormat, ...]:
    from microtensor.harness.engines import BUILTIN

    for fmt, loader in BUILTIN:
        if has(fmt):
            continue
        try:
            factory, info = loader()
        except ImportError as exc:
            log.info("engine for %s is unavailable: %s", fmt.value, exc)
            continue
        register(fmt, factory, info)

    _load_plugins()
    return available()
