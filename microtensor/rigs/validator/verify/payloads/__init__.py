from __future__ import annotations

from importlib import resources


def source(name: str) -> bytes:
    return resources.files(__package__).joinpath(name).read_bytes()


def text(name: str) -> str:
    return source(name).decode("utf-8")
