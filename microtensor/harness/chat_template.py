from __future__ import annotations

import json
from typing import Any


def render(template: str, messages: list[dict[str, str]]) -> str | None:
    if not template:
        return None
    try:
        import jinja2

        environment = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        environment.filters["tojson"] = json.dumps

        def _raise(message: str) -> Any:
            raise ValueError(message)

        environment.globals["raise_exception"] = _raise
        return str(
            environment.from_string(template).render(
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    except Exception:
        return None
