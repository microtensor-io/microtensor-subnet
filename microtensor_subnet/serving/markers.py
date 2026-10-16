from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

RAW: Final[str] = "raw"
PAIRS: Final[str] = "pairs"
JSON_SCALARS: Final[str] = "json"
TEXT_SCALARS: Final[str] = "text"


class MarkerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Family:
    name: str
    think_open: str = ""
    think_close: str = ""
    content_prefix: str = ""
    before_calls: str = ""
    calls_open: str = ""
    call: str = ""
    call_separator: str = ""
    calls_close: str = ""
    arguments: str = RAW
    pair: str = ""
    scalars: str = JSON_SCALARS
    call_id: str = ""
    end: str = ""
    end_tools: str = ""


FAMILIES: Final[dict[str, Family]] = {
    "deepseek": Family(
        name="deepseek",
        think_close="</think>",
        calls_open="<\uff5ctool\u2581calls\u2581begin\uff5c>",
        call="<\uff5ctool\u2581call\u2581begin\uff5c>{name}<\uff5ctool\u2581sep\uff5c>{arguments}<\uff5ctool\u2581call\u2581end\uff5c>",
        calls_close="<\uff5ctool\u2581calls\u2581end\uff5c>",
        end="<\uff5cend\u2581of\u2581sentence\uff5c>",
        end_tools="<\uff5cend\u2581of\u2581sentence\uff5c>",
    ),
    "glm": Family(
        name="glm",
        think_open="\n<think>",
        think_close="</think>",
        content_prefix="\n",
        call="\n<tool_call>{name}\n{arguments}</tool_call>",
        arguments=PAIRS,
        pair="<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n",
        end="<|user|>",
        end_tools="<|observation|>",
    ),
    "kimi": Family(
        name="kimi",
        think_open="<think>",
        think_close="</think>",
        calls_open="<|tool_calls_section_begin|>",
        call="<|tool_call_begin|>{id}<|tool_call_argument_begin|>{arguments}<|tool_call_end|>",
        calls_close="<|tool_calls_section_end|>",
        call_id="functions.{name}:{index}",
        end="<|im_end|>",
        end_tools="<|im_end|>",
    ),
    "qwen": Family(
        name="qwen",
        think_open="<think>\n",
        think_close="\n</think>\n\n",
        before_calls="\n",
        call='<tool_call>\n{{"name": {name_json}, "arguments": {arguments}}}\n</tool_call>',
        call_separator="\n",
        end="<|im_end|>",
        end_tools="<|im_end|>",
    ),
    "qwen_coder": Family(
        name="qwen_coder",
        before_calls="\n\n",
        call="<tool_call>\n<function={name}>\n{arguments}</function>\n</tool_call>",
        call_separator="\n",
        arguments=PAIRS,
        pair="<parameter={key}>\n{value}\n</parameter>\n",
        scalars=TEXT_SCALARS,
        end="<|im_end|>",
        end_tools="<|im_end|>",
    ),
    "hermes": Family(
        name="hermes",
        before_calls="\n",
        call='<tool_call>\n{{"name": {name_json}, "arguments": {arguments}}}\n</tool_call>',
        call_separator="\n",
        end="<|im_end|>",
        end_tools="<|im_end|>",
    ),
}


def family_of(family: str | Family) -> Family:
    if isinstance(family, Family):
        return family
    found = FAMILIES.get(str(family).strip().lower())
    if found is None:
        raise MarkerError(f"no raw output format is known for the family {family!r}")
    return found


def _scalar(value: Any, style: str) -> str:
    if isinstance(value, str):
        return value
    if style == TEXT_SCALARS and not isinstance(value, (Mapping, list, tuple)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _arguments(family: Family, raw: str) -> str:
    if family.arguments == RAW:
        return raw
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except ValueError as exc:
        raise MarkerError("tool call arguments are not a json object") from exc
    if not isinstance(parsed, Mapping):
        raise MarkerError("tool call arguments are not a json object")
    return "".join(
        family.pair.format(key=key, value=_scalar(value, family.scalars))
        for key, value in parsed.items()
    )


def _call(family: Family, call: Mapping[str, Any], position: int) -> str:
    function = call.get("function")
    named: Mapping[str, Any] = function if isinstance(function, Mapping) else call
    name = str(named.get("name") or "")
    raw = named.get("arguments")
    arguments = raw if isinstance(raw, str) else json.dumps(raw or {}, ensure_ascii=False)
    index = call.get("index") if isinstance(call.get("index"), int) else position
    ident = str(call.get("id") or "")
    if family.call_id and not ident.startswith(family.call_id.split("{", 1)[0]):
        ident = family.call_id.format(name=name, index=index)
    return family.call.format(
        name=name,
        name_json=json.dumps(name, ensure_ascii=False),
        id=ident,
        index=index,
        arguments=_arguments(family, arguments),
    )


def reassemble(
    family: str | Family,
    reasoning: str,
    content: str,
    tool_calls: Sequence[Mapping[str, Any]],
) -> str:
    found = family_of(family)
    parts: list[str] = []
    if reasoning:
        parts.append(found.think_open + reasoning + found.think_close)
    if content:
        parts.append(found.content_prefix + content)
    if tool_calls:
        if content:
            parts.append(found.before_calls)
        rendered = [_call(found, call, position) for position, call in enumerate(tool_calls)]
        parts.append(found.calls_open + found.call_separator.join(rendered) + found.calls_close)
    return "".join(parts)
