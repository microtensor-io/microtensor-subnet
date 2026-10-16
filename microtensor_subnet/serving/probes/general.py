from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from microtensor_subnet.core.hashing import DIGEST_PREFIX
from microtensor_subnet.serving.probes.ask import Chatted

if TYPE_CHECKING:
    from microtensor_subnet.serving.verifier import ModelVerifier
    from microtensor_subnet.serving.verify import Calibration

CHARS_PER_TOKEN: Final[int] = 3
STOP: Final[str] = "stop"
TOOL_CALLS: Final[str] = "tool_calls"
LENGTH: Final[str] = "length"
FINISHES: Final[frozenset[str]] = frozenset({STOP, TOOL_CALLS, LENGTH})


@dataclass(frozen=True, slots=True)
class GeneralItem:
    item_id: str
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] = ()
    max_tokens: int = 256

    @property
    def needs(self) -> int:
        text = json.dumps([list(self.messages), list(self.tools)], ensure_ascii=False)
        return len(text) // CHARS_PER_TOKEN + self.max_tokens

    @property
    def names(self) -> frozenset[str]:
        return frozenset(str(_named(tool)[0]) for tool in self.tools)

    def request(self) -> dict[str, Any]:
        found: dict[str, Any] = {
            "messages": copy.deepcopy(list(self.messages)),
            "tools": copy.deepcopy(list(self.tools)),
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        if self.tools:
            found["tool_choice"] = "auto"
        return found


@dataclass(frozen=True, slots=True)
class Target:
    model: str
    artifact_digest: str
    items: tuple[GeneralItem, ...]
    verifier: ModelVerifier | None = None
    calibration: Calibration | None = None

    @property
    def digest(self) -> str:
        return self.artifact_digest.strip().lower().removeprefix(DIGEST_PREFIX)


WEATHER: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
LOCAL_TIME: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": "get_local_time",
        "description": "Current local time in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}
TOOLS: Final[tuple[dict[str, Any], ...]] = (WEATHER, LOCAL_TIME)

SENTENCES: Final[tuple[str, ...]] = (
    "The north loading bay received two pallets of copper wire before the morning shift.",
    "A forklift battery was swapped and logged against the maintenance ticket.",
    "Inventory counts for aisle seven matched the ledger to the last carton.",
    "The night guard reported the east gate closed and the alarm armed at ten.",
    "Three returned parcels were opened, inspected and restocked on shelf twelve.",
    "The heating in the office annex was set back two degrees for the weekend.",
    "A delivery driver asked for directions to the cold store and was escorted.",
    "Packing tape and stretch film were reordered after the weekly stock check.",
)
NEEDLE_CODE: Final[str] = "4817-ALDER"


def notes(entries: int) -> str:
    middle = entries // 2
    lines: list[str] = []
    for index in range(entries):
        picked = [SENTENCES[(index + step) % len(SENTENCES)] for step in (0, 3, 5)]
        line = f"Entry {index + 1}. " + " ".join(picked)
        if index == middle:
            line += f" The archive room access code was changed to {NEEDLE_CODE}."
        lines.append(line)
    return "\n".join(lines)


def _long(item_id: str, entries: int) -> GeneralItem:
    return GeneralItem(
        item_id=item_id,
        messages=(
            {"role": "system", "content": "You answer questions from the notes you are given."},
            {
                "role": "user",
                "content": "Warehouse log:\n\n"
                + notes(entries)
                + "\n\nWhat is the current access code of the archive room? "
                "Answer with the code only.",
            },
        ),
        max_tokens=64,
    )


POOL: Final[tuple[GeneralItem, ...]] = (
    GeneralItem(
        item_id="chat-plain",
        messages=(
            {
                "role": "user",
                "content": "In two sentences, explain why the sky looks blue during the day.",
            },
        ),
        max_tokens=160,
    ),
    GeneralItem(
        item_id="chat-json",
        messages=(
            {"role": "system", "content": "Reply with one JSON object and nothing else."},
            {
                "role": "user",
                "content": "Give the capital and the official language of Portugal as JSON "
                "with the keys capital and language.",
            },
        ),
        max_tokens=96,
    ),
    GeneralItem(
        item_id="tool-one",
        messages=(
            {"role": "user", "content": "What is the weather in Lisbon right now? Use celsius."},
        ),
        tools=TOOLS,
    ),
    GeneralItem(
        item_id="tool-two",
        messages=(
            {"role": "user", "content": "Is it warmer in Lisbon or in Oslo right now?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Lisbon"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"city": "Lisbon", "temperature": 21, "unit": "celsius"}',
            },
        ),
        tools=TOOLS,
    ),
    _long("long-8k", 64),
    _long("long-32k", 256),
)


def items_for(entry: Mapping[str, Any]) -> list[GeneralItem]:
    tools = bool(entry.get("tool_calling"))
    context = int(entry.get("context") or 0)
    return [
        item
        for item in POOL
        if (tools or not item.tools) and (context <= 0 or item.needs <= context)
    ]


def verifiable(entry: Mapping[str, Any]) -> bool:
    found = entry.get("verifiable_on")
    if isinstance(found, str):
        return bool(found.strip())
    return bool(found)


def _named(call: Mapping[str, Any]) -> tuple[str, Any]:
    function = call.get("function")
    named: Mapping[str, Any] = function if isinstance(function, Mapping) else call
    return str(named.get("name") or ""), named.get("arguments")


def format_failure(item: GeneralItem, answered: Chatted) -> str:
    finish = answered.finish_reason
    if finish not in FINISHES:
        return f"the finish reason {finish!r} is not one a chat completion ends with"
    calls = answered.tool_calls
    if finish == TOOL_CALLS and not calls:
        return "the answer ended for tool calls and carried none"
    if not calls:
        return ""
    if not item.tools:
        return "the answer called a tool the request never offered"
    if finish == STOP:
        return "the answer carried tool calls and ended with stop"
    offered = item.names
    for call in calls:
        name, arguments = _named(call)
        if name not in offered:
            return f"the answer called {name!r}, which the request did not offer"
        if finish == LENGTH:
            continue
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError:
            return f"the arguments of {name!r} do not parse as json"
        if not isinstance(parsed, Mapping):
            return f"the arguments of {name!r} are not a json object"
    return ""
