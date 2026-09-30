from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Mapping
from typing import Any, Final, cast

from microtensor.core.outputs import PATTERNS, schema_of
from microtensor.core.tracks import CHOICE, MEDIA, STRUCTURED, TEXT, Track

MEDIA_MAX_BYTES: Final[int] = 16 * 1024 * 1024
PROBABILITY_SLACK: Final[float] = 1e-4

_FENCE: Final[re.Pattern[str]] = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)```", re.DOTALL)

_MAGIC: Final[tuple[tuple[str, int, bytes], ...]] = (
    ("png", 0, b"\x89PNG\r\n\x1a\n"),
    ("jpeg", 0, b"\xff\xd8\xff"),
    ("flac", 0, b"fLaC"),
    ("webm", 0, b"\x1a\x45\xdf\xa3"),
    ("mp4", 4, b"ftyp"),
    ("wav", 8, b"WAVE"),
    ("webp", 8, b"WEBP"),
)

_TYPES: Final[dict[str, tuple[type, ...]]] = {
    "object": (dict,),
    "array": (list, tuple),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
}


def parse(output: Any) -> Any:
    if isinstance(output, dict | list):
        return output
    if not isinstance(output, str):
        return None
    blocks = _FENCE.findall(output)
    candidate = max(blocks, key=len).strip() if blocks else output.strip()
    try:
        return json.loads(candidate)
    except ValueError:
        return None


def conforms(value: Any, schema: Mapping[str, Any], where: str = "$") -> str:
    kind = schema.get("type")
    if kind:
        allowed = _TYPES[str(kind)]
        if not isinstance(value, allowed) or (kind != "boolean" and isinstance(value, bool)):
            return f"{where} is not {kind}"
        if kind == "number" and not math.isfinite(float(cast(float, value))):
            return f"{where} is not a finite number"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where} is not one of the declared values"
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{where} is below {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{where} is above {schema['maximum']}"
    if isinstance(value, dict):
        for key in schema.get("required", ()):
            if key not in value:
                return f"{where}.{key} is missing"
        for key, sub in dict(schema.get("properties", {})).items():
            if key in value:
                reason = conforms(value[key], sub, f"{where}.{key}")
                if reason:
                    return reason
    if isinstance(value, list | tuple):
        if len(value) < int(schema.get("minItems", 0)):
            return f"{where} has too few items"
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            return f"{where} has too many items"
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                reason = conforms(item, items, f"{where}[{index}]")
                if reason:
                    return reason
    return ""


def media_format(data: bytes) -> str:
    for name, offset, magic in _MAGIC:
        if data[offset : offset + len(magic)] == magic:
            return name
    return ""


def check_media(output: Any, max_bytes: int = MEDIA_MAX_BYTES) -> str:
    data: bytes
    declared = ""
    if isinstance(output, bytes | bytearray):
        data = bytes(output)
    elif isinstance(output, Mapping) and isinstance(output.get("data"), str):
        declared = str(output.get("format", "")).lower()
        try:
            data = base64.b64decode(str(output["data"]), validate=True)
        except ValueError:
            return "media is not valid base64"
    else:
        return "media output is neither bytes nor a base64 document"
    if not data:
        return "media output is empty"
    if len(data) > max_bytes:
        return f"media output is {len(data)} bytes, over the {max_bytes} byte limit"
    found = media_format(data)
    if not found:
        return "media output is not a recognised format"
    if declared and declared != found:
        return f"media declares {declared} but is {found}"
    return ""


def _choice(output: Any) -> str:
    answers = output.get("answers") if isinstance(output, Mapping) else None
    if not isinstance(answers, Mapping) or not answers:
        return "no answers"
    for name, answer in answers.items():
        probabilities = answer.get("probabilities") if isinstance(answer, Mapping) else None
        if not isinstance(probabilities, Mapping) or not probabilities:
            return f"answer {name!r} carries no probabilities"
        values = [float(v) for v in probabilities.values()]
        if any(not 0.0 <= v <= 1.0 for v in values):
            return f"answer {name!r} has a probability outside [0, 1]"
        if abs(math.fsum(values) - 1.0) > PROBABILITY_SLACK:
            return f"answer {name!r} probabilities do not sum to one"
    return ""


def _confident(response: Any) -> bool:
    return bool(getattr(response, "logprobs", ())) or bool(getattr(response, "confidence", ()))


def check(track: Track, response: Any) -> str:
    kind = track.output_type
    if not kind:
        return ""
    output = getattr(response, "output", None)
    if kind == CHOICE:
        return _choice(output)
    if kind == MEDIA:
        return check_media(output)
    if kind == STRUCTURED:
        parsed = parse(output)
        if parsed is None:
            return "output is not valid JSON"
        schema = schema_of(track)
        reason = conforms(parsed, schema) if schema is not None else ""
        if reason:
            return reason
    if kind == TEXT:
        if not isinstance(output, str) or not output.strip():
            return "output is empty"
        pattern = PATTERNS.get(track.output_schema)
        if pattern is not None and not pattern.match(output):
            return f"output does not match the {track.output_schema} format"
    if not _confident(response):
        return "output carries no confidence"
    return ""
