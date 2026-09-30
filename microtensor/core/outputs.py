from __future__ import annotations

import json
import re
from typing import Any, Final

from microtensor.core.tracks import STRUCTURED, Track

_UNIT: Final[dict[str, Any]] = {"type": "number", "minimum": 0, "maximum": 1}
_STRINGS: Final[dict[str, Any]] = {"type": "array", "items": {"type": "string"}}

SCHEMAS: Final[dict[str, dict[str, Any]]] = {
    "spans": {
        "type": "object",
        "properties": {"unsupported": _STRINGS},
        "required": ["unsupported"],
    },
    "entities": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}, "type": {"type": "string"}},
                    "required": ["text", "type"],
                },
            }
        },
        "required": ["entities"],
    },
    "tool_calls": {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
                    "required": ["name", "arguments"],
                },
            },
            "covers": _STRINGS,
        },
        "required": ["tool_calls"],
    },
    "boxes": {
        "type": "object",
        "properties": {
            "detections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category_id": {"type": "integer", "minimum": 0},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "score": _UNIT,
                    },
                    "required": ["category_id", "bbox", "score"],
                },
            }
        },
        "required": ["detections"],
    },
    "segments": {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "number", "minimum": 0},
                        "end": {"type": "number", "minimum": 0},
                        "label": {"type": "string"},
                        "score": _UNIT,
                    },
                    "required": ["start", "end", "label", "score"],
                },
            }
        },
        "required": ["segments"],
    },
    "invoice": {"type": "object"},
}

GRAMMARS: Final[dict[str, str]] = {
    "sql": "\n".join(
        (
            'root ::= ws (with sp)? select ws ";"? ws',
            'with ::= [Ww] [Ii] [Tt] [Hh] sp ident sp [Aa] [Ss] ws "(" ws select ws ")"',
            "select ::= [Ss] [Ee] [Ll] [Ee] [Cc] [Tt] sp inner",
            "inner ::= ([^;`()] | paren)*",
            'paren ::= "(" inner ")"',
            "ident ::= [A-Za-z_] [A-Za-z0-9_]*",
            r"sp ::= [ \t\n]+",
            r"ws ::= [ \t\n]*",
        )
    ),
}

PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "sql": re.compile(
        r"^\s*(with\s+\w+\s+as\s*\(\s*select\s[^;`]*\)\s+)?select\s[^;`]*;?\s*$",
        re.IGNORECASE | re.DOTALL,
    ),
}


def schema_of(track: Track) -> dict[str, Any] | None:
    return SCHEMAS.get(track.output_schema) if track.output_type == STRUCTURED else None


def grammar_for(track: Track) -> dict[str, str]:
    schema = schema_of(track)
    if schema is not None:
        return {"json_schema": json.dumps(schema, sort_keys=True)}
    if track.output_schema in GRAMMARS:
        return {"gbnf": GRAMMARS[track.output_schema]}
    return {}


def published(track: Track) -> dict[str, Any]:
    block: dict[str, Any] = {"type": track.output_type}
    if track.output_schema:
        block["schema"] = SCHEMAS.get(track.output_schema) or GRAMMARS.get(
            track.output_schema, track.output_schema
        )
    if track.confidence_output:
        block["confidence"] = track.confidence_output
    return block
