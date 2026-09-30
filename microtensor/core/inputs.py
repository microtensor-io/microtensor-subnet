from __future__ import annotations

from typing import Any, Final

INPUT_FORMATS: Final[dict[str, dict[str, Any]]] = {
    "raw-t1": {"modality": "text", "template": "raw"},
    "chat-t1": {"modality": "text", "template": "chat", "thinking": False},
    "decide-d1": {"modality": "text", "template": "chat", "thinking": False, "prompt": "d1"},
    "audio-a1": {
        "modality": "audio",
        "sample_rate": 16000,
        "channels": 1,
        "encoding": "pcm_s16le",
        "max_seconds": 30,
    },
    "image-i1": {"modality": "vision", "color": "rgb", "encoding": "png", "max_side": 1024},
    "video-v1": {
        "modality": "video",
        "color": "rgb",
        "fps": 2,
        "max_frames": 32,
        "max_side": 448,
    },
}


def input_format(name: str) -> dict[str, Any]:
    if name not in INPUT_FORMATS:
        raise ValueError(
            f"input format {name!r} is not published; expected one of {sorted(INPUT_FORMATS)}"
        )
    return {"id": name, **INPUT_FORMATS[name]}
