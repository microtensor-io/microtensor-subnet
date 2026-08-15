from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


class Modality(str, Enum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    GENERATIVE = "generative"


class Decoding(str, Enum):
    GREEDY = "greedy"
    ARGMAX = "argmax"
    SEEDED = "seeded"


@dataclass(frozen=True, slots=True)
class Track:
    id: str
    modality: Modality
    metric: str
    decoding: Decoding
    emission_share: float
    work_unit: str
    enabled: bool = False

    def __post_init__(self) -> None:
        if self.enabled and self.emission_share <= 0.0:
            raise ValueError(f"track {self.id!r} is enabled but earns nothing")
        if not self.enabled and self.emission_share != 0.0:
            raise ValueError(f"track {self.id!r} is disabled but holds an emission share")


@dataclass(frozen=True, slots=True)
class HardwareClass:
    id: str
    max_size_bytes: int
    max_rss_bytes: int
    max_p95_ms: int
    reference: str
    device_profile: str = ""

    @property
    def ceilings(self) -> tuple[int, int, int]:
        return (self.max_size_bytes, self.max_rss_bytes, self.max_p95_ms)


_GB: Final[int] = 1024**3
_MB: Final[int] = 1024**2


TRACKS: Final[dict[str, Track]] = {
    t.id: t
    for t in (
        Track(
            id="code",
            modality=Modality.TEXT,
            metric="execution_pass_rate",
            decoding=Decoding.GREEDY,
            emission_share=0.30,
            work_unit="generated_tokens",
            enabled=True,
        ),
        Track(
            id="document",
            modality=Modality.TEXT,
            metric="extraction_f1",
            decoding=Decoding.GREEDY,
            emission_share=0.30,
            work_unit="generated_tokens",
            enabled=True,
        ),
        Track(
            id="analytics",
            modality=Modality.TEXT,
            metric="exact_match_numeric",
            decoding=Decoding.GREEDY,
            emission_share=0.20,
            work_unit="generated_tokens",
            enabled=True,
        ),
        Track(
            id="support",
            modality=Modality.TEXT,
            metric="rubric_f1_tool_calls",
            decoding=Decoding.GREEDY,
            emission_share=0.20,
            work_unit="generated_tokens",
            enabled=True,
        ),
        Track(
            id="detect",
            modality=Modality.VISION,
            metric="map_at_iou",
            decoding=Decoding.ARGMAX,
            emission_share=0.0,
            work_unit="images",
        ),
        Track(
            id="vqa",
            modality=Modality.VISION,
            metric="answer_accuracy_grounding_iou",
            decoding=Decoding.GREEDY,
            emission_share=0.0,
            work_unit="images",
        ),
        Track(
            id="speech",
            modality=Modality.AUDIO,
            metric="word_error_rate",
            decoding=Decoding.ARGMAX,
            emission_share=0.0,
            work_unit="audio_seconds",
        ),
        Track(
            id="video",
            modality=Modality.VIDEO,
            metric="temporal_localisation_f1",
            decoding=Decoding.ARGMAX,
            emission_share=0.0,
            work_unit="frames",
        ),
        Track(
            id="image-synth",
            modality=Modality.GENERATIVE,
            metric="reference_perceptual_distance",
            decoding=Decoding.SEEDED,
            emission_share=0.0,
            work_unit="sampling_steps",
        ),
    )
}


CLASSES: Final[dict[str, HardwareClass]] = {
    c.id: c
    for c in (
        HardwareClass(
            id="server-cpu",
            max_size_bytes=8 * _GB,
            max_rss_bytes=16 * _GB,
            max_p95_ms=400,
            reference="x86-64 server, no accelerator",
        ),
        HardwareClass(
            id="edge-gpu",
            max_size_bytes=(5 * _GB) // 2,
            max_rss_bytes=4 * _GB,
            max_p95_ms=120,
            reference="consumer or embedded GPU",
        ),
        HardwareClass(
            id="laptop",
            max_size_bytes=(3 * _GB) // 2,
            max_rss_bytes=3 * _GB,
            max_p95_ms=180,
            reference="developer workstation",
        ),
        HardwareClass(
            id="embedded",
            max_size_bytes=600 * _MB,
            max_rss_bytes=1 * _GB,
            max_p95_ms=300,
            reference="mobile SoC or NPU",
        ),
    )
}


def enabled_tracks() -> list[Track]:
    return [t for t in TRACKS.values() if t.enabled]


def get_track(track_id: str) -> Track:
    try:
        return TRACKS[track_id]
    except KeyError:
        raise KeyError(f"unknown track {track_id!r}; known: {sorted(TRACKS)}") from None


def get_class(class_id: str) -> HardwareClass:
    try:
        return CLASSES[class_id]
    except KeyError:
        raise KeyError(f"unknown class {class_id!r}; known: {sorted(CLASSES)}") from None


def is_competable(track_id: str, class_id: str) -> bool:
    return track_id in TRACKS and TRACKS[track_id].enabled and class_id in CLASSES


def competitions() -> list[tuple[str, str]]:
    return [(t.id, c.id) for t in enabled_tracks() for c in CLASSES.values()]


def validate_registry() -> None:
    live = enabled_tracks()
    if not live:
        raise ValueError("no track is enabled; the subnet would emit nothing")

    total = sum(t.emission_share for t in live)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"enabled emission shares sum to {total!r}, not 1.0 — "
            "emissions would be silently redistributed"
        )

    for cls in CLASSES.values():
        if not 0 < cls.max_size_bytes < cls.max_rss_bytes:
            raise ValueError(
                f"class {cls.id!r}: size ceiling must be positive and below the "
                "RSS ceiling"
            )
