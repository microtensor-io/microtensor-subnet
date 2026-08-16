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
    classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.enabled and self.emission_share <= 0.0:
            raise ValueError(f"track {self.id!r} is enabled but earns nothing")
        if not self.enabled and self.emission_share != 0.0:
            raise ValueError(f"track {self.id!r} is disabled but holds an emission share")

    @property
    def live_classes(self) -> tuple[str, ...]:
        return self.classes or tuple(CLASSES)


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
            emission_share=1.0,
            work_unit="generated_tokens",
            enabled=True,
            classes=("laptop", "edge-gpu"),
        ),
        Track(
            id="document",
            modality=Modality.TEXT,
            metric="extraction_f1",
            decoding=Decoding.GREEDY,
            emission_share=0.0,
            work_unit="generated_tokens",
        ),
        Track(
            id="analytics",
            modality=Modality.TEXT,
            metric="exact_match_numeric",
            decoding=Decoding.GREEDY,
            emission_share=0.0,
            work_unit="generated_tokens",
        ),
        # BLOCKED: rubric_f1_tool_calls is a judged metric wearing a computed
        # name. Before this track is re-enabled, either the rubric becomes fully
        # mechanical (a checklist reducible to string/structure matching) or the
        # metric is renamed to what it actually computes.
        Track(
            id="support",
            modality=Modality.TEXT,
            metric="rubric_f1_tool_calls",
            decoding=Decoding.GREEDY,
            emission_share=0.0,
            work_unit="generated_tokens",
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
    if track_id not in TRACKS or not TRACKS[track_id].enabled:
        return False
    return class_id in TRACKS[track_id].live_classes and class_id in CLASSES


def competitions() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for track in enabled_tracks():
        out.extend((track.id, c) for c in track.live_classes if c in CLASSES)
    return out


def validate_registry() -> None:
    from microtensor.core.constants import CLASS_WEIGHTS

    live = enabled_tracks()
    if not live:
        raise ValueError("no track is enabled; the subnet would emit nothing")

    total = sum(t.emission_share for t in live)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"enabled emission shares sum to {total!r}, not 1.0 — "
            "emissions would be silently redistributed"
        )

    for track in TRACKS.values():
        unknown = [c for c in track.classes if c not in CLASSES]
        if unknown:
            raise ValueError(f"track {track.id!r} names unknown classes {unknown}")

    for track in live:
        if not track.live_classes:
            raise ValueError(f"track {track.id!r} is enabled but yields no competition")
        weighted = [c for c in track.live_classes if c in CLASS_WEIGHTS]
        if len(weighted) == len(track.live_classes):
            weight_total = sum(CLASS_WEIGHTS[c] for c in track.live_classes)
            if abs(weight_total - 1.0) > 1e-9:
                raise ValueError(
                    f"class weights over {track.id!r}'s live classes sum to "
                    f"{weight_total!r}, not 1.0"
                )

    for cls in CLASSES.values():
        if not 0 < cls.max_size_bytes < cls.max_rss_bytes:
            raise ValueError(
                f"class {cls.id!r}: size ceiling must be positive and below the "
                "RSS ceiling"
            )
