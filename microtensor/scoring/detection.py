"""Object-detection scoring: a validated box schema and COCO mAP.

mAP is computed by pycocotools' COCOeval unmodified, so a miner's number is the
one published detector benchmarks report rather than a reimplementation that
would differ in the edge cases COCOeval already settled: tie-breaking on score,
the recall grid, area ranges, the 101-point interpolation. The whole value of
this arena is that comparability, and it survives only if the metric is theirs.

A prediction is parsed under a strict schema. Anything malformed contributes no
detections for that image, which mirrors how malformed code scores zero: a
broken output earns nothing rather than being guessed at.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

BOX_FIELDS = 4


@dataclass(frozen=True, slots=True)
class Detection:
    """One predicted box, in COCO [x, y, width, height] with a class and score."""

    category_id: int
    bbox: tuple[float, float, float, float]
    score: float


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int | float)):
        return float(value)
    return None


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    """A COCO xywh box, or None if it is not one.

    Width and height must be positive; a zero-area box is not a detection and
    would make COCOeval's IoU undefined, so it is rejected rather than clamped.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str | bytes)):
        return None
    if len(value) != BOX_FIELDS:
        return None
    coords = [c for c in (_number(v) for v in value) if c is not None]
    if len(coords) != BOX_FIELDS:
        return None
    x, y, w, h = coords
    if w <= 0.0 or h <= 0.0 or x < 0.0 or y < 0.0:
        return None
    return (x, y, w, h)


def _one(raw: Any) -> Detection | None:
    if not isinstance(raw, Mapping):
        return None
    category = raw.get("category_id")
    if not isinstance(category, int) or isinstance(category, bool) or category < 0:
        return None
    bbox = _bbox(raw.get("bbox"))
    if bbox is None:
        return None
    score = _number(raw.get("score"))
    if score is None or not 0.0 <= score <= 1.0:
        return None
    return Detection(category_id=category, bbox=bbox, score=score)


def parse_detections(output: Any) -> list[Detection]:
    """Validated detections for one image, or an empty list if malformed.

    Accepts either a bare list of detections or an object carrying a
    `detections` list. Strict on purpose: one malformed box rejects the whole
    output, so a miner whose detector emits garbage scores zero on that image
    rather than having a parser rescue a fraction of it.
    """
    items: Any = output.get("detections") if isinstance(output, Mapping) else output
    if not isinstance(items, Sequence) or isinstance(items, (str | bytes)):
        return []

    parsed: list[Detection] = []
    for raw in items:
        detection = _one(raw)
        if detection is None:
            return []
        parsed.append(detection)
    return parsed


def _gold_boxes(gold: Any) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Ground-truth (category, box) pairs for one image from its gold record.

    Gold carries `boxes`, each `{category_id, bbox}`. A malformed gold entry is
    a corpus defect, not a miner one, so it is skipped rather than zeroing the
    image; the corpus is trusted, the submission is not.
    """
    if not isinstance(gold, Mapping):
        return []
    boxes = gold.get("boxes")
    if not isinstance(boxes, Sequence):
        return []
    out: list[tuple[int, tuple[float, float, float, float]]] = []
    for raw in boxes:
        if not isinstance(raw, Mapping):
            continue
        category = raw.get("category_id")
        bbox = _bbox(raw.get("bbox"))
        if isinstance(category, int) and not isinstance(category, bool) and bbox is not None:
            out.append((int(category), bbox))
    return out


def category_ids(golds: Sequence[Any]) -> list[int]:
    found: set[int] = set()
    for gold in golds:
        for category, _ in _gold_boxes(gold):
            found.add(category)
    return sorted(found)


def coco_map(
    predictions: Sequence[Sequence[Detection]],
    golds: Sequence[Any],
    *,
    categories: Sequence[int] | None = None,
) -> float:
    """mAP@[.5:.95] over a set of images, via COCOeval unmodified.

    `predictions[i]` are the detections for image i, `golds[i]` its gold record.
    The two sequences are aligned by index; an image with no gold boxes still
    counts, since a detector that fires on an empty image should be penalised.
    Returns 0.0 when there is nothing to score.
    """
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if len(predictions) != len(golds):
        raise ValueError("predictions and golds must align by image index")
    if not golds:
        return 0.0

    cats = list(categories) if categories is not None else category_ids(golds)
    if not cats:
        return 0.0

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    ann_id = 1

    for image_id, (gold, preds) in enumerate(zip(golds, predictions, strict=True), start=1):
        images.append({"id": image_id})
        for category, bbox in _gold_boxes(gold):
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(category),
                    "bbox": [float(v) for v in bbox],
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        for detection in preds:
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(detection.category_id),
                    "bbox": [float(v) for v in detection.bbox],
                    "score": float(detection.score),
                }
            )

    if not annotations:
        return 0.0

    truth = COCO()
    truth.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": c} for c in cats],
    }
    truth.createIndex()

    if not results:
        return 0.0
    detected = truth.loadRes(results)

    evaluation = COCOeval(truth, detected, iouType="bbox")
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()

    mean_ap = float(evaluation.stats[0])
    if mean_ap < 0.0 or np.isnan(mean_ap):
        return 0.0
    return mean_ap
