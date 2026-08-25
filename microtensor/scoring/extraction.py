"""Entity extraction scoring: a strict entity schema and micro-F1.

Micro-F1 is aggregated across the whole evaluation set, not averaged per
sentence: true positives, false positives and false negatives are summed over
every document and the F1 computed once. That is what published biomedical NER
benchmarks report, so a miner's number compares directly; a per-sentence
average would be a different quantity that no leaderboard uses.

An entity is a (text, type) pair, matched exactly. A prediction is parsed under
a strict schema, and anything malformed contributes no entities for that
document, which scores zero the same way malformed code does.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

Entity = tuple[str, str]


def _entity(raw: Any) -> Entity | None:
    if not isinstance(raw, Mapping):
        return None
    text = raw.get("text")
    kind = raw.get("type")
    if not isinstance(text, str) or not isinstance(kind, str):
        return None
    text = text.strip()
    kind = kind.strip()
    if not text or not kind:
        return None
    return (text, kind)


def parse_entities(output: Any) -> set[Entity] | None:
    """The predicted entity set for one document, or None if malformed.

    Expects `{"entities": [{"text": ..., "type": ...}]}` or a bare list of the
    same. Strict: one malformed entity rejects the whole output, so a detector
    that emits garbage scores zero on that document rather than being partly
    rescued. An empty entity list is valid and distinct from malformed: a
    sentence can legitimately contain no entities.
    """
    if isinstance(output, str):
        # Engine output is text. It must be exactly one JSON document of the
        # declared shape; prose, markdown fences or trailing chatter are
        # malformed, which is the strictness the task states.
        try:
            output = json.loads(output)
        except ValueError:
            return None
    items: Any = output.get("entities") if isinstance(output, Mapping) else output
    if not isinstance(items, Sequence) or isinstance(items, (str | bytes)):
        return None

    entities: set[Entity] = set()
    for raw in items:
        entity = _entity(raw)
        if entity is None:
            return None
        entities.add(entity)
    return entities


def gold_entities(gold: Any) -> set[Entity]:
    """The ground-truth entity set for one document from its gold record.

    The gold arrives as the task sidecar, `{"entry_point": ..., "tests": [...]}`,
    each test carrying an `entities` list; they are unioned. Gold is trusted, so
    a malformed gold entry is skipped rather than zeroing the document.
    """
    tests: Any
    if isinstance(gold, Mapping):
        tests = gold.get("tests")
        if tests is None and "entities" in gold:
            tests = [gold]
    else:
        tests = gold
    if not isinstance(tests, Sequence) or isinstance(tests, (str | bytes)):
        return set()

    out: set[Entity] = set()
    for case in tests:
        if not isinstance(case, Mapping):
            continue
        for raw in case.get("entities", ()) or ():
            entity = _entity(raw)
            if entity is not None:
                out.add(entity)
    return out


def micro_f1(
    predictions: Sequence[set[Entity] | None], golds: Sequence[set[Entity]]
) -> float:
    """Entity-level micro-F1 over a set of documents.

    Counts are summed across documents, then one F1 is computed. A prediction of
    None is a malformed output for that document: it contributes no true or
    false positives, and every gold entity it missed counts against recall.
    """
    if len(predictions) != len(golds):
        raise ValueError("predictions and golds must align by document index")

    tp = fp = fn = 0
    for pred, gold in zip(predictions, golds, strict=True):
        found = pred or set()
        tp += len(found & gold)
        fp += len(found - gold)
        fn += len(gold - found)

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2.0 * precision * recall / (precision + recall)
