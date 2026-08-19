from __future__ import annotations

import logging
from typing import Any

from microtensor.tasks.corpus import Corpus, CorpusError, Task, corpus_digest

log = logging.getLogger("microtensor.validator")

DIGEST_MISMATCH = "the corpus the coordinator served does not hash to the digest it declared"


def parse(payload: dict[str, Any]) -> Corpus:
    """Build a corpus from what the coordinator served, and check it.

    The served digest is recomputed rather than trusted. A coordinator that
    edited a task after publishing its digest would be caught here, and a worker
    that skipped the check would be measuring against a task set nobody else
    agreed to.
    """
    tasks = tuple(
        Task(
            ref=str(row["ref"]),
            prompt=str(row["prompt"]),
            gold=row.get("gold"),
            partition=str(row["partition"]),
            inputs=dict(row.get("inputs") or {}),
            max_output_tokens=int(row.get("max_output_tokens", 512)),
        )
        for row in payload.get("tasks", ())
    )
    corpus = Corpus(
        track=str(payload["track"]), version=str(payload.get("version", "")), tasks=tasks
    )

    declared = str(payload.get("digest", ""))
    if declared and corpus.digest() != declared:
        raise CorpusError(f"{DIGEST_MISMATCH}: {corpus.digest()} against {declared}")

    return corpus


def fetch(client: Any) -> dict[str, Corpus]:
    """Take the round's corpus from the coordinator.

    Every worker measuring the same round measures the same tasks, which is what
    makes two workers disagreeing about a system mean something. A locally
    supplied corpus would make a disagreement unreadable: nobody could tell a
    wrong measurement from a different question.
    """
    index = client.corpus_index()
    tracks = list(index.get("tracks", ()))
    if not tracks:
        return {}

    corpora: dict[str, Corpus] = {}
    for track in tracks:
        served = client.corpus(track)
        if served is None:
            log.warning("the coordinator lists %s but serves no corpus for it", track)
            continue
        corpora[track] = parse(served)

    declared = str(index.get("digest", ""))
    if declared and corpora and corpus_digest(corpora) != declared:
        raise CorpusError(f"{DIGEST_MISMATCH}: {corpus_digest(corpora)} against {declared}")

    log.info("took %d corpora from the coordinator at %s", len(corpora), declared or "no digest")
    return corpora
