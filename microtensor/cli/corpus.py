from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from microtensor.cli.common import fail
from microtensor.core.constants import CORPUS_VERSION, TASKS_PER_ROUND
from microtensor.core.tracks import enabled_tracks, get_track
from microtensor.tasks.corpus import FIXED, ROTATING, Corpus, CorpusError, load_all, load_corpus
from microtensor.tasks.selection import partition_sizes

SEED_ROTATING = 240
SEED_FIXED = 100


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("corpus", help="check, describe and seed task corpora")
    inner = parser.add_subparsers(dest="action", required=True)

    check = inner.add_parser("check", help="verify every corpus a validator would load")
    check.add_argument("directory", type=Path)
    check.add_argument("--tasks-per-round", type=int, default=TASKS_PER_ROUND)
    check.set_defaults(handler=_check)

    stats = inner.add_parser("stats", help="counts and digests per track")
    stats.add_argument("directory", type=Path)
    stats.set_defaults(handler=_stats)

    seed = inner.add_parser(
        "seed", help="write a synthetic smoke-test corpus (NOT for production use)"
    )
    seed.add_argument("directory", type=Path)
    seed.add_argument("--rotating", type=int, default=SEED_ROTATING)
    seed.add_argument("--fixed", type=int, default=SEED_FIXED)
    seed.add_argument("--force", action="store_true")
    seed.set_defaults(handler=_seed)


def _digest(corpus: Corpus) -> str:
    material = "\n".join(sorted(f"{t.ref}:{t.partition}" for t in corpus))
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def _check(args: argparse.Namespace) -> int:
    try:
        corpora = load_all(args.directory)
    except CorpusError as exc:
        return fail(str(exc))

    want_rotating, want_fixed = partition_sizes(args.tasks_per_round)
    open_tracks = {t.id for t in enabled_tracks()}
    problems: list[str] = []

    print(f"{'track':<14}{'total':>8}{'rotating':>10}{'fixed':>8}  {'digest':<18}status")
    for name in sorted(corpora):
        corpus = corpora[name]
        notes: list[str] = []
        if name not in open_tracks:
            notes.append("track not open")
        if len(corpus.rotating) < want_rotating:
            notes.append(f"rotating short of {want_rotating}")
        if len(corpus.fixed) < want_fixed:
            notes.append(f"fixed short of {want_fixed}")

        status = "ok" if not notes else "; ".join(notes)
        problems.extend(f"{name}: {n}" for n in notes)
        print(
            f"{name:<14}{len(corpus):>8}{len(corpus.rotating):>10}"
            f"{len(corpus.fixed):>8}  {_digest(corpus):<18}{status}"
        )

    missing = sorted(open_tracks - set(corpora))
    if missing:
        print(f"\nno corpus for open tracks: {', '.join(missing)} — they will not be scored")

    if problems:
        print(f"\n{len(problems)} problem(s) found")
        return 1
    print("\nevery corpus can fill a round")
    return 0


def _stats(args: argparse.Namespace) -> int:
    try:
        corpora = load_all(args.directory)
    except CorpusError as exc:
        return fail(str(exc))

    for name in sorted(corpora):
        corpus = corpora[name]
        lengths = [len(t.prompt) for t in corpus]
        print(f"{name}  ({get_track(name).metric})")
        print(f"  tasks        {len(corpus)}  ({len(corpus.rotating)} rotating, "
              f"{len(corpus.fixed)} fixed)")
        print(f"  prompt chars min {min(lengths)}, median {sorted(lengths)[len(lengths) // 2]}, "
              f"max {max(lengths)}")
        print(f"  digest       {_digest(corpus)}")
        print(f"  version      {corpus.version}")
    return 0


def _seed(args: argparse.Namespace) -> int:
    directory: Path = args.directory
    directory.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for track in enabled_tracks():
        path = directory / f"{track.id}.jsonl"
        if path.exists() and not args.force:
            print(f"skipping {path.name}: already exists (pass --force to overwrite)")
            continue

        rows = [_synthetic(track.id, i, ROTATING) for i in range(args.rotating)]
        rows += [_synthetic(track.id, 100_000 + i, FIXED) for i in range(args.fixed)]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        written.append(path.name)

    if not written:
        print("nothing written")
        return 0

    for name in written:
        load_corpus(directory / name, name.removesuffix(".jsonl"), CORPUS_VERSION)

    print(f"wrote {len(written)} corpora to {directory}")
    print(
        "\nThese are synthetic smoke-test tasks. They exercise the round loop end to end "
        "and measure nothing real.\nReplace them before validating on mainnet — a public "
        "or guessable corpus measures memorisation, not capability."
    )
    return 0


def _synthetic(track: str, index: int, partition: str) -> dict[str, object]:
    nonce = hashlib.sha256(f"{track}:{index}:{partition}".encode()).hexdigest()[:12]
    return {
        "ref": f"{track}-{partition[:3]}-{index:06d}",
        "prompt": f"[synthetic {track} task {index}] echo the token {nonce}",
        "gold": nonce,
        "partition": partition,
        "max_output_tokens": 64,
    }
