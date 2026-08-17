from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from microtensor.cli.common import fail
from microtensor.core.constants import CORPUS_VERSION, TASKS_PER_ROUND
from microtensor.core.tracks import enabled_tracks, get_track
from microtensor.tasks import contamination, generator
from microtensor.tasks.corpus import FIXED, ROTATING, Corpus, CorpusError, load_all, load_corpus
from microtensor.tasks.selection import partition_sizes

SEED_ROTATING = 240
SEED_FIXED = 100


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("corpus", help="generate, check and seed task corpora")
    inner = parser.add_subparsers(dest="action", required=True)

    gen = inner.add_parser("generate", help="generate a corpus with hidden tests")
    gen.add_argument("track")
    gen.add_argument("--rotating", type=int, default=2000)
    gen.add_argument("--fixed", type=int, default=300)
    gen.add_argument("--train", type=int, default=1000)
    gen.add_argument("--seed", required=True, help="hex or phrase; recorded in the manifest")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--force", action="store_true")
    gen.set_defaults(handler=_generate)

    check = inner.add_parser("check", help="verify every corpus a validator would load")
    check.add_argument("directory", type=Path)
    check.add_argument("--tasks-per-round", type=int, default=TASKS_PER_ROUND)
    check.add_argument(
        "--skip-contamination",
        action="store_true",
        help="smoke-test path only; skips the public-benchmark and partition scans",
    )
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


def _generate(args: argparse.Namespace) -> int:
    if args.track != "code":
        return fail(f"only the code track has a generator today, not {args.track!r}")

    try:
        bundle = generator.generate(
            rotating=args.rotating, fixed=args.fixed, train=args.train, seed=args.seed
        )
        digests = generator.write_bundle(bundle, args.out, force=args.force)
    except generator.GenerationError as exc:
        return fail(str(exc))

    for name, digest in sorted(digests.items()):
        print(f"{name:<22} {digest}")
    print(
        f"\nwrote {len(bundle.tasks)} evaluation tasks and {len(bundle.train)} train tasks "
        f"to {args.out}"
    )
    print(
        "code.jsonl + code.tests.jsonl are the validator bundle and must never be "
        "published.\ncode.train.jsonl is the public split; run "
        "scripts/generate_reference_completions.py against it before release."
    )

    problems = generator.verify_bundle(args.out)
    if problems:
        for problem in problems[:10]:
            print(f"  PROBLEM {problem}")
        return 1
    print("bundle verifies: every ref has tests, digests match, no hidden input leaks")
    return 0


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

    for name in sorted(corpora):
        if (args.directory / f"{name}.tests.jsonl").is_file():
            bundle_problems = generator.verify_bundle(args.directory, name)
            problems.extend(bundle_problems)
            for problem in bundle_problems[:5]:
                print(f"  PROBLEM {problem}")

    if not args.skip_contamination:
        problems.extend(_contamination(corpora))

    missing = sorted(open_tracks - set(corpora))
    if missing:
        print(f"\nno corpus for open tracks: {', '.join(missing)} — they will not be scored")

    if problems:
        print(f"\n{len(problems)} problem(s) found")
        return 1
    print("\nevery corpus can fill a round")
    return 0


def _samples(corpus: Corpus, partition: str | None = None) -> list[contamination.Sample]:
    samples: list[contamination.Sample] = []
    for task in corpus:
        if partition is not None and task.partition != partition:
            continue
        gold = task.gold if isinstance(task.gold, dict) else {}
        tests = gold.get("tests") or []
        samples.append(
            contamination.Sample(
                ref=task.ref,
                prompt=task.prompt,
                solution=str(gold.get("solution", "")),
                instance=(
                    json.dumps([str(gold.get("entry_point", "")), tests], sort_keys=True)
                    if tests
                    else ""
                ),
            )
        )
    return samples


def _report(label: str, flagged: list[contamination.Overlap]) -> list[str]:
    print(f"  {label:<28}{len(flagged)} flagged")
    for overlap in flagged[:5]:
        print(f"    {overlap.describe()}")
    return [f"contamination/{label}: {o.describe()}" for o in flagged]


def _contamination(corpora: dict[str, Corpus]) -> list[str]:
    print("\ncontamination")
    references = contamination.load_reference_fingerprints()
    problems: list[str] = []

    if not references:
        print(
            "  public benchmark overlap    SKIPPED, no fingerprints shipped; build them "
            "with scripts/build_reference_fingerprints.py"
        )
    for name in sorted(corpora):
        corpus = corpora[name]
        if references:
            problems.extend(
                _report(
                    "public benchmark overlap",
                    contamination.scan_overlap(_samples(corpus), references),
                )
            )
        problems.extend(
            _report(
                "fixed/rotating overlap",
                contamination.scan_partitions(
                    _samples(corpus, ROTATING), _samples(corpus, FIXED)
                ),
            )
        )
        problems.extend(
            _report("duplicate prompts", contamination.duplicate_prompts(_samples(corpus)))
        )
    return problems


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
