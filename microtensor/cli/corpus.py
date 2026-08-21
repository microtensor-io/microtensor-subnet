from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

from microtensor.cli.common import fail
from microtensor.core.constants import CONTROL_URL, CORPUS_VERSION, TASKS_PER_ROUND
from microtensor.core.tracks import enabled_tracks, get_track
from microtensor.tasks import bundle, contamination
from microtensor.tasks.corpus import (
    ADMISSION_MINIMUMS,
    FIXED,
    ROTATING,
    Corpus,
    CorpusError,
    load_all,
    load_corpus,
)
from microtensor.tasks.selection import partition_sizes

SEED_ROTATING = 240
SEED_FIXED = 100


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("corpus", help="check and seed task corpora")
    inner = parser.add_subparsers(dest="action", required=True)

    check = inner.add_parser(
        "check",
        help="verify a validator's corpus directory, or an upload bundle with --bundle",
    )
    check.add_argument(
        "directory", type=Path, help="corpus directory, or bundle file with --bundle"
    )
    check.add_argument(
        "--bundle",
        action="store_true",
        help="the path is an upload bundle; validate it against the control plane",
    )
    check.add_argument(
        "--control",
        default=CONTROL_URL,
        help="control plane to validate the bundle against",
    )
    check.add_argument(
        "--credential",
        default=os.environ.get("MTS_OPERATOR_SECRET", ""),
        help="operator credential; defaults to MTS_OPERATOR_SECRET",
    )
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


def _check(args: argparse.Namespace) -> int:
    if args.bundle:
        return _check_bundle(args)

    try:
        corpora = load_all(args.directory)
    except CorpusError as exc:
        return fail(str(exc))

    want_rotating, want_fixed = partition_sizes(args.tasks_per_round)
    open_tracks = {t.id for t in enabled_tracks()}
    problems: list[str] = []
    advisories: list[str] = []

    print(f"{'track':<14}{'total':>8}{'rotating':>10}{'fixed':>8}  {'digest':<18}status")
    for name in sorted(corpora):
        corpus = corpora[name]
        notes: list[str] = []
        soft: list[str] = []

        if name not in open_tracks:
            notes.append("track not open")

        # A real failure: below this the control plane will not accept the
        # corpus, so it can never be attached to an arena.
        if len(corpus.rotating) < ADMISSION_MINIMUMS["rotating"]:
            notes.append(f"rotating below the {ADMISSION_MINIMUMS['rotating']} upload minimum")
        if len(corpus.fixed) < ADMISSION_MINIMUMS["fixed"]:
            notes.append(f"fixed below the {ADMISSION_MINIMUMS['fixed']} upload minimum")

        # Not a failure: the draw takes min(want, available), so a smaller
        # corpus simply supplies a smaller round. Said separately because
        # reporting it as a problem sends people chasing one that is not there.
        if len(corpus.rotating) < want_rotating:
            soft.append(f"rotating {len(corpus.rotating)} of {want_rotating} per round")
        if len(corpus.fixed) < want_fixed:
            soft.append(f"fixed {len(corpus.fixed)} of {want_fixed} per round")

        status = "ok" if not notes else "; ".join(notes)
        if not notes and soft:
            status = "ok, draws short"
        problems.extend(f"{name}: {n}" for n in notes)
        advisories.extend(f"{name}: {n}" for n in soft)
        print(
            f"{name:<14}{len(corpus):>8}{len(corpus.rotating):>10}"
            f"{len(corpus.fixed):>8}  {_digest(corpus):<18}{status}"
        )

    for name in sorted(corpora):
        if (args.directory / f"{name}.tests.jsonl").is_file():
            bundle_problems = bundle.verify_bundle(args.directory, name)
            problems.extend(bundle_problems)
            for problem in bundle_problems[:5]:
                print(f"  PROBLEM {problem}")

    if not args.skip_contamination:
        problems.extend(_contamination(corpora))

    missing = sorted(open_tracks - set(corpora))
    if missing:
        print(f"\nno corpus for open tracks: {', '.join(missing)} — they will not be scored")

    if advisories:
        print(f"\nsmaller than one round's draw at {args.tasks_per_round} tasks per round:")
        for note in advisories:
            print(f"  {note}")
        print("  the draw takes what is there, so this is size, not a defect")

    if problems:
        print(f"\n{len(problems)} problem(s) found")
        return 1
    print("\nevery corpus clears the upload minimums")
    return 0


def _check_bundle(args: argparse.Namespace) -> int:
    """Validate an upload bundle against the control plane, storing nothing.

    Asked of the server rather than reimplemented here on purpose. The local
    loader reads the validator split, which has no train partition and no idea
    what the upload minimums are; a second validator that could disagree with
    the one that decides is worse than a round trip.
    """
    path: Path = args.directory
    if not path.is_file():
        return fail(f"{path} is not a file; --bundle expects the bundle json")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return fail(f"{path} could not be read: {exc}")

    if not isinstance(payload, dict) or "tasks" not in payload:
        return fail(f"{path} is not a bundle; expected an object with a tasks array")

    if not args.credential:
        return fail("set MTS_OPERATOR_SECRET or pass --credential to validate a bundle")

    body = json.dumps(
        {
            "track": str(payload.get("track", "")),
            "description": str(payload.get("description", "")),
            "manifest": payload.get("manifest", {}),
            "tasks": payload.get("tasks", []),
            "tests": payload.get("tests", []),
        }
    ).encode()

    url = f"{str(args.control).rstrip('/')}/v1/operator/corpora/validate"
    if not url.startswith(("http://", "https://")):
        return fail(f"--control needs an http or https URL, got {args.control!r}")

    request = urllib.request.Request(  # noqa: S310
        url,
        data=body,
        headers={"content-type": "application/json", "x-mt-credential": args.credential},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as answer:  # noqa: S310
            report = json.loads(answer.read().decode("utf-8"))
    except Exception as exc:
        return fail(f"{url} could not validate the bundle: {exc}")

    counts = report.get("counts", {})
    print(f"version   {report.get('version', '?')}")
    print(
        "counts    "
        + ", ".join(f"{name} {counts.get(name, 0)}" for name in ("train", "fixed", "rotating"))
    )

    flagged = report.get("contamination", [])
    print(f"overlap   {len(flagged)} flagged against public benchmarks")
    for hit in flagged[:5]:
        print(f"  {hit['ref']} overlaps {hit['against']} (via {hit['signal']})")

    failures = report.get("failures", [])
    if not failures:
        print("\nthe bundle would upload")
        return 0

    print(f"\n{len(failures)} check(s) failed:")
    for failure in failures:
        refs = failure.get("refs", [])
        shown = ", ".join(refs[:6]) + (f" (+{len(refs) - 6} more)" if len(refs) > 6 else "")
        print(f"  [{failure['check']}] {failure['detail']}")
        if shown:
            print(f"    {shown}")
    return 1


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
                contamination.scan_partitions(_samples(corpus, ROTATING), _samples(corpus, FIXED)),
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
        print(
            f"  tasks        {len(corpus)}  ({len(corpus.rotating)} rotating, "
            f"{len(corpus.fixed)} fixed)"
        )
        print(
            f"  prompt chars min {min(lengths)}, median {sorted(lengths)[len(lengths) // 2]}, "
            f"max {max(lengths)}"
        )
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
