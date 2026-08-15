from __future__ import annotations

import argparse
from pathlib import Path

from microtensor.cli.common import add_common_arguments, fail
from microtensor.core.constants import MECHANISM_VERSION
from microtensor.core.tracks import CLASSES, TRACKS, competitions, enabled_tracks
from microtensor.harness.limits import sandbox_available
from microtensor.harness.registry import available, describe, load_builtin
from microtensor.store.state import ValidatorState


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("inspect", help="inspect the mechanism and local state")
    inner = parser.add_subparsers(dest="action", required=True)

    tracks = inner.add_parser("tracks", help="list tracks, classes and emission shares")
    tracks.set_defaults(handler=_tracks)

    engines = inner.add_parser("engines", help="show which engines this host can run")
    engines.set_defaults(handler=_engines)

    rounds = inner.add_parser("rounds", help="show recent rounds from local state")
    add_common_arguments(rounds)
    rounds.add_argument("--limit", type=int, default=10)
    rounds.set_defaults(handler=_rounds)

    hotkey = inner.add_parser("hotkey", help="show one hotkey's evaluation history")
    add_common_arguments(hotkey)
    hotkey.add_argument("hotkey")
    hotkey.add_argument("--limit", type=int, default=20)
    hotkey.set_defaults(handler=_hotkey)


def _tracks(args: argparse.Namespace) -> int:
    print(f"mechanism {MECHANISM_VERSION}\n")
    print(f"{'track':<14}{'modality':<12}{'metric':<32}{'share':>7}  status")
    for track in TRACKS.values():
        status = "open" if track.enabled else "registered, not scored"
        print(
            f"{track.id:<14}{track.modality.value:<12}{track.metric:<32}"
            f"{track.emission_share:>6.0%}  {status}"
        )

    print(f"\n{'class':<14}{'size':>10}{'rss':>10}{'p95':>8}  reference")
    for hardware in CLASSES.values():
        print(
            f"{hardware.id:<14}{hardware.max_size_bytes / 1024**3:>9.1f}G"
            f"{hardware.max_rss_bytes / 1024**3:>9.1f}G{hardware.max_p95_ms:>7}ms"
            f"  {hardware.reference}"
        )

    print(f"\n{len(enabled_tracks())} tracks open, {len(competitions())} competitions")
    return 0


def _engines(args: argparse.Namespace) -> int:
    load_builtin()
    formats = available()
    if not formats:
        print("no engine is available; this host cannot score a round")
    for fmt in formats:
        info = describe(fmt)
        detail = f"{info.name} {info.version} — {info.notes}" if info else "registered"
        print(f"{fmt.value:<14}{detail}")
    print(
        f"\nsandbox     {'enforced' if sandbox_available() else 'UNAVAILABLE on this host'}"
    )
    return 0 if formats else 1


def _state(args: argparse.Namespace) -> ValidatorState:
    path = Path(args.home) / "state" / "validator.sqlite"
    if not path.is_file():
        raise SystemExit(f"no validator state at {path}")
    return ValidatorState(path)


def _rounds(args: argparse.Namespace) -> int:
    try:
        state = _state(args)
    except SystemExit as exc:
        return fail(str(exc))

    try:
        rows = state.db.query(
            "SELECT round_index, seed_block, status, reason FROM rounds "
            "ORDER BY round_index DESC LIMIT ?",
            (args.limit,),
        )
        if not rows:
            print("no rounds recorded yet")
            return 0
        print(f"{'round':>8}{'seed block':>12}  {'status':<11}reason")
        for row in rows:
            print(
                f"{row['round_index']:>8}{row['seed_block']:>12}  "
                f"{row['status']:<11}{row['reason']}"
            )
        return 0
    finally:
        state.close()


def _hotkey(args: argparse.Namespace) -> int:
    try:
        state = _state(args)
    except SystemExit as exc:
        return fail(str(exc))

    try:
        history = state.history(args.hotkey, args.limit)
        if not history:
            print(f"no evaluations recorded for {args.hotkey}")
            return 0
        print(f"{'round':>8}  {'competition':<24}{'admitted':<10}{'score':>8}  reason")
        for row in history:
            competition = f"{row['track']}/{row['hardware_class']}"
            print(
                f"{row['round_index']:>8}  {competition:<24}"
                f"{'yes' if row['admitted'] else 'no':<10}"
                f"{row['score_combined']:>8.4f}  {row['gate_reason']}"
            )
        return 0
    finally:
        state.close()
