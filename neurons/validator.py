from __future__ import annotations

import sys

from microtensor.cli.main import main


def run() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] != "validator":
        argv = ["validator", "run", *argv]
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(run())
