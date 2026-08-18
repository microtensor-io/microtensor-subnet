from __future__ import annotations

import argparse
import sys

from microtensor.cli import coordinator as coordinator_cmd
from microtensor.cli import corpus as corpus_cmd
from microtensor.cli import inspect as inspect_cmd
from microtensor.cli import miner as miner_cmd
from microtensor.cli import update as update_cmd
from microtensor.cli import validator as validator_cmd
from microtensor.cli.common import configure_logging
from microtensor.core.constants import MECHANISM_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mt",
        description="Microtensor subnet: package, publish, evaluate and settle.",
    )
    parser.add_argument("--version", action="version", version=f"microtensor {MECHANISM_VERSION}")

    subparsers = parser.add_subparsers(dest="command", required=True)
    validator_cmd.register(subparsers)
    coordinator_cmd.register(subparsers)
    miner_cmd.register(subparsers)
    corpus_cmd.register(subparsers)
    update_cmd.register(subparsers)
    inspect_cmd.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    configure_logging(getattr(args, "log_level", "INFO"))

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return int(handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, str):
            print(f"error: {code}", file=sys.stderr)
            return 1
        return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
