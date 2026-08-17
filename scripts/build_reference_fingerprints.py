from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from microtensor.tasks.contamination import (  # noqa: E402
    Sample,
    fingerprint,
    fingerprint_dir,
    write_fingerprints,
)

FIELDS = {
    "prompt": ("prompt", "text", "question", "description"),
    "solution": ("canonical_solution", "solution", "code", "completion"),
    "ref": ("task_id", "id", "ref", "name"),
}


def pick(row: dict[str, object], names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return default


def load(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return list(payload.values())
    raise SystemExit(f"{path} is not a list of records")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Turn a local public benchmark (HumanEval, MBPP or similar) into the "
            "fingerprint file mt corpus check compares against. Only fingerprints are "
            "written, never the third-party prompts or solutions themselves."
        )
    )
    parser.add_argument("source", type=Path, help="local .jsonl or .json benchmark dump")
    parser.add_argument("--name", required=True, help="e.g. humaneval, mbpp")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = load(args.source)
    samples = [
        Sample(
            ref=pick(row, FIELDS["ref"], default=f"{args.name}-{index}"),
            prompt=pick(row, FIELDS["prompt"]),
            solution=pick(row, FIELDS["solution"]),
        )
        for index, row in enumerate(rows)
    ]
    usable = [s for s in samples if s.prompt or s.solution]
    if not usable:
        raise SystemExit(f"{args.source} yielded no prompts or solutions to fingerprint")

    out = args.out or (fingerprint_dir() / f"{args.name}.jsonl")
    write_fingerprints(out, [fingerprint(s, source=args.name) for s in usable])

    print(f"wrote {len(usable)} fingerprints to {out}")
    print("this file carries structure only; the benchmark text itself is not stored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
