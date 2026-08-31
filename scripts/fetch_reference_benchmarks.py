from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROWS_API = "https://datasets-server.huggingface.co/rows"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
PAGE = 100

SPECS: dict[str, dict[str, Any]] = {
    "bigcodebench": {
        "repo": "bigcode/bigcodebench",
        "mode": "rows",
        "config": "default",
        "split": "v0.1.4",
        "ref": "task_id",
        "prompt": ("complete_prompt", "instruct_prompt"),
        "solution": ("canonical_solution",),
    },
    "humaneval": {
        "repo": "openai/openai_humaneval",
        "mode": "rows",
        "config": "openai_humaneval",
        "split": "test",
        "ref": "task_id",
        "prompt": ("prompt",),
        "solution": ("canonical_solution",),
    },
    "mbpp": {
        "repo": "google-research-datasets/mbpp",
        "mode": "rows",
        "config": "full",
        "split": "test",
        "ref": "task_id",
        "prompt": ("text",),
        "solution": ("code",),
    },
    "livecodebench": {
        "repo": "livecodebench/code_generation_lite",
        "mode": "stream",
        "files": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl",
                  "test6.jsonl"),
        "ref": "question_id",
        "prompt": ("question_content",),
        "solution": (),
    },
    "apps": {
        "repo": "codeparrot/apps",
        "mode": "stream",
        "files": ("train.jsonl", "test.jsonl"),
        "ref": "problem_id",
        "prompt": ("question",),
        "solution": ("solutions",),
    },
}


def _first(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value:
            head = value[0]
            if isinstance(head, str) and head.strip():
                return head
    return ""


def _solution(row: dict[str, Any], names: tuple[str, ...]) -> str:
    raw = _first(row, names)
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
            return parsed[0]
    return raw


def _rows(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    base = (
        f"{ROWS_API}?dataset={urllib.parse.quote(spec['repo'], safe='')}"
        f"&config={urllib.parse.quote(spec['config'])}"
        f"&split={urllib.parse.quote(spec['split'])}"
    )
    offset = 0
    total = None
    while total is None or offset < total:
        with urllib.request.urlopen(  # noqa: S310
            f"{base}&offset={offset}&length={PAGE}", timeout=120
        ) as answer:
            page = json.load(answer)
        if total is None:
            total = int(page.get("num_rows_total", 0))
        rows = page.get("rows", [])
        if not rows:
            break
        for entry in rows:
            yield entry["row"]
        offset += len(rows)
        print(f"  {offset}/{total}", file=sys.stderr)


def _stream(spec: dict[str, Any], limit: int) -> Iterator[dict[str, Any]]:
    seen = 0
    for name in spec["files"]:
        if seen >= limit:
            return
        url = RESOLVE.format(repo=spec["repo"], path=name)
        request = urllib.request.Request(  # noqa: S310
            url, headers={"Accept-Encoding": "gzip"}
        )
        try:
            answer = urllib.request.urlopen(request, timeout=300)  # noqa: S310
        except OSError as error:
            print(f"  {name}: {error}", file=sys.stderr)
            continue
        stream = (
            gzip.GzipFile(fileobj=answer)
            if answer.headers.get("Content-Encoding") == "gzip"
            else answer
        )
        with answer:
            for line in stream:
                if seen >= limit:
                    return
                text = line.decode("utf-8", "replace").strip()
                if not text:
                    continue
                try:
                    yield json.loads(text)
                except json.JSONDecodeError:
                    continue
                seen += 1
                if seen % 500 == 0:
                    print(f"  {name}: {seen}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download a public code benchmark and write the ref, prompt and solution "
            "of each task as jsonl, ready for build_reference_fingerprints.py. Only the "
            "three fields the fingerprint needs are kept."
        )
    )
    parser.add_argument("name", choices=sorted(SPECS))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=100_000)
    args = parser.parse_args()

    spec = SPECS[args.name]
    out = args.out or Path(f"{args.name}.jsonl")
    source = _rows(spec) if spec["mode"] == "rows" else _stream(spec, args.limit)

    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(source):
            if written >= args.limit:
                break
            record = {
                "ref": str(row.get(spec["ref"]) or f"{args.name}-{index}"),
                "prompt": _first(row, spec["prompt"]),
                "solution": _solution(row, spec["solution"]),
            }
            if not record["prompt"] and not record["solution"]:
                continue
            handle.write(json.dumps(record) + "\n")
            written += 1

    print(f"wrote {written} records to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
