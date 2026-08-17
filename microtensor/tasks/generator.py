from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from microtensor.core.constants import CORPUS_VERSION
from microtensor.core.hashing import canonical_hash, digest_file
from microtensor.tasks import contamination
from microtensor.tasks.corpus import FIXED, ROTATING

GENERATOR_VERSION: Final[str] = "1"
MIN_HIDDEN_TESTS: Final[int] = 7
PUBLIC_EXAMPLES: Final[int] = 2
LEAK_CHECK_MIN_CHARS: Final[int] = 12

MANIFEST_NAME: Final[str] = "corpus.manifest.json"


class GenerationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GeneratedTask:
    ref: str
    partition: str
    prompt: str
    entry_point: str
    solution: str
    hidden: tuple[tuple[tuple[Any, ...], Any], ...]
    public: tuple[tuple[tuple[Any, ...], Any], ...]
    family: str
    difficulty: int

    def task_row(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "prompt": self.prompt,
            "gold": None,
            "partition": self.partition,
            "max_output_tokens": 512,
            "inputs": {
                "entry_point": self.entry_point,
                "tests_digest": self.tests_digest(),
                "public_examples": [
                    {"in": list(args), "out": expected} for args, expected in self.public
                ],
                "family": self.family,
                "difficulty": self.difficulty,
            },
        }

    def train_row(self) -> dict[str, Any]:
        row = self.task_row()
        row["inputs"].pop("tests_digest")
        return row

    def tests_row(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "entry_point": self.entry_point,
            "tests": self._tests_json(),
            "solution": self.solution,
        }

    def _tests_json(self) -> list[dict[str, Any]]:
        return [{"args": list(args), "expected": expected} for args, expected in self.hidden]

    def tests_digest(self) -> str:
        return canonical_hash(
            {"ref": self.ref, "entry_point": self.entry_point, "tests": self._tests_json()}
        )

    def instance_key(self) -> str:
        cases = [
            f"{args!r}->{expected!r}" for args, expected in (*self.hidden, *self.public)
        ]
        return f"{self.family}|{self.entry_point}|" + "|".join(sorted(cases))


@dataclass(frozen=True, slots=True)
class Template:
    entry: str
    signature: str
    behaviour: str
    solution: str
    reference: Callable[..., Any]
    make_input: Callable[[random.Random, int, bool], tuple[Any, ...]]


def _words(rng: random.Random, count: int, alphabet: str = "abcdefghijklmnop") -> list[str]:
    return [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 4 + count)))
        for _ in range(count)
    ]


def _family_rle(rng: random.Random) -> Template:
    entry = rng.choice(["run_length_encode", "encode_runs", "compress_runs"])
    joiner = rng.choice(["", ",", ";"])
    pair = rng.choice(["{c}{n}", "{c}:{n}", "{c}x{n}"])

    def reference(s: str) -> str:
        parts: list[str] = []
        index = 0
        while index < len(s):
            char = s[index]
            run = 1
            while index + run < len(s) and s[index + run] == char:
                run += 1
            parts.append(pair.format(c=char, n=run))
            index += run
        return joiner.join(parts)

    solution = (
        f"def {entry}(s):\n"
        "    parts = []\n"
        "    i = 0\n"
        "    while i < len(s):\n"
        "        c = s[i]\n"
        "        n = 1\n"
        "        while i + n < len(s) and s[i + n] == c:\n"
        "            n += 1\n"
        f"        parts.append({pair!r}.replace('{{c}}', c).replace('{{n}}', str(n)))\n"
        "        i += n\n"
        f"    return {joiner!r}.join(parts)\n"
    )

    def make_input(rng: random.Random, difficulty: int, edge: bool) -> tuple[Any, ...]:
        if edge:
            return ("",)
        length = rng.randint(3, 6 + difficulty * 6)
        alphabet = "ab" if difficulty == 1 else "abcd"
        return ("".join(rng.choice(alphabet) for _ in range(length)),)

    fmt = pair.format(c="<char>", n="<count>")
    return Template(
        entry=entry,
        signature="s",
        behaviour=(
            f"run-length encodes the string s: each maximal run of a repeated character "
            f"becomes {fmt!r}, and runs are joined with {joiner!r}. "
            "An empty string encodes to an empty string."
        ),
        solution=solution,
        reference=reference,
        make_input=make_input,
    )


def _family_intervals(rng: random.Random) -> Template:
    entry = rng.choice(["merge_intervals", "merge_ranges", "collapse_intervals"])
    touching = rng.choice([True, False])

    def reference(intervals: list[list[int]]) -> list[list[int]]:
        merged: list[list[int]] = []
        for start, end in sorted([list(i) for i in intervals]):
            if merged and (start <= merged[-1][1] if touching else start < merged[-1][1]):
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return merged

    op = "<=" if touching else "<"
    solution = (
        f"def {entry}(intervals):\n"
        "    merged = []\n"
        "    for start, end in sorted([list(i) for i in intervals]):\n"
        f"        if merged and start {op} merged[-1][1]:\n"
        "            merged[-1][1] = max(merged[-1][1], end)\n"
        "        else:\n"
        "            merged.append([start, end])\n"
        "    return merged\n"
    )

    def make_input(rng: random.Random, difficulty: int, edge: bool) -> tuple[Any, ...]:
        if edge:
            return ([],)
        count = rng.randint(2, 3 + difficulty * 3)
        spans = []
        for _ in range(count):
            start = rng.randint(0, 30 * difficulty)
            spans.append([start, start + rng.randint(1, 12)])
        return (spans,)

    rule = (
        "intervals that touch (the next start equals the previous end) also merge"
        if touching
        else "intervals that merely touch do not merge; only genuine overlap merges"
    )
    return Template(
        entry=entry,
        signature="intervals",
        behaviour=(
            "merges a list of [start, end] integer intervals: sort them, combine "
            f"overlapping ones, and return the merged list. Note that {rule}. "
            "An empty list returns an empty list."
        ),
        solution=solution,
        reference=reference,
        make_input=make_input,
    )


def _family_group(rng: random.Random) -> Template:
    entry = rng.choice(["group_total", "sum_by_key", "totals_by_group"])
    key_field = rng.choice(["group", "team", "region"])
    value_field = rng.choice(["value", "amount", "score"])
    with_count = rng.choice([True, False])

    def reference(records: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for record in records:
            key = record[key_field]
            if with_count:
                total, count = out.get(key, [0, 0])
                out[key] = [total + record[value_field], count + 1]
            else:
                out[key] = out.get(key, 0) + record[value_field]
        return out

    if with_count:
        body = (
            "        total, count = out.get(k, [0, 0])\n"
            f"        out[k] = [total + r[{value_field!r}], count + 1]\n"
        )
        shape = f"a list [total of {value_field!r}, count of records]"
    else:
        body = f"        out[k] = out.get(k, 0) + r[{value_field!r}]\n"
        shape = f"the total of {value_field!r}"
    solution = (
        f"def {entry}(records):\n"
        "    out = {}\n"
        "    for r in records:\n"
        f"        k = r[{key_field!r}]\n" + body + "    return out\n"
    )

    def make_input(rng: random.Random, difficulty: int, edge: bool) -> tuple[Any, ...]:
        if edge:
            return ([],)
        groups = _words(rng, rng.randint(2, 2 + difficulty))
        return (
            [
                {key_field: rng.choice(groups), value_field: rng.randint(1, 40)}
                for _ in range(rng.randint(3, 4 + difficulty * 4))
            ],
        )

    return Template(
        entry=entry,
        signature="records",
        behaviour=(
            f"takes a list of dicts each carrying {key_field!r} and {value_field!r}, and "
            f"returns a dict mapping each {key_field!r} to {shape}. "
            "An empty list returns an empty dict."
        ),
        solution=solution,
        reference=reference,
        make_input=make_input,
    )


def _family_csv(rng: random.Random) -> Template:
    entry = rng.choice(["parse_table", "rows_to_records", "read_delimited"])
    sep = rng.choice([",", ";", "|"])

    def reference(text: str) -> list[dict[str, str]]:
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return []
        header = lines[0].split(sep)
        return [dict(zip(header, line.split(sep), strict=False)) for line in lines[1:]]

    solution = (
        f"def {entry}(text):\n"
        "    lines = [line for line in text.splitlines() if line.strip()]\n"
        "    if len(lines) < 2:\n"
        "        return []\n"
        f"    header = lines[0].split({sep!r})\n"
        f"    return [dict(zip(header, line.split({sep!r}))) for line in lines[1:]]\n"
    )

    def make_input(rng: random.Random, difficulty: int, edge: bool) -> tuple[Any, ...]:
        if edge:
            return ("",)
        columns = _words(rng, rng.randint(2, 2 + difficulty))
        rows = []
        for _ in range(rng.randint(1, 2 + difficulty * 2)):
            rows.append(sep.join(str(rng.randint(0, 99)) for _ in columns))
        return (sep.join(columns) + "\n" + "\n".join(rows),)

    return Template(
        entry=entry,
        signature="text",
        behaviour=(
            f"parses delimited text whose first non-empty line is a header row, with fields "
            f"separated by {sep!r}, and returns a list of dicts mapping header names to the "
            "string values of each subsequent non-empty row. Text with no data rows "
            "returns an empty list."
        ),
        solution=solution,
        reference=reference,
        make_input=make_input,
    )


def _family_filter_words(rng: random.Random) -> Template:
    entry = rng.choice(["filter_words", "keep_words", "select_words"])
    strict = rng.choice([True, False])
    lower = rng.choice([True, False])

    def reference(text: str, n: int) -> list[str]:
        out = []
        for word in text.split():
            if (len(word) > n) if strict else (len(word) >= n):
                out.append(word.lower() if lower else word)
        return out

    op = ">" if strict else ">="
    transform = ".lower()" if lower else ""
    solution = (
        f"def {entry}(text, n):\n"
        "    out = []\n"
        "    for word in text.split():\n"
        f"        if len(word) {op} n:\n"
        f"            out.append(word{transform})\n"
        "    return out\n"
    )

    def make_input(rng: random.Random, difficulty: int, edge: bool) -> tuple[Any, ...]:
        if edge:
            return ("", 3)
        words = _words(rng, rng.randint(3, 4 + difficulty * 3), "abcdefgHIJKLm")
        return (" ".join(words), rng.randint(2, 5))

    rule = "strictly longer than n characters" if strict else "at least n characters long"
    case = " Lower-case each kept word." if lower else " Preserve the original case."
    return Template(
        entry=entry,
        signature="text, n",
        behaviour=(
            f"splits text on whitespace and returns, in order, the words that are {rule}."
            f"{case} An empty text returns an empty list."
        ),
        solution=solution,
        reference=reference,
        make_input=make_input,
    )


def _family_brackets(rng: random.Random) -> Template:
    entry = rng.choice(["max_depth", "bracket_depth", "nesting_depth"])
    pairs = rng.choice(["()", "()[]"])
    opens = pairs[0::2]
    closes = pairs[1::2]
    match = {c: o for o, c in zip(opens, closes, strict=True)}

    def reference(s: str) -> int:
        stack: list[str] = []
        deepest = 0
        for char in s:
            if char in opens:
                stack.append(char)
                deepest = max(deepest, len(stack))
            elif char in closes:
                if not stack or stack[-1] != match[char]:
                    return -1
                stack.pop()
        return -1 if stack else deepest

    solution = (
        f"def {entry}(s):\n"
        f"    match = {match!r}\n"
        "    stack = []\n"
        "    deepest = 0\n"
        "    for ch in s:\n"
        f"        if ch in {opens!r}:\n"
        "            stack.append(ch)\n"
        "            deepest = max(deepest, len(stack))\n"
        f"        elif ch in {closes!r}:\n"
        "            if not stack or stack[-1] != match[ch]:\n"
        "                return -1\n"
        "            stack.pop()\n"
        "    return -1 if stack else deepest\n"
    )

    def make_input(rng: random.Random, difficulty: int, edge: bool) -> tuple[Any, ...]:
        if edge:
            return ("",)
        length = rng.randint(2, 6 + difficulty * 6)
        alphabet = pairs + ("ab" if difficulty > 1 else "")
        return ("".join(rng.choice(alphabet) for _ in range(length)),)

    kinds = "round brackets" if pairs == "()" else "round and square brackets"
    return Template(
        entry=entry,
        signature="s",
        behaviour=(
            f"returns the maximum nesting depth of {kinds} in s, ignoring other characters. "
            "If the brackets are unbalanced or mismatched, return -1. "
            "An empty string has depth 0."
        ),
        solution=solution,
        reference=reference,
        make_input=make_input,
    )


def _family_sort_records(rng: random.Random) -> Template:
    entry = rng.choice(["sort_records", "order_records", "rank_records"])
    fields = rng.sample(["id", "score", "age", "rank"], 2)
    primary, tiebreak = fields[0], fields[1]
    descending = rng.choice([True, False])

    def reference(records: list[dict[str, int]]) -> list[dict[str, int]]:
        return sorted(
            records,
            key=lambda r: ((-r[primary] if descending else r[primary]), r[tiebreak]),
        )

    sign = "-" if descending else ""
    solution = (
        f"def {entry}(records):\n"
        f"    return sorted(records, key=lambda r: ({sign}r[{primary!r}], r[{tiebreak!r}]))\n"
    )

    def make_input(rng: random.Random, difficulty: int, edge: bool) -> tuple[Any, ...]:
        if edge:
            return ([],)
        return (
            [
                {primary: rng.randint(0, 5 + difficulty * 3), tiebreak: rng.randint(0, 30)}
                for _ in range(rng.randint(3, 4 + difficulty * 3))
            ],
        )

    order = "descending" if descending else "ascending"
    return Template(
        entry=entry,
        signature="records",
        behaviour=(
            f"sorts a list of dicts by {primary!r} in {order} order, breaking ties by "
            f"{tiebreak!r} ascending, and returns the sorted list. "
            "An empty list returns an empty list."
        ),
        solution=solution,
        reference=reference,
        make_input=make_input,
    )


def _family_summary(rng: random.Random) -> Template:
    entry = rng.choice(["summarise_items", "build_summary", "describe_items"])
    value_field = rng.choice(["value", "amount", "qty"])
    id_field = rng.choice(["id", "sku"])

    def reference(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(items),
            "total": sum(item[value_field] for item in items),
            "ids": sorted(item[id_field] for item in items),
        }

    solution = (
        f"def {entry}(items):\n"
        "    return {\n"
        "        'count': len(items),\n"
        f"        'total': sum(item[{value_field!r}] for item in items),\n"
        f"        'ids': sorted(item[{id_field!r}] for item in items),\n"
        "    }\n"
    )

    def make_input(rng: random.Random, difficulty: int, edge: bool) -> tuple[Any, ...]:
        if edge:
            return ([],)
        return (
            [
                {id_field: rng.randint(100, 999), value_field: rng.randint(1, 50)}
                for _ in range(rng.randint(2, 3 + difficulty * 3))
            ],
        )

    return Template(
        entry=entry,
        signature="items",
        behaviour=(
            f"takes a list of dicts each carrying {id_field!r} and {value_field!r} and returns "
            f"a dict with exactly the keys 'count' (number of items), 'total' (sum of "
            f"{value_field!r}), and 'ids' (sorted list of {id_field!r} values). "
            "An empty list returns {'count': 0, 'total': 0, 'ids': []}."
        ),
        solution=solution,
        reference=reference,
        make_input=make_input,
    )


FAMILIES: Final[dict[str, Callable[[random.Random], Template]]] = {
    "rle": _family_rle,
    "intervals": _family_intervals,
    "group": _family_group,
    "csv": _family_csv,
    "filter-words": _family_filter_words,
    "brackets": _family_brackets,
    "sort-records": _family_sort_records,
    "summary": _family_summary,
}


def _rng(seed: str, namespace: str, index: int) -> random.Random:
    material = hashlib.sha256(f"{seed}:{namespace}:{index}".encode()).digest()
    return random.Random(int.from_bytes(material, "big"))  # noqa: S311


def _draw_cases(
    template: Template,
    rng: random.Random,
    difficulty: int,
    count: int,
    *,
    edge_first: bool,
    exclude: frozenset[str] = frozenset(),
) -> list[tuple[tuple[Any, ...], Any]]:
    cases: list[tuple[tuple[Any, ...], Any]] = []
    seen: set[str] = set(exclude)
    while len(cases) < count:
        edge = edge_first and not cases
        args = template.make_input(rng, difficulty, edge)
        key = repr(args)
        if not edge and key in seen:
            continue
        seen.add(key)
        cases.append((args, template.reference(*args)))
    return cases


def build_task(ref: str, partition: str, family: str, seed: str, index: int) -> GeneratedTask:
    rng = _rng(seed, f"{partition}:{family}", index)
    difficulty = rng.randint(1, 3)
    template = FAMILIES[family](rng)

    public = _draw_cases(template, rng, difficulty, PUBLIC_EXAMPLES, edge_first=False)
    hidden = _draw_cases(
        template,
        rng,
        difficulty,
        MIN_HIDDEN_TESTS,
        edge_first=True,
        exclude=frozenset(repr(args) for args, _ in public),
    )

    examples = "\n".join(
        f"  {template.entry}({', '.join(repr(a) for a in args)}) -> {expected!r}"
        for args, expected in public
    )
    prompt = (
        f"Write a Python function `{template.entry}({template.signature})` that "
        f"{template.behaviour}\n"
        f"Examples:\n{examples}\n"
        "Return only the code."
    )

    return GeneratedTask(
        ref=ref,
        partition=partition,
        prompt=prompt,
        entry_point=template.entry,
        solution=template.solution,
        hidden=tuple(hidden),
        public=tuple(public),
        family=family,
        difficulty=difficulty,
    )


@dataclass(frozen=True, slots=True)
class Bundle:
    tasks: tuple[GeneratedTask, ...]
    train: tuple[GeneratedTask, ...]
    seed: str
    version: str = CORPUS_VERSION


REDRAW_ATTEMPTS: Final[int] = 24
REDRAW_STRIDE: Final[int] = 7919


def _as_sample(task: GeneratedTask) -> contamination.Sample:
    return contamination.Sample(
        ref=task.ref,
        prompt=task.prompt,
        solution=task.solution,
        instance=task.instance_key(),
    )


def _redraw_collisions(
    rotating_tasks: list[GeneratedTask],
    fixed_tasks: list[GeneratedTask],
    seed: str,
    names: list[str],
) -> list[GeneratedTask]:
    anchors = [_as_sample(t) for t in fixed_tasks]
    by_ref = {t.ref: index for index, t in enumerate(rotating_tasks)}
    flagged: list[contamination.Overlap] = []

    for attempt in range(1, REDRAW_ATTEMPTS + 1):
        flagged = contamination.scan_partitions(
            [_as_sample(t) for t in rotating_tasks], anchors
        )
        if not flagged:
            return rotating_tasks
        for overlap in flagged:
            position = by_ref[overlap.ref]
            original = rotating_tasks[position]
            index = int(original.ref.rsplit("-", 1)[1]) + attempt * REDRAW_STRIDE
            rotating_tasks[position] = build_task(
                original.ref, ROTATING, names[index % len(names)], seed, index
            )

    raise GenerationError(
        f"{len(flagged)} rotating tasks still instantiate the same cases as the fixed "
        f"partition after {REDRAW_ATTEMPTS} redraws; widen the template parameter space"
    )


def generate(
    *,
    rotating: int,
    fixed: int,
    train: int,
    seed: str,
    version: str = CORPUS_VERSION,
) -> Bundle:
    if rotating < 1 or fixed < 1:
        raise GenerationError("a corpus needs at least one rotating and one fixed task")

    names = sorted(FAMILIES)

    def batch(prefix: str, partition: str, count: int) -> list[GeneratedTask]:
        return [
            build_task(
                f"code-{prefix}-{i:06d}", partition, names[i % len(names)], seed, i
            )
            for i in range(count)
        ]

    fixed_tasks = batch("f", FIXED, fixed)
    rotating_tasks = _redraw_collisions(
        batch("r", ROTATING, rotating), fixed_tasks, seed, names
    )

    training = batch("t", ROTATING, train) if train else []
    return Bundle(
        tasks=tuple(rotating_tasks + fixed_tasks),
        train=tuple(training),
        seed=seed,
        version=version,
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
    )


def write_bundle(bundle: Bundle, out: Path, *, force: bool = False) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    targets = {
        "code.jsonl": [t.task_row() for t in bundle.tasks],
        "code.tests.jsonl": [t.tests_row() for t in bundle.tasks],
        "code.train.jsonl": [t.train_row() for t in bundle.train],
    }

    existing = [name for name in targets if (out / name).exists()]
    if existing and not force:
        raise GenerationError(
            f"{', '.join(existing)} already exist; the fixed partition is generated once "
            "and committed, so pass --force only if you mean to replace the corpus"
        )

    for name, rows in targets.items():
        _write_jsonl(out / name, rows)

    digests = {name: digest_file(out / name) for name in targets}
    manifest = {
        "version": bundle.version,
        "generator": GENERATOR_VERSION,
        "seed": bundle.seed,
        "counts": {
            "rotating": sum(1 for t in bundle.tasks if t.partition == ROTATING),
            "fixed": sum(1 for t in bundle.tasks if t.partition == FIXED),
            "train": len(bundle.train),
        },
        "files": digests,
        "reference_model": None,
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return digests


def verify_bundle(directory: Path, track: str = "code") -> list[str]:
    problems: list[str] = []
    corpus_path = directory / f"{track}.jsonl"
    tests_path = directory / f"{track}.tests.jsonl"
    if not corpus_path.is_file():
        return [f"{corpus_path.name} is missing"]
    if not tests_path.is_file():
        return [f"{tests_path.name} is missing"]

    tests: dict[str, dict[str, Any]] = {}
    for line in tests_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            tests[str(row["ref"])] = row

    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ref = str(row["ref"])
        declared = row.get("inputs", {}).get("tests_digest", "")
        entry = tests.get(ref)
        if entry is None:
            problems.append(f"{ref}: no hidden tests")
            continue
        actual = canonical_hash(
            {"ref": ref, "entry_point": entry["entry_point"], "tests": entry["tests"]}
        )
        if declared != actual:
            problems.append(f"{ref}: tests_digest does not match the tests file")
        prompt = str(row.get("prompt", ""))
        for case in entry["tests"]:
            for arg in case.get("args", []):
                rendered = repr(arg)
                if len(rendered) >= LEAK_CHECK_MIN_CHARS and rendered in prompt:
                    problems.append(f"{ref}: hidden test input appears in the prompt")
    return problems
