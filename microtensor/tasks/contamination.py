from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

AST_THRESHOLD: Final[float] = 0.90
EDIT_THRESHOLD: Final[float] = 0.85
MIN_TOKENS: Final[int] = 8

FINGERPRINT_DIR: Final[str] = "reference_fingerprints"
TOKEN_SALT: Final[str] = "microtensor/contamination/1:"  # noqa: S105

_COMMENT = re.compile(r"#[^\n]*")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")

NAME = "N"
CONST = "C"


class ContaminationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Sample:
    ref: str
    prompt: str
    solution: str = ""
    instance: str = ""

    @property
    def text(self) -> str:
        return f"{self.prompt}\n{self.solution}".strip()


@dataclass(frozen=True, slots=True)
class Fingerprint:
    ref: str
    ast: str
    prompt_tokens: tuple[str, ...]
    solution_tokens: tuple[str, ...]
    source: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.ref,
            "ast": self.ast,
            "prompt": list(self.prompt_tokens),
            "solution": list(self.solution_tokens),
            "source": self.source,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Fingerprint:
        return cls(
            ref=str(row["id"]),
            ast=str(row.get("ast", "")),
            prompt_tokens=tuple(str(t) for t in row.get("prompt", [])),
            solution_tokens=tuple(str(t) for t in row.get("solution", [])),
            source=str(row.get("source", "")),
        )


@dataclass(frozen=True, slots=True)
class Overlap:
    ref: str
    against: str
    ast_similarity: float
    edit_similarity: float
    signal: str

    def describe(self) -> str:
        return (
            f"{self.ref} overlaps {self.against} "
            f"(ast {self.ast_similarity:.2f}, edit {self.edit_similarity:.2f}, "
            f"via {self.signal})"
        )


def _is_docstring(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


class _Normalise(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=NAME, ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(ast.arg(arg=NAME, annotation=None), node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.copy_location(ast.Constant(value=CONST), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        return ast.copy_location(
            ast.Attribute(value=node.value, attr=NAME, ctx=node.ctx), node
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        body = [n for n in node.body if not _is_docstring(n)]
        return ast.copy_location(
            ast.FunctionDef(
                name=NAME,
                args=node.args,
                body=body or [ast.Pass()],
                decorator_list=[],
                returns=None,
                type_comment=None,
            ),
            node,
        )


def ast_fingerprint(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    tree = _Normalise().visit(tree)
    ast.fix_missing_locations(tree)
    body = [n for n in tree.body if not _is_docstring(n)]
    return ast.dump(ast.Module(body=body, type_ignores=[]), annotate_fields=False)


def tokenise(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(_COMMENT.sub(" ", text).lower()))


def digest_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        hashlib.blake2s(f"{TOKEN_SALT}{token}".encode(), digest_size=4).hexdigest()
        for token in tokenise(text)
    )


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (left != right))
            )
        previous = current
    return previous[-1]


def _could_reach(a: Sequence[str], b: Sequence[str], threshold: float) -> bool:
    longest = max(len(a), len(b))
    if longest == 0:
        return False
    if min(len(a), len(b)) / longest < threshold:
        return False
    shared = sum((Counter(a) & Counter(b)).values())
    return (shared / longest) >= threshold


def _ratio(a: Sequence[str], b: Sequence[str], threshold: float = 0.0) -> float:
    if not a or not b:
        return 0.0
    if threshold > 0.0 and not _could_reach(a, b, threshold):
        return 0.0
    return 1.0 - (_edit_distance(a, b) / max(len(a), len(b)))


def normalised_edit_ratio(a: str, b: str) -> float:
    return _ratio(tokenise(a), tokenise(b))


def ast_similarity(a: str, b: str, threshold: float = 0.0) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return _ratio(tokenise(a), tokenise(b), threshold)


def fingerprint(sample: Sample, source: str = "") -> Fingerprint:
    return Fingerprint(
        ref=sample.ref,
        ast=ast_fingerprint(sample.solution) if sample.solution else "",
        prompt_tokens=digest_tokens(sample.prompt),
        solution_tokens=digest_tokens(sample.solution),
        source=source,
    )


def scan_overlap(
    corpus_items: Sequence[Sample],
    reference_items: Sequence[Fingerprint],
    ast_threshold: float = AST_THRESHOLD,
    edit_threshold: float = EDIT_THRESHOLD,
) -> list[Overlap]:
    flagged: list[Overlap] = []
    if not reference_items:
        return flagged

    for item in corpus_items:
        probe = fingerprint(item)
        if len(probe.prompt_tokens) + len(probe.solution_tokens) < MIN_TOKENS:
            continue

        for reference in reference_items:
            ast_score = (
                ast_similarity(probe.ast, reference.ast, ast_threshold) if probe.ast else 0.0
            )
            edit_score = max(
                _ratio(probe.prompt_tokens, reference.prompt_tokens, edit_threshold),
                _ratio(probe.solution_tokens, reference.solution_tokens, edit_threshold),
            )
            over_ast = ast_score >= ast_threshold
            over_edit = edit_score >= edit_threshold
            if not (over_ast or over_edit):
                continue
            signal = "both" if over_ast and over_edit else ("ast" if over_ast else "edit")
            against = f"{reference.source}:{reference.ref}" if reference.source else reference.ref
            flagged.append(
                Overlap(
                    ref=item.ref,
                    against=against,
                    ast_similarity=ast_score,
                    edit_similarity=edit_score,
                    signal=signal,
                )
            )
            break
    return flagged


INSTANCE_THRESHOLD: Final[float] = 0.95


def scan_partitions(
    rotating: Sequence[Sample],
    fixed: Sequence[Sample],
    threshold: float = INSTANCE_THRESHOLD,
) -> list[Overlap]:
    """Two tasks drawn from the same template are supposed to read alike; that is
    what combinatorial instantiation means, and it is the defence against
    contamination rather than a symptom of it. What must not happen is the same
    *instance* appearing in both partitions, because the fixed set is the
    cross-round anchor and stops anchoring anything if the rotating draw can
    contain it. So this compares the concrete parameterisation, not the prose."""
    anchors: dict[str, str] = {}
    for item in fixed:
        anchors[" ".join(digest_tokens(item.instance or item.text))] = item.ref

    flagged: list[Overlap] = []
    for item in rotating:
        key = " ".join(digest_tokens(item.instance or item.text))
        exact = anchors.get(key)
        if exact is not None:
            flagged.append(
                Overlap(
                    ref=item.ref,
                    against=f"fixed:{exact}",
                    ast_similarity=0.0,
                    edit_similarity=1.0,
                    signal="instance",
                )
            )
            continue

        probe = digest_tokens(item.instance or item.text)
        for anchor_key, anchor_ref in anchors.items():
            score = _ratio(probe, anchor_key.split(), threshold)
            if score >= threshold:
                flagged.append(
                    Overlap(
                        ref=item.ref,
                        against=f"fixed:{anchor_ref}",
                        ast_similarity=0.0,
                        edit_similarity=score,
                        signal="instance",
                    )
                )
                break
    return flagged


def duplicate_prompts(items: Sequence[Sample]) -> list[Overlap]:
    seen: dict[str, str] = {}
    flagged: list[Overlap] = []
    for item in items:
        key = " ".join(tokenise(item.prompt))
        if key in seen:
            flagged.append(
                Overlap(
                    ref=item.ref,
                    against=seen[key],
                    ast_similarity=0.0,
                    edit_similarity=1.0,
                    signal="duplicate-prompt",
                )
            )
        else:
            seen[key] = item.ref
    return flagged


def fingerprint_dir() -> Path:
    return Path(__file__).resolve().parent / FINGERPRINT_DIR


def load_reference_fingerprints(directory: Path | None = None) -> list[Fingerprint]:
    root = directory or fingerprint_dir()
    if not root.is_dir():
        return []
    loaded: list[Fingerprint] = []
    for path in sorted(root.glob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                loaded.append(Fingerprint.from_row(json.loads(line)))
            except (ValueError, KeyError, TypeError) as exc:
                raise ContaminationError(f"{path.name}:{number} is malformed: {exc}") from exc
    return loaded


def write_fingerprints(path: Path, fingerprints: Sequence[Fingerprint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(f.to_row(), sort_keys=True) for f in fingerprints) + "\n",
        encoding="utf-8",
    )
