from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from microtensor_subnet.artifact import contract, gguf
from microtensor_subnet.artifact.params import (
    ParameterError,
    count_parameters,
    model_parameters,
    safetensors_in,
)

_SHA = re.compile(r"^(sha256:)?[0-9a-f]{64}$")
_GIT = re.compile(r"^[0-9a-f]{40}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)\s*==\s*([A-Za-z0-9_.+!-]+)\s*$")


class ManifestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SlmPin:
    repo: str
    revision: str
    files: dict[str, str]


@dataclass(frozen=True, slots=True)
class Manifest:
    arena: str
    components: tuple[str, ...]
    hardware_class: str
    files: dict[str, str]
    slm: SlmPin | None = None

    @property
    def has(self) -> frozenset[str]:
        return frozenset(self.components)

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema_version": contract.SCHEMA_VERSION,
            "arena": self.arena,
            "components": list(self.components),
            "hardware_class": self.hardware_class,
            "files": dict(sorted(self.files.items())),
        }
        if self.slm is not None:
            body["slm"] = {
                "repo": self.slm.repo,
                "revision": self.slm.revision,
                "files": dict(sorted(self.slm.files.items())),
            }
        return body

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Manifest:
        if int(raw.get("schema_version", contract.SCHEMA_VERSION)) != contract.SCHEMA_VERSION:
            raise ManifestError("unknown manifest schema version")
        arena = str(raw.get("arena", "")).strip()
        if not arena:
            raise ManifestError("manifest.json names no arena")
        components = tuple(str(c) for c in raw.get("components") or [])
        unknown = [c for c in components if c not in contract.COMPONENTS]
        if not components or unknown or len(set(components)) != len(components):
            raise ManifestError(f"components must be a non-empty subset of {contract.COMPONENTS}")
        hardware_class = str(raw.get("hardware_class", ""))
        if hardware_class not in contract.HARDWARE_CLASSES:
            raise ManifestError(f"hardware_class must be one of {contract.HARDWARE_CLASSES}")
        files = _digests(raw.get("files"), "files")
        slm: SlmPin | None = None
        pin = raw.get("slm")
        if "slm" in components:
            if not isinstance(pin, Mapping):
                raise ManifestError("a submission with an slm must pin its repository")
            repo = str(pin.get("repo", ""))
            revision = str(pin.get("revision", "")).lower()
            if not _REPO.match(repo):
                raise ManifestError(f"slm.repo {repo!r} is not a hugging face repository id")
            if not _GIT.match(revision):
                raise ManifestError("slm.revision must be the 40 hex git commit sha")
            slm_files = _digests(pin.get("files"), "slm.files")
            if not any(name.endswith(".safetensors") for name in slm_files):
                raise ManifestError("slm.files must hash at least one safetensors file")
            slm = SlmPin(repo=repo, revision=revision, files=slm_files)
        elif pin is not None:
            raise ManifestError("slm is pinned but not submitted")
        return cls(
            arena=arena,
            components=components,
            hardware_class=hardware_class,
            files=files,
            slm=slm,
        )


def _digests(raw: Any, where: str) -> dict[str, str]:
    if not isinstance(raw, Mapping) or not raw:
        raise ManifestError(f"{where} must map paths to sha256 digests")
    found: dict[str, str] = {}
    for name, value in raw.items():
        path = str(name)
        digest = str(value).lower()
        if path.startswith("/") or ".." in path.split("/"):
            raise ManifestError(f"{where} entry {path!r} is not a relative path")
        if not _SHA.match(digest):
            raise ManifestError(f"{where} entry {path!r} carries no sha256 digest")
        found[path] = digest if digest.startswith("sha256:") else f"sha256:{digest}"
    return found


UNTRACKED_DIRS = frozenset({".git", ".cache"})
UNTRACKED_FILES = frozenset({".gitattributes", ".complete"})


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def tree_files(root: Path, *, skip: Iterable[str] = ()) -> list[str]:
    skipped = set(skip)
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and not UNTRACKED_DIRS & set(p.parts)
        and p.name not in UNTRACKED_FILES
        and p.relative_to(root).as_posix() not in skipped
    )


def untracked(name: str) -> bool:
    parts = name.split("/")
    return (
        bool(UNTRACKED_DIRS & set(parts))
        or parts[-1] in UNTRACKED_FILES
        or (len(parts) > 1 and parts[0] == contract.SLM_FOLDER)
    )


def tracked(files: dict[str, str]) -> dict[str, str]:
    return {name: digest for name, digest in files.items() if not untracked(name)}


def hash_tree(root: Path, *, skip: Iterable[str] = ()) -> dict[str, str]:
    return {
        name: sha256_of(root / name) for name in tree_files(root, skip=skip) if not untracked(name)
    }


@dataclass(slots=True)
class Findings:
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def load_manifest(root: Path) -> Manifest:
    path = root / contract.MANIFEST_FILE
    if not path.is_file():
        raise ManifestError(f"{contract.MANIFEST_FILE} is missing")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ManifestError(f"{contract.MANIFEST_FILE} is not valid json: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{contract.MANIFEST_FILE} must be a json object")
    try:
        return Manifest.from_dict(raw)
    except ManifestError:
        raise
    except (TypeError, ValueError, AttributeError) as exc:
        raise ManifestError(f"{contract.MANIFEST_FILE} is malformed: {exc}") from exc


def check_artifact(root: Path, manifest: Manifest) -> Findings:
    findings = Findings()
    present = set(tree_files(root, skip=(contract.MANIFEST_FILE,)))
    for name in (contract.CONFIG_FILE, contract.MAIN_FILE, contract.REQUIREMENTS_FILE):
        if name not in present:
            findings.problems.append(f"{name} is missing")
    required: list[str] = []
    if "router" in manifest.has:
        required += list(contract.ROUTER_FILES)
    if "harness" in manifest.has:
        required += list(contract.HARNESS_FILES)
    if "escalation" in manifest.has:
        required += list(contract.ESCALATION_FILES)
    for name in required:
        if name not in present:
            findings.problems.append(f"{name} is required by the components submitted")
    listed = set(tracked(manifest.files))
    for name in sorted(present - listed):
        findings.problems.append(f"{name} is in the repository but not in manifest.files")
    for name in sorted(listed - present):
        findings.problems.append(f"{name} is in manifest.files but not in the repository")
    for name in sorted(listed & present):
        if sha256_of(root / name) != manifest.files[name]:
            findings.problems.append(f"{name} does not match its manifest digest")
    code_bytes = sum(
        (root / name).stat().st_size
        for name in present
        if name.split("/")[0] in contract.CODE_DIRS and (root / name).is_file()
    )
    if code_bytes > contract.CODE_LIMIT_BYTES:
        findings.problems.append(
            f"harness and escalation code is {code_bytes} bytes, over {contract.CODE_LIMIT_BYTES}"
        )
    if contract.CONFIG_FILE in present:
        findings.problems += check_config(root / contract.CONFIG_FILE)
    if contract.REQUIREMENTS_FILE in present:
        findings.problems += check_requirements(root / contract.REQUIREMENTS_FILE)
    if "router" in manifest.has and contract.ROUTER_FILES[1] in present:
        try:
            parameters = count_parameters([root / contract.ROUTER_FILES[1]])
            if parameters > contract.ROUTER_MAX_PARAMETERS:
                findings.problems.append(
                    f"router has {parameters} parameters, over {contract.ROUTER_MAX_PARAMETERS}"
                )
            findings.notes.append(f"router parameters {parameters}")
        except ParameterError as exc:
            findings.problems.append(str(exc))
    for name in present:
        if _looks_secret(root / name):
            findings.problems.append(f"{name} looks like it carries a credential")
    return findings


def check_config(path: Path) -> list[str]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"{contract.CONFIG_FILE} is not valid json: {exc}"]
    if not isinstance(config, Mapping):
        return [f"{contract.CONFIG_FILE} must be an object"]
    problems: list[str] = []
    for section, limits in contract.CONFIG_LIMITS.items():
        block = config.get(section) or {}
        if not isinstance(block, Mapping):
            problems.append(f"config.{section} must be an object")
            continue
        for key, ceiling in limits.items():
            value = block.get(key)
            if value is None:
                continue
            try:
                number = int(value)
            except (TypeError, ValueError):
                problems.append(f"config.{section}.{key} must be an integer")
                continue
            if number <= 0 or number > ceiling:
                problems.append(f"config.{section}.{key} must lie in 1..{ceiling}")
    return problems


def check_requirements(path: Path) -> list[str]:
    problems: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = _REQUIREMENT.match(line)
        if match is None:
            problems.append(f"requirement {line!r} is not pinned with ==")
            continue
        name = match.group(1).lower().replace("_", "-")
        version = match.group(2)
        allowed = contract.ALLOWED_PACKAGES.get(name)
        if allowed is None:
            problems.append(f"package {name} is not on the allowed list")
        elif allowed != version:
            problems.append(f"package {name} must be pinned to {allowed}, not {version}")
    return problems


_SECRET_MARKERS = (
    b"sk-",
    b"AKIA",
    b"-----BEGIN",
    b"api_key=",
    b"ANTHROPIC_API_KEY=",
    b"OPENAI_API_KEY=",
)


def _looks_secret(path: Path) -> bool:
    if path.suffix in (".safetensors", ".bin", ".pt", ".db", ".sqlite"):
        return False
    try:
        head = path.read_bytes()[:65536]
    except OSError:
        return False
    return any(marker in head for marker in _SECRET_MARKERS)


def check_slm_repo(
    root: Path,
    pin: SlmPin | None,
    allowed_bases: Iterable[str] = (),
    *,
    bounds: tuple[int, int] | None = None,
    formats: Iterable[str] = (),
) -> Findings:
    findings = Findings()
    present = set(tree_files(root))
    weights = safetensors_in(root)
    ggufs = gguf.gguf_in(root)
    required = (
        contract.SLM_GGUF_REQUIRED_FILES if ggufs and not weights else contract.SLM_REQUIRED_FILES
    )
    for name in required:
        if name not in present:
            findings.problems.append(f"{name} is missing from the slm repository")
    for name in contract.SLM_TRAINING_FILES:
        if name not in present:
            findings.problems.append(f"{name} is required for the archive")
    if not list(root.glob(contract.SLM_METRICS_GLOB)):
        findings.problems.append("training/runs/<id>/metrics.jsonl is required for the archive")
    low, high = bounds or (contract.SLM_MIN_PARAMETERS, contract.SLM_MAX_PARAMETERS)
    if not weights and not ggufs:
        findings.problems.append("no model.safetensors (or shards) or model.gguf in the root")
    else:
        try:
            parameters = model_parameters(root)
            findings.notes.append(f"slm parameters {parameters}")
            if not low <= parameters <= high:
                findings.problems.append(
                    f"slm has {parameters} parameters; the arena allows {low} to {high}"
                )
        except ParameterError as exc:
            findings.problems.append(str(exc))
    if ggufs and not weights:
        findings.problems += _gguf_problems(ggufs[0], [f.upper() for f in formats])
    if pin is not None:
        for name, digest in tracked(pin.files).items():
            if name not in present:
                findings.problems.append(f"pinned slm file {name} is missing")
            elif sha256_of(root / name) != digest:
                findings.problems.append(f"pinned slm file {name} does not match its digest")
    bases = [b.lower() for b in allowed_bases]
    recipe = root / "training" / "recipe.json"
    if bases and recipe.is_file():
        try:
            base = str(json.loads(recipe.read_text(encoding="utf-8")).get("base_model", ""))
        except ValueError:
            base = ""
        if base.lower() not in bases:
            findings.problems.append(f"base model {base!r} is not on the arena's allowed list")
    return findings


def _gguf_problems(path: Path, formats: list[str]) -> list[str]:
    try:
        header = gguf.read(path)
    except (OSError, gguf.GgufError) as exc:
        return [str(exc)]
    problems: list[str] = []
    low = header.below_four_bits
    if low:
        problems.append(f"{len(low)} tensors are below 4 bit, for example {low[0]}")
    if formats and header.file_type.upper() not in formats:
        problems.append(f"GGUF type {header.file_type or 'unknown'} is not one of {formats}")
    return problems


def build_manifest(
    root: Path,
    *,
    arena: str,
    components: Iterable[str],
    hardware_class: str = contract.HARDWARE_CLASSES[0],
    slm_repo: str = "",
    slm_revision: str = "",
    slm_root: Path | None = None,
) -> Manifest:
    pin: SlmPin | None = None
    parts = tuple(components)
    if "slm" in parts:
        if slm_root is None or not slm_repo or not slm_revision:
            raise ManifestError("an slm submission needs the slm repository, revision and files")
        files = {
            name: sha256_of(slm_root / name)
            for name in tree_files(slm_root)
            if not name.startswith("training/")
        }
        pin = SlmPin(repo=slm_repo, revision=slm_revision.lower(), files=files)
    return Manifest(
        arena=arena,
        components=parts,
        hardware_class=hardware_class,
        files=hash_tree(root, skip=(contract.MANIFEST_FILE,)),
        slm=pin,
    )


def write_manifest(root: Path, manifest: Manifest) -> Path:
    path = root / contract.MANIFEST_FILE
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def manifest_sha256(root: Path) -> str:
    return sha256_of(root / contract.MANIFEST_FILE)
