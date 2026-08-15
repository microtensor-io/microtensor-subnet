from __future__ import annotations

import json
from pathlib import Path

import pytest

from microtensor.chain.commitment import build_commitment
from microtensor.core.hashing import digest_tree
from microtensor.core.protocol import ArtifactFormat, DeclaredEnvelope, LoadManifest
from microtensor.registry import (
    ArtifactCache,
    ArtifactManifest,
    ArtifactMismatch,
    CacheError,
    FetchError,
    ManifestError,
    Unfetchable,
    build_manifest,
    materialise,
    parse_source,
    verify_tree,
)
from microtensor.registry.fetch import FETCHERS, fetcher_for

LOAD = LoadManifest(
    format=ArtifactFormat.ONNX,
    quantization="int8",
    entrypoint="model.onnx",
    max_input={"tokens": 4096},
    preprocessing={"tokenizer": "tokenizer.json"},
)
DECLARED = DeclaredEnvelope(size_bytes=1 << 20, peak_rss_bytes=1 << 30, p95_latency_ms=150)


@pytest.fixture
def artifact_dir(tmp_path: Path) -> Path:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "model.onnx").write_bytes(b"graph" * 100)
    (root / "tokenizer.json").write_text('{"model":{}}', encoding="utf-8")
    return root


def _manifest(root: Path, **kw: object) -> ArtifactManifest:
    fields: dict[str, object] = {
        "hotkey": "hk-miner",
        "round_index": 7,
        "track": "code",
        "hardware_class": "laptop",
        "source": "https:models.example.com/mt-code",
        "load": LOAD,
        "declared": DECLARED,
    }
    fields.update(kw)
    return build_manifest(root, **fields)  # type: ignore[arg-type]


def test_manifest_digest_matches_the_tree_on_disk(artifact_dir: Path) -> None:
    assert _manifest(artifact_dir).artifact_digest == digest_tree(artifact_dir)


def test_manifest_round_trips_through_json(artifact_dir: Path) -> None:
    manifest = _manifest(artifact_dir).signed_with("deadbeef")
    assert ArtifactManifest.from_json(manifest.to_json()) == manifest


def test_manifest_rejects_a_tampered_digest(artifact_dir: Path) -> None:
    payload = json.loads(_manifest(artifact_dir).to_json())
    payload["artifact_digest"] = "sha256:" + "00" * 32
    with pytest.raises(ManifestError):
        ArtifactManifest.from_json(json.dumps(payload))


def test_manifest_rejects_a_path_that_escapes_the_artifact(artifact_dir: Path) -> None:
    payload = json.loads(_manifest(artifact_dir).to_json())
    payload["files"][0]["path"] = "../../etc/passwd"
    payload.pop("artifact_digest")
    with pytest.raises(ManifestError):
        ArtifactManifest.from_json(json.dumps(payload))


def test_manifest_demands_the_entrypoint_be_shipped(artifact_dir: Path) -> None:
    load = LoadManifest(
        format=ArtifactFormat.ONNX,
        quantization="int8",
        entrypoint="absent.onnx",
        max_input={"tokens": 4096},
    )
    with pytest.raises(ManifestError):
        _manifest(artifact_dir, load=load)


def test_manifest_rejects_a_closed_competition(artifact_dir: Path) -> None:
    with pytest.raises(ManifestError):
        _manifest(artifact_dir, track="speech")


def test_manifest_refuses_a_size_declaration_below_its_own_files(artifact_dir: Path) -> None:
    manifest = _manifest(
        artifact_dir,
        declared=DeclaredEnvelope(size_bytes=1, peak_rss_bytes=1 << 30, p95_latency_ms=150),
    )
    fits, reason = manifest.fits_class()
    assert not fits
    assert reason


def test_manifest_is_bound_to_its_commitment(artifact_dir: Path) -> None:
    manifest = _manifest(artifact_dir)
    commitment = build_commitment(7, "code", "laptop", manifest.digest(), manifest.source)
    ok, reason = manifest.matches(commitment)
    assert ok, reason


def test_a_commitment_for_another_round_does_not_match(artifact_dir: Path) -> None:
    manifest = _manifest(artifact_dir)
    commitment = build_commitment(8, "code", "laptop", manifest.digest(), manifest.source)
    ok, reason = manifest.matches(commitment)
    assert not ok
    assert "round" in reason


def test_verify_tree_catches_a_swapped_file(artifact_dir: Path) -> None:
    manifest = _manifest(artifact_dir)
    (artifact_dir / "model.onnx").write_bytes(b"other")
    ok, reason = verify_tree(artifact_dir, manifest)
    assert not ok
    assert "model.onnx" in reason


def test_verify_tree_catches_a_smuggled_file(artifact_dir: Path) -> None:
    manifest = _manifest(artifact_dir)
    (artifact_dir / "extra.bin").write_bytes(b"payload")
    ok, reason = verify_tree(artifact_dir, manifest)
    assert not ok
    assert "unlisted" in reason


def test_cache_orders_by_access_not_by_wall_clock(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "cache", cap_bytes=8192)
    for index in range(3):
        staged = tmp_path / f"s{index}"
        staged.mkdir()
        (staged / "blob").write_bytes(b"x" * 100)
        cache.adopt("sha256:" + f"{index:02d}" * 32, staged)

    cache.get("sha256:" + "00" * 32)
    order = [e.digest for e in cache.entries()]
    assert order[-1] == "sha256:" + "00" * 32
    assert order[0] == "sha256:" + "01" * 32


def test_cache_evicts_least_recently_used_first(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "cache", cap_bytes=2048)
    for index in range(2):
        staged = tmp_path / f"stage{index}"
        staged.mkdir()
        (staged / "blob").write_bytes(b"x" * 1000)
        cache.adopt("sha256:" + f"{index:02d}" * 32, staged)

    cache.get("sha256:" + "00" * 32)
    staged = tmp_path / "stage2"
    staged.mkdir()
    (staged / "blob").write_bytes(b"x" * 1000)
    cache.adopt("sha256:" + "02" * 32, staged)

    assert cache.has("sha256:" + "00" * 32)
    assert not cache.has("sha256:" + "01" * 32)


def test_cache_refuses_an_artifact_larger_than_its_cap(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "cache", cap_bytes=100)
    with pytest.raises(CacheError):
        cache.reserve(1000)


def test_cache_survives_a_corrupt_index(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    cache = ArtifactCache(root, cap_bytes=4096)
    staged = tmp_path / "stage"
    staged.mkdir()
    (staged / "blob").write_bytes(b"x" * 100)
    cache.adopt("sha256:" + "ab" * 32, staged)

    (root / "index.json").write_text("not json", encoding="utf-8")
    rebuilt = ArtifactCache(root, cap_bytes=4096)
    assert rebuilt.has("sha256:" + "ab" * 32)
    assert rebuilt.total_bytes == 100


def test_cache_forgets_a_directory_that_vanished(tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "cache"
    cache = ArtifactCache(root, cap_bytes=4096)
    staged = tmp_path / "stage"
    staged.mkdir()
    (staged / "blob").write_bytes(b"x" * 100)
    path = cache.adopt("sha256:" + "cd" * 32, staged)

    shutil.rmtree(path)
    assert not ArtifactCache(root, cap_bytes=4096).has("sha256:" + "cd" * 32)


def test_cache_rejects_a_malformed_digest(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "cache")
    with pytest.raises(CacheError):
        cache.has("not-a-digest")


def test_source_parsing_demands_a_locator() -> None:
    assert parse_source("hf:acme/model@rev") == ("hf", "acme/model@rev")
    with pytest.raises(FetchError):
        parse_source("hf:")


def test_unknown_scheme_is_unfetchable() -> None:
    with pytest.raises(Unfetchable):
        fetcher_for("ftp")


def test_materialise_serves_a_cache_hit_without_fetching(
    tmp_path: Path, artifact_dir: Path
) -> None:
    manifest = _manifest(artifact_dir)
    cache = ArtifactCache(tmp_path / "cache", cap_bytes=1 << 20)

    staged = tmp_path / "stage"
    staged.mkdir()
    for entry in manifest.files:
        (staged / entry.path).write_bytes((artifact_dir / entry.path).read_bytes())
    cache.adopt(manifest.artifact_digest, staged)

    path = materialise(manifest, cache, workdir=tmp_path / "work")
    assert path.is_dir()
    assert (path / "model.onnx").exists()


def test_materialise_verifies_what_it_fetched(
    tmp_path: Path, artifact_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(artifact_dir)
    cache = ArtifactCache(tmp_path / "cache", cap_bytes=1 << 20)
    workdir = tmp_path / "work"
    workdir.mkdir()

    def liar(locator: str, relpath: str, destination: Path, timeout: int) -> None:
        destination.write_bytes(b"not what was promised")

    monkeypatch.setitem(FETCHERS, "https", liar)
    with pytest.raises(ArtifactMismatch):
        materialise(manifest, cache, workdir=workdir, sleep=lambda _: None)


def test_materialise_abstains_when_the_source_is_down(
    tmp_path: Path, artifact_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(artifact_dir)
    cache = ArtifactCache(tmp_path / "cache", cap_bytes=1 << 20)
    workdir = tmp_path / "work"
    workdir.mkdir()

    def down(locator: str, relpath: str, destination: Path, timeout: int) -> None:
        raise Unfetchable("connection refused")

    monkeypatch.setitem(FETCHERS, "https", down)
    with pytest.raises(Unfetchable):
        materialise(manifest, cache, workdir=workdir, attempts=2, sleep=lambda _: None)


def test_materialise_stores_a_clean_fetch(
    tmp_path: Path, artifact_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(artifact_dir)
    cache = ArtifactCache(tmp_path / "cache", cap_bytes=1 << 20)
    workdir = tmp_path / "work"
    workdir.mkdir()

    def honest(locator: str, relpath: str, destination: Path, timeout: int) -> None:
        destination.write_bytes((artifact_dir / relpath).read_bytes())

    monkeypatch.setitem(FETCHERS, "https", honest)
    path = materialise(manifest, cache, workdir=workdir, sleep=lambda _: None)
    ok, reason = verify_tree(path, manifest)
    assert ok, reason
    assert cache.has(manifest.artifact_digest)
