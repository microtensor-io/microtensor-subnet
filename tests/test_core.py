from __future__ import annotations

import pytest

from microtensor.core import (
    CLASSES,
    TRACKS,
    ArtifactFormat,
    DeclaredEnvelope,
    GateFailure,
    LoadManifest,
    MeasuredEnvelope,
    canonical_json,
    competitions,
    enabled_tracks,
    evaluate_gate,
    round_seed,
    select_deterministic,
    validate_registry,
)

_CLASS = CLASSES["edge-gpu"]

_DECLARED = DeclaredEnvelope(
    size_bytes=850 * 1024**2,
    peak_rss_bytes=2100 * 1024**2,
    p95_latency_ms=45,
)


def _measured(**over: object) -> MeasuredEnvelope:
    base: dict[str, object] = {
        "size_bytes": 800 * 1024**2,
        "peak_rss_bytes": 2 * 1024**3,
        "input_at_peak": {"context": 8192},
        "ttft_p50_ms": 30,
        "ttft_p95_ms": 41,
        "tokens_per_second": 112.4,
        "cold_start_ms": 900,
        "device_profile": "sha256:ref",
        "conforming": True,
    }
    base.update(over)
    return MeasuredEnvelope(**base)  # type: ignore[arg-type]


def test_enabled_shares_sum_to_one() -> None:
    validate_registry()
    assert sum(t.emission_share for t in enabled_tracks()) == pytest.approx(1.0)


def test_disabled_tracks_earn_nothing() -> None:
    for track in TRACKS.values():
        if not track.enabled:
            assert track.emission_share == 0.0


def test_size_ceiling_below_rss_ceiling() -> None:
    for cls in CLASSES.values():
        assert cls.max_size_bytes < cls.max_rss_bytes


def test_the_launch_scope_is_code_on_two_classes() -> None:
    assert competitions() == [("code", "laptop"), ("code", "edge-gpu")]


def test_class_gating_bounds_is_competable() -> None:
    from microtensor.core import is_competable

    assert is_competable("code", "laptop")
    assert is_competable("code", "edge-gpu")
    assert not is_competable("code", "embedded")
    assert not is_competable("code", "server-cpu")
    assert not is_competable("document", "laptop")


def test_registry_rejects_a_track_naming_an_unknown_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from microtensor.core.tracks import Decoding, Modality, Track

    bogus = Track(
        id="bogus",
        modality=Modality.TEXT,
        metric="extraction_f1",
        decoding=Decoding.GREEDY,
        emission_share=0.0,
        work_unit="generated_tokens",
        classes=("no-such-class",),
    )
    monkeypatch.setitem(TRACKS, "bogus", bogus)
    with pytest.raises(ValueError, match="unknown classes"):
        validate_registry()


def test_canonical_json_is_key_order_independent() -> None:
    a = {"b": 1, "a": 2, "c": {"z": 1, "y": 2}}
    b = {"c": {"y": 2, "z": 1}, "a": 2, "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_canonical_json_normalises_composed_and_decomposed_unicode() -> None:
    assert canonical_json({"name": "café"}) == canonical_json({"name": "café"})


def test_canonical_json_normalises_keys() -> None:
    assert canonical_json({"café": 1}) == canonical_json({"café": 1})


def test_canonical_json_has_no_insignificant_whitespace() -> None:
    assert b" " not in canonical_json({"a": 1, "b": [1, 2]})


def test_same_seed_draws_the_same_tasks() -> None:
    corpus = [f"task-{i:04d}" for i in range(500)]
    seed = round_seed("0xdeadbeef", "code", "edge-gpu")
    assert select_deterministic(corpus, seed, 30) == select_deterministic(corpus, seed, 30)


def test_different_competitions_draw_different_tasks() -> None:
    corpus = [f"task-{i:04d}" for i in range(500)]
    a = select_deterministic(corpus, round_seed("0xabc", "code", "edge-gpu"), 30)
    b = select_deterministic(corpus, round_seed("0xabc", "code", "laptop"), 30)
    assert a != b


def test_selection_returns_the_requested_count() -> None:
    corpus = [f"task-{i}" for i in range(100)]
    assert len(select_deterministic(corpus, "seed", 25)) == 25


def test_selection_degrades_to_the_whole_corpus() -> None:
    corpus = ["a", "b", "c"]
    assert select_deterministic(corpus, "seed", 10) == sorted(corpus)


def test_gate_admits_a_truthful_artifact() -> None:
    assert evaluate_gate(_measured(), _DECLARED, _CLASS).admitted


def test_gate_rejects_over_class_ceiling() -> None:
    result = evaluate_gate(_measured(peak_rss_bytes=5 * 1024**3), _DECLARED, _CLASS)
    assert not result.admitted
    assert GateFailure.RSS_CEILING in result.failures


def test_gate_rejects_under_declaration_inside_the_ceiling() -> None:
    result = evaluate_gate(_measured(peak_rss_bytes=3500 * 1024**2), _DECLARED, _CLASS)
    assert not result.admitted
    assert GateFailure.RSS_OVER_DECLARED in result.failures
    assert GateFailure.RSS_CEILING not in result.failures


def test_declaration_tolerance_absorbs_device_noise() -> None:
    nudged = int(_DECLARED.peak_rss_bytes * 1.01)
    assert evaluate_gate(_measured(peak_rss_bytes=nudged), _DECLARED, _CLASS).admitted


def test_tolerance_never_applies_to_a_class_ceiling() -> None:
    over = int(_CLASS.max_rss_bytes * 1.01)
    generous = DeclaredEnvelope(
        size_bytes=_DECLARED.size_bytes,
        peak_rss_bytes=over * 2,
        p95_latency_ms=_DECLARED.p95_latency_ms,
    )
    result = evaluate_gate(_measured(peak_rss_bytes=over), generous, _CLASS)
    assert not result.admitted
    assert GateFailure.RSS_CEILING in result.failures


def test_latency_declaration_gets_a_relative_and_an_absolute_floor() -> None:
    assert evaluate_gate(_measured(ttft_p95_ms=59), _DECLARED, _CLASS).admitted
    result = evaluate_gate(_measured(ttft_p95_ms=61), _DECLARED, _CLASS)
    assert GateFailure.LATENCY_OVER_DECLARED in result.failures


def test_latency_slack_never_reaches_the_class_ceiling() -> None:
    declared = DeclaredEnvelope(
        size_bytes=_DECLARED.size_bytes,
        peak_rss_bytes=_DECLARED.peak_rss_bytes,
        p95_latency_ms=_CLASS.max_p95_ms,
    )
    result = evaluate_gate(_measured(ttft_p95_ms=_CLASS.max_p95_ms + 5), declared, _CLASS)
    assert GateFailure.LATENCY_CEILING in result.failures


def test_non_conforming_host_checks_ceilings_not_declarations() -> None:
    result = evaluate_gate(
        _measured(peak_rss_bytes=3500 * 1024**2, conforming=False), _DECLARED, _CLASS
    )
    assert result.admitted


def test_gate_reports_every_failing_axis() -> None:
    result = evaluate_gate(
        _measured(size_bytes=9 * 1024**3, peak_rss_bytes=9 * 1024**3, ttft_p95_ms=999),
        _DECLARED,
        _CLASS,
    )
    assert len(result.failures) >= 3


def test_manifest_requires_a_declared_maximum_input() -> None:
    with pytest.raises(ValueError, match="max_input"):
        LoadManifest(
            format=ArtifactFormat.ONNX,
            quantization="int8",
            entrypoint="model.onnx",
            max_input={},
        )


def test_declared_envelope_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError):
        DeclaredEnvelope(size_bytes=0, peak_rss_bytes=1, p95_latency_ms=1)
