from __future__ import annotations

from itertools import pairwise

import pytest

from microtensor.core.constants import PAID_RANKS, RANK_DECAY, TRACK_THRESHOLD
from microtensor.scoring import (
    Candidate,
    allocate,
    apply_concentration_cap,
    apply_incumbent_decay,
    blend,
    combine_competitions,
    eligible,
    geometric_shares,
    normalise,
    origin_group,
    quantise,
    rank_with_hysteresis,
    score_task,
    to_uid_weights,
)


def _c(hotkey: str, score: float, **kw: int) -> Candidate:
    return Candidate(hotkey=hotkey, score=score, **kw)


def test_geometric_shares_sum_to_one() -> None:
    assert sum(geometric_shares()) == pytest.approx(1.0)


def test_geometric_shares_are_strictly_decreasing() -> None:
    assert all(a > b for a, b in pairwise(geometric_shares()))


def test_last_paid_rank_remains_worth_competing_for() -> None:
    shares = geometric_shares()
    assert shares[0] / shares[-1] < 5.0


def test_geometric_shares_reject_invalid_decay() -> None:
    with pytest.raises(ValueError):
        geometric_shares(PAID_RANKS, 1.0)


def test_eligible_drops_below_threshold() -> None:
    survivors = eligible([_c("a", TRACK_THRESHOLD - 0.01), _c("b", TRACK_THRESHOLD)])
    assert [c.hotkey for c in survivors] == ["b"]


def test_eligible_drops_unobserved_artifacts() -> None:
    survivors = eligible([_c("new", 0.9, rounds_observed=0), _c("seen", 0.5)])
    assert [c.hotkey for c in survivors] == ["seen"]


def test_hysteresis_holds_rank_against_a_marginal_challenger() -> None:
    ranked = rank_with_hysteresis(
        [_c("holder", 0.700), _c("challenger", 0.702)], previous=["holder"]
    )
    assert ranked[0].hotkey == "holder"


def test_hysteresis_yields_to_a_material_challenger() -> None:
    ranked = rank_with_hysteresis(
        [_c("holder", 0.700), _c("challenger", 0.760)], previous=["holder"]
    )
    assert ranked[0].hotkey == "challenger"


def test_hysteresis_protects_every_rank_not_only_the_first() -> None:
    candidates = [_c("a", 0.90), _c("b", 0.700), _c("copy", 0.702)]
    ranked = rank_with_hysteresis(candidates, previous=["a", "b"])
    assert [c.hotkey for c in ranked[:2]] == ["a", "b"]


def test_hysteresis_fills_a_rank_whose_holder_left() -> None:
    ranked = rank_with_hysteresis([_c("b", 0.5)], previous=["gone", "b"])
    assert ranked[0].hotkey == "b"


def test_incumbent_decay_moves_share_to_active_ranks() -> None:
    ranked = [_c("stale", 0.9, stale_rounds=4), _c("active", 0.8)]
    shares = geometric_shares(2, RANK_DECAY)
    adjusted = apply_incumbent_decay(shares, ranked)
    assert adjusted[0] < shares[0]
    assert adjusted[1] > shares[1]
    assert sum(adjusted) == pytest.approx(sum(shares))


def test_incumbent_decay_is_inert_without_staleness() -> None:
    ranked = [_c("a", 0.9), _c("b", 0.8)]
    shares = geometric_shares(2, RANK_DECAY)
    assert apply_incumbent_decay(shares, ranked) == shares


def test_allocation_pays_only_the_paid_ranks() -> None:
    candidates = [_c(f"m{i}", 0.9 - i * 0.01) for i in range(PAID_RANKS + 5)]
    allocation = allocate(candidates)
    assert len(allocation) == PAID_RANKS
    assert sum(allocation.values()) == pytest.approx(1.0)


def test_allocation_is_empty_when_nobody_clears_the_threshold() -> None:
    assert allocate([_c("a", 0.01), _c("b", 0.02)]) == {}


def test_concentration_cap_zeroes_the_excess_of_one_origin() -> None:
    weights = {f"m{i}": 0.125 for i in range(8)}
    origins = {f"m{i}": ("10.0" if i < 5 else f"9{i}.0") for i in range(8)}
    capped = apply_concentration_cap(weights, origins)
    survivors = [h for h in capped if h.startswith("m") and capped[h] > 0 and origins[h] == "10.0"]
    assert len(survivors) == 2


def test_concentration_cap_keeps_the_strongest_of_a_group() -> None:
    weights = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    origins = dict.fromkeys(("a", "b", "c", "d"), "10.0")
    capped = apply_concentration_cap(weights, origins)
    assert capped.get("a", 0) > 0
    assert capped.get("d", 0) == 0


def test_concentration_cap_is_inert_when_origins_are_spread() -> None:
    weights = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    origins = {"a": "10.0", "b": "20.0", "c": "30.0", "d": "40.0"}
    assert apply_concentration_cap(weights, origins) == pytest.approx(normalise(weights))


def test_origin_group_is_the_coldkey_itself() -> None:
    assert origin_group("5ColdkeyAAAA") == "5ColdkeyAAAA"
    assert origin_group("  5ColdkeyAAAA ") == "5ColdkeyAAAA"
    assert origin_group("") == ""


def test_three_hotkeys_under_one_coldkey_keep_exactly_two() -> None:
    weights = {f"m{i}": (8 - i) / 36 for i in range(8)}
    origins = {f"m{i}": ("ck-one" if i in (0, 3, 5) else f"ck-{i}") for i in range(8)}
    capped = apply_concentration_cap(weights, origins, fraction=0.25)
    survivors = [h for h in ("m0", "m3", "m5") if capped.get(h, 0.0) > 0.0]
    assert survivors == ["m0", "m3"]
    assert capped.get("m5", 0.0) == 0.0


def test_blend_falls_faster_than_it_rises() -> None:
    prior = {"a": 0.5, "b": 0.5}
    fall = 0.5 - blend({"a": 0.0, "b": 0.5}, prior)["a"]
    rise = blend({"a": 1.0, "b": 0.5}, prior)["a"] - 0.5
    assert fall > rise


def test_blend_normalises() -> None:
    assert sum(blend({"a": 0.7, "b": 0.3}, {"a": 0.5, "b": 0.5}).values()) == pytest.approx(1.0)


def test_blend_drops_vanishing_holders() -> None:
    assert "gone" not in blend({}, {"gone": 1e-12})


def test_combine_competitions_honours_the_class_weights() -> None:
    per = {
        ("code", "laptop"): {"a": 1.0},
        ("code", "edge-gpu"): {"b": 1.0},
    }
    combined = combine_competitions(per)
    assert combined["a"] == pytest.approx(0.60)
    assert combined["b"] == pytest.approx(0.40)
    assert sum(combined.values()) == pytest.approx(1.0)


def test_combine_competitions_falls_back_to_even_for_unweighted_classes() -> None:
    per = {
        ("code", "server-cpu"): {"a": 1.0},
        ("code", "embedded"): {"b": 1.0},
    }
    combined = combine_competitions(per)
    assert combined["a"] == pytest.approx(combined["b"])


def test_a_disabled_track_contributes_no_emission() -> None:
    per = {
        ("code", "laptop"): {"a": 1.0},
        ("document", "laptop"): {"b": 1.0},
    }
    combined = combine_competitions(per)
    assert combined["a"] == pytest.approx(1.0)
    assert "b" not in combined


def test_to_uid_weights_reports_unmatched_hotkeys() -> None:
    uid_weights, dropped = to_uid_weights({"a": 0.5, "ghost": 0.5}, {"a": 7})
    assert uid_weights == {7: pytest.approx(1.0)}
    assert dropped == ["ghost"]


def test_quantise_clamps_and_rounds() -> None:
    assert quantise(1.5) == 1.0
    assert quantise(-0.2) == 0.0
    assert quantise(0.123456789) == 0.1235


def test_two_validators_agree_after_quantisation() -> None:
    a = 0.1 + 0.2
    b = 0.3
    assert a != b
    assert quantise(a) == quantise(b)


def test_score_task_survives_malformed_output() -> None:
    assert score_task("extraction_f1", None, {"a"}) == 0.0


def test_exact_match_numeric_tolerates_formatting() -> None:
    assert score_task("exact_match_numeric", "the answer is 42", {"value": 42}) == 1.0
    assert score_task("exact_match_numeric", "41", {"value": 42}) == 0.0
