from __future__ import annotations

import dataclasses
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from calico.domain import Evidence
from calico.evaluation import (
    BAR_AT,
    DISAGREEMENT_GAP,
    GEOMETRY_BAND,
    OUTSIDE_SHORTLIST,
    PATCH_SAMPLE_BANDS,
    PRECISION_AT,
    RUNNING_AFTER,
    RUNNING_LINE,
    TASTE_FIRST,
    TASTE_RANDOM,
    TASTE_SAMPLE,
    TOP_SHARE,
    VALUE_QUOTAS,
    VALUE_STRATA,
    AnchorRating,
    Averaged,
    ClearCalls,
    ComparedPhoto,
    Drawn,
    Evaluated,
    ExampleEvaluated,
    FinishedReport,
    FinishedStory,
    FirstPicks,
    LabelledSearch,
    LabelledView,
    LocationRatings,
    Pool,
    RatedStory,
    Scored,
    SearchScore,
    SignalledStory,
    StoryArmKey,
    StoryVerdict,
    TastePhoto,
    TasteReport,
    TasteRow,
    Threshold,
    _Judged,
    _top_share,
    apart_from,
    auc,
    average,
    best_threshold,
    choose_examples,
    clear_calls,
    count_copies_once,
    criteria_intervals,
    cut_at,
    example_intervals,
    finished_intervals,
    learn_stories,
    learned_apart,
    measure_anchors,
    measure_compare,
    measure_examples,
    measure_signals,
    measure_stories,
    measure_taste,
    measure_view_geometry,
    measure_views,
    ndcg_at,
    pair_weighted_auc,
    photos_to_compare,
    precision_at,
    sample_by_match,
    sample_by_value,
    sample_for_anchors,
    sample_for_labelling,
    score_search,
    shipped_strength,
    stories_to_rate,
    swap_wins,
    taste_apart,
    taste_disagreements,
    view_differences,
)
from calico.models.taste import TasteWeights, taste_probability
from calico.selection.copies import COPY_SIMILARITY
from tests import factories as f

if TYPE_CHECKING:
    from collections.abc import Sequence

    from calico.features.base import Embedder


def test_precision_counts_partly_as_half() -> None:
    assert precision_at([2, 1, 0, 2], k=4) == pytest.approx(0.625)
    assert precision_at([2, 2], k=10) == pytest.approx(1.0)  # fewer than k: over what exists


def test_ndcg_is_one_for_the_ideal_order_and_lower_otherwise() -> None:
    assert ndcg_at([2, 2, 1, 0], k=10) == pytest.approx(1.0)
    worse = ndcg_at([0, 1, 2, 2], k=10)
    assert 0 < worse < 1
    # Only the top k counts.
    assert ndcg_at([0, 0, 2], k=2) == 0.0


def test_ndcg_without_any_match_is_undefined() -> None:
    assert math.isnan(ndcg_at([0, 0, 0], k=10))


def test_auc_is_the_chance_a_match_outranks_a_non_match() -> None:
    assert auc([0.9, 0.8, 0.1, 0.2], [2, 2, 0, 0]) == pytest.approx(1.0)
    assert auc([0.1, 0.2, 0.9, 0.8], [2, 2, 0, 0]) == pytest.approx(0.0)
    assert auc([0.5, 0.5], [2, 0]) == pytest.approx(0.5)  # a tie is a coin toss
    assert auc([0.9, 0.5, 0.1], [2, 1, 0]) == pytest.approx(1.0)  # "partly" is left out


def test_auc_needs_both_matches_and_non_matches() -> None:
    assert math.isnan(auc([0.9, 0.8], [2, 2]))


def test_best_threshold_separates_matches_from_the_rest() -> None:
    cut = best_threshold([0.30, 0.28, 0.26, 0.20, 0.18], [2, 2, 2, 0, 0])
    assert cut.f1 == pytest.approx(1.0)
    assert 0.20 < cut.threshold <= 0.26


def test_sample_starts_with_the_top_and_reaches_into_the_rest() -> None:
    ranked = [f"k{i:03d}" for i in range(500)]
    sample = sample_for_labelling(ranked, seed=7, top=36, near=12, rest=12)
    assert len(sample) == len(set(sample)) == 60
    chosen = {int(k[1:]) for k in sample}
    assert set(range(36)) <= chosen
    assert len([i for i in chosen if 36 <= i < 300]) == 12
    assert len([i for i in chosen if i >= 300]) == 12
    assert sample != sorted(sample)  # shown shuffled, so rank cannot sway the judgement


def keys(drawn: Sequence[Drawn]) -> list[str]:
    return [d.key for d in drawn]


def place(n: int = 400) -> list[str]:
    """A place's photos, one per burst, best first by General's strength."""
    return [f"p{i:03d}" for i in range(n)]


def test_a_session_by_value_goes_mostly_to_what_the_templates_show_first() -> None:
    ranked = place()
    general = ranked[:40]  # General shows its strongest first
    yours = ranked[20:40] + ranked[200:220]  # Yours agrees on 20, and differs on 20
    drawn = sample_by_value(ranked, general, yours, seed=7)
    assert len(keys(drawn)) == len(set(keys(drawn))) == 60
    strata = Counter(d.stratum for d in drawn)
    assert strata == {
        "random": 15,
        "first: General only": 15,
        "first: Yours only": 15,
        "first: both": 5,
        "top quarter": 10,
    }
    random = {d.key for d in drawn if d.stratum == "random"}
    for d in drawn:
        if d.stratum == "first: General only":
            assert d.key in ranked[:20]
            assert d.inclusion == 15 / len(set(ranked[:20]) - random)
        elif d.stratum == "first: Yours only":
            assert d.key in ranked[200:220]
        elif d.stratum == "first: both":
            assert d.key in ranked[20:40]
        elif d.stratum == "top quarter":
            assert d.key in ranked[40:100]
        else:
            assert d.inclusion == 15 / 400  # random: any photo at the place, alike
    assert keys(drawn) != sorted(keys(drawn))  # shuffled: the order hints at no stratum


def test_turns_a_stratum_cannot_fill_pass_to_the_next() -> None:
    """Without a trained Yours there is nothing to disagree with, so General's first picks
    take their turns and the top quarter the rest."""
    ranked = place()
    drawn = sample_by_value(ranked, ranked[:40], (), seed=7)
    assert Counter(d.stratum for d in drawn) == {
        "random": 15,
        "first: General only": 15,
        "top quarter": 30,
    }


def test_a_session_by_value_leaves_out_what_was_rated() -> None:
    ranked = place()
    rated = set(ranked[:30])  # most of General's first picks: a second session there
    drawn = sample_by_value(ranked, ranked[:40], ranked[200:240], seed=3, exclude=rated)
    assert not rated & set(keys(drawn))
    assert len(drawn) == 60
    general_only = [d for d in drawn if d.stratum == "first: General only"]
    assert {d.key for d in general_only} <= set(ranked[30:40])
    top = {d.key for d in drawn if d.stratum == "top quarter"}
    assert top
    assert top <= set(ranked[40:100])  # the top quarter of the place, not of what is left


def test_a_small_place_by_value_is_taken_whole_and_counts_as_the_whole_range() -> None:
    ranked = place(50)
    drawn = sample_by_value(ranked, ranked[:40], ranked[10:50], seed=1)
    assert sorted(keys(drawn)) == ranked
    assert {(d.stratum, d.inclusion) for d in drawn} == {("whole place", 1.0)}
    assert "whole place" not in VALUE_STRATA
    assert sample_by_value(ranked, ranked, ranked, seed=1, exclude=set(ranked)) == []


def test_a_small_place_by_value_is_shuffled_so_the_order_does_not_show_generals_ranking() -> None:
    ranked = place(50)  # best first by General's strength
    drawn = keys(sample_by_value(ranked, ranked[:40], ranked[10:50], seed=1))
    assert sorted(drawn) == ranked
    assert drawn != ranked  # strongest first would tell the rater what General thinks
    assert drawn != ranked[::-1]
    assert drawn == keys(sample_by_value(ranked, ranked[:40], ranked[10:50], seed=1))
    assert drawn != keys(sample_by_value(ranked, ranked[:40], ranked[10:50], seed=2))


def test_a_session_by_value_is_repeatable_and_every_stratum_but_random_is_for_value() -> None:
    ranked = place()
    drawn = sample_by_value(ranked, ranked[:40], ranked[100:140], seed=2)
    assert sample_by_value(ranked, ranked[:40], ranked[100:140], seed=2) == drawn
    assert TASTE_RANDOM + sum(quota for _, quota in VALUE_QUOTAS) == TASTE_SAMPLE
    assert {d.stratum for d in drawn} - {"random"} <= VALUE_STRATA
    assert "random" not in VALUE_STRATA


def test_sample_is_repeatable_and_small_pools_are_taken_whole() -> None:
    ranked = [f"k{i}" for i in range(500)]
    assert sample_for_labelling(ranked, seed=1) == sample_for_labelling(ranked, seed=1)
    assert sorted(sample_for_labelling(ranked[:20], seed=1)) == sorted(ranked[:20])


@given(st.lists(st.sampled_from([0, 1, 2]), min_size=1, max_size=30))
def test_metrics_stay_in_range(labels: list[int]) -> None:
    for value in (precision_at(labels, k=10), ndcg_at(labels, k=10)):
        assert math.isnan(value) or 0 <= value <= 1


def test_a_labelled_search_is_scored_by_ranking_its_labelled_photos() -> None:
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.1, "unlabelled": 0.95}
    labels = {"a": 2, "b": 0, "c": 1, "d": 2}
    result = score_search(scores, labels)
    assert result.labelled == 4
    assert result.matches == 2
    assert result.precision_at_10 == pytest.approx((1 + 0 + 0.5 + 1) / 4)
    assert result.auc == pytest.approx(0.5)  # a beats b; d loses to b
    assert 0 < result.ndcg_at_10 < 1


def test_a_search_with_nothing_to_find_has_no_precision() -> None:
    """ "snow" in a summer folder: precision would be 0 however well it ranks; it is undefined."""
    nothing = score_search({"a": 0.3, "b": 0.1}, {"a": 0, "b": 0})
    assert math.isnan(nothing.precision_at_10)
    assert math.isnan(nothing.ndcg_at_10)


def test_ties_break_by_key_so_scores_are_repeatable() -> None:
    labels = {"a": 0, "b": 2}
    tied = score_search({"a": 0.5, "b": 0.5}, labels)
    assert tied.precision_at_10 == pytest.approx(0.5)
    assert tied.ndcg_at_10 == pytest.approx(score_search({"a": 0.9, "b": 0.1}, labels).ndcg_at_10)


def test_clear_calls_count_the_photos_called_clear_by_how_they_were_labelled() -> None:
    labels = {"a": 2, "b": 2, "c": 1, "d": 0, "e": 0}
    clear = {"a": True, "b": False, "c": True, "d": True, "e": False}
    calls = clear_calls(clear, labels)
    assert calls == ClearCalls(matches=1, partly=1, wrong=1)
    assert calls.called == 3
    assert calls + ClearCalls(matches=2, partly=0, wrong=0) == ClearCalls(3, 1, 1)
    assert clear_calls(dict.fromkeys(labels, False), labels).called == 0


# -- examples -------------------------------------------------------------------------------


def test_examples_are_chosen_repeatably_whatever_the_order_given() -> None:
    keys = [f"k{i}" for i in range(10)]
    chosen = choose_examples(keys, 3, seed=7)
    assert len(set(chosen)) == 3
    assert set(chosen) <= set(keys)
    assert chosen == choose_examples(list(reversed(keys)), 3, seed=7)
    assert any(choose_examples(keys, 3, seed=s) != chosen for s in range(8))


def test_asking_for_more_examples_than_there_are_takes_them_all() -> None:
    assert sorted(choose_examples(["b", "a"], 5, seed=1)) == ["a", "b"]


def test_photos_shot_close_to_an_example_are_left_out() -> None:
    t = datetime(2021, 3, 6, 10)
    taken = {
        "burst": t + timedelta(seconds=30),
        "later": t + timedelta(minutes=10),
        "undated": None,
    }
    assert apart_from(taken, [t, None], seconds=120) == {"later", "undated"}


def searched(
    set_id: str,
    folder: str,
    labels: dict[str, int],
    frames: dict[str, list[float]],
    taken: dict[str, datetime] | None = None,
) -> LabelledSearch:
    """A labelled search read by examples alone; ``frames`` are its photos' copy vectors."""
    keys = list(labels)
    vectors = f.unit([frames[k] for k in keys])
    return LabelledSearch(
        set_id=set_id,
        text="fog",
        folders=[Path(folder)],
        keys=keys,
        taken=[(taken or {}).get(k) for k in keys],
        labels=labels,
        view=f.view(vectors),
        text_scores=None,
        frames=vectors,
    )


class ByEmbedding:
    """Compares photos on their embeddings, as an embedder with no feature of its own."""

    name = "by embedding"


def test_a_copy_of_an_example_from_another_site_is_never_scored() -> None:
    other = searched("b", "/b", {"b1": 2, "b2": 0}, {"b1": [1, 0, 0], "b2": [0, 1, 0]})
    here = searched(
        "a",
        "/a",
        {"b1": 2, "export": 2, "a1": 2, "a2": 0, "a3": 0},  # b1: the same file, in both
        {
            "b1": [1, 0, 0],
            "export": [3, 0, 0],  # saved twice: another file, the same frame
            "a1": [1, 1, 0],
            "a2": [0, 0, 1],
            "a3": [0, 1, 1],
        },
    )
    rows = measure_examples(
        [here, other], cast("Embedder", ByEmbedding()), k=1, draws=2, weights=()
    )
    (row,) = [r for r in rows if r.protocol == "other site" and r.set_id == "a"]
    assert row.copies == 2.0
    assert row.score.labelled == 3


def test_a_copy_of_an_example_in_its_own_search_is_left_out_and_counted() -> None:
    t = datetime(2021, 3, 6, 10)
    search = searched(
        "a",
        "/a",
        {"m": 2, "export": 2, "burst": 0, "x": 0, "z": 0},
        {"m": [1, 0, 0], "export": [1, 0, 0], "burst": [1, 0.1, 0], "x": [0, 1, 0], "z": [0, 0, 1]},
        taken={"m": t, "export": t + timedelta(days=700), "burst": t + timedelta(seconds=60)},
    )
    rows = measure_examples([search], cast("Embedder", ByEmbedding()), k=1, draws=4, weights=())
    (row,) = rows
    # Whichever is the example, the other is its copy; the burst is near the original only.
    assert row.copies == 1.0
    assert row.copies + row.nearby + row.score.labelled == 4
    assert row.score.labelled >= 2


def test_scores_over_several_draws_are_averaged_ignoring_undefined_ones() -> None:
    def score(p: float, auc_: float) -> SearchScore:
        return SearchScore(10, 4, p, p, auc_, Threshold(math.nan, 0, 0, 0))

    mean = average([score(0.5, 0.8), score(1.0, math.nan)])
    assert mean.precision_at_10 == pytest.approx(0.75)
    assert mean.auc == pytest.approx(0.8)
    assert mean.labelled == 10


# -- same view -------------------------------------------------------------------------------


def test_a_same_view_sample_spans_the_threshold_band_by_band() -> None:
    keys = [f"k{i:03d}" for i in range(300)]
    match = [i / 300 for i in range(300)]  # evenly from 0 to ~1
    bands = ((0.85, 4), (0.80, 4), (0.70, 6), (-math.inf, 3))
    sample = sample_by_match(keys, match, seed=3, bands=bands)
    by_key = dict(zip(keys, match, strict=True))
    values = [by_key[k] for k in sample]
    assert sum(v >= 0.85 for v in values) == 4
    assert sum(0.80 <= v < 0.85 for v in values) == 4
    assert sum(0.70 <= v < 0.80 for v in values) == 6
    assert sum(v < 0.70 for v in values) == 3
    assert sample == sample_by_match(keys, match, seed=3, bands=bands)
    assert sample != sorted(sample)  # shuffled: the order gives nothing away


def test_a_thin_band_is_taken_whole() -> None:
    sample = sample_by_match(
        ["a", "b", "c"], [0.9, 0.5, 0.4], seed=1, bands=((0.85, 5), (-math.inf, 1))
    )
    assert "a" in sample
    assert len(sample) == 2


def test_patch_bands_cover_both_tiers_and_what_lies_just_outside() -> None:
    lowers = [lower for lower, _ in PATCH_SAMPLE_BANDS]
    assert lowers == sorted(lowers, reverse=True)
    assert {0.25, 0.18} <= set(lowers)  # the sure and possible cuts
    assert OUTSIDE_SHORTLIST in lowers
    assert lowers[-1] == -math.inf  # a couple from anywhere: plainly not the view
    assert sum(count for _, count in PATCH_SAMPLE_BANDS) == 60


def test_precision_and_recall_at_a_given_cut() -> None:
    cut = cut_at([0.9, 0.85, 0.82, 0.7, 0.6], [2, 0, 2, 2, 1], 0.8)
    assert cut.precision == pytest.approx(2 / 3)
    assert cut.recall == pytest.approx(2 / 3)
    assert cut.threshold == 0.8


def test_locations_count_by_how_many_like_dislike_pairs_they_have() -> None:
    rows = [
        LocationRatings("a", rated=4, liked=2, disliked=2, auc={"x": 1.0}),  # 4 pairs
        LocationRatings("b", rated=2, liked=1, disliked=1, auc={"x": 0.0}),  # 1 pair
        LocationRatings("c", rated=2, liked=1, disliked=1, auc={"y": 0.0}),
    ]
    assert pair_weighted_auc(rows, "x") == pytest.approx(0.8)
    assert math.isnan(pair_weighted_auc(rows, "z"))


def labelled_view(set_id: str, same: list[float], other: list[float]) -> LabelledView:
    match = {f"{set_id}s{i}": v for i, v in enumerate(same)}
    match |= {f"{set_id}o{i}": v for i, v in enumerate(other)}
    labels = {k: 2 if "s" in k[len(set_id) :] else 0 for k in match}
    return LabelledView(set_id, f"Same view as {set_id}", match, labels)


def test_each_search_is_also_judged_by_a_cut_chosen_on_the_others() -> None:
    searches = [
        labelled_view("a", [0.9, 0.85], [0.5]),
        labelled_view("b", [0.8], [0.6]),
        labelled_view("c", [0.72], [0.4]),
    ]
    report = measure_views(searches, default=0.75)
    by_id = {r.set_id: r for r in report.rows}
    # Chosen on a and b (same views from 0.8, others below 0.6): c's 0.72 falls short.
    assert by_id["c"].held_out.threshold == pytest.approx(0.8)
    assert by_id["c"].held_out.recall == 0.0
    # Chosen on b and c (same views from 0.72): a keeps both.
    assert by_id["a"].held_out.recall == 1.0
    assert by_id["c"].at_pooled.threshold == report.pooled.threshold


def test_one_search_alone_has_nothing_to_be_held_out_from() -> None:
    (row,) = measure_views([labelled_view("a", [0.9], [0.5])], default=0.75).rows
    assert math.isnan(row.held_out.threshold)


def test_same_view_headlines_have_intervals_by_reference() -> None:
    searches = [
        labelled_view(name, [0.9, 0.8, 0.7], [0.5, 0.6, 0.4]) for name in ("a", "b", "c", "d")
    ]
    report = measure_views(searches, default=0.75, resamples=100)
    assert report.auc is not None
    assert report.auc.value == 1.0
    assert (report.auc.places, report.auc.rated, report.auc.unit) == (4, 24, "reference")
    assert report.recall is not None
    assert report.recall.value == report.pooled_at_default.recall
    assert report.precision is not None
    assert report.precision.value == report.pooled_at_default.precision


def test_ways_of_matching_are_compared_with_the_first_photo_by_photo() -> None:
    sharp = [labelled_view(n, [0.9, 0.8], [0.5, 0.6]) for n in ("a", "b", "c")]
    blunt = [labelled_view(n, [0.6, 0.5], [0.9, 0.8]) for n in ("a", "b", "c")]
    first, second = view_differences(
        [measure_views(sharp, default=0.75), measure_views(blunt, default=0.75)], resamples=100
    )
    assert first.versus is None
    assert second.versus is not None
    assert second.versus.value == -1.0
    assert second.versus.excludes_zero


def test_matches_in_place_are_compared_with_the_share_overall_and_where_it_is_unsure() -> None:
    # The share ranks every look-alike in the band above a same view there; the matches in
    # place do not. Outside the band both are right.
    share = [labelled_view(n, [0.5, 0.2], [0.25, 0.05]) for n in ("a", "b", "c")]
    consistent = [labelled_view(n, [0.45, 0.15], [0.02, 0.01]) for n in ("a", "b", "c")]
    measured = measure_view_geometry(share, consistent, sure=0.25, resamples=100)
    assert measured.band == GEOMETRY_BAND == (0.12, 0.30)
    today, in_place = measured.whole
    assert (today.embedder, in_place.embedder) == ("share", "in place")
    assert in_place.versus is not None
    assert in_place.versus.value == pytest.approx(0.25)
    band_today, band_in_place = measured.within_band
    # Only the 0.2 same view and the 0.25 look-alike of each search are in the band.
    assert measured.in_band == 6
    assert [r.labelled for r in band_today.rows] == [2, 2, 2]
    assert band_today.auc is not None
    assert band_today.auc.value == 0.0
    assert band_in_place.versus is not None
    assert band_in_place.versus.value == 1.0
    assert band_in_place.default == today.default == 0.25


def test_a_search_with_nothing_in_the_band_is_left_out_of_it() -> None:
    share = [labelled_view("a", [0.9], [0.01]), labelled_view("b", [0.2], [0.15])]
    consistent = [labelled_view("a", [0.8], [0.0]), labelled_view("b", [0.2], [0.0])]
    measured = measure_view_geometry(share, consistent, sure=0.25, resamples=100)
    assert [r.set_id for r in measured.within_band[0].rows] == ["b"]
    assert measured.in_band == 2


def test_scorers_are_compared_with_the_first_on_the_same_labelled_photos() -> None:
    labels = {f"k{i}": 2 if i < 4 else 0 for i in range(8)}
    good = {f"k{i}": 1.0 - i / 10 for i in range(8)}
    bad = {f"k{i}": i / 10 for i in range(8)}
    rows = [
        Evaluated(
            set_id=search,
            text=search,
            scorer=name,
            score=score_search(scores, labels),
            missing=0,
            scores=scores,
            labels=labels,
        )
        for search in ("s1", "s2", "s3")
        for name, scores in (("baseline", good), ("parts", bad))
    ]
    measured = criteria_intervals(rows, resamples=100)
    assert measured.baseline == "baseline"
    assert measured.auc["baseline"].value == 1.0
    assert (measured.auc["parts"].places, measured.auc["parts"].unit) == (3, "search")
    assert measured.versus["parts"].value == -1.0
    assert "baseline" not in measured.versus


def test_example_readings_are_compared_with_text_alone_search_by_search() -> None:
    def row(set_id: str, variant: str, value: float) -> ExampleEvaluated:
        return ExampleEvaluated(
            set_id=set_id,
            text="fog",
            site=set_id,
            protocol="same search",
            source=set_id,
            variant=variant,
            score=Averaged(labelled=10, precision_at_10=0.5, ndcg_at_10=0.5, auc=value),
        )

    rows = [row(s, v, x) for s in ("a", "b", "c") for v, x in (("text", 0.6), ("examples", 0.8))]
    measured = example_intervals(rows, resamples=100)
    text, versus = measured["same search", "text"]
    assert (text.value, text.places, text.unit) == (0.6, 3, "search")
    assert versus is None
    examples, versus = measured["same search", "examples"]
    assert examples.value == pytest.approx(0.8)
    assert versus is not None
    assert versus.value == pytest.approx(0.2)


def test_copies_of_one_frame_count_once_and_burst_frames_do_not() -> None:
    # As measured: copies of one frame >= 0.99998 on the view embedder, the closest
    # burst frames <= 0.99973.
    frame, burst, other = f.unit([[1.0, 0.0], [1.0, 0.023], [0.0, 1.0]])
    assert float(frame @ burst) < COPY_SIMILARITY
    search = LabelledView(
        "a",
        "Same view as a",
        {"frame": 0.6, "export": 0.6, "burst": 0.5, "other": 0.1},
        {"frame": 2, "export": 2, "burst": 2, "other": 0},
    )
    counted, copies = count_copies_once(
        search, ["export", "frame", "burst", "other"], np.stack([frame, frame, burst, other])
    )
    assert copies == 1
    assert counted.labels == {"export": 2, "burst": 2, "other": 0}  # the first copy stays
    assert counted.match == {"export": 0.6, "burst": 0.5, "other": 0.1}


def test_copies_labelled_differently_count_once_at_the_higher_label() -> None:
    # Called the same view at least once: "partly" for one copy would drop the frame from
    # recall, and which file came first decides nothing about the view.
    search = LabelledView("a", "Same view as a", {"heic": 0.3, "jpg": 0.3}, {"heic": 1, "jpg": 2})
    counted, copies = count_copies_once(search, ["heic", "jpg"], f.unit([[1, 0], [1, 0]]))
    assert (counted.labels, copies) == ({"heic": 2}, 1)


def rated_places(
    places: dict[str, int], *, session: bool = True, seed: int = 0
) -> list[TastePhoto]:
    """Photos you like lean along the first axis everywhere; the general score is noise."""
    rng = np.random.default_rng(seed)
    out = []
    for place, n in places.items():
        x = f.unit(rng.normal(size=(n, 8)))
        liked = x[:, 0] + 0.2 * rng.normal(size=n) > 0
        out += [
            TastePhoto(
                place,
                bool(liked[i]),
                session,
                x[i],
                float(rng.random()),
                float(rng.random()),
                content_key=f"{place}{i}",
            )
            for i in range(n)
        ]
    return out


def test_your_taste_is_judged_at_each_place_by_a_model_that_never_saw_it() -> None:
    photos = rated_places({"Mill": 60, "Harbor": 60, "Fern": 60, "Birch": 60})
    report = measure_taste(photos, feature="clip@1", quality_weight=0.35, l2=1.0)
    assert [r.location for r in report.rows] == ["Birch", "Fern", "Harbor", "Mill"]
    for row in report.rows:
        assert row.rated == 60
        assert row.liked + row.disliked == 60
        assert row.yours_aesthetic > 0.8 > row.general_aesthetic + 0.2
    assert report.yours_strength > report.general_strength
    assert report.yours_aesthetic > report.general_aesthetic


def test_each_judged_photo_keeps_the_forecast_of_a_model_that_never_saw_its_place() -> None:
    photos = rated_places({"Mill": 60, "Harbor": 60, "Fern": 60})
    photos += rated_places({"Signal Hill": 80}, session=False, seed=1)  # trained on, not judged
    report = measure_taste(photos, feature="clip@1", quality_weight=0.35, l2=1.0, resamples=10)
    assert len(report.forecasts) == 180  # what a probability is calibrated on, and no more
    assert all(0 <= s.yours <= 1 for s in report.forecasts)
    assert np.mean([s.yours for s in report.forecasts if s.liked]) > 0.6
    assert np.mean([s.yours for s in report.forecasts if not s.liked]) < 0.4
    assert not any(s.first_pick for s in report.forecasts)  # all of these sample the whole range
    assert {s.place for s in report.forecasts} == {"Mill", "Harbor", "Fern"}
    chosen = [dataclasses.replace(p, whole_range=False) for p in photos if p.location == "Mill"]
    others = [p for p in photos if p.location != "Mill"]
    again = measure_taste(
        others + chosen, feature="clip@1", quality_weight=0.35, l2=1.0, resamples=10
    )
    assert {s.place for s in again.forecasts if s.first_pick} == {"Mill"}  # a taste's picks, apart


def test_ratings_on_results_train_but_are_never_judged() -> None:
    photos = rated_places({"Mill": 60, "Harbor": 60})
    photos += rated_places({"Signal Hill": 80}, session=False, seed=1)
    report = measure_taste(photos, feature="clip@1", quality_weight=0.35, l2=1.0)
    assert [r.location for r in report.rows] == ["Harbor", "Mill"]
    assert report.trained_on == 200


def test_a_place_without_dislikes_cannot_be_judged() -> None:
    photos = rated_places({"Mill": 60, "Harbor": 60})
    photos += [dataclasses.replace(p, location="Fern", liked=True) for p in photos[:20]]
    report = measure_taste(photos, feature="clip@1", quality_weight=0.35, l2=1.0)
    assert "Fern" not in [r.location for r in report.rows]
    assert report.one_sided == ["Fern"]


FIVE = {"Mill": 60, "Harbor": 60, "Fern": 60, "Birch": 60, "Tivoli": 60}


def test_each_taste_measure_has_its_interval_and_yours_minus_general_is_paired() -> None:
    photos = rated_places({"Mill": 60, "Harbor": 60, "Fern": 60})
    report = measure_taste(photos, feature="clip@1", quality_weight=0.35, l2=1.0, resamples=200)
    for name, measured in (("strength", report.strength), ("aesthetic", report.aesthetic)):
        assert measured is not None
        yours, general = measured.a, measured.b
        assert math.isclose(yours.value, getattr(report, f"yours_{name}"))
        assert math.isclose(general.value, getattr(report, f"general_{name}"))
        assert general.lo <= general.value <= general.hi
        assert math.isclose(measured.difference.value, yours.value - general.value)
        assert (measured.difference.places, measured.difference.rated) == (3, 180)
    assert report.aesthetic is not None
    assert report.aesthetic.difference.excludes_zero  # Yours learned the axis; General is noise
    assert set(report.macro) == {
        "general_strength",
        "yours_strength",
        "general_aesthetic",
        "yours_aesthetic",
    }


def test_the_macro_average_counts_each_place_once() -> None:
    photos = rated_places({"Mill": 60, "Harbor": 20})
    report = measure_taste(photos, feature="clip@1", quality_weight=0.35, l2=1.0, resamples=10)
    mill, harbor = sorted(report.rows, key=lambda r: r.location != "Mill")
    assert math.isclose(
        report.macro["yours_aesthetic"], (mill.yours_aesthetic + harbor.yours_aesthetic) / 2
    )


def test_yours_wins_only_when_it_clearly_beats_general_by_strength_at_five_places() -> None:
    four = dict(list(FIVE.items())[:4])
    report = measure_taste(
        rated_places(four), feature="clip@1", quality_weight=0.35, l2=1.0, resamples=200
    )
    assert report.strength is not None
    assert report.strength.difference.lo > 0
    assert not report.yours_wins  # clear, but at too few places

    report = measure_taste(
        rated_places(FIVE), feature="clip@1", quality_weight=0.35, l2=1.0, resamples=200
    )
    assert report.yours_wins
    assert report.strength is not None
    unclear = dataclasses.replace(
        report.strength, difference=dataclasses.replace(report.strength.difference, lo=-0.01)
    )
    assert not dataclasses.replace(report, strength=unclear).yours_wins


def test_nothing_judged_wins_nothing() -> None:
    report = measure_taste([], feature="clip@1", quality_weight=0.35, resamples=10)
    assert not report.yours_wins


def test_first_picks_shares_have_intervals_and_a_paired_difference() -> None:
    photos = rated_places({"Mill": 30, "Harbor": 30})

    def first_picks(location: str, taste: TasteWeights | None, k: int) -> list[str]:
        here = [p for p in photos if p.location == location]
        chosen = here if taste is not None else list(reversed(here))
        return [p.content_key for p in chosen][:k]

    report = measure_taste(
        photos,
        feature="clip@1",
        quality_weight=0.35,
        l2=1.0,
        first_picks=first_picks,
        at=(4, 8),
        resamples=50,
    )
    assert len(report.first) == 2
    for measured, general, yours in zip(
        report.first, report.general_first, report.yours_first, strict=True
    ):
        assert measured.a.value == yours.share
        assert measured.b.value == general.share
        assert measured.a.rated == 60


def test_first_picks_count_the_photos_you_rated_among_them() -> None:
    photos = rated_places({"Mill": 30, "Harbor": 30})
    liked = {p.content_key for p in photos if p.liked}

    def first_picks(location: str, taste: TasteWeights | None, k: int) -> list[str]:
        # Unrated photos and one rated elsewhere are shown too, but count for nothing.
        rated = [f"{location}{i}" for i in range(k // 2)]
        return [*rated, "Elsewhere0" if location == "Mill" else "Mill0", "unrated"][:k]

    report = measure_taste(
        photos, feature="clip@1", quality_weight=0.35, l2=1.0, first_picks=first_picks, at=(4, 8)
    )
    assert report.first_at == (4, 8)
    for row in report.rows:
        at4, at8 = row.general_first
        assert at4 == FirstPicks(
            rated=2, liked=sum(f"{row.location}{i}" in liked for i in range(2))
        )
        assert at8.rated == 4
        assert row.yours_first == row.general_first  # this callback ignores the template
    assert report.general_first[1].rated == 8  # each rated photo counts once, all places
    assert report.general_first[1].liked == sum(r.general_first[1].liked for r in report.rows)
    assert FirstPicks(rated=4, liked=3).share == 0.75
    assert math.isnan(FirstPicks(rated=0, liked=0).share)


def test_photos_drawn_for_their_value_count_in_first_picks_not_in_the_whole_range_auc() -> None:
    """A session by value (``sample_by_value``) rates mostly the top of a place, so its
    photos would weigh the top more than the rest in an AUC over the whole range."""
    photos = rated_places({"Mill": 60, "Harbor": 60, "Fern": 60})
    for_value = [  # at Mill, ranked as its whole-range photos are, but judged the other way
        dataclasses.replace(p, content_key=f"value{i}", liked=not p.liked, whole_range=False)
        for i, p in enumerate(photos[:20])
    ]

    def first_picks(location: str, taste: TasteWeights | None, k: int) -> list[str]:
        return [p.content_key for p in photos + for_value if p.location == location][:k]

    def mill(rated: list[TastePhoto]) -> TasteRow:
        report = measure_taste(
            rated,
            feature="clip@1",
            quality_weight=0.35,
            l2=1.0,
            first_picks=first_picks,
            at=(80,),
            resamples=10,
        )
        (row,) = [r for r in report.rows if r.location == "Mill"]
        return row

    before, after = mill(photos), mill(photos + for_value)
    assert after.general_aesthetic == before.general_aesthetic
    assert after.yours_aesthetic == before.yours_aesthetic  # learned at Harbor and Fern
    assert (after.rated, after.for_value, before.for_value) == (80, 20, 0)
    assert after.general_first[0].rated == before.general_first[0].rated + 20


def test_a_place_whose_whole_range_photos_are_one_sided_still_counts_its_first_picks() -> None:
    photos = rated_places({"Mill": 60, "Harbor": 60, "Fern": 60})
    photos = [
        dataclasses.replace(p, whole_range=p.liked) if p.location == "Mill" else p for p in photos
    ]
    report = measure_taste(photos, feature="clip@1", quality_weight=0.35, l2=1.0, resamples=10)
    (row,) = [r for r in report.rows if r.location == "Mill"]
    assert math.isnan(row.general_strength)  # only likes over the whole range: no pairs
    assert not math.isnan(report.general_strength)  # and the pooled AUC goes without it


def test_yours_shows_first_what_a_taste_learned_elsewhere_ranks_highest() -> None:
    photos = rated_places({"Mill": 60, "Harbor": 60, "Fern": 60})
    seen: list[tuple[str, int]] = []

    def first_picks(location: str, taste: TasteWeights | None, k: int) -> list[str]:
        here = [p for p in photos if p.location == location]
        if taste is None:
            return [p.content_key for p in sorted(here, key=lambda p: -p.general)][:k]
        seen.append((location, taste.rated))
        scores = taste_probability(taste, np.stack([p.vector for p in here]))
        return [here[i].content_key for i in np.argsort(-scores)][:k]

    report = measure_taste(
        photos, feature="clip@1", quality_weight=0.35, l2=1.0, first_picks=first_picks
    )
    assert report.first_at == TASTE_FIRST
    assert set(seen) == {(place, 120) for place in ("Fern", "Harbor", "Mill")}  # never itself
    at20 = TASTE_FIRST.index(20)
    assert report.yours_first[at20].share > 0.8 > report.general_first[at20].share


def test_held_out_places_are_judged_only_when_asked_and_never_trained_on() -> None:
    photos = rated_places({"Mill": 60, "Harbor": 60, "Fern": 60, "Birch": 60})
    trained: list[tuple[str, int]] = []

    def first_picks(location: str, taste: TasteWeights | None, k: int) -> list[str]:
        if taste is not None:
            trained.append((location, taste.rated))
        return []

    report = measure_taste(
        photos,
        feature="clip@1",
        quality_weight=0.35,
        l2=1.0,
        first_picks=first_picks,
        judged=["Mill"],
        untrained=["Mill", "Harbor"],
        resamples=10,
    )
    assert [r.location for r in report.rows] == ["Mill"]
    assert set(trained) == {("Mill", 120)}  # Fern and Birch only: never Harbor
    assert report.trained_on == 120


def own_taste(place: str, n: int = 60, *, seed: int = 5) -> list[TastePhoto]:
    """Photos you like at ``place`` lean along the second axis: nowhere else teaches it."""
    rng = np.random.default_rng(seed)
    x = f.unit(rng.normal(size=(n, 8)))
    liked = x[:, 1] + 0.2 * rng.normal(size=n) > 0
    return [
        TastePhoto(place, bool(liked[i]), True, x[i], 0.5, 0.5, content_key=f"{place}{i}")
        for i in range(n)
    ]


def mill_judged(report: TasteReport) -> float:
    (mill,) = [r for r in report.rows if r.location == "Mill"]
    return mill.yours_aesthetic


def test_a_copy_of_a_judged_photo_is_never_learned_from() -> None:
    mill = own_taste("Mill")
    elsewhere = rated_places({"Harbor": 60, "Fern": 60})
    # The mill's photos saved twice, rated again on results elsewhere.
    exports = [
        dataclasses.replace(p, location="Exports", from_session=False, content_key=f"e{i}")
        for i, p in enumerate(mill)
    ]
    leaked = measure_taste(
        [*mill, *elsewhere, *exports], feature="clip@1", quality_weight=0.35, l2=1.0, resamples=10
    )
    assert mill_judged(leaked) > 0.7  # without knowing they are copies, the answer leaks
    copies = [dataclasses.replace(p, same_frame=f"Mill{i}") for i, p in enumerate(exports)]
    report = measure_taste(
        [*mill, *elsewhere, *copies], feature="clip@1", quality_weight=0.35, l2=1.0, resamples=10
    )
    assert mill_judged(report) < 0.6
    assert (report.copies_left_out, report.same_place_left_out) == (60, 0)


def test_a_place_grouped_with_the_judged_one_is_never_learned_from() -> None:
    fire = own_taste("Mill (after the fire)", seed=6)  # one trip, filed under two names
    elsewhere = rated_places({"Harbor": 60, "Fern": 60})
    leaked = measure_taste(
        [*own_taste("Mill"), *fire, *elsewhere],
        feature="clip@1",
        quality_weight=0.35,
        l2=1.0,
        resamples=10,
    )
    assert mill_judged(leaked) > 0.7
    mill = [dataclasses.replace(p, same_place="Mill") for p in own_taste("Mill")]
    fire = [dataclasses.replace(p, same_place="Mill") for p in fire]
    report = measure_taste(
        [*mill, *fire, *elsewhere], feature="clip@1", quality_weight=0.35, l2=1.0, resamples=10
    )
    assert mill_judged(report) < 0.6
    assert report.same_place_left_out == 120  # each name's photos, when the other is judged
    assert report.same_places == (("Mill", "Mill (after the fire)"),)
    assert report.copies_left_out == 0


def test_copies_and_grouped_places_of_the_hold_out_are_never_learned_from() -> None:
    mill = [dataclasses.replace(p, same_place="Mill") for p in own_taste("Mill")]
    fire = [
        dataclasses.replace(p, location="Mill (after the fire)", same_place="Mill")
        for p in own_taste("Mill (after the fire)", n=10, seed=6)
    ]
    copy = dataclasses.replace(
        mill[0], location="Harbor", content_key="e0", same_frame="Mill0", same_place=""
    )
    elsewhere = rated_places({"Harbor": 60, "Fern": 60})
    report = measure_taste(
        [*mill, *fire, *elsewhere, copy],
        feature="clip@1",
        quality_weight=0.35,
        judged=["Mill"],
        untrained=["Mill"],
        resamples=10,
    )
    assert report.trained_on == 120  # Harbor and Fern, without the mill's copy
    assert (report.copies_left_out, report.same_place_left_out) == (1, 10)


def test_photos_apart_from_places_leave_out_their_copies_and_grouped_names() -> None:
    mill = own_taste("Mill", n=4)
    copy = dataclasses.replace(mill[0], location="Harbor", content_key="e0", same_frame="Mill0")
    fire = dataclasses.replace(mill[1], location="Mill (fire)", content_key="f", same_place="Mill")
    mill = [dataclasses.replace(p, same_place="Mill") for p in mill]
    kept = taste_apart([*mill, copy, fire, *rated_places({"Fern": 3})], ["Mill"])
    assert [p.location for p in kept.photos] == ["Fern"] * 3
    assert (kept.copies, kept.same_place) == ({"e0"}, {"f"})


def test_a_candidate_quality_is_measured_against_todays_paired() -> None:
    photos = rated_places({"Mill": 60, "Harbor": 60, "Fern": 60})
    # Today's quality is noise; the candidate's follows what you liked.
    better = [dataclasses.replace(p, candidate_quality=0.9 if p.liked else 0.1) for p in photos]
    report = measure_taste(better, feature="clip@1", quality_weight=0.5, resamples=100)
    assert report.candidate is not None
    assert math.isclose(report.candidate.b.value, report.general_strength)
    assert report.candidate.difference.excludes_zero
    assert report.candidate.difference.value > 0
    assert report.quality_weight == 0.5
    plain = measure_taste(photos, feature="clip@1", quality_weight=0.5, resamples=10)
    assert plain.candidate is None


def test_a_candidate_aesthetic_for_general_is_measured_against_todays_paired() -> None:
    photos = rated_places({"Mill": 60, "Harbor": 60, "Fern": 60})
    # Today's General is noise; the candidate's aesthetic follows what you liked.
    better = [dataclasses.replace(p, candidate_general=7.0 if p.liked else 3.0) for p in photos]
    report = measure_taste(better, feature="clip@1", quality_weight=0.35, resamples=100)
    for measured, today in (
        (report.candidate_general, report.general_strength),
        (report.candidate_general_alone, report.general_aesthetic),
    ):
        assert measured is not None
        assert math.isclose(measured.b.value, today)
        assert measured.difference.excludes_zero
        assert measured.difference.value > 0
    assert report.candidate is None  # the technical candidate is another measure
    plain = measure_taste(photos, feature="clip@1", quality_weight=0.35, resamples=10)
    assert (plain.candidate_general, plain.candidate_general_alone) == (None, None)


def test_what_quality_adds_to_strength_is_measured_against_the_aesthetic_alone() -> None:
    photos = rated_places({"Mill": 60, "Harbor": 60, "Fern": 60})
    # Quality follows what you liked, so mixing it in ranks your ratings better.
    telling = [dataclasses.replace(p, quality=0.9 if p.liked else 0.1) for p in photos]
    report = measure_taste(telling, feature="clip@1", quality_weight=0.5, resamples=100)
    for measured, strength, aesthetic in (
        (report.general_with_quality, report.general_strength, report.general_aesthetic),
        (report.yours_with_quality, report.yours_strength, report.yours_aesthetic),
    ):
        assert measured is not None
        assert math.isclose(measured.a.value, strength)
        assert math.isclose(measured.b.value, aesthetic)
        assert measured.difference.excludes_zero
        assert measured.difference.value > 0
    # At weight 0 strength is the aesthetic alone: nothing is added.
    alone = measure_taste(telling, feature="clip@1", quality_weight=0.0, resamples=10)
    assert alone.general_with_quality is not None
    assert alone.general_with_quality.difference.value == 0


def pool_around(rated: list[TastePhoto], location: str, size: int = 600, *, seed: int = 0) -> Pool:
    """Everything catalogued at ``location``: its rated photos, and many more that look worse."""
    rng = np.random.default_rng(seed)
    here = [p for p in rated if p.location == location]
    more = size - len(here)
    return Pool(
        keys=tuple(p.content_key for p in here) + tuple(f"{location}-more{i}" for i in range(more)),
        vectors=np.concatenate(
            [np.stack([p.vector for p in here]), f.unit(rng.normal(size=(more, 8)))]
        ),
        general=np.concatenate([[p.general for p in here], rng.uniform(0, 0.5, more)]),
        quality=np.concatenate([[p.quality for p in here], rng.uniform(0.2, 0.8, more)]),
    )


def top_of_the_pool(places: dict[str, int]) -> list[TastePhoto]:
    """Rated photos all at the top of their pool's aesthetic, where only quality tells: what you
    liked has the better quality, and the aesthetic is noise among them."""
    rng = np.random.default_rng(5)
    return [
        dataclasses.replace(
            p,
            general=float(0.90 + 0.09 * rng.random()),
            quality=float(0.9 if p.liked else 0.1) + 0.01 * float(rng.random()),
        )
        for p in rated_places(places)
    ]


def test_strength_is_scored_over_the_whole_pool_as_strongest_ranks_it() -> None:
    # In a search the aesthetic's percentile is taken over everything at the location, so every
    # photo at the top of it sits near 1 and quality tells them apart; among the rated photos
    # alone the same percentile stretches them over 0 to 1 and drowns quality (10 November 2026
    # audit, B-7: General's first-pick AUC read 0.667 and ships as 0.638).
    photos = top_of_the_pool({"Mill": 60, "Harbor": 60, "Fern": 60})
    pools = {name: pool_around(photos, name) for name in ("Mill", "Harbor", "Fern")}
    among_rated = measure_taste(photos, feature="clip@1", quality_weight=0.2, l2=1.0)
    over_pool = measure_taste(photos, feature="clip@1", quality_weight=0.2, l2=1.0, pools=pools)
    assert among_rated.general_strength < 0.8
    assert over_pool.general_strength > 0.95
    assert over_pool.pooled == 3
    assert among_rated.pooled == 0
    for row in over_pool.rows:
        assert row.general_strength > 0.95
        assert row.yours_aesthetic == next(
            r.yours_aesthetic for r in among_rated.rows if r.location == row.location
        )  # the aesthetics alone are ranked among the rated photos either way


def test_a_location_without_a_pool_is_scored_among_its_rated_photos_as_before() -> None:
    photos = top_of_the_pool({"Mill": 60, "Harbor": 60, "Fern": 60})
    only_mill = {"Mill": pool_around(photos, "Mill")}
    alone = measure_taste(photos, feature="clip@1", quality_weight=0.2, l2=1.0)
    partly = measure_taste(photos, feature="clip@1", quality_weight=0.2, l2=1.0, pools=only_mill)
    by_place = {r.location: r for r in partly.rows}
    before = {r.location: r for r in alone.rows}
    assert partly.pooled == 1
    assert by_place["Mill"].general_strength > 0.95 > before["Mill"].general_strength
    for name in ("Harbor", "Fern"):
        assert by_place[name].general_strength == before[name].general_strength
    short = Pool(
        keys=only_mill["Mill"].keys[1:],
        vectors=only_mill["Mill"].vectors[1:],
        general=only_mill["Mill"].general[1:],
        quality=only_mill["Mill"].quality[1:],
    )  # a pool that lacks a rated photo cannot score it, so it is not used
    missing = measure_taste(
        photos, feature="clip@1", quality_weight=0.2, l2=1.0, pools={"Mill": short}
    )
    assert missing.pooled == 0
    assert {r.location: r.general_strength for r in missing.rows} == {
        r.location: r.general_strength for r in alone.rows
    }


def test_a_candidate_is_measured_against_todays_among_the_rated_photos_it_was_computed_for() -> (
    None
):
    # A candidate is read for the rated photos alone, so it cannot be scored over a pool: it and
    # today's are compared the same way, or the pair would measure two things.
    photos = top_of_the_pool({"Mill": 60, "Harbor": 60, "Fern": 60})
    pools = {name: pool_around(photos, name) for name in ("Mill", "Harbor", "Fern")}
    candidate = [dataclasses.replace(p, candidate_quality=p.quality) for p in photos]
    with_pools = measure_taste(candidate, feature="clip@1", quality_weight=0.2, l2=1.0, pools=pools)
    without = measure_taste(candidate, feature="clip@1", quality_weight=0.2, l2=1.0)
    assert with_pools.pooled == 0
    assert with_pools.general_strength == without.general_strength
    fitted = measure_taste(
        photos, feature="clip@1", quality_weight=0.2, l2=1.0, fit="offset", pools=pools
    )
    assert fitted.pooled == 0


def test_strength_over_a_pool_is_the_mix_strongest_makes_of_its_quality_and_aesthetic() -> None:
    quality = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)
    aesthetic = np.array([0.2, 0.9, 0.9, 0.4], dtype=np.float32)
    mixed = shipped_strength(quality, aesthetic, 0.2)
    assert mixed == pytest.approx(0.2 * quality + 0.8 * np.array([0.0, 5 / 6, 5 / 6, 1 / 3]))
    assert shipped_strength(quality, aesthetic, 0.0) == pytest.approx([0.0, 5 / 6, 5 / 6, 1 / 3])
    assert shipped_strength(quality, aesthetic, 1.0) == pytest.approx(quality)


def confounded_places(seed: int = 0) -> list[TastePhoto]:
    """What you like leans along the first axis at every place, but whole places sit apart
    on the second, and you liked more of the places further along it: pooled, the second
    axis tells likes best, and inside a place it tells nothing."""
    rng = np.random.default_rng(seed)
    out = []
    for i, (place, share) in enumerate(
        [("Mill", 0.9), ("Harbor", 0.2), ("Fern", 0.8), ("Dock", 0.3), ("Quarry", 0.7)]
    ):
        shift = np.zeros(8, dtype=np.float32)
        shift[1] = 4.0 * (share - 0.5)
        x = rng.normal(size=(60, 8)).astype(np.float32)
        x[:, 0] *= 0.3  # what you like is a small difference beside how places differ
        liked = x[:, 0] > np.quantile(x[:, 0], 1 - share)
        out += [
            TastePhoto(
                place,
                bool(liked[j]),
                True,
                x[j] + shift,
                float(rng.random()),
                float(rng.random()),
                content_key=f"{place}{j}",
                place_mean=shift + 0.01 * i,
            )
            for j in range(60)
        ]
    return out


def test_centred_on_each_place_yours_learns_what_you_like_within_a_place() -> None:
    photos = confounded_places()
    report = measure_taste(photos, feature="clip@1", quality_weight=0.0, resamples=100)
    assert (report.fit, report.fitted) == ("", None)
    centred = measure_taste(
        photos, feature="clip@1", quality_weight=0.0, resamples=100, fit="centred"
    )
    assert centred.fit == "centred"
    assert centred.fitted is not None
    assert math.isclose(centred.fitted.b.value, report.yours_strength)
    assert centred.fitted.a.value > 0.9
    assert centred.fitted.difference.excludes_zero
    assert centred.fitted.difference.value > 0
    assert centred.uncentred == ()


def test_with_general_as_an_offset_yours_learns_what_general_misses() -> None:
    rng = np.random.default_rng(0)
    photos = []
    for place in ("Mill", "Harbor", "Fern", "Dock", "Quarry"):
        x = f.unit(rng.normal(size=(60, 8)))
        general = rng.normal(size=60)
        liked = x[:, 0] * 3 + general > 0  # your taste is General's plus what it misses
        photos += [
            TastePhoto(
                place,
                bool(liked[j]),
                True,
                x[j],
                float(1 / (1 + np.exp(-general[j]))),
                float(rng.random()),
                content_key=f"{place}{j}",
            )
            for j in range(60)
        ]
    report = measure_taste(
        photos, feature="clip@1", quality_weight=0.0, l2=10, resamples=100, fit="offset"
    )
    assert report.fit == "offset"
    assert report.fitted is not None
    assert report.fitted.difference.excludes_zero
    assert report.fitted.difference.value > 0
    assert report.uncentred == ()
    # Against today's Yours as it is used (l2 1000), not today's at the offset's l2.
    today = measure_taste(photos, feature="clip@1", quality_weight=0.0, resamples=10)
    assert math.isclose(report.fitted.b.value, today.yours_strength)
    assert not math.isclose(report.yours_strength, today.yours_strength)


def test_a_place_without_a_mean_is_learned_from_as_it_is_and_named() -> None:
    photos = [
        dataclasses.replace(p, place_mean=None) if p.location == "Dock" else p
        for p in confounded_places()
    ]
    report = measure_taste(
        photos, feature="clip@1", quality_weight=0.0, resamples=10, fit="centred"
    )
    assert report.uncentred == ("Dock",)
    assert report.fitted is not None


def test_without_first_picks_only_the_range_is_judged() -> None:
    report = measure_taste(
        rated_places({"Mill": 60, "Harbor": 60}), feature="clip@1", quality_weight=0.35
    )
    assert report.first_at == ()
    assert all(r.general_first == r.yours_first == () for r in report.rows)


def test_photos_to_compare_are_shown_once_credited_to_each_template_and_shuffled() -> None:
    general = [f"g{i}" for i in range(10)] + ["both"]
    yours = ["both", *(f"y{i}" for i in range(10))]
    chosen = photos_to_compare([general, yours], seed=3)
    keys = [c.content_key for c in chosen]
    assert sorted(keys) == sorted({*general, *yours})  # once each
    by_key = {c.content_key: c for c in chosen}
    assert by_key["both"].shown_by == ((0, 10), (1, 0))  # (template, rank) for each that showed it
    assert by_key["g2"].shown_by == ((0, 2),)
    assert keys != sorted(keys, key=lambda k: min(r for _, r in by_key[k].shown_by))
    assert photos_to_compare([general, yours], seed=3) == chosen  # repeatable
    assert keys != [c.content_key for c in photos_to_compare([general, yours], seed=4)]


def test_compare_counts_the_photos_you_liked_among_each_templates_picks() -> None:
    def photo(liked: bool, *shown: str) -> ComparedPhoto:
        return ComparedPhoto(liked=liked, shown_by=frozenset(shown))

    mill = [
        photo(True, "general", "yours"),
        photo(True, "general"),
        photo(False, "general"),
        photo(True, "yours"),
        photo(True, "yours"),
    ]
    harbor = [photo(False, "general"), photo(True, "yours")]
    report = measure_compare([("Mill", mill), ("Harbor", harbor), ("Nowhere", [])])
    assert [r.place for r in report.rows] == ["Mill", "Harbor"]
    row = report.rows[0]
    assert (row.rated, row.general, row.yours) == (5, FirstPicks(3, 2), FirstPicks(3, 3))
    assert (row.general_only, row.yours_only, row.both) == (
        FirstPicks(2, 1),
        FirstPicks(2, 2),
        FirstPicks(1, 1),
    )
    assert report.general == FirstPicks(4, 2)
    assert report.yours == FirstPicks(4, 4)
    assert report.general_only == FirstPicks(3, 1)
    assert report.yours_only == FirstPicks(3, 3)
    assert report.empty == ["Nowhere"]


def compared(liked: bool, *shown: str) -> ComparedPhoto:
    return ComparedPhoto(liked=liked, shown_by=frozenset(shown))


def test_the_liked_share_down_each_list_is_read_in_tens_with_how_far_it_falls() -> None:
    def session(place: str) -> tuple[str, list[ComparedPhoto]]:
        photos = []
        for rank in range(30):
            # Yours falls from 9 liked in 10 to 5; General is flat at 7.
            yours_liked = rank % 10 < (9, 7, 5)[rank // 10]
            general_liked = rank % 10 < 7
            photos.append(ComparedPhoto(yours_liked, frozenset({"yours"}), (("yours", rank),)))
            photos.append(
                ComparedPhoto(general_liked, frozenset({"general"}), (("general", rank),))
            )
        return place, photos

    report = measure_compare([session(p) for p in ("Mill", "Harbor", "Fern")], resamples=200)
    assert [(f.rated, f.liked) for f in report.curve["yours"]] == [(30, 27), (30, 21), (30, 15)]
    assert [(f.rated, f.liked) for f in report.curve["general"]] == [(30, 21)] * 3
    assert report.fall["yours"].value == pytest.approx(0.9 - 0.5)
    assert report.fall["general"].value == pytest.approx(0.0)


def listed(
    place: str,
    yours: list[bool],
    general: list[bool] | None = None,
    *,
    marker: int | None = None,
) -> tuple[str, list[ComparedPhoto]]:
    """A session whose lists are rated in order, one verdict per rank: Yours's (with the
    marker its run drew, if any) and General's."""
    photos = [
        ComparedPhoto(liked, frozenset({"yours"}), (("yours", rank),), marker=marker)
        for rank, liked in enumerate(yours)
    ]
    photos += [
        ComparedPhoto(liked, frozenset({"general"}), (("general", rank),))
        for rank, liked in enumerate(general or [])
    ]
    return place, photos


PLACES3 = ("Mill", "Harbor", "Fern")


def test_precision_is_the_liked_share_of_each_lists_first_10_20_and_30() -> None:
    falling = [rank % 10 < (9, 7, 5)[rank // 10] for rank in range(30)]  # 0.9, then 0.7, then 0.5
    flat = [rank % 10 < 7 for rank in range(30)]
    report = measure_compare([listed(p, falling, flat) for p in PLACES3], resamples=100)
    assert PRECISION_AT == (10, 20, 30, 60)
    assert sorted(report.precision) == [10, 20, 30]  # no list reaches 60
    at_10, at_20, at_30 = (report.precision[k] for k in (10, 20, 30))
    assert (at_10.a.value, at_20.a.value, at_30.a.value) == pytest.approx((0.9, 0.8, 0.7))
    assert all(report.precision[k].b.value == pytest.approx(0.7) for k in (10, 20, 30))
    assert at_10.difference.value == pytest.approx(0.2)
    assert at_30.difference.value == pytest.approx(0.0)
    assert at_30.a.places == 3  # three places, each counting once in the interval


def test_a_list_of_sixty_has_a_precision_at_sixty() -> None:
    first_half = [rank < 30 for rank in range(60)]
    report = measure_compare([listed(p, first_half, first_half) for p in PLACES3], resamples=50)
    assert report.precision[30].a.value == 1.0
    assert report.precision[60].a.value == pytest.approx(0.5)


def test_precision_at_sixty_reads_only_the_places_whose_lists_reach_sixty() -> None:
    # 12 November 2026: a place of 60 made every session's whole list "the first 60", so the
    # figure was the share of everything shown, not what the first 60 of a list hold.
    sixty = [rank < 30 for rank in range(60)]  # 30 liked, then 30 not
    thirty = [True] * 30
    report = measure_compare(
        [
            listed("Mill", thirty, thirty),
            listed("Harbor", thirty, thirty),
            listed("Fern", sixty, sixty),
        ],
        resamples=50,
    )
    assert (report.precision[30].a.places, report.precision[30].a.value) == (3, 1.0)
    assert report.precision[60].a.places == 1  # the lists of 30 have no precision at 60
    assert report.precision[60].a.value == pytest.approx(0.5)


def test_precision_at_a_rank_needs_both_lists_to_reach_it_at_that_place() -> None:
    sixty = [rank < 30 for rank in range(60)]
    report = measure_compare(
        [listed("Mill", sixty, sixty), listed("Harbor", sixty, [True] * 30)], resamples=50
    )
    assert report.precision[60].a.places == 1  # General's list at Harbor stops at 30
    assert report.precision[60].a.value == pytest.approx(0.5)
    assert report.precision[30].a.places == 2


def test_no_precision_is_read_at_a_rank_no_place_reaches() -> None:
    report = measure_compare([listed(p, [True] * 30, [True] * 30) for p in PLACES3], resamples=20)
    assert 60 not in report.precision


def test_the_headline_counts_each_lists_first_thirty_and_the_rest_is_read_apart() -> None:
    sixty = [rank < 30 for rank in range(60)]
    _, photos = listed("Mill", sixty, sixty)
    early_and_late = ComparedPhoto(
        True, frozenset({"general", "yours"}), (("general", 5), ("yours", 45))
    )
    report = measure_compare([("Mill", [*photos, early_and_late])], resamples=20)
    assert BAR_AT == 30
    assert report.yours == FirstPicks(30, 30)  # the photo Yours had at 45 is not in its first 30
    assert report.general == FirstPicks(31, 31)
    assert (report.rows[0].both, report.rows[0].general_only) == (
        FirstPicks(0, 0),
        FirstPicks(31, 31),
    )
    assert report.shown is not None
    assert report.shown.a.value == report.precision[30].a.value == 1.0
    assert report.precision[60].a.value == pytest.approx(31 / 61)  # but it is within Yours's 60


def test_a_list_shorter_than_the_bar_is_counted_whole() -> None:
    report = measure_compare([listed("Mill", [True] * 12, [False] * 12)], resamples=10)
    assert (report.yours, report.general) == (FirstPicks(12, 12), FirstPicks(12, 0))


def pooled(place: str, bursts: int | None) -> tuple[str, list[ComparedPhoto]]:
    """A session of one liked pick of Yours's and one disliked pick of General's, from a pool
    that held ``bursts`` bursts."""
    return place, [
        ComparedPhoto(True, frozenset({"yours"}), (("yours", 0),), bursts=bursts),
        ComparedPhoto(False, frozenset({"general"}), (("general", 0),), bursts=bursts),
    ]


def test_each_session_says_how_many_bursts_its_pool_held() -> None:
    report = measure_compare([pooled("Mill", 300), pooled("Harbor", None)], resamples=10)
    assert [r.bursts for r in report.rows] == [300, None]
    assert report.small is None  # no size was asked for


def test_places_with_fewer_bursts_than_asked_for_are_reported_apart() -> None:
    report = measure_compare(
        [pooled("Mill", 300), pooled("Edge", 100), pooled("Harbor", 99), pooled("Fern", None)],
        min_bursts=100,
        resamples=10,
    )
    assert [r.place for r in report.rows] == ["Mill", "Edge", "Fern"]  # 100 is not under 100
    assert report.small is not None
    assert [(r.place, r.bursts) for r in report.small.rows] == [("Harbor", 99)]
    assert report.general == FirstPicks(3, 0)  # the small place is not in the headline
    assert report.small.general == FirstPicks(1, 0)


def test_a_size_nothing_falls_under_leaves_nothing_apart() -> None:
    report = measure_compare([pooled("Mill", 300)], min_bursts=100, resamples=10)
    assert report.small is None
    assert [r.place for r in report.rows] == ["Mill"]


def test_the_share_running_down_a_list_first_falls_below_80_percent_at_a_pick() -> None:
    stops = [rank < 15 for rank in range(30)]  # 15 liked, then none: 15/19 is the first under 0.8
    good = [True] * 30
    report = measure_compare([listed(p, stops, good) for p in PLACES3], resamples=50)
    assert RUNNING_LINE == 0.8
    assert report.drop["yours"].positions == (19, 19, 19)
    assert report.drop["yours"].median == 19
    assert report.drop["general"].positions == (None, None, None)  # never, within the list
    assert math.isnan(report.drop["general"].median)


def test_the_running_share_is_read_only_once_ten_photos_are_rated() -> None:
    late = [False] + [True] * 29  # 0 of 1 rated at the first pick, but 9 of 10 by the tenth
    short = [True] * 6  # too few rated photos to read a running share from
    report = measure_compare([listed("Mill", late), listed("Harbor", short)], resamples=20)
    assert RUNNING_AFTER == 10
    assert report.drop["yours"].positions == (None,)  # Harbor is left out, Mill never fell


def test_the_marker_is_checked_against_the_liked_share_above_it_and_below_it() -> None:
    until_the_line = [rank < 20 for rank in range(30)]
    report = measure_compare(
        [listed(p, until_the_line, marker=20) for p in PLACES3]
        + [listed("Nowhere", until_the_line)],  # a session whose list drew no marker
        resamples=100,
    )
    check = report.marker
    assert check is not None
    assert (check.above, check.below) == (FirstPicks(60, 60), FirstPicks(30, 0))
    assert check.difference.difference.value == pytest.approx(1.0)
    assert check.at == (20, 20, 20)
    assert check.fell == (26, 26, 26)  # 20 of 25 is 0.8; the 26th pick is the first under it


def test_the_pick_at_the_markers_rank_is_below_the_line() -> None:
    report = measure_compare([listed("Mill", [True, True, False, False], marker=2)], resamples=10)
    assert report.marker is not None
    assert (report.marker.above, report.marker.below) == (FirstPicks(2, 2), FirstPicks(2, 0))


def test_without_a_marker_there_is_nothing_to_check() -> None:
    report = measure_compare([listed("Mill", [True] * 12)], resamples=10)
    assert report.marker is None


def test_a_photo_both_templates_showed_counts_down_each_list_at_its_own_place() -> None:
    both = ComparedPhoto(True, frozenset({"general", "yours"}), (("general", 25), ("yours", 3)))
    report = measure_compare([("Mill", [both])], resamples=10)
    assert [f.rated for f in report.curve["yours"]] == [1]  # in its first ten
    assert [f.rated for f in report.curve["general"]] == [0, 0, 1]  # in its last
    assert report.fall == {}  # one ten, or none at the start: nothing to fall from


def test_photos_without_a_place_in_the_list_make_no_curve() -> None:
    report = measure_compare([("Mill", [compared(True, "yours"), compared(False, "general")])])
    assert report.curve == {}
    assert report.fall == {}


def test_compare_shares_have_intervals_by_place_and_yours_minus_general_is_paired() -> None:
    def session(place: str) -> tuple[str, list[ComparedPhoto]]:
        yours = [compared(i % 4 != 0, "yours") for i in range(12)]
        general = [compared(i % 2 == 0, "general") for i in range(12)]
        return place, [*yours, *general, compared(True, "general", "yours")]

    report = measure_compare([session(p) for p in ("Mill", "Harbor", "Fern")], resamples=200)
    assert report.shown is not None
    assert report.shown.a.value == report.yours.share
    assert report.shown.b.value == report.general.share
    assert math.isclose(report.shown.difference.value, report.yours.share - report.general.share)
    assert (report.shown.difference.places, report.shown.difference.rated) == (3, 75)
    assert report.only is not None
    assert report.only.a.value == report.yours_only.share
    assert report.only.b.value == report.general_only.share
    assert report.macro == {"general": report.general.share, "yours": report.yours.share}


def test_two_sessions_at_one_place_are_drawn_together() -> None:
    mill = [compared(True, "yours"), compared(False, "general")]
    report = measure_compare([("Mill", mill), ("Mill", mill), ("Harbor", mill)], resamples=20)
    assert report.shown is not None
    assert (report.shown.a.places, report.shown.a.rated) == (2, 6)
    assert len(report.rows) == 3  # still one row per session


# -- stories -----------------------------------------------------------------------


def trio(similarity: float) -> np.ndarray:
    """Three unit vectors whose every pair has cosine ``similarity``."""
    a = math.sqrt(similarity)
    b = math.sqrt(1 - similarity)
    return f.unit(np.hstack([np.full((3, 1), a), b * np.eye(3)]))


def story(
    run: str, marks: tuple[int, int, int], similarity: float, *, connection: float = 0.9
) -> RatedStory:
    return RatedStory(
        run_id=run,
        vectors=trio(similarity),
        marks=marks,
        evidence=(
            Evidence(signal="strength", value=0.8, weight=0.4),
            Evidence(signal="theme", value=connection, weight=0.3),
            Evidence(signal="mood", value=connection, weight=0.3),
        ),
    )


def test_a_story_is_liked_rejected_or_mixed_by_how_many_photos_did_not_fit() -> None:
    assert story("r", (1, 1, 1), 0.7).verdict == "liked"
    assert story("r", (1, 0, 1), 0.7).verdict == "liked"
    assert story("r", (1, -1, 1), 0.7).verdict == "mixed"
    assert story("r", (-1, 1, -1), 0.7).verdict == "rejected"  # an anchor's two companions
    assert story("r", (-1, -1, -1), 0.7).verdict == "rejected"


def test_variety_ranks_liked_stories_above_rejected_sets_of_look_alikes() -> None:
    stories = [
        story("r", (1, 1, 1), 0.75, connection=0.85),
        story("r", (1, 1, 1), 0.78, connection=0.88),
        story("r", (-1, -1, -1), 0.92, connection=0.99),
        story("r", (-1, -1, -1), 0.90, connection=0.98),
    ]
    report = measure_stories(stories, [(0.8, 0.25)], near_duplicate=0.95)
    today, varied = report.rows
    assert (today.too_alike, today.weight) == (None, 0.0)
    assert today.auc == 0.0  # today the look-alikes win
    assert (varied.too_alike, varied.weight) == (0.8, 0.25)
    assert varied.auc == 1.0
    assert (report.liked, report.rejected, report.mixed) == (2, 2, 0)


def test_stories_are_compared_within_their_own_run() -> None:
    stories = [
        story("a", (1, 1, 1), 0.7),
        story("a", (-1, -1, -1), 0.9),
        story("b", (1, 1, 1), 0.9),  # no rejected story in run b: nothing to compare
    ]
    report = measure_stories(stories, [(0.8, 0.5)], near_duplicate=0.95)
    assert report.rows[1].auc == 1.0
    assert report.runs == 2


def test_without_liked_and_rejected_stories_in_one_run_nothing_is_measured() -> None:
    report = measure_stories([story("a", (1, 1, 1), 0.7)], [(0.8, 0.5)], near_duplicate=0.95)
    assert all(math.isnan(row.auc) for row in report.rows)


def test_a_photo_that_did_not_fit_is_counted_when_it_was_the_least_alike() -> None:
    e = np.eye(4)
    odd = RatedStory(
        run_id="r",
        vectors=f.unit([e[0] + e[1], e[0] + 0.9 * e[1] + 0.1 * e[2], e[3]]),
        marks=(1, 1, -1),
        evidence=(
            Evidence(signal="strength", value=0.8, weight=0.4),
            Evidence(signal="theme", value=0.9, weight=0.6),
        ),
    )
    alike = dataclasses.replace(odd, marks=(-1, 1, 1))
    report = measure_stories([odd, alike], [], near_duplicate=0.95)
    assert (report.doesnt_fit, report.odd_one_out) == (2, 1)


def test_a_story_rated_whole_keeps_its_rating_and_not_sure_is_left_out() -> None:
    works = dataclasses.replace(story("s", (1, -1, 1), 0.7), stated="liked")
    doesnt = dataclasses.replace(story("s", (1, 1, 1), 0.9), stated="rejected")
    unsure = dataclasses.replace(story("s", (1, 1, 1), 0.8), stated="unsure")
    assert (works.verdict, doesnt.verdict, unsure.verdict) == ("liked", "rejected", "unsure")
    report = measure_stories([works, doesnt, unsure], [(0.8, 0.5)], near_duplicate=0.95)
    assert (report.liked, report.rejected, report.mixed, report.unsure) == (1, 1, 0, 1)
    assert report.rows[1].auc == 1.0
    assert report.doesnt_fit == 1  # one photo alone did not fit a story that works


def test_session_stories_are_counted_by_how_they_were_told() -> None:
    general, plain = ("general", True, "today"), ("general", False, "today")
    yours, roles = ("yours", True, "today"), ("general", True, "roles")
    stories = [
        dataclasses.replace(story("s", (1, 1, 1), 0.7), stated="liked", arms=(general, yours)),
        dataclasses.replace(story("s", (1, 1, 1), 0.9), stated="rejected", arms=(plain,)),
        dataclasses.replace(story("s", (1, 1, 1), 0.8), stated="unsure", arms=(general,)),
        dataclasses.replace(story("s", (1, 1, 1), 0.6), stated="liked", arms=(roles,)),
        story("r", (1, 1, 1), 0.7),  # rated on a result: told one way, not recorded
    ]
    report = measure_stories(stories, [], near_duplicate=0.95)
    rows = [(a.template, a.variety, a.way, a.works, a.unsure, a.doesnt) for a in report.arms]
    assert rows == [
        ("general", False, "today", 0, 0, 1),
        ("general", True, "roles", 1, 0, 0),
        ("general", True, "today", 1, 1, 0),
        ("yours", True, "today", 1, 0, 0),
    ]


def test_variety_rows_have_intervals_by_run_and_each_is_paired_with_no_variety() -> None:
    stories = [
        story(run, marks, sim, connection=conn)
        for run in ("r1", "r2", "r3")
        for marks, sim, conn in (
            ((1, 1, 1), 0.75, 0.85),
            ((1, 1, 1), 0.78, 0.88),
            ((-1, -1, -1), 0.92, 0.99),
            ((-1, -1, -1), 0.90, 0.98),
        )
    ]
    report = measure_stories(stories, [(0.8, 0.25)], near_duplicate=0.95, resamples=100)
    today, varied = report.rows
    assert today.interval is not None
    assert varied.interval is not None
    assert today.interval.value == today.auc
    assert (varied.interval.places, varied.interval.rated, varied.interval.unit) == (
        3,
        12,
        "run or session",
    )
    assert today.versus is None  # the baseline
    assert varied.versus is not None
    assert varied.versus.value == 1.0
    assert varied.versus.excludes_zero


def test_each_way_of_telling_has_its_share_that_works_and_is_paired_with_todays() -> None:
    today, roles = ("general", True, "today"), ("general", True, "roles")
    told: list[tuple[StoryArmKey, StoryVerdict]] = [
        (today, "liked"),
        (today, "rejected"),
        (roles, "liked"),
        (roles, "liked"),
        (roles, "unsure"),
    ]
    stories = [
        dataclasses.replace(story(session, (1, 1, 1), 0.7), stated=verdict, arms=(arm,))
        for session in ("s1", "s2", "s3")
        for arm, verdict in told
    ]
    report = measure_stories(stories, [], near_duplicate=0.95, resamples=100)
    by_way = {a.way: a for a in report.arms}
    assert by_way["today"].works_share is not None
    assert by_way["today"].works_share.value == 0.5  # not sure is left out
    assert by_way["today"].versus_today is None
    versus = by_way["roles"].versus_today
    assert versus is not None
    assert versus.value == 0.5
    assert versus.places == 3


def test_a_way_is_compared_with_todays_only_in_the_sessions_that_told_it() -> None:
    today, deeper = ("general", True, "today"), ("general", True, "deeper")
    earlier = [
        dataclasses.replace(story(session, (1, 1, 1), 0.7), stated="liked", arms=(today,))
        for session in ("s1", "s2")
    ]
    told: list[tuple[StoryArmKey, StoryVerdict]] = [
        (today, "liked"),
        (today, "rejected"),
        (today, "rejected"),
        (today, "rejected"),
        (deeper, "liked"),
        (deeper, "rejected"),
    ]
    later = [
        dataclasses.replace(story(session, (1, 1, 1), 0.7), stated=verdict, arms=(arm,))
        for session in ("s3", "s4")
        for arm, verdict in told
    ]
    report = measure_stories([*earlier, *later], [], near_duplicate=0.95, resamples=100)
    by_way = {a.way: a for a in report.arms}
    versus = by_way["deeper"].versus_today
    assert versus is not None
    assert versus.value == 0.25  # 0.5 against 0.25 in s3 and s4, not 0.4 in all four
    assert versus.places == 2
    share = by_way["deeper"].works_share
    assert share is not None
    assert share.places == 2


def test_stories_told_without_variety_are_compared_with_todays_way_with_it() -> None:
    today, plain = ("yours", True, "today"), ("yours", False, "today")
    told: list[tuple[StoryArmKey, StoryVerdict]] = [
        (today, "liked"),
        (today, "liked"),
        (plain, "liked"),
        (plain, "rejected"),
    ]
    stories = [
        dataclasses.replace(story(session, (1, 1, 1), 0.7), stated=verdict, arms=(arm,))
        for session in ("s1", "s2")
        for arm, verdict in told
    ]
    report = measure_stories(stories, [], near_duplicate=0.95, resamples=50)
    (without,) = [a for a in report.arms if not a.variety]
    assert without.versus_today is not None
    assert without.versus_today.value == -0.5


def test_each_story_is_rated_once_with_every_way_that_told_it() -> None:
    told = [
        (("general", True, "today"), [("a", "b", "c"), ("d", "e", "f")]),
        (("general", True, "roles"), [("c", "b", "a"), ("g", "h", "i")]),
    ]
    chosen = stories_to_rate(told, seed=1)
    assert {c.told: c.arms for c in chosen} == {
        (0, 0): (("general", True, "today"), ("general", True, "roles")),
        (0, 1): (("general", True, "today"),),
        (1, 1): (("general", True, "roles"),),
    }
    assert stories_to_rate(told, seed=1) == chosen
    orders = {tuple(c.told for c in stories_to_rate(told, seed=s)) for s in range(20)}
    assert len(orders) > 1  # shuffled, so the order cannot say how a story was told


# -- finished stories ------------------------------------------------------------------


def test_a_story_wins_each_swap_it_outscores_and_ties_count_half() -> None:
    assert swap_wins(0.8, [0.5, 0.7, 0.8, 0.9]) == pytest.approx((1 + 1 + 0.5) / 4)
    assert math.isnan(swap_wins(0.8, []))


def test_a_finished_story_wins_over_its_sides_and_says_what_was_shortlisted() -> None:
    story = FinishedStory(
        name="A",
        score=0.8,
        swaps=((0.5, 0.9), (), (0.1, 0.2, 0.3, 0.9)),
        shortlisted=(True, False, True),
    )
    assert story.wins[0] == pytest.approx(0.5)
    assert math.isnan(story.wins[1])  # nothing could take the middle's place
    assert story.win == pytest.approx((0.5 + 0.75) / 2)  # sides without swaps do not count
    report = FinishedReport(
        template="general",
        variant="defaults",
        stories=[story, dataclasses.replace(story, name="B", shortlisted=(True, True, True))],
        missing=["W"],
    )
    assert report.mean_win == pytest.approx(story.win)
    assert report.shortlisted == pytest.approx(5 / 6)
    assert report.all_three_shortlisted == pytest.approx(0.5)
    assert report.anchors_tried == 0.0
    tried = dataclasses.replace(report, stories=[dataclasses.replace(story, anchor_tried=True)])
    assert tried.anchors_tried == 1.0


def finished(template: str, wins: dict[str, float]) -> FinishedReport:
    """A report whose stories each beat ``wins[name]`` of their swaps (1, 0, or a tie)."""
    stories = [
        FinishedStory(
            name=name, score=0.5, swaps=((1.0 - win,),) * 3, shortlisted=(True, True, True)
        )
        for name, win in wins.items()
    ]
    return FinishedReport(template=template, variant="defaults", stories=stories, missing=[])


def test_finished_stories_have_an_interval_by_story_and_each_report_is_paired_with_the_first(
) -> None:  # fmt: skip
    first = finished("general", {f"s{i}": 0.5 for i in range(8)})
    better = finished("yours", {f"s{i}": 1.0 for i in range(8)})
    same = finished("yours", {f"s{i}": (1.0 if i % 2 else 0.0) for i in range(8)})
    measured = finished_intervals([first, better, same], resamples=100)
    assert measured[0].win_interval is not None
    assert measured[0].win_interval.value == first.mean_win
    assert (measured[0].win_interval.places, measured[0].win_interval.unit) == (8, "story")
    assert measured[0].versus is None  # the first is what the others are compared with
    assert measured[1].versus is not None
    assert measured[1].versus.value == pytest.approx(0.5)
    assert measured[1].versus.excludes_zero
    assert measured[2].versus is not None
    assert not measured[2].versus.excludes_zero


def test_anchor_samples_take_each_kind_of_photo_in_turn_spread_over_strength() -> None:
    ranked = [(f"a{i}", "wide") for i in range(10)] + [(f"m{i}", "macro") for i in range(10)]
    ranked += [("p0", "wide, a person"), ("p1", "wide, a person")]
    chosen = keys(sample_for_anchors(ranked, seed=3, size=12))
    assert Counter(k[0] for k in chosen) == {"a": 5, "m": 5, "p": 2}
    assert {"a0", "a9", "m0", "m9"} <= set(chosen)  # the strongest and weakest of each
    assert keys(sample_for_anchors(ranked, seed=3, size=12)) == chosen
    orders = {tuple(keys(sample_for_anchors(ranked, seed=s, size=12))) for s in range(10)}
    assert len(orders) > 1  # shuffled, so the order says nothing of strength or kind
    small = keys(sample_for_anchors(ranked[:5], seed=3, size=12))
    assert sorted(small) == [f"a{i}" for i in range(5)]


def test_an_anchor_sample_leaves_out_what_was_rated_and_says_how_each_photo_was_drawn() -> None:
    ranked = [(f"a{i}", "wide") for i in range(10)] + [(f"m{i}", "macro") for i in range(10)]
    drawn = sample_for_anchors(ranked, seed=3, size=6, exclude={"a0", "a1", "m5"})
    assert not {"a0", "a1", "m5"} & set(keys(drawn))
    assert Counter(d.stratum for d in drawn) == {"wide": 3, "macro": 3}
    assert {d.inclusion for d in drawn} == {3 / 8, 3 / 9}
    assert {d.inclusion for d in drawn if d.stratum == "wide"} == {3 / 8}


def test_anchor_ratings_say_which_signal_finds_anchors_within_each_place() -> None:
    rows = [
        AnchorRating(place="A", could=True, signals={"strength": 0.2, "person or animal": 0.9}),
        AnchorRating(place="A", could=False, signals={"strength": 0.8, "person or animal": 0.1}),
        AnchorRating(place="B", could=True, signals={"strength": 0.1, "person or animal": 0.6}),
        AnchorRating(place="B", could=False, signals={"strength": 0.9, "person or animal": 0.7}),
        AnchorRating(place="B", could=False, signals={"strength": 0.0}),
    ]
    report = measure_anchors(rows)
    found = {s.signal: s for s in report.signals}
    assert found["strength"].auc == pytest.approx(1 / 3)  # A: lost; B: lost, won
    assert found["strength"].pairs == 3
    assert found["person or animal"].auc == pytest.approx(0.5)
    assert found["person or animal"].pairs == 2  # a photo without the signal is left out
    assert (report.could, report.could_not, report.places) == (2, 3, 2)
    measured = report.measured
    assert measured is not None
    assert measured.auc["strength"].value == pytest.approx(1 / 3)
    assert (measured.auc["strength"].places, measured.auc["strength"].rated) == (2, 5)


def test_each_signal_is_measured_within_places_with_its_interval_and_against_a_baseline() -> None:
    rng = np.random.default_rng(0)
    by_place = {}
    for place in ("Mill", "Harbor", "Fern", "Birch"):
        liked = rng.random(40) > 0.5
        by_place[place] = [
            Scored(
                liked=bool(y),
                scores={"strength": float(y) + rng.normal(), "noise": float(rng.random())},
            )
            for y in liked
        ]
    measured = measure_signals(by_place, baseline="strength", resamples=200)
    assert list(measured.auc) == ["strength", "noise"]
    strength = measured.auc["strength"]
    assert 0.5 < strength.lo <= strength.value <= strength.hi
    assert (strength.places, strength.rated) == (4, 160)
    assert measured.auc["noise"].lo < 0.5 < measured.auc["noise"].hi
    assert list(measured.versus) == ["noise"]  # the baseline is not compared with itself
    versus = measured.versus["noise"]
    assert versus.value == pytest.approx(measured.auc["noise"].value - strength.value)
    assert versus.hi < 0
    assert set(measured.macro) == {"strength", "noise"}


def test_signals_pool_like_dislike_pairs_as_the_location_table_does() -> None:
    by_place = {
        "a": [Scored(True, {"x": 1.0}), Scored(True, {"x": 1.0}),
              Scored(False, {"x": 0.0}), Scored(False, {"x": 0.0})],
        "b": [Scored(True, {"x": 0.0}), Scored(False, {"x": 1.0})],
    }  # fmt: skip
    measured = measure_signals(by_place, resamples=10)
    assert measured.auc["x"].value == pytest.approx(0.8)  # 4 pairs right, 1 wrong
    assert measured.macro["x"] == pytest.approx(0.5)  # each place once
    assert measured.versus == {}


def told(
    session: str | None, place: str, works: bool, *, bond: float, theme: float
) -> SignalledStory:
    """A story whose pairs say one thing (``theme``, which today's way reads) and whose
    close pair another (``bond``)."""
    return SignalledStory(
        session=session,
        places=frozenset({place}),
        signals={"strength": 0.5, "theme": theme, "bond": bond},
        works=works,
        today=theme,
    )


def sessions_where_the_close_pair_decides(n: int = 4) -> list[SignalledStory]:
    stories = []
    for i in range(n):
        for j in range(3):
            stories.append(told(f"s{i}", f"P{i}", True, bond=0.8 + 0.05 * j, theme=0.3))
            stories.append(told(f"s{i}", f"P{i}", False, bond=0.2 + 0.05 * j, theme=0.6))
    return stories


def test_a_scorer_learned_from_other_sessions_judges_each_session_against_today() -> None:
    report = learn_stories(sessions_where_the_close_pair_decides(), resamples=50)
    assert [s.session for s in report.sessions] == ["s0", "s1", "s2", "s3"]
    assert all(s.auc["learned"] == 1.0 and s.auc["today"] == 0.0 for s in report.sessions)
    assert report.auc["learned"].value == 1.0
    assert report.versus["learned"].value == 1.0  # learned minus today, paired by session
    assert report.versus["learned"].unit == "session"
    assert report.weights["bond"] > 0 > report.weights["theme"]
    assert report.signals == ("strength", "theme", "bond")


def test_a_session_is_never_judged_by_a_scorer_that_learned_from_its_places() -> None:
    stories = [
        told("a", "P", True, bond=0.9, theme=0.1),
        told("b", "P", True, bond=0.9, theme=0.1),  # another session at the same place
        told(None, "P", True, bond=0.9, theme=0.1),  # a story you finished there
        told("c", "Q", False, bond=0.1, theme=0.9),
        told(None, "R", True, bond=0.9, theme=0.1),
    ]
    assert learned_apart(stories, "a") == [3, 4]
    assert learned_apart(stories, "c") == [0, 1, 2, 4]


def test_finished_stories_are_learned_from_as_stories_that_work_and_never_judged() -> None:
    stories = [
        *sessions_where_the_close_pair_decides(3),
        told(None, "F", True, bond=0.9, theme=0.2),
        told(None, "G", True, bond=0.85, theme=0.3),
    ]
    report = learn_stories(stories, resamples=50)
    assert report.finished == 2
    assert None not in {s.session for s in report.sessions}
    assert set(report.auc) == {"today", "learned", "learned, sessions only"}


def test_a_signal_not_measured_for_every_story_is_left_out_and_said() -> None:
    stories = sessions_where_the_close_pair_decides(3)
    stories[0] = dataclasses.replace(stories[0], signals={"strength": 0.5, "theme": 0.3})
    report = learn_stories(stories, resamples=50)
    assert report.signals == ("strength", "theme")
    assert report.left_out == ("bond",)


def test_a_session_with_nothing_of_both_kinds_to_learn_from_is_not_scored() -> None:
    stories = [
        told("a", "P", True, bond=0.9, theme=0.1),
        told("a", "P", False, bond=0.1, theme=0.9),
        told("b", "Q", True, bond=0.9, theme=0.1),  # what a learns from: only works
    ]
    report = learn_stories(stories, resamples=50)
    a, _ = report.sessions
    assert "learned" not in a.auc
    assert a.auc["today"] == 0.0


def test_swapped_stories_are_learned_from_as_doesnt_by_their_own_scorer_alone() -> None:
    stories = sessions_where_the_close_pair_decides(3)
    without = learn_stories(stories, resamples=50)
    assert "learned, with swaps" not in without.auc
    assert without.swaps == 0
    finished = told(None, "F", True, bond=0.9, theme=0.2)
    # A finished story with one photo swapped out: everything the scorers see but the answer.
    swap = dataclasses.replace(finished, works=False, swapped=True)
    report = learn_stories([*stories, finished, swap, swap], resamples=50)
    assert (report.finished, report.swaps) == (1, 2)
    assert set(report.auc) == {"today", "learned", "learned, sessions only", "learned, with swaps"}
    assert "learned, with swaps" in report.versus
    assert None not in {s.session for s in report.sessions}  # swaps are never judged
    assert report.weights == learn_stories([*stories, finished], resamples=50).weights


def test_today_s_way_with_other_options_is_judged_against_today_too() -> None:
    stories = [
        dataclasses.replace(s, variants={"distinct_weight=0.3": s.signals["bond"]})
        for s in sessions_where_the_close_pair_decides(3)
    ]
    report = learn_stories(stories, resamples=50)
    assert report.variants == ("distinct_weight=0.3",)
    assert report.versus["distinct_weight=0.3"].value == 1.0
    assert all(s.auc["distinct_weight=0.3"] == 1.0 for s in report.sessions)


def explored_and_travelled() -> list[TastePhoto]:
    """Five places of one kind (EXPLORE), close together in the embedding, where what you
    like lies along an axis Yours can learn; and three trips (TRAVEL), each somewhere no
    rating is, where what you like is what General sees."""
    rng = np.random.default_rng(0)
    out = []
    places = [(f"Mill {i}", "EXPLORE", 0) for i in range(5)]
    places += [(f"Trip {i}", "TRAVEL", 3 + i) for i in range(3)]
    for place, collection, axis in places:
        x = 0.4 * rng.normal(size=(60, 8))
        x[:, axis] += 1.5
        general = rng.normal(size=60)
        if collection == "EXPLORE":
            liked = x[:, 2] + 0.3 * general > 0
        else:
            liked = general + 0.3 * rng.normal(size=60) > 0
        out += [
            TastePhoto(
                place,
                bool(liked[j]),
                True,
                f.unit(x)[j],
                float(1 / (1 + np.exp(-general[j]))),
                float(rng.random()),
                content_key=f"{place}{j}",
                collection=collection,
            )
            for j in range(60)
        ]
    return out


def test_leaning_on_general_where_yours_has_seen_little_helps_on_trips() -> None:
    """S6: Yours learned from abandoned places knows little about a trip somewhere new, so
    there it leans on General; where its ratings are typical it is Yours, as today."""
    report = measure_taste(
        explored_and_travelled(),
        feature="clip@1",
        quality_weight=0.0,
        resamples=100,
        fit="leaning",
    )
    assert report.fit == "leaning"
    assert report.fitted is not None
    by = dict(report.fitted_by_collection)
    assert set(by) == {"EXPLORE", "TRAVEL"}
    assert by["TRAVEL"].difference.value > 0.1
    assert abs(by["EXPLORE"].difference.value) < 0.05
    assert by["TRAVEL"].difference.places == 3
    rows = {r.location: r for r in report.rows}
    assert rows["Trip 0"].leaned > 0.5 > rows["Mill 0"].leaned
    assert rows["Trip 0"].fitted_strength > rows["Trip 0"].yours_strength


def test_a_candidate_general_is_shown_by_collection_too() -> None:
    """A General kept for trips must not cost the abandoned places (the bar's rule)."""
    photos = explored_and_travelled()
    better = [
        dataclasses.replace(
            p, candidate_general=7.0 if p.liked and p.collection == "TRAVEL" else 3.0
        )
        for p in photos
    ]
    report = measure_taste(better, feature="clip@1", quality_weight=0.0, resamples=100)
    by = dict(report.candidate_general_by_collection)
    assert set(by) == {"EXPLORE", "TRAVEL"}
    assert by["TRAVEL"].difference.value > by["EXPLORE"].difference.value
    plain = measure_taste(photos, feature="clip@1", quality_weight=0.0, resamples=10)
    assert plain.candidate_general_by_collection == ()


def test_yours_learned_on_another_embedding_is_measured_against_todays_paired() -> None:
    """A larger CLIP is judged against today's Yours on the same photos, so the difference
    has its own interval, not two reports read side by side."""
    photos = rated_places({"Mill": 60, "Harbor": 60, "Fern": 60})
    rng = np.random.default_rng(3)
    # Today's vectors say little; the candidate's first axis follows what you liked.
    sharper = [
        dataclasses.replace(
            p,
            candidate_vector=np.asarray(
                [1.0 if p.liked else -1.0, *rng.normal(size=7)], dtype=np.float32
            ),
        )
        for p in photos
    ]
    report = measure_taste(
        sharper, feature="clip@1", quality_weight=0.0, resamples=100, fit="embedding"
    )
    assert report.fit == "embedding"
    assert report.fitted is not None
    assert math.isclose(report.fitted.b.value, report.yours_strength)
    assert report.fitted.difference.value > 0
    assert report.top_fitted is not None
    with pytest.raises(ValueError, match="candidate embedding"):
        measure_taste(photos, feature="clip@1", quality_weight=0.0, fit="embedding")


def test_yours_fitted_on_pairs_among_first_picks_is_measured_against_todays_paired() -> None:
    """Pairs come only from photos a taste chose to show (``whole_range`` false); the AUCs
    judge the other rated photos, so a pair is never judged."""
    photos = rated_places({"Mill": 60, "Harbor": 60, "Fern": 60, "Birch": 60})
    chosen = [dataclasses.replace(p, whole_range=i % 2 == 0) for i, p in enumerate(photos)]
    report = measure_taste(
        chosen, feature="clip@1", quality_weight=0.0, resamples=50, fit="top_pairs"
    )
    assert report.fit == "top_pairs"
    assert report.fitted is not None
    assert math.isclose(report.fitted.b.value, report.yours_strength)
    assert report.fitted.a.value > 0.5  # it learned the first axis from the pairs
    assert report.top_fitted is not None
    with pytest.raises(ValueError, match="first picks"):
        measure_taste(photos, feature="clip@1", quality_weight=0.0, fit="top_pairs")


def test_another_fit_is_shown_place_by_place_and_by_collection_only_when_measured() -> None:
    photos = explored_and_travelled()
    plain = measure_taste(photos, feature="clip@1", quality_weight=0.0, resamples=10)
    assert plain.fitted_by_collection == ()
    assert all(math.isnan(r.fitted_strength) and math.isnan(r.leaned) for r in plain.rows)
    centred = measure_taste(
        [dataclasses.replace(p, place_mean=np.zeros(8, dtype=np.float32)) for p in photos],
        feature="clip@1",
        quality_weight=0.0,
        resamples=10,
        fit="centred",
    )
    assert {name for name, _ in centred.fitted_by_collection} == {"EXPLORE", "TRAVEL"}
    assert all(not math.isnan(r.fitted_strength) and math.isnan(r.leaned) for r in centred.rows)
    one = [dataclasses.replace(p, collection="") for p in photos]
    alone = measure_taste(one, feature="clip@1", quality_weight=0.0, resamples=10, fit="leaning")
    assert alone.fitted_by_collection == ()  # one collection: the overall difference says it


def general_blind() -> list[TastePhoto]:
    """Five places where what you like lies along an axis Yours learns and General's view is
    noise: General misses what Yours sees."""
    rng = np.random.default_rng(0)
    out = []
    for place in ("Mill", "Harbor", "Fern", "Dock", "Quarry"):
        x = f.unit(rng.normal(size=(40, 8)))
        liked = x[:, 0] > 0
        out += [
            TastePhoto(
                place,
                bool(liked[j]),
                True,
                x[j],
                float(rng.random()),
                float(rng.random()),
                content_key=f"{place}{j:02d}",
            )
            for j in range(40)
        ]
    return out


def test_disagreements_are_the_photos_one_taste_ranks_far_above_the_other() -> None:
    """S5: to name what Yours sees that General misses, look at the rated photos the two rank
    furthest apart at places neither learned from, grouped by whom you sided with."""
    found = taste_disagreements(general_blind(), feature="clip@1", quality_weight=0.0, count=500)
    assert found.places == 5
    assert found.rated == 200
    for group, liked, yours_higher in (
        (found.general_missed, True, True),
        (found.general_wrong, False, False),
        (found.yours_missed, True, False),
        (found.yours_wrong, False, True),
    ):
        for p in group:
            assert p.liked == liked
            assert (p.yours - p.general if yours_higher else p.general - p.yours) >= (
                DISAGREEMENT_GAP
            )
    yours_right = len(found.general_missed) + len(found.general_wrong)
    general_right = len(found.yours_missed) + len(found.yours_wrong)
    assert yours_right > 3 * general_right  # General is blind here, so Yours is mostly right
    assert all(0 <= p.general <= 1 and 0 <= p.yours <= 1 for p in found.general_missed)


def test_disagreements_put_the_widest_first_and_keep_count_of_each_group() -> None:
    every = taste_disagreements(general_blind(), feature="clip@1", quality_weight=0.0, count=500)
    few = taste_disagreements(general_blind(), feature="clip@1", quality_weight=0.0, count=3)
    assert len(few.general_missed) == 3
    assert few.general_missed == every.general_missed[:3]
    gaps = [p.yours - p.general for p in every.general_missed]
    assert gaps == sorted(gaps, reverse=True)
    assert few.counts == every.counts  # how many there are, beyond those shown
    assert few.counts[0] == len(every.general_missed)


def scaled_places() -> list[TastePhoto]:
    """Five places where close-ups and macros lean one way in the embedding, you like few
    close-ups and most macros, and what you like within every scale is one axis."""
    rng = np.random.default_rng(0)
    out = []
    for place in ("Mill", "Harbor", "Fern", "Dock", "Quarry"):
        scales = rng.choice(["wide", "close", "macro"], size=80, p=[0.5, 0.35, 0.15])
        x = rng.normal(size=(80, 8))
        x[scales != "wide", 1] += 2.0
        rate = np.asarray([{"wide": 0.8, "close": 0.3, "macro": 0.75}[s] for s in scales])
        cut = np.asarray([np.quantile(x[:, 0], 1 - r) for r in rate])
        liked = x[:, 0] > cut
        out += [
            TastePhoto(
                place,
                bool(liked[j]),
                True,
                x[j].astype(np.float32),
                float(rng.random()),
                float(rng.random()),
                content_key=f"{place}{j:02d}",
                scale=str(scales[j]),
            )
            for j in range(80)
        ]
    return out


def test_yours_fitted_by_scale_ranks_liked_macros_where_they_belong() -> None:
    """S5: fitted as today, Yours learns "close" as a dislike and ranks macros down with it."""
    report = measure_taste(
        scaled_places(), feature="clip@1", quality_weight=0.0, resamples=100, fit="by_scale"
    )
    assert report.fit == "by_scale"
    assert report.fitted is not None
    assert report.fitted.difference.value > 0
    ranks = {scale: (today, fitted) for scale, today, fitted in report.liked_rank_by_scale}
    assert set(ranks) == {"wide", "close", "macro"}
    assert ranks["macro"][1] > ranks["macro"][0] + 0.05
    assert all(0 <= v <= 1 for pair in ranks.values() for v in pair)


def test_fitting_by_scale_needs_every_photo_s_scale() -> None:
    photos = [
        dataclasses.replace(p, scale="") if i == 3 else p for i, p in enumerate(scaled_places())
    ]
    with pytest.raises(ValueError, match="scale"):
        measure_taste(photos, feature="clip@1", quality_weight=0.0, resamples=10, fit="by_scale")
    plain = measure_taste(scaled_places(), feature="clip@1", quality_weight=0.0, resamples=10)
    assert plain.liked_rank_by_scale == ()


def two_views(seed: int = 0, *, candidate_noise: float = 1.0) -> list[TastePhoto]:
    """Five places where what you like is one hidden quality, which today's General and a
    candidate each see through their own noise: neither view is the other."""
    rng = np.random.default_rng(seed)
    out = []
    for place in ("Mill", "Harbor", "Fern", "Dock", "Quarry"):
        quality = rng.normal(size=60)
        liked = quality > np.quantile(quality, 0.3)
        today = quality + rng.normal(size=60)
        candidate = quality + candidate_noise * rng.normal(size=60)
        out += [
            TastePhoto(
                place,
                bool(liked[j]),
                True,
                f.unit(rng.normal(size=8))[0],
                float(1 / (1 + np.exp(-today[j]))),
                float(rng.random()),
                content_key=f"{place}{j:02d}",
                candidate_general=float(candidate[j]),
            )
            for j in range(60)
        ]
    return out


def test_general_blended_with_a_candidate_by_rank_beats_either_view_alone() -> None:
    """1b: two different views of a photo, averaged by rank within each place, untuned."""
    photos = two_views()
    alone = measure_taste(photos, feature="clip@1", quality_weight=0.0, resamples=100)
    blended = measure_taste(
        photos, feature="clip@1", quality_weight=0.0, resamples=100, general_blend=True
    )
    assert blended.general_blend
    assert alone.candidate_general is not None
    assert blended.candidate_general is not None
    assert blended.candidate_general.a.value > alone.candidate_general.a.value
    assert blended.candidate_general.difference.value > 0  # better than today's alone
    with pytest.raises(ValueError, match="candidate"):
        measure_taste(
            [dataclasses.replace(p, candidate_general=None) for p in photos],
            feature="clip@1",
            quality_weight=0.0,
            general_blend=True,
        )


def test_an_offset_yours_reads_the_candidate_general_when_one_is_measured() -> None:
    """1c: Yours on top of the General being tried, learning only where your taste differs."""
    rng = np.random.default_rng(1)
    photos = []
    for place in ("Mill", "Harbor", "Fern", "Dock", "Quarry"):
        x = f.unit(rng.normal(size=(60, 8)))
        seen = rng.normal(size=60)  # what the candidate sees, and today's General does not
        liked = x[:, 0] * 2 + seen > 0
        photos += [
            TastePhoto(
                place,
                bool(liked[j]),
                True,
                x[j],
                float(rng.random()),
                float(rng.random()),
                content_key=f"{place}{j:02d}",
                candidate_general=float(seen[j]),
            )
            for j in range(60)
        ]
    report = measure_taste(
        photos, feature="clip@1", quality_weight=0.0, l2=10, resamples=100, fit="offset"
    )
    assert report.offset_on_candidate
    assert report.fitted is not None
    assert report.fitted.difference.lo > 0
    plain = [dataclasses.replace(p, candidate_general=None) for p in photos]
    today = measure_taste(
        plain, feature="clip@1", quality_weight=0.0, l2=10, resamples=10, fit="offset"
    )
    assert not today.offset_on_candidate


def test_the_top_of_each_place_is_judged_beside_the_whole_range() -> None:
    """Decision 3 (7 November 2026): Strongest's bar is about its first picks, so a candidate
    is also judged by the share you liked among the top of each place's rated photos."""
    photos = [
        dataclasses.replace(p, candidate_general=float(p.liked) + 0.01 * i)
        for i, p in enumerate(two_views())
    ]
    report = measure_taste(photos, feature="clip@1", quality_weight=0.0, resamples=100)
    assert report.top_candidate_general is not None
    assert report.top_candidate_general.a.value == 1.0  # everything it puts on top was liked
    assert report.top_candidate_general.difference.value > 0
    assert 0 < TOP_SHARE < 1
    plain = measure_taste(two_views(), feature="clip@1", quality_weight=0.0, resamples=10)
    assert plain.top_fitted is None


# -- survivors of the mutation run, 10 November 2026 --------------------


def judged(score: float, *, liked: bool, whole_range: bool = True) -> _Judged:
    return _Judged(
        liked=liked,
        general_aesthetic=score,
        yours_aesthetic=score,
        general_strength=score,
        yours_strength=score,
        general_first=(),
        yours_first=(),
        whole_range=whole_range,
    )


def test_the_top_share_of_a_place_leaves_out_photos_drawn_for_their_value() -> None:
    whole = [judged(0.1 * i, liked=i >= 7) for i in range(1, 9)]  # the top two are liked
    for_value = [judged(s, liked=False, whole_range=False) for s in (0.95, 0.9)]
    assert _top_share("general_strength")([[*whole, *for_value]]) == 1.0


def test_the_top_quarter_of_a_place_rounds_up_and_a_small_place_has_a_top_photo() -> None:
    ten = [judged(0.1 * i, liked=i >= 9) for i in range(1, 11)]  # a quarter is 2.5: three count
    assert _top_share("general_strength")([ten]) == pytest.approx(2 / 3)
    assert _top_share("general_strength")([[judged(0.5, liked=True)] * 3]) == 1.0


def test_yours_does_not_win_when_its_range_just_touches_zero() -> None:
    report = measure_taste(
        rated_places(FIVE), feature="clip@1", quality_weight=0.35, l2=1.0, resamples=200
    )
    assert report.yours_wins
    assert report.strength is not None
    touching = dataclasses.replace(
        report.strength, difference=dataclasses.replace(report.strength.difference, lo=0.0)
    )
    assert not dataclasses.replace(report, strength=touching).yours_wins


def test_copies_labelled_differently_count_at_the_higher_label_whichever_comes_first() -> None:
    search = LabelledView("a", "Same view as a", {"heic": 0.3, "jpg": 0.3}, {"heic": 2, "jpg": 1})
    counted, copies = count_copies_once(search, ["heic", "jpg"], f.unit([[1, 0], [1, 0]]))
    assert (counted.labels, copies) == ({"heic": 2}, 1)


def test_a_place_is_one_sided_when_what_is_left_to_learn_from_is_too() -> None:
    mill = rated_places({"Mill": 60})
    harbor = [dataclasses.replace(p, liked=True) for p in rated_places({"Harbor": 60})]
    report = measure_taste(
        [*mill, *harbor], feature="clip@1", quality_weight=0.35, l2=1.0, resamples=10
    )
    assert sorted(report.one_sided) == ["Harbor", "Mill"]  # Mill has nothing to learn dislikes from
    assert report.rows == []
