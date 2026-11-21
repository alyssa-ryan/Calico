"""A taste measured on the photos a taste chose to show, and a file of scores measured beside
General and Yours (audit 11 November 2026, steps 4 and 5): ``measure_taste`` reads the photos
a ranking chose apart from the whole range, and a candidate that is one number per photo
(``TastePhoto.candidate_score``) is judged with the same leave-one-location-out machinery."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from calico.evaluation import TastePhoto, TasteReport, mean_z, measure_taste
from tests import factories as f

PLACES = ("Mill", "Harbor", "Fisher")


def separated(
    place: str,
    n: int,
    *,
    general_high_for_liked: bool,
    whole_range: bool = True,
    from_session: bool = True,
    tag: str = "w",
    seed: int = 0,
) -> list[TastePhoto]:
    """``n`` photos, alternately liked and not, whose General aesthetic ranks the liked ones
    above the others (or exactly below them), with quality the same for all: strength then
    orders them as the aesthetic does, so an AUC is exactly 1 or 0."""
    x = f.unit(np.random.default_rng(seed).normal(size=(n, 8)))
    return [
        TastePhoto(
            place,
            i % 2 == 0,
            from_session,
            x[i],
            general=0.9 if (i % 2 == 0) == general_high_for_liked else 0.1,
            quality=0.5,
            content_key=f"{place}-{tag}{i}",
            whole_range=whole_range,
        )
        for i in range(n)
    ]


def scored(photos: list[TastePhoto], *, liked_high: bool) -> list[TastePhoto]:
    """Each photo with a score that ranks the liked ones above the others (or below them)."""
    return [
        dataclasses.replace(p, candidate_score=1.0 if p.liked == liked_high else 0.0)
        for p in photos
    ]


def measured(photos: list[TastePhoto]) -> TasteReport:
    return measure_taste(photos, feature="clip@1", quality_weight=0.35, l2=1.0, resamples=20)


def test_what_a_taste_chose_to_show_is_read_apart_from_the_whole_range() -> None:
    photos = []
    for place in PLACES:
        photos += separated(place, 20, general_high_for_liked=True, tag="w")
        photos += separated(
            place, 20, general_high_for_liked=False, whole_range=False, tag="c", seed=1
        )
    report = measured(photos)
    assert report.strength is not None
    assert report.strength.b.value == 1.0  # General ranks the whole range right ...
    assert report.chosen is not None
    assert report.chosen.b.value == 0.0  # ... and the photos a taste chose the wrong way round
    assert (report.chosen.a.places, report.chosen.a.rated) == (3, 60)
    assert report.chosen.difference.value == pytest.approx(report.chosen.a.value - 0.0)


def test_nothing_is_read_apart_when_no_photo_was_chosen_by_a_taste() -> None:
    photos = [p for place in PLACES for p in separated(place, 20, general_high_for_liked=True)]
    assert measured(photos).chosen is None


def test_a_location_with_one_answer_among_the_chosen_photos_does_not_count_among_them() -> None:
    photos = []
    for place in PLACES:
        photos += separated(place, 20, general_high_for_liked=True, tag="w")
    only_likes = [  # at the Mill, every photo a taste chose was liked
        dataclasses.replace(p, liked=True)
        for p in separated("Mill", 6, general_high_for_liked=True, whole_range=False, tag="c")
    ]
    both = separated("Harbor", 6, general_high_for_liked=True, whole_range=False, tag="c")
    report = measured([*photos, *only_likes, *both])
    assert report.chosen is not None
    assert report.chosen.a.places == 1  # Harbor: it has a like and a dislike among them


def test_a_file_of_scores_is_read_beside_general_and_yours_in_each_population() -> None:
    photos = []
    for place in PLACES:
        photos += scored(
            separated(place, 20, general_high_for_liked=False, tag="w"), liked_high=True
        )
        photos += scored(
            separated(place, 20, general_high_for_liked=True, whole_range=False, tag="c", seed=1),
            liked_high=False,
        )
    report = measured(photos)
    assert report.scores is not None
    assert report.scores.scored == 120
    whole, chosen = report.scores.whole, report.scores.chosen
    assert whole.versus_general.a.value == 1.0  # the scores rank the whole range right ...
    assert whole.versus_general.b.value == 0.0  # ... where General has it the wrong way round
    assert chosen.versus_general.a.value == 0.0
    assert chosen.versus_general.b.value == 1.0
    assert whole.versus_general.difference.value == 1.0
    assert chosen.versus_general.difference.value == -1.0
    assert whole.versus_general.a.few  # three locations are too few to trust a range


def test_a_file_of_scores_is_read_location_by_location_too() -> None:
    photos = []
    for i, place in enumerate(PLACES):
        photos += scored(
            separated(place, 20, general_high_for_liked=False, tag="w", seed=i), liked_high=True
        )
    photos += scored(  # only the Harbor has photos a taste chose
        separated("Harbor", 10, general_high_for_liked=True, whole_range=False, tag="c", seed=9),
        liked_high=False,
    )
    report = measured(photos)
    assert report.scores is not None
    rows = {r.location: r for r in report.scores.rows}
    assert sorted(rows) == sorted(PLACES)
    mill, harbor = rows["Mill"], rows["Harbor"]
    assert (mill.whole.liked, mill.whole.disliked) == (10, 10)
    assert (mill.whole.scores, mill.whole.general) == (1.0, 0.0)
    assert (mill.chosen.liked, mill.chosen.disliked) == (0, 0)
    assert math.isnan(mill.chosen.scores)  # no pair there to count
    assert (harbor.chosen.liked, harbor.chosen.disliked) == (5, 5)
    assert (harbor.chosen.scores, harbor.chosen.general) == (0.0, 1.0)


def test_a_photo_without_a_score_teaches_but_is_not_judged() -> None:
    taught = f.unit(np.random.default_rng(3).normal(size=(60, 8)))
    labelled = [  # liked along the first axis: what a taste learned here tells at the Mill
        TastePhoto(
            place,
            bool(x[0] > 0),
            True,
            x,
            general=0.5,
            quality=0.5,
            content_key=f"{place}-{i}",
        )
        for place in ("Mill", "Harbor")
        for i, x in enumerate(taught)
    ]
    for_mill = [dataclasses.replace(p, candidate_score=p.vector[0]) for p in labelled[:60]]
    report = measured([*for_mill, *labelled[60:]])
    (row,) = report.rows  # the Harbor has no score, so it is not judged ...
    assert (row.location, row.rated) == ("Mill", 60)
    assert report.trained_on == 120  # ... but every rating teaches
    assert row.yours_aesthetic > 0.9  # Yours learned at the Harbor, which no score covers


def test_a_photo_thumbed_on_a_result_is_judged_with_a_score_and_counts_as_chosen() -> None:
    photos = []
    for place in PLACES:
        photos += separated(place, 20, general_high_for_liked=True, tag="w")
        photos += separated(
            place, 10, general_high_for_liked=True, from_session=False, tag="r", seed=2
        )
    rows = measured(photos).rows
    assert [r.rated for r in rows] == [20, 20, 20]  # a result's thumb is learned from, not judged
    judged = measured(scored(photos, liked_high=True))
    assert [r.rated for r in judged.rows] == [30, 30, 30]  # with a score, it is judged
    assert judged.scores is not None
    assert (
        judged.scores.chosen.versus_general.a.places,
        judged.scores.chosen.versus_general.a.rated,
    ) == (3, 30)
    assert judged.scores.whole.versus_general.a.rated == 60  # the whole range is the sessions'


def test_a_score_that_tells_nothing_leaves_yours_as_it_is() -> None:
    photos = []
    for place in PLACES:
        photos += separated(place, 20, general_high_for_liked=True, tag="w", seed=4)
        photos += separated(
            place, 20, general_high_for_liked=True, whole_range=False, tag="c", seed=5
        )
    report = measured([dataclasses.replace(p, candidate_score=0.0) for p in photos])
    assert report.scores is not None
    for reading in (report.scores.whole, report.scores.chosen):
        assert reading.versus_general.a.value == 0.5  # no better than a coin
        assert reading.blend_versus_yours.difference.value == 0.0  # and adds nothing to Yours


def test_yours_plus_perfect_scores_ranks_better_than_yours_alone() -> None:
    photos = []
    for place in PLACES:
        photos += scored(separated(place, 20, general_high_for_liked=True, seed=6), liked_high=True)
    report = measured(photos)
    assert report.scores is not None
    # perfect scores added to a Yours that learned noise: the blend ranks far better than Yours
    assert report.scores.whole.blend_versus_yours.difference.value > 0.2


def test_a_taste_is_not_judged_on_scores_when_none_is_given() -> None:
    photos = [p for place in PLACES for p in separated(place, 20, general_high_for_liked=True)]
    assert measured(photos).scores is None


def test_mean_z_is_the_mean_of_each_columns_distance_from_its_mean_in_standard_deviations() -> None:
    assert mean_z([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx([0.0, 0.0, 0.0])
    assert mean_z([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(
        [-1.2247448713915892, 0.0, 1.2247448713915892]
    )
    assert mean_z([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) == pytest.approx(
        [-1.2247448713915892 / 2, 0.0, 1.2247448713915892 / 2]
    )  # a column that says nothing counts as zero, but still takes its half
