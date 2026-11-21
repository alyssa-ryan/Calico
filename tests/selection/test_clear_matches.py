from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from calico.criteria import CriteriaScores
from calico.domain import Group, Pick, SelectionRequest
from calico.selection.clear_matches import mark_clear_matches, narrow_to_clear_matches
from calico.selection.context import Trace
from calico.selection.picks import make_pick
from tests import factories as f

if TYPE_CHECKING:
    from calico.domain import Photo
    from calico.selection.context import FeatureView


def request(
    text: str = "snow", *, only: bool = False, examples: tuple[Path, ...] = ()
) -> SelectionRequest:
    return SelectionRequest(
        task="best",
        folders=(Path("/archive"),),
        criteria=text,
        only_clear_matches=only,
        examples=examples,
    )


def pool(n: int = 4) -> tuple[list[Photo], FeatureView]:
    return [f.photo(f"{i}.jpg") for i in range(n)], f.view(np.eye(n))


def scored(*match: float) -> CriteriaScores:
    return CriteriaScores(
        scores=np.array(match, dtype=np.float32), match=np.array(match, dtype=np.float32)
    )


def test_only_clear_matches_narrows_the_pool_and_says_so() -> None:
    photos, v = pool()
    trace = Trace()
    kept, view, criteria = narrow_to_clear_matches(
        request(only=True), photos, v, scored(0.9, 0.2, 0.6, 0.4), trace
    )
    assert kept == [photos[0], photos[2]]
    assert len(view) == 2
    assert criteria is not None
    np.testing.assert_allclose(criteria.scores, [0.9, 0.6])
    assert trace.notes == ["Only the 2 of 4 photos that clearly match “snow” were considered."]
    np.testing.assert_allclose(trace.columns["criteria_match"], [0.9, 0.6])


def test_without_the_switch_every_photo_stays() -> None:
    photos, v = pool()
    trace = Trace()
    kept, view, _ = narrow_to_clear_matches(request(), photos, v, scored(0.9, 0.2, 0.6, 0.4), trace)
    assert kept == photos
    assert len(view) == 4
    assert trace.notes == []
    assert len(trace.columns["criteria_match"]) == 4


def test_when_nothing_clearly_matches_the_closest_are_shown_and_it_says_so() -> None:
    photos, v = pool()
    for only in (False, True):
        trace = Trace()
        kept, _, _ = narrow_to_clear_matches(
            request("a bicycle", only=only), photos, v, scored(0.01, 0.2, 0.3, 0.4), trace
        )
        assert kept == photos
        (note,) = trace.notes
        assert note.startswith("Calico isn't confident any photo here clearly matches “a bicycle”")
        assert "closest" in note


def test_with_examples_the_note_says_they_still_shaped_the_order() -> None:
    photos, v = pool()
    trace = Trace()
    examples = (Path("/archive/e.jpg"),)
    narrow_to_clear_matches(
        request("brooding", examples=examples), photos, v, scored(0.01, 0.2, 0.3, 0.4), trace
    )
    (note,) = trace.notes
    assert note.endswith("Your example photos still shaped the order.")


def test_the_switch_without_a_description_is_noted() -> None:
    photos, v = pool()
    trace = Trace()
    kept, _, _ = narrow_to_clear_matches(request("", only=True), photos, v, None, trace)
    assert kept == photos
    assert trace.notes == ["“Only clear matches” needs a description in “Describe it”; ignored."]


def test_the_switch_with_a_model_that_cannot_tell_is_noted() -> None:
    photos, v = pool()
    ranks_only = CriteriaScores(scores=np.zeros(4, dtype=np.float32))
    for criteria in (ranks_only, None):
        trace = Trace()
        kept, _, _ = narrow_to_clear_matches(request(only=True), photos, v, criteria, trace)
        assert kept == photos
        assert trace.notes == [
            (
                "Calico cannot tell which photos clearly match “snow”, "
                "so “Only clear matches” was ignored."
            )
        ]


def test_an_empty_pool_adds_no_note() -> None:
    trace = Trace()
    kept, _, _ = narrow_to_clear_matches(
        request(only=True), [], f.view(np.zeros((0, 2))), scored(), trace
    )
    assert kept == []
    assert trace.notes == []


def test_picks_that_clearly_match_are_marked() -> None:
    photos, _ = pool(3)
    reference = Pick(
        path=Path("/refs/0.jpg"), content_key="ref", location="", captured_at=None, score=1
    )
    groups = (
        Group(title="g", score=1, picks=(make_pick(photos, 2, 1, ()), make_pick(photos, 1, 1, ()))),
        Group(title="h", score=1, picks=(reference,)),
    )
    marked = mark_clear_matches(groups, photos, scored(0.1, 0.4, 0.8))
    assert [p.clear_match for p in marked[0].picks] == [True, False]
    assert marked[1].picks[0].clear_match is None  # not in the pool: not judged


def test_picks_are_unmarked_when_the_model_cannot_tell() -> None:
    photos, _ = pool(1)
    groups = (Group(title="g", score=1, picks=(make_pick(photos, 0, 1, ()),)),)
    assert mark_clear_matches(groups, photos, None) == groups
    ranks_only = CriteriaScores(scores=np.zeros(1, dtype=np.float32))
    assert mark_clear_matches(groups, photos, ranks_only)[0].picks[0].clear_match is None
