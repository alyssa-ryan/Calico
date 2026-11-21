from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from calico.selection.strength import photo_strength
from tests.factories import FakeModels, FeatureLabels, FeatureScore, context, photo, view

if TYPE_CHECKING:
    from calico.models.base import Model
    from calico.selection.context import TaskContext


def make(
    quality: list[float],
    strong: list[float] | None = None,
    *,
    criteria: list[float] | None = None,
    with_quality: bool = True,
) -> TaskContext:
    n = len(quality)
    feats = {"q": np.array(quality), "a": np.stack([strong or [0.5] * n, [0.5] * n], axis=1)}
    models = FakeModels()
    found: dict[str, Model] = {}
    if with_quality:
        found["quality"] = FeatureScore("quality", "q")
    if strong is not None:
        found["aesthetic"] = FeatureLabels("aesthetic", "a", ("strong", "weak"))
    models = FakeModels(found)
    photos = [photo(f"{i}.jpg", location=f"L{i}") for i in range(n)]
    return context(photos, view(np.eye(n) + 0.01, feats), models=models, criteria=criteria)


def test_quality_and_aesthetic_are_mixed_by_weight() -> None:
    ctx = make([1.0, 0.0], [0.0, 1.0])
    scores, signals = photo_strength(
        ctx, [0, 1], quality_weight=0.35, aesthetic_weight=0.65, criteria_share=0
    )
    assert {s.name: s.weight for s in signals} == pytest.approx(
        {"quality": 0.35, "aesthetic": 0.65}
    )
    assert scores.tolist() == pytest.approx([0.35, 0.65])


def test_criteria_share_is_their_share_of_the_score() -> None:
    ctx = make([0.5, 0.5], [0.5, 0.5], criteria=[1.0, 0.0])
    _, signals = photo_strength(
        ctx, [0, 1], quality_weight=0.35, aesthetic_weight=0.65, criteria_share=0.4
    )
    weights = {s.name: s.weight for s in signals}
    assert weights == pytest.approx(
        {"quality": 0.6 * 0.35, "aesthetic": 0.6 * 0.65, "criteria": 0.4}
    )


def test_a_full_share_ranks_by_the_criteria_alone() -> None:
    ctx = make([1.0, 0.0], [1.0, 0.0], criteria=[0.0, 1.0])
    scores, signals = photo_strength(
        ctx, [0, 1], quality_weight=0.35, aesthetic_weight=0.65, criteria_share=1.0
    )
    assert [s.name for s in signals] == ["criteria"]
    assert scores[1] > scores[0]


def test_without_criteria_the_share_goes_to_the_rest() -> None:
    ctx = make([1.0, 0.0], [0.0, 1.0])
    scores, _ = photo_strength(
        ctx, [0, 1], quality_weight=0.35, aesthetic_weight=0.65, criteria_share=0.8
    )
    assert scores.tolist() == pytest.approx([0.35, 0.65])


def test_missing_model_is_noted_and_weights_renormalise() -> None:
    ctx = make([0.2, 0.9])
    _, signals = photo_strength(
        ctx, [0, 1], quality_weight=0.35, aesthetic_weight=0.65, criteria_share=0
    )
    assert [s.name for s in signals] == ["quality"]
    assert signals[0].weight == pytest.approx(1.0)
    assert any("aesthetic" in n for n in ctx.trace.notes)


def test_no_signal_gives_neutral_scores_and_a_note() -> None:
    ctx = make([0.2, 0.9], with_quality=False)
    scores, signals = photo_strength(
        ctx, [0, 1], quality_weight=0.35, aesthetic_weight=0.65, criteria_share=0
    )
    assert signals == []
    assert scores.tolist() == [0.5, 0.5]
    assert ctx.trace.notes


def test_non_finite_model_output_counts_as_zero_with_a_note() -> None:
    ctx = make([np.nan, 0.8, np.inf])
    scores, _ = photo_strength(
        ctx, [0, 1, 2], quality_weight=1, aesthetic_weight=0, criteria_share=0
    )
    assert scores.tolist() == pytest.approx([0.0, 0.8, 0.0])
    assert any("quality returned 2 invalid scores; treated as 0" in n for n in ctx.trace.notes)


def test_subset_of_indices_is_scored_in_that_order() -> None:
    ctx = make([0.1, 0.9, 0.5])
    scores, _ = photo_strength(ctx, [2, 0], quality_weight=1, aesthetic_weight=0, criteria_share=0)
    assert scores.tolist() == pytest.approx([0.5, 0.1])


# -- survivors of the mutation run, 10 November 2026 --------------------


def test_criteria_take_their_share_and_the_rest_split_what_is_left_whatever_their_weights() -> None:
    ctx = make([0.2, 0.9], [0.4, 0.6], criteria=[0.3, 0.7])
    _, signals = photo_strength(
        ctx, [0, 1], quality_weight=0.35, aesthetic_weight=0.15, criteria_share=0.5
    )
    assert {s.name: s.weight for s in signals} == pytest.approx(
        {"quality": 0.35, "aesthetic": 0.15, "criteria": 0.5}
    )
