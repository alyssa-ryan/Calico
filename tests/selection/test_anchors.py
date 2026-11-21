from __future__ import annotations

import numpy as np
import pytest

from calico.selection.anchors import SCALES, anchor_prior, expected_scale, wide_view


def test_a_photo_sure_of_its_scale_sits_at_that_scale() -> None:
    probs = np.eye(len(SCALES), dtype=np.float32)
    assert expected_scale(SCALES, probs) == pytest.approx([0, 1, 2, 3, 4])


def test_scale_is_the_expected_position_whatever_order_the_labels_come_in() -> None:
    labels = ("macro", "close", "medium", "wide", "aerial")
    halfway = np.array([[0, 0.5, 0, 0.5, 0]], dtype=np.float32)  # half close, half wide
    assert expected_scale(labels, halfway) == pytest.approx([2.0])


def test_a_vocabulary_missing_a_scale_gives_no_scale() -> None:
    probs = np.full((3, 4), 0.25, dtype=np.float32)
    assert expected_scale(("aerial", "wide", "medium", "close"), probs) is None


def test_wide_view_runs_from_aerial_to_macro() -> None:
    assert wide_view(np.array([0, 2, 4], dtype=np.float32)) == pytest.approx([1.0, 0.5, 0.0])


def test_the_prior_is_half_strength_half_person_or_wide_view() -> None:
    strength = np.array([0.8, 0.2, 0.4], dtype=np.float32)
    living = np.array([0.0, 0.9, 0.1], dtype=np.float32)
    scale = np.array([4.0, 4.0, 0.0], dtype=np.float32)  # macro, macro, aerial
    prior = anchor_prior(strength, scale=scale, living=living)
    assert prior.dtype == np.float32
    assert prior == pytest.approx([0.4, 0.55, 0.7])


def test_without_scale_or_subject_the_prior_is_half_strength() -> None:
    strength = np.array([0.6, 0.2], dtype=np.float32)
    assert anchor_prior(strength) == pytest.approx([0.3, 0.1])
