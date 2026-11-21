from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from calico.selection.variety import variety


def test_photos_no_more_alike_than_too_alike_are_fully_varied() -> None:
    out = variety(np.array([0.2, 0.6, 0.8]), too_alike=0.8, near_duplicate=0.95)
    assert out.tolist() == [1.0, 1.0, 1.0]


def test_variety_falls_to_nothing_at_the_near_duplicate_cut() -> None:
    out = variety(np.array([0.875, 0.95, 0.99]), too_alike=0.8, near_duplicate=0.95)
    assert out.tolist() == pytest.approx([0.5, 0.0, 0.0])


def test_variety_is_float32_and_keeps_the_shape() -> None:
    out = variety(np.full((2, 3), 0.9), too_alike=0.8, near_duplicate=0.95)
    assert out.shape == (2, 3)
    assert out.dtype == np.float32


def test_a_cut_at_or_above_the_near_duplicate_one_never_penalises() -> None:
    out = variety(np.array([0.9, 0.97]), too_alike=0.95, near_duplicate=0.95)
    assert out.tolist() == [1.0, 1.0]


@given(st.floats(-1, 1), st.floats(-1, 1))
def test_variety_is_in_unit_range_and_never_rises_with_similarity(a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    out = variety(np.array([lo, hi]), too_alike=0.8, near_duplicate=0.95)
    assert 0.0 <= out[1] <= out[0] <= 1.0
