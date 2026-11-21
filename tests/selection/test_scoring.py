from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from calico.selection.scoring import (
    Signal,
    bhattacharyya,
    combine,
    evidence_for,
    histogram_intersection,
    pairwise_percentile,
    percentile_rank,
)


def test_percentile_rank_spans_zero_to_one() -> None:
    assert percentile_rank(np.array([3.0, 1.0, 2.0])).tolist() == pytest.approx([1.0, 0.0, 0.5])


def test_percentile_rank_ties_share_their_mean_rank() -> None:
    assert percentile_rank(np.array([1.0, 2.0, 2.0, 3.0])).tolist() == pytest.approx(
        [0.0, 0.5, 0.5, 1.0]
    )


@pytest.mark.parametrize("values", [[5.0], [2.0, 2.0, 2.0]])
def test_percentile_rank_without_spread_is_neutral(values: list[float]) -> None:
    assert percentile_rank(np.array(values)).tolist() == pytest.approx([0.5] * len(values))


def test_percentile_rank_maps_nan_to_zero_and_ranks_the_rest() -> None:
    assert percentile_rank(np.array([np.nan, 1.0, 2.0])).tolist() == pytest.approx([0, 0, 1])


def test_percentile_rank_of_nothing_is_empty() -> None:
    assert percentile_rank(np.array([])).shape == (0,)


@given(arrays(np.float64, st.integers(1, 40), elements=st.floats(-1e6, 1e6)))
def test_percentile_rank_is_bounded_and_order_preserving(values: np.ndarray) -> None:
    ranks = percentile_rank(values)
    assert ranks.min() >= 0
    assert ranks.max() <= 1
    order = np.argsort(values, kind="stable")
    assert np.all(np.diff(ranks[order]) >= -1e-6)


def test_pairwise_percentile_is_symmetric_with_unit_diagonal() -> None:
    sim = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.5], [0.1, 0.5, 1.0]])
    out = pairwise_percentile(sim)
    assert np.allclose(out, out.T)
    assert np.allclose(np.diag(out), 1)
    assert out[0, 1] == pytest.approx(1.0)
    assert out[0, 2] == pytest.approx(0.0)


def test_pairwise_percentile_of_one_item() -> None:
    assert pairwise_percentile(np.ones((1, 1))).tolist() == [[1.0]]


def test_bhattacharyya_is_one_for_identical_and_zero_for_disjoint() -> None:
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = bhattacharyya(p, q)
    assert out[0, 0] == pytest.approx(1)
    assert out[0, 1] == pytest.approx(0)
    assert 0 < out[1, 0] < 1


def test_histogram_intersection() -> None:
    a = np.array([[0.5, 0.5, 0.0]])
    b = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    assert histogram_intersection(a, b).tolist()[0] == pytest.approx([1.0, 0.0])


def test_combine_normalises_weights_and_drops_zero_weight_signals() -> None:
    scores, used = combine(
        [
            Signal("a", np.array([1.0, 0.0], np.float32), 3),
            Signal("b", np.array([0.0, 1.0], np.float32), 1),
            Signal("c", np.array([1.0, 1.0], np.float32), 0),
        ]
    )
    assert scores.tolist() == pytest.approx([0.75, 0.25])
    assert [(s.name, s.weight) for s in used] == [("a", 0.75), ("b", 0.25)]


def test_combine_needs_a_positive_weight() -> None:
    with pytest.raises(ValueError, match="positive weight"):
        combine([Signal("a", np.zeros(2, np.float32), 0)])


def test_evidence_sums_to_the_combined_score() -> None:
    signals = [
        Signal("q", np.array([0.2, 0.9], np.float32), 1, "sharp"),
        Signal("c", np.array([0.6, 0.1], np.float32), 1),
    ]
    scores, used = combine(signals)
    ev = evidence_for(0, used, {"c": "matches 'snow'"})
    assert sum(e.contribution for e in ev) == pytest.approx(float(scores[0]), abs=1e-3)
    assert [e.detail for e in ev] == ["sharp", "matches 'snow'"]
