from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from calico.selection.diversity import mmr
from tests.factories import unit


def test_zero_diversity_is_plain_ranking() -> None:
    rel = np.array([0.2, 0.9, 0.5, 0.7])
    emb = unit(np.ones((4, 3)))
    assert mmr(rel, emb, 3, diversity=0.0) == [1, 3, 2]


def test_diversity_prefers_a_different_photo_over_a_near_duplicate() -> None:
    emb = unit([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]])
    rel = np.array([1.0, 0.9, 0.8])
    assert mmr(rel, emb, 2, diversity=0.0) == [0, 1]
    assert mmr(rel, emb, 2, diversity=0.7) == [0, 2]


def test_max_similarity_makes_near_duplicates_ineligible() -> None:
    emb = unit([[1.0, 0.0], [0.99, 0.14], [0.98, 0.2]])
    rel = np.array([1.0, 0.9, 0.8])
    assert mmr(rel, emb, 3, diversity=0.0, max_similarity=0.9) == [0]


def test_stops_at_k() -> None:
    assert len(mmr(np.arange(5.0), unit(np.eye(5)), 2, diversity=0.3)) == 2


def test_k_larger_than_pool_returns_everything_once() -> None:
    assert sorted(mmr(np.arange(3.0), unit(np.eye(3)), 10, diversity=0.3)) == [0, 1, 2]


def test_nothing_to_pick() -> None:
    assert mmr(np.array([]), np.zeros((0, 2), np.float32), 3, diversity=0.3) == []
    assert mmr(np.array([1.0]), unit([[1.0, 0.0]]), 0, diversity=0.3) == []


def test_ties_break_by_lower_index() -> None:
    assert mmr(np.array([0.5, 0.5, 0.5]), unit(np.eye(3)), 3, diversity=0.0) == [0, 1, 2]


def test_first_pick_is_most_relevant_whatever_the_diversity() -> None:
    rel = np.array([0.1, 0.8, 0.3])
    assert mmr(rel, unit(np.eye(3)), 1, diversity=1.0) == [1]


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(1, 15),
    k=st.integers(0, 20),
    diversity=st.floats(0, 1),
    seed=st.integers(0, 1000),
    cap=st.one_of(st.none(), st.floats(0.1, 1.0)),
)
def test_mmr_returns_distinct_indices_at_most_k(
    n: int, k: int, diversity: float, seed: int, cap: float | None
) -> None:
    rng = np.random.default_rng(seed)
    emb = unit(rng.normal(size=(n, 4)))
    picked = mmr(rng.random(n), emb, k, diversity=diversity, max_similarity=cap)
    assert len(picked) == len(set(picked)) <= min(k, n)
    assert all(0 <= i < n for i in picked)
    if cap is not None:
        sims = emb[picked] @ emb[picked].T
        np.fill_diagonal(sims, -1)
        assert (sims < cap + 1e-6).all()


@pytest.mark.parametrize("diversity", [0.0, 0.5, 1.0])
def test_mmr_is_deterministic(diversity: float) -> None:
    rng = np.random.default_rng(1)
    emb, rel = unit(rng.normal(size=(8, 3))), rng.random(8)
    assert mmr(rel, emb, 5, diversity=diversity) == mmr(rel, emb, 5, diversity=diversity)


def test_nan_relevance_is_never_selected() -> None:
    rel = np.array([np.nan, 0.2, 0.9, np.nan])
    assert mmr(rel, unit(np.eye(4)), 4, diversity=0.3) == [2, 1]


def test_all_nan_relevance_selects_nothing() -> None:
    assert mmr(np.full(3, np.nan), unit(np.eye(3)), 3, diversity=0.3) == []


def test_nan_embeddings_count_as_dissimilar() -> None:
    emb = np.eye(3, dtype=np.float32)
    emb[1] = np.nan
    assert mmr(np.array([0.9, 0.8, 0.1]), emb, 3, diversity=0.5) == [0, 1, 2]
