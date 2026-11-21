from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest

from calico.selection.bursts import group_bursts, representatives
from tests.factories import photo, unit

if TYPE_CHECKING:
    from calico.domain import Photo

T0 = datetime(2024, 5, 1, 12, 0, 0)


def shots(offsets: list[float], *, location: str = "Mill") -> list[Photo]:
    return [
        photo(f"{i}.jpg", location=location, captured_at=T0 + timedelta(seconds=s))
        for i, s in enumerate(offsets)
    ]


def same(n: int) -> np.ndarray:
    return unit(np.tile([1.0, 0.0], (n, 1)))


def test_frames_within_the_window_form_one_burst() -> None:
    photos = shots([0, 10, 20])
    assert group_bursts(photos, same(3), window_seconds=30, min_similarity=0.9) == [[0, 1, 2]]


def test_burst_splits_on_time_gap() -> None:
    photos = shots([0, 10, 200, 205])
    result = group_bursts(photos, same(4), window_seconds=30, min_similarity=0.9)
    assert result == [[0, 1], [2, 3]]


def test_without_a_camera_a_burst_splits_on_folder() -> None:
    photos = [*shots([0], location="A"), *shots([1], location="B")]
    result = group_bursts(photos, same(2), window_seconds=30, min_similarity=0.9)
    assert result == [[0], [1]]


def test_one_cameras_frames_are_one_burst_across_folders() -> None:
    """An edit filed in "Lightroom Edits" or copied to "iPhone X" beside its original in
    "Canon RP": the same camera, seconds apart, so the same moment."""
    canon = "Canon EOS RP"
    photos = [
        photo("IMG_6245.jpg", location="Canon RP", captured_at=T0, camera=canon),
        photo(
            "NSFL5500.JPEG",
            location="iPhone X",
            captured_at=T0 + timedelta(seconds=4),
            camera=canon,
        ),
        photo(
            "IMG_9000.jpg", location="Phone", captured_at=T0 + timedelta(seconds=2), camera="iPhone"
        ),
    ]
    result = group_bursts(photos, same(3), window_seconds=30, min_similarity=0.9)
    assert result == [[0, 1], [2]]


def test_burst_splits_on_dissimilarity() -> None:
    photos = shots([0, 5])
    emb = unit([[1.0, 0.0], [0.0, 1.0]])
    assert group_bursts(photos, emb, window_seconds=30, min_similarity=0.9) == [[0], [1]]


def test_leader_rule_prevents_chaining_drift() -> None:
    # each neighbour is similar to the next, but the last is far from the first
    angles = np.radians([0, 20, 40, 60])
    emb = unit(np.stack([np.cos(angles), np.sin(angles)], axis=1))
    photos = shots([0, 5, 10, 15])
    result = group_bursts(photos, emb, window_seconds=30, min_similarity=0.9)
    assert result == [[0, 1], [2, 3]]


def test_undated_photos_are_singletons() -> None:
    photos = [photo("a.jpg"), photo("b.jpg"), *shots([0, 1])]
    result = group_bursts(photos, same(4), window_seconds=30, min_similarity=0.9)
    assert result == [[0], [1], [2, 3]]


def test_zero_window_makes_every_photo_a_singleton() -> None:
    photos = shots([0, 0, 0])
    result = group_bursts(photos, same(3), window_seconds=0, min_similarity=0.0)
    assert result == [[0], [1], [2]]


def test_members_are_in_time_order_and_bursts_ordered_by_smallest_index() -> None:
    photos = shots([20, 0, 10, 500])
    result = group_bursts(photos, same(4), window_seconds=30, min_similarity=0.9)
    assert result == [[1, 2, 0], [3]]


def test_every_index_appears_exactly_once() -> None:
    photos = [*shots([0, 1, 100]), photo("u.jpg")]
    result = group_bursts(photos, same(4), window_seconds=30, min_similarity=0.9)
    assert sorted(i for b in result for i in b) == [0, 1, 2, 3]


def test_no_photos_no_bursts() -> None:
    assert (
        group_bursts([], np.zeros((0, 2), np.float32), window_seconds=5, min_similarity=0.5) == []
    )


def test_representative_is_best_scored_with_others_best_first() -> None:
    reps = representatives([[0, 1, 2], [3]], np.array([0.2, 0.9, 0.5, 0.1]))
    assert reps == [(1, [2, 0]), (3, [])]


def test_representative_tie_goes_to_lower_index() -> None:
    assert representatives([[2, 0, 1]], np.array([0.5, 0.5, 0.5])) == [(0, [1, 2])]


@pytest.mark.parametrize("window", [0, 10])
def test_bursts_are_deterministic(window: int) -> None:
    photos = shots([0, 3, 50, 52])
    a = group_bursts(photos, same(4), window_seconds=window, min_similarity=0.5)
    b = group_bursts(photos, same(4), window_seconds=window, min_similarity=0.5)
    assert a == b


def test_mixed_naive_and_aware_timestamps_do_not_crash() -> None:
    photos = [
        photo("a.jpg", location="Mill", captured_at=datetime(2024, 5, 1, 12, 0, 0)),
        photo("b.jpg", location="Mill", captured_at=datetime(2024, 5, 1, 12, 0, 5, tzinfo=UTC)),
    ]
    assert group_bursts(photos, same(2), window_seconds=30, min_similarity=0.9) == [[0, 1]]


# -- survivors of the mutation run, 10 November 2026 --------------------


def test_a_gap_of_exactly_the_window_still_joins_the_burst() -> None:
    joined = group_bursts(shots([0, 120]), same(2), window_seconds=120, min_similarity=0.9)
    apart = group_bursts(shots([0, 121]), same(2), window_seconds=120, min_similarity=0.9)
    assert (joined, apart) == ([[0, 1]], [[0], [1]])


def test_a_burst_is_timed_from_the_previous_frame() -> None:
    chain = shots([0, 20, 40])  # 20 s between neighbours, 40 s from the first
    assert group_bursts(chain, same(3), window_seconds=30, min_similarity=0.9) == [[0, 1, 2]]


def test_frames_exactly_as_alike_as_asked_still_join_the_burst() -> None:
    emb = unit([[1.0, 0.0], [0.8, 0.6]])
    as_alike = float(emb[0] @ emb[1])
    assert group_bursts(shots([0, 1]), emb, window_seconds=30, min_similarity=as_alike) == [[0, 1]]


def test_frames_taken_at_the_same_second_are_ordered_by_path() -> None:
    photos = [
        photo("b.jpg", location="Mill", captured_at=T0),
        photo("a.jpg", location="Mill", captured_at=T0),
    ]
    assert group_bursts(photos, same(2), window_seconds=30, min_similarity=0.9) == [[1, 0]]
