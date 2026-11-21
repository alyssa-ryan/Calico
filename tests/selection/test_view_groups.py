from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from calico.selection.view_groups import group_views, split_occasions

T0 = datetime(2024, 5, 1, 9, 0, 0)
HOUR = 3600.0


def pairs(*rows: tuple[int, int, float]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([[a, b] for a, b, _ in rows], dtype=np.intp).reshape(-1, 2),
        np.asarray([s for _, _, s in rows], dtype=np.float32),
    )


def test_photos_that_share_enough_on_average_are_one_view() -> None:
    index, shares = pairs((0, 1, 0.4), (0, 2, 0.3), (1, 2, 0.35), (3, 4, 0.5))
    assert group_views(5, index, shares, min_share=0.17) == [[0, 1, 2], [3, 4]]


def test_a_look_alike_bridge_does_not_chain_two_views_together() -> None:
    # 0-1 one view, 2-3 another; 1 and 2 look alike, but nothing else does.
    index, shares = pairs((0, 1, 0.6), (2, 3, 0.6), (1, 2, 0.3))
    assert group_views(4, index, shares, min_share=0.17) == [[0, 1], [2, 3]]


def test_pairs_never_compared_or_unreadable_count_as_sharing_nothing() -> None:
    index, shares = pairs((0, 1, float("nan")), (1, 2, 0.5))
    assert group_views(3, index, shares, min_share=0.17) == [[1, 2]]


def test_photos_alone_are_not_groups() -> None:
    index, shares = pairs()
    assert group_views(3, index, shares, min_share=0.17) == []


def test_a_new_occasion_starts_after_a_long_gap_between_photos_of_the_view() -> None:
    times = [T0, T0 + timedelta(minutes=50), T0 + timedelta(hours=9), T0 + timedelta(days=30)]
    assert split_occasions([times[3], times[0], times[2], times[1]], 2 * HOUR) == [
        [1, 3],
        [2],
        [0],
    ]


def test_a_morning_and_an_evening_visit_are_two_occasions() -> None:
    morning, evening = T0, T0 + timedelta(hours=8)
    assert split_occasions([morning, evening], 2 * HOUR) == [[0], [1]]


def test_a_gap_of_exactly_the_limit_stays_one_occasion() -> None:
    assert split_occasions([T0, T0 + timedelta(hours=2)], 2 * HOUR) == [[0, 1]]


def test_photos_close_on_either_clock_are_one_occasion() -> None:
    # Can Bros Church: five hours apart by the cameras' clocks, together in UTC.
    local = [T0, T0 + timedelta(hours=5)]
    utc = [T0 + timedelta(hours=4), T0 + timedelta(hours=4, minutes=15)]
    assert split_occasions(local, 2 * HOUR, utc=utc) == [[0, 1]]
    # A frame and its export (no time zone), together by the clock, apart in UTC.
    assert split_occasions([T0, T0], 2 * HOUR, utc=[T0 + timedelta(hours=4), T0]) == [[0, 1]]


def test_photos_apart_on_both_clocks_are_separate_occasions_ordered_by_the_camera_clock() -> None:
    local = [T0 + timedelta(hours=9), T0]
    utc = [T0 + timedelta(hours=13), T0 + timedelta(hours=4)]
    assert split_occasions(local, 2 * HOUR, utc=utc) == [[1], [0]]


def test_joining_on_either_clock_chains_through_a_photo_close_to_both() -> None:
    local = [T0, T0 + timedelta(hours=3), T0 + timedelta(hours=6)]
    utc = [T0, T0 + timedelta(hours=1), T0 + timedelta(hours=6)]
    assert split_occasions(local, 2 * HOUR, utc=utc) == [[0, 1], [2]]


def _whole_matrix(
    n: int, a: np.ndarray, b: np.ndarray, share: np.ndarray, t: float
) -> list[list[int]]:
    """The reference: average linkage over every pair at once (what group_views did)."""
    from scipy.cluster.hierarchy import fcluster, linkage  # noqa: PLC0415

    distance = np.ones(n * (n - 1) // 2)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    distance[n * lo - lo * (lo + 1) // 2 + (hi - lo - 1)] = 1 - share
    labels = fcluster(linkage(distance, method="average"), t=1 - t, criterion="distance")
    found: dict[int, list[int]] = {}
    for i, label in enumerate(labels.tolist()):
        found.setdefault(label, []).append(i)
    return sorted((m for m in found.values() if len(m) > 1), key=lambda m: m[0])


def test_views_are_the_same_when_each_cluster_of_compared_photos_is_grouped_alone() -> None:
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(2, 40))
        count = int(rng.integers(0, n * 2))
        a, b = rng.integers(0, n, count), rng.integers(0, n, count)
        keep = a != b
        a, b = a[keep], b[keep]
        _, first = np.unique(
            np.stack([np.minimum(a, b), np.maximum(a, b)], 1), axis=0, return_index=True
        )
        a, b = a[first], b[first]
        share = rng.random(len(a))
        got = group_views(n, np.stack([a, b], 1).astype(np.intp), share, min_share=0.3)
        assert got == _whole_matrix(n, a, b, share, 0.3)


def test_a_place_of_twenty_thousand_photos_groups_in_little_memory() -> None:
    """The whole-matrix form needed 1.6 GB here; photos only meet within their cluster."""
    import tracemalloc  # noqa: PLC0415

    n = 20_000
    a = np.arange(0, n - 1, 2, dtype=np.intp)
    pairs = np.stack([a, a + 1], axis=1)
    tracemalloc.start()
    groups = group_views(n, pairs, np.full(len(pairs), 0.5), min_share=0.17)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert len(groups) == n // 2
    assert peak < 100_000_000
