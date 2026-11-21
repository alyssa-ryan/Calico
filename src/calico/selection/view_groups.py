"""Which photos show one view, and on how many occasions it was shot.

**Views.** Photos are grouped by average linkage of their patch shares
(``selection.patch_match``): two groups join only if their photos share enough *on
average*. Joining whenever any two photos match would chain look-alikes: at a 480-photo
site it made one group of 1,613 photos and pulled in 69 of 219 photos labelled "not the same
view"; average linkage at 0.17 pulled in 2, and kept 64 of 69 labelled repeats with their
reference.

**Occasions.** Within a view, a photo more than a gap after the previous one starts a new
occasion. Measured within the view, not across everything shot that day, so a morning and
an evening visit to one room count twice even if the day between was spent shooting
elsewhere. Pure: arrays in, indices out.
"""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

CANDIDATE_SIMILARITY = 0.82
"""Whole-photo similarity (the view embedder) from which two photos are compared part by
part: recurring views compares such photos taken hours apart, Strongest such photos among
its strongest. Every labelled same view taken two or more hours from its reference scored
at least 0.839 (96 of them); over the archive this leaves 478k pairs among 34.6k of 109.5k
photos."""

GROUP_SHARE = 0.17
"""How much of their details a view's photos share on average (average linkage). On the
labels, 64 of 69 repeats grouped with their reference and 2 of 219 "not the same view"
photos pulled in (0.16: 66 and 6; 0.12: 69 and 33). Chosen on those labels, then reviewed
by eye: 213 of 229 groups were one view."""


def group_views(
    n: int, pairs: NDArray[np.intp], shares: NDArray[np.floating], *, min_share: float
) -> list[list[int]]:
    """Groups of two or more of ``n`` photos whose ``shares`` (one per row of ``pairs``)
    average at least ``min_share``; photos never compared, or unreadable (NaN), share
    nothing. Members ascending; groups by their first member."""
    if n < 2 or not len(pairs):
        return []
    a, b = np.minimum(pairs[:, 0], pairs[:, 1]), np.maximum(pairs[:, 0], pairs[:, 1])
    if (a == b).any():
        raise ValueError("a photo cannot be paired with itself")
    found = np.nan_to_num(shares.astype(np.float64), nan=0.0)
    # Photos never compared share nothing, so only photos linked by compared pairs can
    # join: each cluster of them is grouped alone, in memory its own size squared, where
    # the whole place at once needed n(n-1)/2 distances (1.6 GB at 20,000 photos).
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _, cluster = connected_components(graph, directed=False)
    groups: list[list[int]] = []
    for label in np.unique(cluster[a]):
        nodes = np.flatnonzero(cluster == label)
        position = {int(node): k for k, node in enumerate(nodes)}
        inside = cluster[a] == label
        groups.extend(
            [int(nodes[k]) for k in members]
            for members in _average_linkage(
                len(nodes),
                np.asarray([position[int(x)] for x in a[inside]], dtype=np.intp),
                np.asarray([position[int(y)] for y in b[inside]], dtype=np.intp),
                found[inside],
                min_share,
            )
        )
    return sorted(groups, key=lambda m: m[0])


def _average_linkage(
    n: int,
    a: NDArray[np.intp],
    b: NDArray[np.intp],
    shares: NDArray[np.float64],
    min_share: float,
) -> list[list[int]]:
    """Groups of two or more of ``n`` photos (``a < b`` pairs) by average linkage."""
    distance = np.ones(n * (n - 1) // 2, dtype=np.float64)
    distance[n * a - a * (a + 1) // 2 + (b - a - 1)] = 1.0 - shares
    labels = fcluster(linkage(distance, method="average"), t=1.0 - min_share, criterion="distance")
    members: dict[int, list[int]] = {}
    for i, label in enumerate(labels.tolist()):
        members.setdefault(int(label), []).append(i)
    return [sorted(m) for m in members.values() if len(m) > 1]


def split_occasions(
    times: Sequence[datetime], gap_seconds: float, *, utc: Sequence[datetime] | None = None
) -> list[list[int]]:
    """Indices into ``times``, oldest first, cut wherever two in a row are more than
    ``gap_seconds`` apart. Aware times are compared in UTC, naive ones as they are.

    With ``utc`` (the same photos on one clock, ``selection.clock``), photos within the
    gap on *either* clock share an occasion: a zone can join a visit that two cameras'
    clocks split (a phone abroad beside a camera on home time), but never split one
    that the clocks agree on (a frame tagged with its zone beside an export without)."""
    if not times:
        return []
    parent = list(range(len(times)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for clock in (times, utc) if utc is not None else (times,):
        order = sorted(range(len(clock)), key=lambda i: (_naive(clock[i]), i))
        for prev, i in pairwise(order):
            if (_naive(clock[i]) - _naive(clock[prev])).total_seconds() <= gap_seconds:
                parent[root(i)] = root(prev)
    members: dict[int, list[int]] = {}
    for i in sorted(range(len(times)), key=lambda i: (_naive(times[i]), i)):
        members.setdefault(root(i), []).append(i)
    return list(members.values())


def seconds(t: datetime) -> float:
    """``t`` as seconds on one clock: aware times in UTC, naive ones as they are."""
    return (_naive(t) - _EPOCH).total_seconds()


_EPOCH = datetime(1970, 1, 1)


def _naive(t: datetime) -> datetime:
    return t if t.tzinfo is None else t.astimezone(UTC).replace(tzinfo=None)
