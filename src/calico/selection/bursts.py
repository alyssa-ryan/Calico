"""Burst detection: frames shot seconds apart of the same thing are one choice.

Selecting from raw frames fills a result with near-identical shots. Collapsing
bursts first means every later stage (ranking, diversity) sees one candidate
per moment, and the other frames survive as ``alternates``.

A moment can span folders: an edit filed in "Lightroom Edits" or copied to an
"iPhone X" folder keeps its original's camera and time, so Strongest showed both
(at one site, 6 such pairs in one top 40; on one trip, 8,396 pairs of one camera's
frames seconds apart in different folders). Photos from one camera therefore burst
across folders; without a camera, two folders may be two people, so they do not.
"""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    import numpy as np
    from numpy.typing import NDArray

    from calico.domain import Photo


def group_bursts(
    photos: Sequence[Photo],
    embeddings: NDArray[np.float32],
    *,
    window_seconds: float,
    min_similarity: float,
) -> list[list[int]]:
    """Partition every index into bursts.

    Photos burst together only if dated, and only from one camera (any folder) or,
    with no camera recorded, from one folder. In
    (time, path) order a photo joins the current burst when it is within
    ``window_seconds`` of the previous member *and* similar enough to the burst's
    first frame. Comparing with the first frame, not the previous one, stops a
    slow pan from chaining unrelated views into one burst.
    """
    bursts: list[list[int]] = []
    sources: dict[object, list[int]] = {}
    for i, p in enumerate(photos):
        if p.captured_at is None or window_seconds <= 0:
            bursts.append([i])
        else:
            sources.setdefault(("camera", p.camera) if p.camera else p.path.parent, []).append(i)

    for members in sources.values():
        members.sort(key=lambda i: (_when(photos[i]), photos[i].path))
        current = [members[0]]
        for i in members[1:]:
            gap = (_when(photos[i]) - _when(photos[current[-1]])).total_seconds()
            similar = float(embeddings[current[0]] @ embeddings[i]) >= min_similarity
            if gap <= window_seconds and similar:
                current.append(i)
            else:
                bursts.append(current)
                current = [i]
        bursts.append(current)

    bursts.sort(key=min)
    return bursts


def _when(p: Photo) -> datetime:
    """Capture time as naive UTC, so naive and timezone-aware stamps compare.

    EXIF times are usually naive local time; some sources carry an offset. Within
    a burst window (seconds) the choice of reference cannot matter, so aware
    values are converted to UTC and stripped, and naive ones are taken as-is.
    """
    t = p.captured_at
    if t is None:  # only dated photos reach the folder passes
        raise ValueError(f"{p.path} has no capture time")
    return t if t.tzinfo is None else t.astimezone(UTC).replace(tzinfo=None)


def representatives(
    bursts: list[list[int]], scores: NDArray[np.floating]
) -> list[tuple[int, list[int]]]:
    """Per burst: the best-scored frame (ties: lower index) and the rest, best first."""
    out: list[tuple[int, list[int]]] = []
    for burst in bursts:
        ranked = sorted(burst, key=lambda i: (-float(scores[i]), i))
        out.append((ranked[0], ranked[1:]))
    return out
