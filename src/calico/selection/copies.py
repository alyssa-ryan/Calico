"""One frame saved twice: a re-export, a HEIC also kept as JPEG, or an edit of it.

Two files, so two content keys, but the same pixels. Counted twice, one view would count
as two, and a re-export carries the date it was exported, so it can look like a second
visit. Frames of a burst are not copies: each is a shot.

An edit (cropped, recoloured) is no longer the same pixels, so ``copy_of`` misses it.
Only an edit exported without its camera data needs catching: one that keeps them keeps
its capture date too, so it already falls on its original's occasion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

COPY_SIMILARITY = 0.9999
"""Two photos this alike (cosine, view embedder) are one frame saved twice. Measured on
the 19 labelled same-view sets (DINOv2): copies at 0.99998 or more, the closest frames of
a burst at 0.99973 or less."""

EDIT_SHARE = 0.50
"""A photo with no camera recorded that shares at least this much of its details with
another (patch share, ``selection.patch_match``) is an edit of it. Judged by eye on
same-place pairs taken more than 2 hours apart, one of them without a camera: at 0.50 or
more, 19 of 20 in a fresh sample were one frame edited (the miss: two frames of one
facade, 9 hours apart); between 0.40 and 0.50 about 6 in 20 were still edits (heavy
crops), but as many were the view shot again, so a lower line trades one mistake for the
other. All 36 such pairs with a camera on both (October 2026, EXPLORE): 29 were the view
shot again (14 of them probably one visit, two cameras' clocks about 5 hours apart), 6
were edits that kept the camera but not the date, which this does not catch, and 1 was
two black frames."""

_BLOCK = 1024


def copy_of(vectors: NDArray[np.floating]) -> NDArray[np.intp]:
    """For each row, the first row (in the given order) showing the same frame: itself
    when it is the first."""
    n = len(vectors)
    out = np.arange(n, dtype=np.intp)
    if n == 0:
        return out
    unit = _unit(vectors)
    later: dict[int, list[int]] = {}
    for start in range(0, n, _BLOCK):
        similar = unit[start : start + _BLOCK] @ unit.T >= COPY_SIMILARITY
        for i, j in zip(*np.nonzero(similar), strict=True):
            if j > start + i:
                later.setdefault(start + int(i), []).append(int(j))
    for i in sorted(later):
        if out[i] == i:
            for j in later[i]:
                if out[j] == j:
                    out[j] = i
    return out


def copies_of(vectors: NDArray[np.floating], frames: NDArray[np.floating]) -> NDArray[np.bool_]:
    """Which rows of ``vectors`` show one of ``frames`` (``COPY_SIMILARITY``)."""
    if not len(vectors) or not len(frames):
        return np.zeros(len(vectors), dtype=np.bool_)
    similar = _unit(vectors) @ _unit(frames).T >= COPY_SIMILARITY
    return np.asarray(similar.any(axis=1), dtype=np.bool_)


def edited_from(
    has_camera: Sequence[bool],
    pairs: NDArray[np.intp],
    shares: NDArray[np.floating],
    *,
    min_share: float = EDIT_SHARE,
) -> NDArray[np.intp]:
    """For each photo, the photo whose capture date it should carry: itself, or, for a
    photo with no camera sharing at least ``min_share`` with another, the photo with a
    camera it shares most with, reached through other edits if need be. Edits only of
    each other carry the first's date. Unreadable (NaN) shares never match."""
    links = _links(pairs, shares, min_share)
    edits = [i for i in range(len(has_camera)) if not has_camera[i] and i in links]
    source = {i: j for i in edits if (j := _best(links[i], has_camera.__getitem__)) is not None}
    changed = True
    while changed:
        changed = False
        for i in edits:
            if i not in source and (j := _best(links[i], source.__contains__)) is not None:
                source[i], changed = source[j], True
    for i in edits:
        if i not in source:
            component = _component(links, i)
            source |= dict.fromkeys(component, min(component))
    out = np.arange(len(has_camera), dtype=np.intp)
    for i, j in source.items():
        out[i] = j
    return out


def _unit(vectors: NDArray[np.floating]) -> NDArray[np.floating]:
    norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return np.asarray(vectors / norms)


def _links(
    pairs: NDArray[np.intp], shares: NDArray[np.floating], min_share: float
) -> dict[int, list[tuple[float, int]]]:
    """Each photo's partners sharing at least ``min_share`` with it, with the share."""
    links: dict[int, list[tuple[float, int]]] = {}
    for (a, b), share in zip(pairs.tolist(), np.nan_to_num(shares, nan=-1.0).tolist(), strict=True):
        if share >= min_share:
            links.setdefault(a, []).append((share, b))
            links.setdefault(b, []).append((share, a))
    return links


def _best(partners: list[tuple[float, int]], allowed: Callable[[int], bool]) -> int | None:
    """The allowed partner sharing most (ties: the first)."""
    found = [(-share, j) for share, j in partners if allowed(j)]
    return min(found)[1] if found else None


def _component(links: dict[int, list[tuple[float, int]]], start: int) -> set[int]:
    component, todo = {start}, [start]
    while todo:
        for _, j in links[todo.pop()]:
            if j not in component:
                component.add(j)
                todo.append(j)
    return component
