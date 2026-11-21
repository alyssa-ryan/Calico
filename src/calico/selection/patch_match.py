"""How much of a photo is the reference, part by part: the share of its patches that are
each other's closest match with a patch of the reference.

A whole-photo embedding averages a place away: two rooms with the same windows score
alike, and a hallway seen from its other end looks the same as seen from this one.
Comparing patches keeps what is where. On the labelled same views this separated same
views from look-alikes at every site (see ``tasks.same_view.SURE_SHARE``); averaged
DINOv2 could not at a symmetric hallway or among identical hospital windows.

"Mutual" is what makes it a share of *the reference*: a patch of sky matches some patch
of sky anywhere, but only one patch of the photo can be the reference patch's best match
in return. Pure: arrays in, one number per photo out.

The share still ignores *where* the matches are, so a room with the same windows in other
places can share as much as the view itself (audit A10-7). ``consistent_shares`` keeps only
the matches that agree on one placement of the reference in the photo. Its constants are
chosen, not yet measured: ``calico eval same-view --geometry`` measures the whole measure
against the share before any of them is tuned.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

CONSISTENT_TOLERANCE = 1.5
"""How far, in grid cells, a match may land from where the placement puts it. A DINOv2
cell is 14 px of a 448 px square, so 1.5 cells forgives a little perspective (the camera a
step to one side) without letting scattered matches line up by chance."""

MAX_SCALE = 3.0
"""The most one frame may be closer than the other: a step in, or a longer lens."""

MAX_TURN = math.radians(30)
"""The most one frame may be turned against the other: a tilted camera, not a photo on its
side (frames are upright once decoded)."""

PLACEMENTS = 256
"""Placements tried per pair, each from two matches. With every match in place the first
try finds it; with a third of its matches in place, 256 tries miss it about once in 10^13."""

type PatchMeasure = Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.float32]]
"""Photos' patches and the references' in, one number per photo out (``patch_shares``,
``consistent_shares``)."""

_SEED = 20261102
"""Placements are drawn with a fixed seed, so a pair always scores the same (rule 7)."""


def patch_shares(photos: NDArray[np.floating], refs: NDArray[np.floating]) -> NDArray[np.float32]:
    """For each photo (``photos[i]``: one unit row per patch), the share of its patches
    that are mutual nearest neighbours of the closest reference's patches, in [0, 1]."""
    references = [r.astype(np.float32) for r in refs]
    out = np.zeros(len(photos), dtype=np.float32)
    for i, photo in enumerate(photos):
        mine = photo.astype(np.float32)
        out[i] = max(_mutual_share(mine, ref) for ref in references)
    return out


def pair_share(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """How much two photos (one unit row per patch, same grid) share, part by part: the
    share of their patches that are each other's closest match. The same either way round
    when both have as many patches, as photos from one extractor do."""
    return _mutual_share(a.astype(np.float32), b.astype(np.float32))


def consistent_shares(
    photos: NDArray[np.floating], refs: NDArray[np.floating]
) -> NDArray[np.float32]:
    """For each photo, the share of its patches that are mutual nearest neighbours of the
    most consistent reference's patches *and* sit where one placement of that reference
    puts them, in [0, 1]. At most the patch share.

    The placement is a shift, a zoom and a small turn on the patch grid (a similarity),
    fitted to the matches by RANSAC. A similarity rather than a perspective warp: two matches
    fix it, so matches scattered at random can rarely agree on one by chance, which is what
    a look-alike room's matches are. Grids must be square (row by row), as DINOv2's are."""
    references = [r.astype(np.float32) for r in refs]
    out = np.zeros(len(photos), dtype=np.float32)
    for i, photo in enumerate(photos):
        mine = photo.astype(np.float32)
        out[i] = max(_consistent_share(mine, ref) for ref in references)
    return out


def _mutual(a: NDArray[np.float32], b: NDArray[np.float32]) -> tuple[NDArray[np.intp], ...]:
    """The patches of ``a`` that are mutual best matches, and their matches in ``b``."""
    similarity = a @ b.T
    a_best, b_best = similarity.argmax(axis=1), similarity.argmax(axis=0)
    mine = np.flatnonzero(b_best[a_best] == np.arange(len(a)))
    return mine, a_best[mine]


def _mutual_share(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    mine, _ = _mutual(a, b)
    return len(mine) / len(a)


def _consistent_share(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    mine, theirs = _mutual(a, b)
    return _most_in_place(_places(mine, len(a)), _places(theirs, len(b))) / len(a)


def _places(index: NDArray[np.intp], patches: int) -> NDArray[np.complex128]:
    """Grid positions as ``column + row·i``: a similarity is then ``z ↦ c·z + t``."""
    side = math.isqrt(patches)
    if side * side != patches:
        raise ValueError(f"{patches} patches are not a square grid")
    return (index % side) + 1j * (index // side)


def _most_in_place(za: NDArray[np.complex128], zb: NDArray[np.complex128]) -> int:
    """How many matches ``za[k] → zb[k]`` the best plausible placement puts in place."""
    m = len(za)
    if m < 2:
        return m
    rng = np.random.default_rng(_SEED)
    p = rng.integers(0, m, PLACEMENTS)
    q = (p + rng.integers(1, m, PLACEMENTS)) % m  # never p: distinct matches, distinct cells
    c = (zb[q] - zb[p]) / (za[q] - za[p])
    t = zb[p] - c * za[p]
    plausible = _plausible(c)
    if not plausible.any():
        return 1
    c, t = c[plausible], t[plausible]
    in_place = np.abs(c[:, None] * za[None, :] + t[:, None] - zb[None, :]) <= CONSISTENT_TOLERANCE
    counts = in_place.sum(axis=1)
    best = int(counts.argmax())
    return max(int(counts[best]), _refitted(za, zb, in_place[best]))


def _refitted(
    za: NDArray[np.complex128], zb: NDArray[np.complex128], chosen: NDArray[np.bool_]
) -> int:
    """How many matches are in place for the least-squares placement of ``chosen``: two
    matches fix a placement only roughly, all of its matches together more closely."""
    if chosen.sum() < 2:
        return 0
    ma, mb = za[chosen].mean(), zb[chosen].mean()
    da, db = za[chosen] - ma, zb[chosen] - mb
    c = np.sum(np.conj(da) * db) / np.sum(np.abs(da) ** 2)
    if not _plausible(np.asarray([c]))[0]:
        return 0
    return int((np.abs(c * (za - ma) + mb - zb) <= CONSISTENT_TOLERANCE).sum())


def _plausible(c: NDArray[np.complex128]) -> NDArray[np.bool_]:
    scale = np.abs(c)
    return (scale >= 1 / MAX_SCALE) & (scale <= MAX_SCALE) & (np.abs(np.angle(c)) <= MAX_TURN)
