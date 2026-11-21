"""Variety: three photos that are related, not three frames of one moment.

Story connections are percentiles ("how alike, compared with every other pair
here"), so on their own they favour the most alike photos a place has. A set of
three frames shot a minute apart scores highest and tells nothing. Stories you
rejected outright were such sets (every pair 0.80-0.94 by CLIP), while stories
you kept could hold one close pair if the third photo differed. So variety is
judged on the set's mean similarity, not on any one pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def variety(
    mean_similarity: ArrayLike, *, too_alike: float, near_duplicate: float
) -> NDArray[np.float32]:
    """1 where photos are no more alike than ``too_alike`` (cosine), falling linearly to 0
    at ``near_duplicate`` (photos past it never share a story anyway)."""
    sim = np.asarray(mean_similarity, dtype=np.float64)
    span = near_duplicate - too_alike
    if span <= 0:
        return np.ones(sim.shape, dtype=np.float32)
    return np.clip((near_duplicate - sim) / span, 0.0, 1.0).astype(np.float32)
