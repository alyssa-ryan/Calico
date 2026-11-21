"""How closely photos match a reference view: the one definition, shared by the
``same_view`` task and the calibration of its ``min_similarity``.

Similarity mixes "same subject" (embedding cosine) with "same camera angle"
(gradient-layout cosine, robust to light and season).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from calico.errors import CapabilityError
from calico.selection.strength import finite_or_zero

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from calico.selection.context import FeatureView, Trace

LAYOUT = "layout"


@dataclass(frozen=True)
class ViewMatch:
    """How closely each photo matches the closest reference: as a subject, as a camera
    angle, and the two mixed (``match``, what ``min_similarity`` is compared with)."""

    semantic: NDArray[np.float32]
    layout: NDArray[np.float32]
    match: NDArray[np.float32]
    layout_known: bool


def view_match(
    photos: FeatureView,
    refs: FeatureView,
    layout_weight: float,
    trace: Trace,
    *,
    semantic: str | None = None,
) -> ViewMatch:
    """The one definition of a same-view match, used by the task and by its calibration.

    "Same subject" comes from the main embedding, or from the extractor named by
    ``semantic`` (an image-only one such as DINOv2 tells this exact place from places
    like it, where a caption-trained one sees them as alike).
    """
    if semantic is None:
        mine, theirs = photos.embeddings, refs.embeddings
    elif semantic in photos.features and semantic in refs.features:
        mine, theirs = photos.features[semantic], refs.features[semantic]
    else:
        raise CapabilityError(
            f"No {semantic!r} features for these photos. Add {semantic!r} to the extractors "
            "Calico computes, then index the folders again: calico index FOLDER"
        )
    subject = finite_or_zero(trace, "semantic", best_cosine(mine, theirs))
    if LAYOUT in photos.features and LAYOUT in refs.features:
        raw = best_cosine(photos.features[LAYOUT], refs.features[LAYOUT])
        layout, w, known = finite_or_zero(trace, "layout", raw), layout_weight, True
    else:
        trace.note("No layout features are available, so only subject similarity is used.")
        layout, w, known = np.zeros_like(subject), 0.0, False
    match = ((1 - w) * subject + w * layout).astype(np.float32)
    return ViewMatch(subject, layout, match, known)


def best_cosine(pool: NDArray[np.float32], refs: NDArray[np.float32]) -> NDArray[np.float32]:
    """Closest reference per photo, clipped to [0, 1] (negative similarity means nothing here)."""
    best: NDArray[np.float32] = (pool @ refs.T).max(axis=1)
    return np.clip(best, 0.0, 1.0).astype(np.float32)
