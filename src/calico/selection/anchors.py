"""What makes a photo likely to anchor a story: its scale, how wide a view it is, and the
prior that stories told by roles try anchors by.

Both the story task and ``calico eval anchors`` read these, so the measurement always
measures what the task does: a change here changes both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

SCALES = ("aerial", "wide", "medium", "close", "macro")
"""The ``scale`` vocabulary's labels, farthest first: a photo's scale is its expected
position (0 aerial ... 4 macro)."""


def expected_scale(labels: Sequence[str], probs: NDArray[np.float32]) -> NDArray[np.float32] | None:
    """Each photo's expected position on ``SCALES`` from a scale model's probabilities,
    or None when the model's labels do not name every scale."""
    if not set(SCALES) <= set(labels):
        return None
    order = [labels.index(s) for s in SCALES]
    return (probs[:, order] @ np.arange(len(SCALES))).astype(np.float32)


def wide_view(scale: NDArray[np.float32]) -> NDArray[np.float32]:
    """How wide a view each photo is: 1 aerial ... 0 macro."""
    return (1 - scale / (len(SCALES) - 1)).astype(np.float32)


def anchor_prior(
    strength: NDArray[np.float32],
    scale: NDArray[np.float32] | None = None,
    living: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """How likely each photo is to anchor a story: half its strength, half a person or
    animal in it or a wide view, whichever is surer. A part that cannot be measured counts
    as 0. Strength alone reached 2 of your 28 anchors."""
    lift = np.zeros(len(strength), dtype=np.float32)
    if living is not None:
        lift = np.maximum(lift, living)
    if scale is not None:
        lift = np.maximum(lift, wide_view(scale))
    return (0.5 * strength + 0.5 * lift).astype(np.float32)
