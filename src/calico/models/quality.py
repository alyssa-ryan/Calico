"""Technical quality from the ``technical`` extractor's columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from calico.selection.scoring import percentile_rank

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from calico.selection.context import FeatureView


class TechnicalQuality:
    """Focus, exposure and clipping folded into one score in [0, 1].

    Pool-relative: sharpness is a percentile rank within the view being scored,
    because raw focus values are not comparable between subjects. The same
    photo can therefore score differently in different pools. With very small
    pools the sharpness term carries little information.
    """

    name = "quality"
    version = "1"
    description = "Technical quality: focus (pool-relative), exposure and clipping"

    def score(self, view: FeatureView) -> NDArray[np.float32]:
        sharp = percentile_rank(np.log1p(view.column("technical", "sharpness")))
        brightness = view.column("technical", "brightness")
        exposure = 1.0 - np.clip(np.abs(brightness - 0.45) / 0.45, 0.0, 1.0)
        clipped = view.column("technical", "highlight_clip") + view.column(
            "technical", "shadow_clip"
        )
        no_clip = 1.0 - np.clip(4.0 * clipped, 0.0, 1.0)
        score = 0.6 * sharp + 0.25 * exposure + 0.15 * no_clip
        out: NDArray[np.float32] = np.clip(score, 0.0, 1.0).astype(np.float32)
        return out
