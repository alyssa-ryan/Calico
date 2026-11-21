from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from calico.models.base import ScoreModel
from calico.models.quality import TechnicalQuality
from tests.factories import view

if TYPE_CHECKING:
    from calico.selection.context import FeatureView

COLS = ("sharpness", "brightness", "contrast", "highlight_clip", "shadow_clip")


def make(rows: list[list[float]]) -> FeatureView:
    arr = np.asarray(rows, np.float32)
    return view(np.eye(len(arr)), {"technical": arr}, {"technical": COLS})


def test_sharp_well_exposed_wins() -> None:
    v = make(
        [
            [5.0, 0.45, 0.5, 0.0, 0.0],  # sharp, well exposed
            [0.1, 0.45, 0.5, 0.0, 0.0],  # blurry
            [5.0, 0.95, 0.5, 0.3, 0.0],  # sharp, blown out
        ]
    )
    m = TechnicalQuality()
    assert isinstance(m, ScoreModel)
    s = m.score(v)
    assert s.dtype == np.float32
    assert s.shape == (3,)
    assert ((s >= 0) & (s <= 1)).all()
    assert s[0] > s[1]
    assert s[0] > s[2]
    assert abs(s[0] - 0.85) < 1e-5
