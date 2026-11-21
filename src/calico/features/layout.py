"""Viewpoint/composition descriptor, deliberately blind to light and season.

A 64x64 grey image yields gradient orientation (unsigned, 8 bins, linearly
shared between the two nearest bins so a small rotation moves weight smoothly)
weighted by magnitude. These are pooled over a spatial pyramid of 1x1, 2x2, 4x4
and 8x8 cells. Coarse levels survive small re-aims (a few degrees of rotation
or a shift of a few percent); fine levels tell similar-looking scenes apart.
Each cell is normalised on its own, so overall brightness or a colour tint
(which scale every gradient in a cell alike) drops out; each level is then
normalised, weighted and concatenated, and the whole vector has unit length so
dot product is cosine similarity.

A featureless image (no gradients at all) maps to one fixed constant unit
vector, so two blank frames are identical rather than a 0/0 non-answer.

Measured over seeds 0-9 of ``tests.factories.scene``, cosine to the original
(different = all 90 ordered pairs of distinct seeds):

* rotated 5 degrees (expand=False): mean 0.81, min 0.75
* shifted 8% of the width:           mean 0.80, min 0.74
* brightness 0.6 + warm tint:        mean 0.99, min 0.97
* different scene:                   mean 0.51, max 0.62

Re-aims beat the best different-scene match by 0.18 on average and 0.12 in the
worst case. A single 8x8 grid (version 1) managed 0.07 and -0.05; the gain comes
from the coarse pyramid levels, 3 px of blur before the gradient, and weighting
the fine levels higher.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFilter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

_SIDE = 64
_BINS = 8
_LEVELS = (1, 2, 4, 8)
_LEVEL_WEIGHTS = (0.5, 1.0, 2.0, 3.0)
_EPS = 1e-6
_BLUR = 3.0
_DIM = sum(g * g for g in _LEVELS) * _BINS


def _constant_vector() -> NDArray[np.float32]:
    vec = np.ones(_DIM, dtype=np.float32)
    return vec / np.linalg.norm(vec)


_FLAT = _constant_vector()


class LayoutExtractor:
    name = "layout"
    version = "2"
    columns: tuple[str, ...] | None = None
    dim = _DIM

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, image in enumerate(images):
            out[i] = self._one(image)
        return out

    @staticmethod
    def _one(image: Image.Image) -> NDArray[np.float32]:
        gray = image.convert("L").resize((_SIDE, _SIDE), resample=Image.Resampling.BOX)
        if _BLUR > 0:
            gray = gray.filter(ImageFilter.GaussianBlur(_BLUR))
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        gy, gx = np.gradient(arr)
        magnitude = np.hypot(gx, gy).ravel()
        if float(magnitude.sum()) < _EPS:
            return _FLAT
        angle = np.mod(np.arctan2(gy, gx), np.pi).ravel() / np.pi * _BINS
        lo = np.floor(angle).astype(np.intp)
        frac = angle - lo
        lo %= _BINS
        hi = (lo + 1) % _BINS
        coords = np.arange(_SIDE)
        parts: list[NDArray[np.float64]] = []
        for grid, weight in zip(_LEVELS, _LEVEL_WEIGHTS, strict=True):
            rows = coords // (_SIDE // grid)
            cell = (rows[:, None] * grid + rows[None, :]).ravel()
            size = grid * grid * _BINS
            hist = np.bincount(cell * _BINS + lo, weights=magnitude * (1 - frac), minlength=size)
            hist += np.bincount(cell * _BINS + hi, weights=magnitude * frac, minlength=size)
            hist = hist.reshape(grid * grid, _BINS)
            hist /= np.linalg.norm(hist, axis=1, keepdims=True) + _EPS
            level = hist.ravel()
            parts.append(weight * level / (np.linalg.norm(level) + _EPS))
        vec = np.concatenate(parts)
        return (vec / (np.linalg.norm(vec) + _EPS)).astype(np.float32)
