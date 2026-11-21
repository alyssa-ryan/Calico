"""Colour palette as an HSV histogram (compare with histogram intersection).

Layout: 12 hues x 2 saturation bands for coloured pixels, then 4 brightness
bins for pixels too grey or too dark for their hue to mean anything. Hue is
meaningless there, and mixing them in would make a grey wall "reddish".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from PIL import Image

_SIDE = 128
_HUES = 12
_MIN_SAT = 0.2
_HIGH_SAT = 0.6
_MIN_VAL = 0.15
_VAL_BINS = 4
_CHROMA_BINS = _HUES * 2


class PaletteExtractor:
    name = "palette"
    version = "2"
    columns: tuple[str, ...] | None = None
    dim = _CHROMA_BINS + _VAL_BINS

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, image in enumerate(images):
            out[i] = self._one(image)
        return out

    def _one(self, image: Image.Image) -> NDArray[np.float32]:
        w, h = image.size
        scale = _SIDE / max(w, h)
        small = image.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        hsv = np.asarray(small.convert("HSV"), dtype=np.float32).reshape(-1, 3) / 255.0
        hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        chromatic = (sat >= _MIN_SAT) & (val >= _MIN_VAL)
        # Hue is circular: share each pixel between its two nearest bin centres
        # (wrapping 11 -> 0) so red at 359 degrees and at 1 degree look alike.
        position = hue * _HUES - 0.5
        lower = np.floor(position)
        frac = position - lower
        lo_bin = lower.astype(np.intp) % _HUES
        hi_bin = (lo_bin + 1) % _HUES
        sat_bin = (sat >= _HIGH_SAT).astype(np.intp)
        val_bin = np.minimum((val * _VAL_BINS).astype(np.intp), _VAL_BINS - 1)
        grey_index = _CHROMA_BINS + val_bin
        hist = np.zeros(self.dim, dtype=np.float64)
        for bins, weight in ((lo_bin, 1.0 - frac), (hi_bin, frac)):
            index = np.where(chromatic, bins * 2 + sat_bin, grey_index)
            share = np.where(chromatic, weight, 0.5 * weight)
            hist += np.bincount(index, weights=share, minlength=self.dim)
        return (hist / hist.sum()).astype(np.float32)
