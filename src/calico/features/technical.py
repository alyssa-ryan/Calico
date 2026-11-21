"""Technical image statistics: focus, exposure, clipping and colour cast.

These are cheap, interpretable columns (not an embedding) that quality models
and tasks read by name. Everything is measured on a copy whose long side is
fixed at 512 px, otherwise sharpness would depend on the decode size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

_SIDE = 512
_CLIP_HIGH = 0.98
_CLIP_LOW = 0.02


def _resized(image: Image.Image, side: int) -> NDArray[np.float32]:
    """RGB float array in [0, 1], long side at most ``side``.

    Never upscales (that would invent smoothness and depress sharpness) and
    always uses LANCZOS, so every input size is reduced the same way. Any mode
    (L, RGBA, P, ...) is converted to RGB first; the input is not mutated.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, side / max(w, h))
    if scale < 1.0:
        rgb = rgb.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))),
            resample=Image.Resampling.LANCZOS,
        )
    return np.asarray(rgb, dtype=np.float32) / 255.0


NOISE_MARGIN = 2.0
"""How many times the noise estimate is taken off the Laplacian variance (version 3). Once
gives too little: in a dark frame the noise is clipped at black and the estimate falls
short, and 11 of 120 synthetic dark noisy blurs still outscored their sharp frame. Twice
put every one of them below it, and below the same frame darkened with the same noise,
while leaving clean sharp frames as they were (their flat areas have no noise to take
off). Synthetic; whether it helps on real photos is for ``calico eval taste --technical
3`` to show."""


def _laplacian(gray: NDArray[np.float32]) -> NDArray[np.float32]:
    padded = np.pad(gray, 1, mode="edge")
    out: NDArray[np.float32] = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * gray
    )
    return out


def _noise_variance(lap: NDArray[np.float32]) -> float:
    """The Laplacian's variance from noise alone: its robust spread (the MAD, scaled to a
    standard deviation), which the few strong edges of a photo barely move, squared."""
    mad = float(np.median(np.abs(lap - np.median(lap))))
    return (1.4826 * mad) ** 2


class TechnicalExtractor:
    """Eight named technical measurements per image.

    ``colorfulness`` is the Hasler-Suesstrunk metric M divided by 100 (typically
    0-1.5, unbounded above). ``sharpness`` is unitless and only comparable
    between images of similar content.
    """

    name = "technical"
    columns: tuple[str, ...] | None = (
        "sharpness",
        "brightness",
        "contrast",
        "highlight_clip",
        "shadow_clip",
        "colorfulness",
        "warmth",
        "saturation",
    )
    dim = 8

    def __init__(self, *, noise_aware: bool = False) -> None:
        """``noise_aware`` (version 3, opt-in): take the noise off sharpness first. In a
        dark or flat frame noise dominates the Laplacian, so a dark, blurred, high-ISO
        frame outscored a sharp one (audit R-05)."""
        self.noise_aware = noise_aware
        self.version = "3" if noise_aware else "2"

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, image in enumerate(images):
            out[i] = self._one(_resized(image, _SIDE))
        return out

    def _one(self, rgb: NDArray[np.float32]) -> NDArray[np.float32]:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        lap = _laplacian(gray)
        detail = float(lap.var())
        if self.noise_aware:
            detail = max(detail - NOISE_MARGIN * _noise_variance(lap), 0.0)
        # Dividing by the grey variance makes focus independent of contrast, so a
        # low-contrast sharp photo is not mistaken for a blurry one.
        sharpness = detail / (float(gray.var()) + 1e-6)
        lo, hi = np.percentile(gray, [2, 98])
        # Hasler & Suesstrunk (2003) colourfulness on 0-255 channels.
        rg = (r - g) * 255.0
        yb = (0.5 * (r + g) - b) * 255.0
        colorfulness = (np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())) / 100.0
        mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
        saturation = np.where(mx > 0, (mx - mn) / np.where(mx > 0, mx, 1.0), 0.0)
        return np.array(
            [
                sharpness,
                gray.mean(),
                hi - lo,
                (gray > _CLIP_HIGH).mean(),
                (gray < _CLIP_LOW).mean(),
                colorfulness,
                (r - b).mean(),
                saturation.mean(),
            ],
            dtype=np.float32,
        )
