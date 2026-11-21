"""The no-PyTorch embedder: composition + colour, unit length.

It cannot read text, so the criteria box is unavailable with it, but it is
deterministic, instant and good enough to find "the same view" in tests and on
machines without the ml extra.

The vector is ``concat(sqrt(0.6) * layout, sqrt(0.4) * sqrt(palette))``. Layout
is unit-norm and so is ``sqrt(palette)`` (the Hellinger map of an L1 histogram),
hence the cosine of two vectors is exactly
``0.6 * layout_cosine + 0.4 * Bhattacharyya(palette)``: composition leads,
colour breaks ties.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from calico.errors import CapabilityError
from calico.features.layout import LayoutExtractor
from calico.features.palette import PaletteExtractor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from PIL import Image


LAYOUT_WEIGHT = 0.6
GRID = 4
PALETTE_WEIGHT = 0.4


NO_TEXT = (
    "The 'tiny' embedder cannot read text. Install the ml extra "
    '(uv sync --extra ml) and set embedder = "clip" in calico.toml.'
)


class TinyEmbedder:
    name = "tiny"
    version = f"3+l{LayoutExtractor.version}+p{PaletteExtractor.version}"
    """Its own version and its parts': bumping layout or palette changes these vectors."""
    supports_text = False
    columns: tuple[str, ...] | None = None

    def __init__(self) -> None:
        self._layout = LayoutExtractor()
        self._palette = PaletteExtractor()
        self.dim = self._layout.dim + self._palette.dim

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        layout = self._layout.extract(images)
        palette = np.sqrt(self._palette.extract(images))
        joined = np.concatenate(
            [np.sqrt(LAYOUT_WEIGHT) * layout, np.sqrt(PALETTE_WEIGHT) * palette], axis=1
        )
        norms = np.linalg.norm(joined, axis=1, keepdims=True)
        out: NDArray[np.float32] = (joined / np.maximum(norms, 1e-12)).astype(np.float32)
        return out

    def extract_patches(self, images: Sequence[Image.Image]) -> NDArray[np.float16]:
        """The same vector for each cell of a 4x4 grid: coarse, but enough to match views
        part by part without PyTorch (tests, the sandbox)."""
        out = np.zeros((len(images), GRID * GRID, self.dim), dtype=np.float16)
        for i, image in enumerate(images):
            w, h = image.size
            cells = [
                image.crop((x * w // GRID, y * h // GRID, (x + 1) * w // GRID, (y + 1) * h // GRID))
                for y in range(GRID)
                for x in range(GRID)
            ]
            out[i] = self.extract(cells).astype(np.float16)
        return out

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        raise CapabilityError(NO_TEXT)

    def readable_prefix(self, text: str) -> str:
        raise CapabilityError(NO_TEXT)

    def match_probability(self, similarities: NDArray[np.float32]) -> NDArray[np.float32] | None:
        return None
