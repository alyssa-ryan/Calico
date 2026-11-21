"""Feature extractors: decoded image -> fixed-length float32 vector.

An extractor is identified by ``name`` (how tasks and settings refer to it) and
``version`` (bumped whenever its output would change). Cached vectors are keyed
by ``feature_key(extractor)`` = ``"name@version"``, so a version bump makes the
old cache invisible instead of silently mixing old and new outputs.

Extractors receive images that are already RGB, EXIF-oriented, and no larger
than ``Settings.decode_max_side`` (see ``calico.imaging``). They must not mutate
them, must be deterministic, and must return one row per input image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray
    from PIL import Image


@runtime_checkable
class ImageExtractor(Protocol):
    name: str
    version: str
    dim: int
    columns: tuple[str, ...] | None
    """Names of the output columns when they are individually meaningful
    (e.g. ``("sharpness", "exposure", ...)``); ``None`` for opaque embeddings."""

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        """Return an array of shape ``(len(images), dim)``."""
        ...


@runtime_checkable
class Embedder(ImageExtractor, Protocol):
    """An extractor whose rows are unit-norm, so dot product = cosine similarity.

    If ``supports_text`` is true, ``encode_texts`` returns unit-norm vectors in
    the *same* space, which is what makes the criteria box work.
    """

    supports_text: bool

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return ``(len(texts), dim)``. Raise ``CapabilityError`` if unsupported."""
        ...

    def readable_prefix(self, text: str) -> str:
        """The leading words of ``text`` that ``encode_texts`` reads; it ignores the rest.

        Text encoders have a fixed context, and silently cut what does not fit.
        Returns ``text`` unchanged when it all fits. Raise ``CapabilityError`` if
        text is unsupported.
        """
        ...

    def match_probability(self, similarities: NDArray[np.float32]) -> NDArray[np.float32] | None:
        """How likely each image matches each text, from their cosine similarities.

        None when the model only ranks. CLIP was trained to pick the right caption
        among many, so its similarities order images but no value means "matches";
        SigLIP was trained to say yes or no to each pair, so it can.
        """
        ...


@runtime_checkable
class PatchExtractor(ImageExtractor, Protocol):
    """An extractor that can also describe an image part by part, for matching what is
    where (``selection.patch_match``). Patches are computed when asked for, not cached:
    a photo's patches are hundreds of vectors."""

    def extract_patches(self, images: Sequence[Image.Image]) -> NDArray[np.float16]:
        """``(n, patches, dim)``: one unit row per patch, in the same grid order for every
        image."""
        ...


def feature_key(extractor: ImageExtractor) -> str:
    return f"{extractor.name}@{extractor.version}"
