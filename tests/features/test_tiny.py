from __future__ import annotations

from calico.features.layout import LayoutExtractor
from calico.features.palette import PaletteExtractor
from calico.features.tiny import TinyEmbedder


def test_tiny_is_versioned_by_the_parts_it_is_built_from() -> None:
    """Bumping layout or palette changes tiny's vectors too, so cached ones must go."""
    assert f"l{LayoutExtractor.version}" in TinyEmbedder.version
    assert f"p{PaletteExtractor.version}" in TinyEmbedder.version
