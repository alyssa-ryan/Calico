from __future__ import annotations

import colorsys

import numpy as np

from calico.features.base import ImageExtractor
from calico.features.palette import PaletteExtractor
from calico.selection.scoring import histogram_intersection
from tests.factories import scene, solid

EX = PaletteExtractor()


def test_shape_and_sum() -> None:
    assert isinstance(EX, ImageExtractor)
    out = EX.extract([scene(1), solid((10, 10, 10)), solid((255, 0, 0))])
    assert out.shape == (3, 28)
    assert out.dtype == np.float32
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_red_vs_blue() -> None:
    red, red2, blue = (
        EX.extract([solid(c)]) for c in [(220, 20, 20), (230, 30, 25), (20, 20, 220)]
    )
    assert histogram_intersection(red, red2)[0, 0] > 0.95
    assert histogram_intersection(red, blue)[0, 0] < 0.05


def test_grey_is_achromatic() -> None:
    row = EX.extract([solid((128, 128, 128))])[0]
    assert row[24:].sum() > 0.999
    assert row[:24].sum() < 1e-6


def test_empty() -> None:
    assert EX.extract([]).shape == (0, 28)


def test_hue_is_circular() -> None:
    """Red just below and just above 0 degrees must still look alike."""

    def rgb(deg: float) -> tuple[int, int, int]:
        r, g, b = colorsys.hsv_to_rgb(deg / 360, 0.9, 0.9)
        return round(r * 255), round(g * 255), round(b * 255)

    a = EX.extract([solid(rgb(1))])
    b = EX.extract([solid(rgb(359))])
    assert histogram_intersection(a, b)[0, 0] > 0.8
