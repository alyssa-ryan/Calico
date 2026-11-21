from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from calico.features.base import ImageExtractor
from calico.features.layout import LayoutExtractor
from tests.factories import scene, solid

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image

EX = LayoutExtractor()


def cos(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    return float(np.dot(a, b))


def vec(img: Image.Image) -> NDArray[np.float32]:
    out: NDArray[np.float32] = EX.extract([img])[0]
    return out


def test_shape_and_norm() -> None:
    assert isinstance(EX, ImageExtractor)
    out = EX.extract([scene(1), scene(2)])
    assert out.shape == (2, 680)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)


def test_robust_to_light() -> None:
    base = vec(scene(4))
    lit = vec(scene(4, brightness=0.6, tint=(1.1, 1.0, 0.85)))
    assert cos(base, lit) > 0.9


def test_small_shift_closer_than_other_scene() -> None:
    base = vec(scene(4))
    assert cos(base, vec(scene(4, shift=(8, 0)))) > cos(base, vec(scene(5)))


def test_different_scenes_differ() -> None:
    assert cos(vec(scene(4)), vec(scene(9))) < 0.8


def test_flat_is_finite() -> None:
    assert np.isfinite(vec(solid((90, 90, 90)))).all()


def test_blank_images_are_identical() -> None:
    a, b = vec(solid((10, 10, 10))), vec(solid((240, 240, 240)))
    assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-5)
    assert cos(a, b) > 0.999


def test_reaims_clearly_above_different_scenes() -> None:
    """Locks the separation documented in the module docstring (seeds 0-9)."""
    seeds = range(10)
    base = {s: vec(scene(s)) for s in seeds}
    reaim = [
        cos(base[s], vec(variant))
        for s in seeds
        for variant in (scene(s).rotate(5, expand=False), scene(s, shift=(26, 0)))
    ]
    light = [cos(base[s], vec(scene(s, brightness=0.6, tint=(1.1, 1.0, 0.85)))) for s in seeds]
    different = [cos(base[s], base[t]) for s in seeds for t in seeds if s != t]
    assert float(np.mean(reaim)) - max(different) > 0.12
    assert min(reaim) - max(different) > 0.05
    assert min(light) > max(different) + 0.25
