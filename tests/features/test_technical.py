from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFilter

if TYPE_CHECKING:
    from numpy.typing import NDArray

from calico.features.base import ImageExtractor
from calico.features.technical import TechnicalExtractor
from tests.factories import scene, solid

EX = TechnicalExtractor()


def col(images: list[Image.Image], name: str) -> NDArray[np.float32]:
    assert EX.columns is not None
    out: NDArray[np.float32] = EX.extract(images)[:, EX.columns.index(name)]
    return out


def test_protocol_and_shape() -> None:
    assert isinstance(EX, ImageExtractor)
    out = EX.extract([scene(1), scene(2)])
    assert out.shape == (2, 8)
    assert out.dtype == np.float32
    assert EX.dim == len(EX.columns or ())


def test_blur_lowers_sharpness() -> None:
    sharp = scene(3)
    blurred = sharp.filter(ImageFilter.GaussianBlur(4))
    assert col([blurred], "sharpness")[0] < col([sharp], "sharpness")[0]


def test_sharpness_comparable_across_sizes() -> None:
    small = col([scene(3, (320, 240))], "sharpness")[0]
    large = col([scene(3, (1024, 768))], "sharpness")[0]
    assert large < small * 10  # both resized to 512: same order of magnitude


def test_white_is_clipped() -> None:
    white = solid((255, 255, 255))
    assert col([white], "highlight_clip")[0] > 0.99
    assert col([solid((0, 0, 0))], "shadow_clip")[0] > 0.99


def test_orange_warmer_than_blue() -> None:
    assert col([solid((240, 140, 40))], "warmth")[0] > col([solid((40, 100, 240))], "warmth")[0]


def test_grey_has_no_colour() -> None:
    grey = solid((128, 128, 128))
    assert col([grey], "colorfulness")[0] < 1e-3
    assert col([grey], "saturation")[0] < 1e-3
    assert abs(col([grey], "brightness")[0] - 128 / 255) < 0.01


def test_colourful_scene_has_contrast_and_colour() -> None:
    assert col([scene(5)], "contrast")[0] > 0.05
    assert col([scene(5)], "colorfulness")[0] > 0.0


def test_deterministic_and_empty() -> None:
    a = EX.extract([scene(7)])
    assert np.array_equal(a, EX.extract([scene(7)]))
    assert EX.extract([]).shape == (0, 8)


def test_sharpness_stable_across_decode_sizes() -> None:
    master = scene(3, (1600, 1200))
    values = [
        col([master.resize((w, w * 3 // 4), Image.Resampling.LANCZOS)], "sharpness")[0]
        for w in (400, 1000, 1600)
    ]
    assert max(values) / min(values) < 1.15 / 0.85


def test_other_modes_do_not_crash() -> None:
    base = scene(3)
    for mode in ("L", "RGBA", "P"):
        out = EX.extract([base.convert(mode)])
        assert out.shape == (1, 8)
        assert np.isfinite(out).all()


def test_small_images_are_not_upscaled() -> None:
    tiny = solid((10, 200, 30), (40, 30))
    assert np.isfinite(EX.extract([tiny])).all()


def dark_noisy_blur(image: Image.Image, seed: int = 0) -> Image.Image:
    """``image`` out of focus, at a tenth of its light, with sensor noise: a dark, high-ISO
    interior shot that missed focus (audit R-05)."""
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(4)), dtype=np.float32)
    noise = np.random.default_rng(seed).normal(0, 3, blurred.shape)
    return Image.fromarray(np.clip(blurred * 0.1 + noise, 0, 255).astype(np.uint8))


def sharpness(extractor: TechnicalExtractor, image: Image.Image) -> float:
    assert extractor.columns is not None
    return float(extractor.extract([image])[0, extractor.columns.index("sharpness")])


NOISE_AWARE = TechnicalExtractor(noise_aware=True)


def test_today_noise_reads_as_detail() -> None:
    # Why the variant exists: today a dark noisy blur outscores the sharp frame.
    assert sharpness(EX, dark_noisy_blur(scene(3))) > sharpness(EX, scene(3))


def test_a_dark_noisy_blur_scores_below_a_sharp_frame_once_noise_is_subtracted() -> None:
    for seed in (1, 3, 10):
        sharp = scene(seed)
        assert sharpness(NOISE_AWARE, dark_noisy_blur(sharp)) < sharpness(NOISE_AWARE, sharp)


def test_noise_aware_sharpness_keeps_a_clean_frame_and_its_blur_apart() -> None:
    sharp = scene(3)
    assert np.isclose(sharpness(NOISE_AWARE, sharp), sharpness(EX, sharp), rtol=0.01)
    blurred = sharp.filter(ImageFilter.GaussianBlur(2))
    assert sharpness(NOISE_AWARE, blurred) < sharpness(NOISE_AWARE, sharp)


def test_the_noise_aware_variant_is_its_own_version_with_the_same_columns() -> None:
    assert (EX.version, NOISE_AWARE.version) == ("2", "3")
    assert NOISE_AWARE.columns == EX.columns
    others = [c for c in EX.columns or () if c != "sharpness"]
    image = dark_noisy_blur(scene(1))
    for name in others:  # only sharpness changes
        index = (EX.columns or ()).index(name)
        assert EX.extract([image])[0, index] == NOISE_AWARE.extract([image])[0, index]
