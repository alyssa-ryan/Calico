from __future__ import annotations

import numpy as np
import pytest

from calico.errors import CapabilityError
from calico.features.dinov2 import Dinov2Embedder
from calico.selection.patch_match import patch_shares
from tests.factories import scene

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module")
def dinov2() -> Dinov2Embedder:
    return Dinov2Embedder(device="cpu")


def test_rows_are_unit_norm_and_versioned_by_what_shapes_them(dinov2: Dinov2Embedder) -> None:
    out = dinov2.extract([scene(1), scene(2)])
    assert out.shape == (2, dinov2.dim)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)
    assert dinov2.version == "vit_base_patch14_reg4_dinov2.lvd142m/448/avg"
    assert np.array_equal(out, dinov2.extract([scene(1), scene(2)]))  # deterministic


def test_the_same_scene_moved_and_darkened_is_closer_than_another(dinov2: Dinov2Embedder) -> None:
    ref, moved, other = dinov2.extract(
        [scene(1), scene(1, shift=(6, 0), brightness=0.75), scene(2)]
    )
    assert ref @ moved > ref @ other


def test_it_reads_images_only(dinov2: Dinov2Embedder) -> None:
    assert dinov2.supports_text is False
    assert dinov2.match_probability(np.zeros((1, 1), dtype=np.float32)) is None
    with pytest.raises(CapabilityError):
        dinov2.encode_texts(["a staircase"])
    with pytest.raises(CapabilityError):
        dinov2.readable_prefix("a staircase")


def test_the_class_token_at_224_px_is_still_available() -> None:
    token = Dinov2Embedder(pool="token", image_size=224, device="cpu")
    out = token.extract([scene(1)])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)
    assert token.version.endswith("/224/token")


def test_patches_are_unit_rows_on_a_32_by_32_grid(dinov2: Dinov2Embedder) -> None:
    patches = dinov2.extract_patches([scene(1), scene(1, shift=(6, 0)), scene(2)])
    assert patches.shape == (3, 32 * 32, dinov2.dim)
    assert patches.dtype == np.float16
    assert np.allclose(np.linalg.norm(patches.astype(np.float32), axis=2), 1.0, atol=1e-2)
    _, moved, other = patch_shares(patches, patches[:1])
    assert moved > other
