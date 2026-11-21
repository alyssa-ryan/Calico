from __future__ import annotations

import numpy as np
import pytest

from calico.features.clip import ClipEmbedder, SiglipEmbedder
from tests.factories import solid

pytestmark = pytest.mark.ml


def test_shapes_and_text_alignment() -> None:
    emb = ClipEmbedder(device="cpu")
    out = emb.extract([solid((255, 0, 0), (224, 224)), solid((0, 0, 255), (224, 224))])
    assert out.shape == (2, emb.dim)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)
    text = emb.encode_texts(["a photo of a red square"])
    assert text.shape == (1, emb.dim)
    assert np.allclose(np.linalg.norm(text, axis=1), 1.0, atol=1e-4)
    assert text[0] @ out[0] > text[0] @ out[1]


def test_readable_prefix_is_what_the_text_encoder_actually_reads() -> None:
    emb = ClipEmbedder(device="cpu")
    short = "warm light on an old building"
    assert emb.readable_prefix(short) == short
    long = " ".join(["mossy stone wall"] * 40) + " and a red door"
    kept = emb.readable_prefix(long)
    assert long.startswith(kept)
    assert 0 < len(kept) < len(long)
    whole, start = emb.encode_texts([long, kept])
    assert whole @ start > 0.99  # everything after the prefix made no difference


@pytest.fixture(scope="module")
def siglip() -> SiglipEmbedder:
    return SiglipEmbedder(device="cpu")


def test_siglip_says_how_likely_an_image_matches(siglip: SiglipEmbedder) -> None:
    images = siglip.extract([solid((255, 0, 0), (224, 224)), solid((0, 0, 255), (224, 224))])
    assert images.shape == (2, siglip.dim)
    assert np.allclose(np.linalg.norm(images, axis=1), 1.0, atol=1e-4)
    text = siglip.encode_texts(["a red square", "a bicycle"])
    p = siglip.match_probability(images @ text.T)
    assert p is not None
    assert ((p >= 0) & (p <= 1)).all()
    assert p[0, 0] > p[1, 0]  # red matches the red square better than blue does
    assert p[:, 1].max() < 0.5  # neither is a bicycle


def test_siglip_reads_each_description_as_a_caption(siglip: SiglipEmbedder) -> None:
    """Measured: "a photo of {}." lifts SigLIP on labelled searches (AUC 0.840 -> 0.847)."""
    plain = SiglipEmbedder(device="cpu", text_template="{}")
    (captioned,) = siglip.encode_texts(["a red square"])
    (written_out,) = plain.encode_texts(["a photo of a red square."])
    assert captioned @ written_out > 0.999


def test_siglip_readable_prefix_counts_the_caption_words(siglip: SiglipEmbedder) -> None:
    short = "warm light on an old building"
    assert siglip.readable_prefix(short) == short
    long = " ".join(["mossy stone wall"] * 30) + " and a red door"
    kept = siglip.readable_prefix(long)
    assert long.startswith(kept)
    assert 0 < len(kept) < len(long)
    whole, start, shorter = siglip.encode_texts([long, kept, " ".join(kept.split()[:-3])])
    assert whole @ start > 0.999  # everything after the prefix made no difference
    assert whole @ shorter < whole @ start  # and the prefix is not cut short
