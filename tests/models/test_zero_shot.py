from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from calico.models.base import LabelModel
from calico.models.taxonomy import Taxonomy
from calico.models.zero_shot import ZeroShotLabelModel
from tests.factories import unit, view

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from PIL import Image

WORDS: dict[str, list[float]] = {"red": [1, 0, 0], "crimson": [0.9, 0.1, 0], "blue": [0, 1, 0]}


class FakeEmbedder:
    name = "fake"
    version = "9"
    dim = 3
    columns: tuple[str, ...] | None = None
    supports_text = True

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        raise NotImplementedError

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls += 1
        return unit([WORDS[t] for t in texts])

    def readable_prefix(self, text: str) -> str:
        return text

    def match_probability(self, similarities: NDArray[np.float32]) -> NDArray[np.float32] | None:
        return None


TAX = Taxonomy(
    name="colour",
    description="Colours",
    version="3",
    labels=("red", "blue"),
    prompts=(("red", "crimson"), ("blue",)),
)


def test_predict() -> None:
    emb = FakeEmbedder()
    m = ZeroShotLabelModel(TAX, emb)
    assert isinstance(m, LabelModel)
    assert emb.calls == 1
    assert (m.name, m.labels, m.description) == ("colour", ("red", "blue"), "Colours")
    assert m.version == "zs3-t0.01-fake@9"
    v = view(np.array([[1, 0.05, 0], [0.05, 1, 0], [1000, 0, 0]]))
    p = m.predict_proba(v)
    assert p.shape == (3, 2)
    assert p.dtype == np.float32
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert p[0].argmax() == 0
    assert p[1].argmax() == 1
    assert np.isfinite(p).all()


def test_version_includes_temperature() -> None:
    a = ZeroShotLabelModel(TAX, FakeEmbedder(), temperature=0.01)
    b = ZeroShotLabelModel(TAX, FakeEmbedder(), temperature=0.05)
    assert a.version == "zs3-t0.01-fake@9"
    assert a.version != b.version
