from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from calico.errors import ConfigError
from calico.features import (
    ExtractorFactory,
    build_extractors,
    describe_embedder_of,
    embedder_of,
    extractor_registry,
)
from calico.plugins import Registry
from calico.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from PIL import Image


class Dummy:
    version = "1"
    dim = 2
    columns: tuple[str, ...] | None = None

    def __init__(self, scale: float = 1.0) -> None:
        self.name = "dummy"
        self.scale = scale

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        return np.zeros((len(images), 2), np.float32)


class DummyEmbedder(Dummy):
    supports_text = False

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.zeros((len(texts), 2), np.float32)

    def readable_prefix(self, text: str) -> str:
        return text

    def match_probability(self, similarities: NDArray[np.float32]) -> NDArray[np.float32] | None:
        return None


def registry(embedder: ExtractorFactory) -> Registry[ExtractorFactory]:
    plugins: dict[str, ExtractorFactory] = {
        "technical": Dummy,
        "layout": Dummy,
        "palette": Dummy,
        "tiny": embedder,
    }
    return Registry.from_mapping("extractor", plugins)


def test_real_entry_points(tmp_path: Path) -> None:
    names = extractor_registry().names()
    for n in ("technical", "layout", "palette", "tiny"):
        assert n in names
    s = Settings(home=tmp_path, embedder="tiny", describe_embedder="tiny", view_embedder="tiny")
    built = build_extractors(s)
    assert list(built) == ["technical", "layout", "palette", "tiny"]
    assert embedder_of(built, s).name == "tiny"
    assert describe_embedder_of(built, s).name == "tiny"


def test_describe_it_can_have_an_embedder_of_its_own(tmp_path: Path) -> None:
    s = Settings(home=tmp_path, embedder="tiny", describe_embedder="words", view_embedder="tiny")
    plugins: dict[str, ExtractorFactory] = {
        "technical": Dummy,
        "layout": Dummy,
        "palette": Dummy,
        "tiny": DummyEmbedder,
        "words": DummyEmbedder,
    }
    built = build_extractors(s, Registry.from_mapping("extractor", plugins))
    assert list(built) == ["technical", "layout", "palette", "tiny", "words"]
    assert describe_embedder_of(built, s) is built["words"]


def test_a_describe_extractor_that_is_not_an_embedder_is_rejected(tmp_path: Path) -> None:
    s = Settings(home=tmp_path, embedder="tiny", describe_embedder="layout", view_embedder="tiny")
    with pytest.raises(ConfigError, match="describe_embedder 'layout'"):
        build_extractors(s, registry(DummyEmbedder))


def test_options_passed(tmp_path: Path) -> None:
    s = Settings(
        home=tmp_path,
        embedder="tiny",
        describe_embedder="tiny",
        view_embedder="tiny",
        extractor_options={"layout": {"scale": 3}},
    )
    built = build_extractors(s, registry(DummyEmbedder))
    layout = built["layout"]
    assert isinstance(layout, Dummy)
    assert layout.scale == 3


def test_non_embedder_rejected(tmp_path: Path) -> None:
    s = Settings(home=tmp_path, embedder="tiny", describe_embedder="tiny", view_embedder="tiny")
    with pytest.raises(ConfigError, match="tiny"):
        build_extractors(s, registry(Dummy))
    with pytest.raises(ConfigError, match="tiny"):
        embedder_of({"tiny": Dummy()}, s)
