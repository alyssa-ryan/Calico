from __future__ import annotations

import pytest

from calico.errors import CapabilityError, UnknownPluginError
from calico.models.quality import TechnicalQuality
from calico.models.registry import Models
from calico.models.taxonomy import Taxonomy
from calico.models.zero_shot import ZeroShotLabelModel
from calico.settings import Settings
from tests.factories import FeatureScore, unit
from tests.models.test_zero_shot import TAX, FakeEmbedder


class NoText(FakeEmbedder):
    supports_text = False


def test_code_precedence_cache_versions() -> None:
    made: list[int] = []

    def factory() -> FeatureScore:
        made.append(1)
        return FeatureScore("colour", "f", version="c1")

    m = Models(FakeEmbedder(), {"colour": TAX}, {"colour": factory})
    assert m.versions() == {}
    assert m.get("colour") is m.get("colour")
    assert len(made) == 1
    assert m.versions() == {"colour": "c1"}


def test_taxonomy_cached() -> None:
    m = Models(FakeEmbedder(), {"colour": TAX}, {})
    got = m.get("colour")
    assert isinstance(got, ZeroShotLabelModel)
    assert m.get("colour") is got
    assert m.versions() == {"colour": got.version}


def test_needs_text() -> None:
    m = Models(NoText(), {"colour": TAX}, {})
    with pytest.raises(CapabilityError, match="zero-shot"):
        m.get("colour")


def test_unknown_and_names() -> None:
    other = Taxonomy("a", "d", "1", ("x", "y"), (("x",), ("y",)))
    m = Models(FakeEmbedder(), {"colour": TAX, "a": other}, {"quality": TechnicalQuality})
    assert m.names() == ["a", "colour", "quality"]
    with pytest.raises(UnknownPluginError, match="a, colour, quality"):
        m.get("nope")


def test_from_settings(tmp_path) -> None:  # type: ignore[no-untyped-def]
    s = Settings(home=tmp_path, embedder="tiny", describe_embedder="tiny", view_embedder="tiny")
    m = Models.from_settings(s, NoText())
    assert {"quality", "shot_kind", "mood", "aesthetic", "place_type"} <= set(m.names())
    assert isinstance(m.get("quality"), TechnicalQuality)
    _ = unit
