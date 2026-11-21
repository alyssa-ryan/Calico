"""How Dinov2Embedder fails without PyTorch (runs without the ml extra)."""

from __future__ import annotations

import sys

import pytest

from calico.errors import CapabilityError
from calico.features.dinov2 import Dinov2Embedder


def test_without_the_ml_extra_it_says_how_to_get_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "timm", None)
    with pytest.raises(CapabilityError, match="uv sync"):
        Dinov2Embedder()


def test_an_unknown_pooling_is_refused_before_loading_anything() -> None:
    with pytest.raises(CapabilityError, match="pool"):
        Dinov2Embedder(pool="max")
