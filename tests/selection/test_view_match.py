from __future__ import annotations

import numpy as np
import pytest

from calico.errors import CapabilityError
from calico.selection.context import Trace
from calico.selection.view_match import view_match
from tests.factories import view


def test_the_subject_comes_from_the_main_embedding_by_default() -> None:
    photos = view([[1.0, 0.0], [0.0, 1.0]], {"layout": np.ones((2, 2))})
    refs = view([[1.0, 0.0]], {"layout": np.ones((1, 2))})
    vm = view_match(photos, refs, 0.0, Trace())
    assert vm.semantic.tolist() == pytest.approx([1.0, 0.0])


def test_the_subject_can_come_from_another_extractor() -> None:
    dino = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    photos = view([[1.0, 0.0], [0.0, 1.0]], {"layout": np.ones((2, 2)), "dino": dino})
    refs = view([[1.0, 0.0]], {"layout": np.ones((1, 2)), "dino": np.asarray([[1.0, 0.0]])})
    vm = view_match(photos, refs, 0.0, Trace(), semantic="dino")
    assert vm.semantic.tolist() == pytest.approx([0.0, 1.0])
    assert vm.match.tolist() == pytest.approx([0.0, 1.0])


def test_a_subject_extractor_without_features_says_how_to_get_them() -> None:
    photos = view([[1.0, 0.0]], {"layout": np.ones((1, 2))})
    refs = view([[1.0, 0.0]], {"layout": np.ones((1, 2))})
    with pytest.raises(CapabilityError, match="dino"):
        view_match(photos, refs, 0.5, Trace(), semantic="dino")
