"""Each ML extractor still answers as it did when its golden output was made (A10-10).

``uv.lock`` pins the libraries, but the weights come from the hub by name, so a replaced
file, or a preprocessing change the lock lets in, would change every vector while the
extractor's ``version`` (the key cached features are stored under) stayed the same, and
the cache would mix old and new under one key. CI downloads the weights afresh each run,
so it sees such a change. One synthetic photo per extractor, on the CPU, against the
output stored in ``golden.json``.

After an intended change (bump the extractor's ``version``), write them again with
    CALICO_WRITE_GOLDEN=1 uv run pytest -m ml tests/features/test_golden.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from calico.features import extractor_registry
from tests.factories import scene

pytestmark = pytest.mark.ml

GOLDEN = Path(__file__).with_name("golden.json")
WRITE_ENV = "CALICO_WRITE_GOLDEN"
EXTRACTORS = ("clip", "siglip", "dinov2", "laion_aesthetic")
SAME_DIRECTION = 0.9999
"""Cosine to the stored vector. The CPU's float rounding differs by machine by far less;
0.999 is too loose: the same photo 10% darker is 0.9993 on DINOv2."""
SAME_SCORE = 0.01
"""For LAION's one-column score (about 1 to 10), where a direction means nothing."""


def write(name: str, version: str, row: np.ndarray) -> None:
    golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    golden[name] = {"version": version, "row": [round(float(x), 6) for x in row]}
    GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")


@pytest.mark.parametrize("name", EXTRACTORS)
def test_the_extractor_answers_as_it_did(name: str) -> None:
    extractor = extractor_registry().get(name)(device="cpu")
    (row,) = extractor.extract([scene(1)])
    if os.environ.get(WRITE_ENV):
        write(name, extractor.version, row)
        return
    golden = json.loads(GOLDEN.read_text())[name]
    assert extractor.version == golden["version"], (
        f"{name} is now {extractor.version}: write its golden output again ({WRITE_ENV}=1)"
    )
    expected = np.asarray(golden["row"], dtype=np.float32)
    if len(expected) == 1:
        assert abs(float(row[0]) - float(expected[0])) < SAME_SCORE
    else:
        cosine = float(row @ expected) / float(np.linalg.norm(row) * np.linalg.norm(expected))
        assert cosine > SAME_DIRECTION, f"{name} answers differently (cosine {cosine:.4f})"
