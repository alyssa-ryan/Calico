"""The documentation quotes measured constants as the code has them (A10-9).

A constant tuned by a measurement is changed in code and quoted in the docs, and the two
drift apart unnoticed. Each entry names a constant and the phrase a document quotes it in,
with ``{}`` where its value goes (``str.format``: ``{:.2f}``, ``{0[1]}`` for a tuple).
Add an entry whenever a doc quotes a constant's value.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
ARCHITECTURE = "docs/architecture.md"

MANIFEST: tuple[tuple[str, str, str, str], ...] = (
    (
        "calico.selection.strength",
        "QUALITY_WEIGHT",
        ARCHITECTURE,
        "`selection.strength.QUALITY_WEIGHT` {}",
    ),
    ("calico.selection.strength", "AESTHETIC_WEIGHT", ARCHITECTURE, "`AESTHETIC_WEIGHT` {}"),
    ("calico.criteria", "EXAMPLE_WEIGHT", ARCHITECTURE, "`EXAMPLE_WEIGHT` ({})"),
    ("calico.selection.copies", "EDIT_SHARE", ARCHITECTURE, "`EDIT_SHARE` {:.2f}"),
    ("calico.tasks.best", "VIEW_REACH", ARCHITECTURE, "`VIEW_REACH` ({})"),
    (
        "calico.selection.view_groups",
        "CANDIDATE_SIMILARITY",
        ARCHITECTURE,
        "`view_similarity` {}",
    ),
    ("calico.selection.view_groups", "GROUP_SHARE", ARCHITECTURE, "`view_share` {}"),
    ("calico.evaluation", "TASTE_FIRST", ARCHITECTURE, "first {0[0]} and {0[1]} each template"),
    ("calico.evaluation", "WIN_PLACES", ARCHITECTURE, "`WIN_PLACES` ({})"),
    ("calico.evaluation", "COMPARE_PICKS", ARCHITECTURE, "`COMPARE_PICKS`, {} each"),
    ("calico.evaluation", "CURVE_STEP", ARCHITECTURE, "tens (`CURVE_STEP`, {})"),
    ("calico.evaluation", "BAR_AT", ARCHITECTURE, "first `BAR_AT` ({})"),
    ("calico.evaluation", "PRECISION_AT", ARCHITECTURE, "`PRECISION_AT` {}"),
    ("calico.evaluation", "RUNNING_LINE", ARCHITECTURE, "`RUNNING_LINE` {}"),
    ("calico.evaluation", "RUNNING_AFTER", ARCHITECTURE, "`RUNNING_AFTER` {}"),
    ("calico.evaluation", "STORY_SESSION_STORIES", ARCHITECTURE, "`STORY_SESSION_STORIES` ({})"),
    (
        "calico.evaluation",
        "CANDIDATE_SESSION_STORIES",
        ARCHITECTURE,
        "`CANDIDATE_SESSION_STORIES` ({})",
    ),
    ("calico.evaluation", "TASTE_RANDOM", ARCHITECTURE, "{} at random from the whole place"),
    ("calico.evaluation", "VALUE_DEPTH", ARCHITECTURE, "(`VALUE_DEPTH`, {})"),
    ("calico.evaluation", "GEOMETRY_BAND", ARCHITECTURE, "(`GEOMETRY_BAND`, {0[0]} to {0[1]:.2f})"),
    ("calico.models.taste", "TASTE_NEIGHBOURS", ARCHITECTURE, "`TASTE_NEIGHBOURS` ({})"),
    ("calico.models.taste", "FAMILIAR_AT", ARCHITECTURE, "`FAMILIAR_AT` ({})"),
    ("calico.models.taste", "TASTE_L2", ARCHITECTURE, "held back hard (l2 {:g})"),
    ("calico.models.calibration", "MIN_FORECASTS", ARCHITECTURE, "`MIN_FORECASTS` ({})"),
    ("calico.models.calibration", "MIN_PLACES", ARCHITECTURE, "`MIN_PLACES` ({})"),
    ("calico.evaluation", "DISAGREEMENT_GAP", ARCHITECTURE, "`DISAGREEMENT_GAP` ({})"),
    ("calico.evaluation", "TOP_SHARE", ARCHITECTURE, "`TOP_SHARE` ({})"),
    ("calico.evaluation.controls", "CONTROL_POOL", ARCHITECTURE, "`CONTROL_POOL` ({})"),
    (
        "calico.evaluation.controls",
        "CONTROLS_PER_SESSION",
        ARCHITECTURE,
        "`CONTROLS_PER_SESSION` ({})",
    ),
    ("calico.evaluation.holdout", "NEAR_SHARE", ARCHITECTURE, "`NEAR_SHARE` ({})"),
    ("calico.evaluation.holdout", "NEAR_FILES", ARCHITECTURE, "`NEAR_FILES` ({})"),
)


@pytest.mark.parametrize(
    ("module", "name", "doc", "phrase"), MANIFEST, ids=[e[1] for e in MANIFEST]
)
def test_the_docs_quote_the_constant_as_the_code_has_it(
    module: str, name: str, doc: str, phrase: str
) -> None:
    value = getattr(importlib.import_module(module), name)
    quoted = phrase.format(value)
    text = " ".join((ROOT / doc).read_text().split())  # a phrase may wrap across lines
    assert quoted in text, (
        f"{doc} does not say {quoted!r}: {name} is {value!r} in {module}. Update the doc, "
        "or this manifest if the doc now words it another way."
    )
