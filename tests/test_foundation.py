"""Tests for the small shared contracts: plugin registry, domain values, context, rollup."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from calico.domain import Evidence, Pick, SelectionRequest, SelectionResult
from calico.errors import UnknownPluginError
from calico.models.base import rollup
from calico.plugins import TASKS_GROUP, Registry
from calico.selection.context import FeatureView, Trace
from tests import factories as f

# -- plugins -----------------------------------------------------------------


def test_registry_lists_and_loads_plugins_lazily() -> None:
    calls: list[str] = []

    def loader() -> str:
        calls.append("loaded")
        return "plugin"

    registry = Registry[str]("thing", {"b": loader, "a": lambda: "other"})
    assert registry.names() == ["a", "b"]
    assert calls == []
    assert registry.get("b") == "plugin"
    assert registry.get("b") == "plugin"
    assert calls == ["loaded"]


def test_unknown_plugin_error_lists_what_exists() -> None:
    registry = Registry.from_mapping("task", {"best": object()})
    with pytest.raises(UnknownPluginError, match="Known tasks: best"):
        registry.get("worst")


def test_builtin_tasks_are_registered_as_entry_points() -> None:
    names = Registry[object].from_entry_points(TASKS_GROUP, "task").names()
    assert {"best", "same_view", "triptych"} <= set(names)


# -- domain ------------------------------------------------------------------


def test_request_needs_at_least_one_folder() -> None:
    with pytest.raises(ValidationError):
        SelectionRequest(task="best", folders=())


def test_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SelectionRequest.model_validate({"task": "best", "folders": ["/a"], "colour": "red"})


def test_result_round_trips_through_json() -> None:
    pick = Pick(
        path=Path("/a/b.jpg"),
        content_key="k",
        location="Old Mill",
        captured_at=datetime(2021, 3, 4, 5, 6, tzinfo=UTC),
        score=0.8,
        evidence=(Evidence(signal="quality", value=0.8, weight=1.0),),
    )
    result = SelectionResult(
        run_id="r1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        request=SelectionRequest(task="best", folders=(Path("/a"),)),
        groups=(),
        notes=("n",),
    )
    assert SelectionResult.model_validate_json(result.model_dump_json()) == result
    assert Pick.model_validate_json(pick.model_dump_json()) == pick


def test_evidence_contribution_is_value_times_weight() -> None:
    assert Evidence(signal="s", value=0.5, weight=0.4).contribution == pytest.approx(0.2)


# -- selection context ---------------------------------------------------------


def test_feature_view_rejects_misaligned_features() -> None:
    with pytest.raises(ValueError, match="rows"):
        FeatureView(
            embeddings=np.zeros((3, 2), np.float32), features={"t": np.zeros((2, 1), np.float32)}
        )


def test_feature_view_take_keeps_rows_aligned() -> None:
    v = f.view([[1, 0], [0, 1], [1, 1]], {"q": [0.1, 0.2, 0.3]})
    sub = v.take([2, 0])
    assert sub.features["q"][:, 0].tolist() == pytest.approx([0.3, 0.1])
    assert len(sub) == 2


def test_feature_view_column_by_name() -> None:
    v = f.view([[1, 0], [0, 1]], {"t": [[1, 2], [3, 4]]}, {"t": ("a", "b")})
    assert v.column("t", "b").tolist() == [2, 4]
    with pytest.raises(KeyError, match="no column"):
        v.column("t", "c")


def test_context_rejects_photo_feature_mismatch() -> None:
    with pytest.raises(ValueError, match="feature rows"):
        f.context([f.photo("a.jpg")], f.view([[1, 0], [0, 1]]))


def test_trace_records_stage_counts_and_dedupes_notes() -> None:
    trace = Trace()
    with trace.stage("filter", 10) as stage:
        stage.count_out = 4
    trace.note("x")
    trace.note("x")
    trace.record("quality", [0.1, 0.2])
    assert (trace.stages[0].count_in, trace.stages[0].count_out) == (10, 4)
    assert trace.notes == ["x"]
    assert trace.columns["quality"].dtype == np.float32


# -- models -------------------------------------------------------------------


def test_rollup_sums_sub_labels_into_parents_in_first_seen_order() -> None:
    labels = ("establishing:wide", "detail", "establishing:close")
    names, probs = rollup(labels, np.array([[0.5, 0.3, 0.2]], dtype=np.float32))
    assert names == ("establishing", "detail")
    assert probs.tolist()[0] == pytest.approx([0.7, 0.3])


def test_rollup_depth_two_is_identity_for_two_level_labels() -> None:
    labels = ("a:x", "a:y")
    names, probs = rollup(labels, np.array([[0.4, 0.6]], dtype=np.float32), depth=2)
    assert names == labels
    assert probs.tolist()[0] == pytest.approx([0.4, 0.6])


def test_rollup_rejects_depth_zero() -> None:
    with pytest.raises(ValueError, match="depth"):
        rollup(("a",), np.ones((1, 1), np.float32), depth=0)


def test_request_limit_defaults_to_none_meaning_the_tasks_default() -> None:
    assert SelectionRequest(task="best", folders=(Path("/a"),)).limit is None
    for bad in (0, 501):
        with pytest.raises(ValidationError):
            SelectionRequest(task="best", folders=(Path("/a"),), limit=bad)
