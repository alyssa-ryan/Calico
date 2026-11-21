from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from calico.domain import Group, Pick, SelectionRequest, SelectionResult
from calico.errors import CalicoError, RunNotFoundError, RunUnreadableError
from calico.runs import RunStore, left_out_because
from tests import factories as f


def result(run_id: str, picked: str) -> SelectionResult:
    pick = Pick(
        path=Path("/archive/Old Mill") / picked,
        content_key=f"key-Old Mill-{picked}",
        location="Old Mill",
        captured_at=None,
        score=0.9,
    )
    return SelectionResult(
        run_id=run_id,
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
        request=SelectionRequest(task="best", folders=(Path("/archive"),), criteria="snow"),
        groups=(Group(title="g", score=0.9, picks=(pick,)),),
    )


def test_run_ids_sort_by_time() -> None:
    early = RunStore.new_run_id(datetime(2026, 1, 1, 9))
    late = RunStore.new_run_id(datetime(2026, 1, 1, 10))
    assert early < late


def test_save_load_and_list(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    photos = [f.photo("a.jpg"), f.photo("b.jpg")]
    store.save(result("20260101-000000-aaaa", "a.jpg"), photos, {})
    store.save(result("20260102-000000-bbbb", "b.jpg"), photos, {})
    assert store.load("20260101-000000-aaaa").request.criteria == "snow"
    assert [s.run_id for s in store.recent()] == ["20260102-000000-bbbb", "20260101-000000-aaaa"]
    assert store.recent()[0].picks == 1


def test_signals_mark_where_each_photo_landed_and_blank_nan(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    photos = [f.photo("a.jpg"), f.photo("b.jpg")]
    store.save(
        result("r", "a.jpg"), photos, {"score": np.array([0.91234, np.nan], dtype=np.float32)}
    )
    rows = store.signals("r")
    assert (rows[0]["picked"], rows[0]["score"]) == ("1.1", "0.9123")
    assert (rows[1]["picked"], rows[1]["score"]) == ("", "")
    assert [r["path"] for r in store.explain("r", "b.jpg")] == [str(photos[1].path)]


def test_missing_run_says_how_to_find_runs(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    assert store.recent() == []
    with pytest.raises(RunNotFoundError, match="calico runs list"):
        store.load("nope")
    with pytest.raises(RunNotFoundError):
        store.signals("nope")


def test_signals_of_an_unknown_run_say_the_run_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(RunNotFoundError, match="No run 'nosuchrun'"):
        RunStore(tmp_path).signals("nosuchrun")


def from_another_version(store: RunStore, run_id: str) -> None:
    """Make a saved run look like one written by a Calico with a request field this one
    does not know (a run saved on another branch or a newer version)."""
    path = store.path(run_id) / "result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["request"]["from_a_newer_calico"] = True
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_run_from_another_version_says_so_when_loaded(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    store.save(result("20260101-000000-aaaa", "a.jpg"), [f.photo("a.jpg")], {})
    from_another_version(store, "20260101-000000-aaaa")
    with pytest.raises(RunUnreadableError, match=r"20260101-000000-aaaa.*another version"):
        store.load("20260101-000000-aaaa")


def test_runs_that_cannot_be_read_are_left_out_of_the_list_and_named(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    photos = [f.photo("a.jpg")]
    store.save(result("20260101-000000-aaaa", "a.jpg"), photos, {})
    store.save(result("20260102-000000-bbbb", "a.jpg"), photos, {})
    from_another_version(store, "20260102-000000-bbbb")
    assert [s.run_id for s in store.recent()] == ["20260101-000000-aaaa"]
    ((run_id, why),) = store.unreadable()
    assert run_id == "20260102-000000-bbbb"
    assert "from_a_newer_calico" in why


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ({"dropped": "1"}, "no capture time"),
        ({"dropped": "2"}, "a copy of a frame saved twice"),
        ({"kind_probability": "0.1200", "score": ""}, "not the kind asked for (probability 0.12)"),
        ({"burst_lead": "0.0000"}, "a stronger frame of its burst was shown"),
        ({"view_lead": "0.0000", "burst_lead": "0.0000"}, "a stronger frame of its view was shown"),
        ({"burst_lead": "1.0000", "strength_rank": "31.0000"}, "31st strongest moment"),
        ({"uses": "0.0000", "in_working_set": "1.0000"}, "in no story chosen"),
        ({"score": "0.4000"}, None),
    ],
)
def test_runs_explain_says_why_a_photo_was_left_out(
    row: dict[str, str], reason: str | None
) -> None:
    found = left_out_because({"path": "/a/x.jpg", "picked": "", **row})
    assert (found is None) if reason is None else (reason in str(found))


def test_a_second_run_with_the_same_id_never_replaces_the_first(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    photos = [f.photo("a.jpg")]
    store.save(result("20260101-000000-aaaa", "a.jpg"), photos, {})
    with pytest.raises(CalicoError, match="already exists"):
        store.save(result("20260101-000000-aaaa", "b.jpg"), photos, {})
    assert store.load("20260101-000000-aaaa").groups[0].picks[0].path.name == "a.jpg"


def test_a_saved_run_leaves_no_temporary_files(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    run_dir = store.save(result("20260101-000000-aaaa", "a.jpg"), [f.photo("a.jpg")], {})
    assert sorted(p.name for p in run_dir.iterdir()) == ["result.json", "signals.csv"]
