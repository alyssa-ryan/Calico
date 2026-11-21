"""Counting what was tried, so a best-of-N is read as one (plan, Workstream 7: A10-3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from calico.experiments import Tried, record, tried_so_far

if TYPE_CHECKING:
    from pathlib import Path

AT = datetime(2026, 10, 29, 16, tzinfo=UTC)


def test_each_setting_counts_once_however_often_it_is_measured(tmp_path: Path) -> None:
    log = tmp_path / "experiments.jsonl"
    assert tried_so_far(log, "taste") == Tried(settings=0, holdout_looks=0)
    record(log, "taste", ["clip l2=1000"], holdout=False, at=AT)
    record(log, "taste", ["clip l2=1000", "clip l2=30"], holdout=False, at=AT)
    assert record(log, "stories", ["0.8 0.25"], holdout=False, at=AT) == Tried(1, 0)
    assert tried_so_far(log, "taste") == Tried(settings=2, holdout_looks=0)


def test_looks_at_the_hold_out_are_counted_apart(tmp_path: Path) -> None:
    log = tmp_path / "experiments.jsonl"
    record(log, "taste", ["clip l2=1000"], holdout=False, at=AT)
    record(log, "taste", ["clip l2=1000"], holdout=True, at=AT)
    assert record(log, "taste", ["clip l2=30"], holdout=True, at=AT) == Tried(1, 2)


def test_looks_are_counted_at_the_hold_out_drawn_last(tmp_path: Path) -> None:
    # A hold-out retired and drawn again starts with no looks: the old one's were at other places.
    log = tmp_path / "experiments.jsonl"
    record(log, "taste", ["a"], holdout=True, at=AT)
    record(log, "taste", ["b"], holdout=False, at=AT + timedelta(days=5))
    redrawn = AT + timedelta(days=12)
    assert tried_so_far(log, "taste", since=redrawn) == Tried(settings=1, holdout_looks=0)
    later = redrawn + timedelta(days=1)
    again = record(log, "taste", ["a"], holdout=True, at=later, since=redrawn)
    assert again == Tried(settings=1, holdout_looks=1)
    assert tried_so_far(log, "taste") == Tried(settings=1, holdout_looks=2)  # none was erased


def test_a_damaged_line_is_skipped(tmp_path: Path) -> None:
    log = tmp_path / "experiments.jsonl"
    record(log, "taste", ["a"], holdout=False, at=AT)
    with log.open("a", encoding="utf-8") as fh:
        fh.write('{"command": "taste", "tri\n')
    record(log, "taste", ["b"], holdout=False, at=AT)
    assert tried_so_far(log, "taste").settings == 2
