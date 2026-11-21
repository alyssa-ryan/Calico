from __future__ import annotations

import fcntl
from typing import TYPE_CHECKING

import pytest

from calico.storage import append_lines, create_exclusive, locked, write_atomic

if TYPE_CHECKING:
    from pathlib import Path


def test_an_atomic_write_replaces_the_file_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "set.json"
    write_atomic(target, "first")
    write_atomic(target, b"second")
    assert target.read_text() == "second"
    assert [p.name for p in target.parent.iterdir()] == ["set.json"]


def test_an_exclusive_create_never_replaces_an_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "set.json"
    create_exclusive(target, "mine")
    with pytest.raises(FileExistsError):
        create_exclusive(target, "someone else's")
    assert target.read_text() == "mine"
    assert [p.name for p in tmp_path.iterdir()] == ["set.json"]


def test_the_lock_keeps_out_another_process_until_it_is_released(tmp_path: Path) -> None:
    """flock locks belong to an open file, so a second open stands in for a second process."""
    lock = tmp_path / ".lock"
    with locked(lock), lock.open("a") as other, pytest.raises(BlockingIOError):
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with lock.open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_a_line_is_appended_whole_and_ends_the_file(tmp_path: Path) -> None:
    path = tmp_path / "journal" / "2026-10.jsonl"
    append_lines(path, ['{"a": 1}'])
    append_lines(path, ['{"a": 2}', '{"a": 3}'])
    append_lines(path, [])
    assert path.read_text(encoding="utf-8") == '{"a": 1}\n{"a": 2}\n{"a": 3}\n'


def test_a_line_after_a_torn_one_starts_on_its_own(tmp_path: Path) -> None:
    path = tmp_path / "2026-10.jsonl"
    path.write_text('{"a": 1}\n{"a": ', encoding="utf-8")  # a crash mid-write
    append_lines(path, ['{"a": 2}'])
    assert path.read_text(encoding="utf-8").splitlines() == ['{"a": 1}', '{"a": ', '{"a": 2}']


def test_each_line_must_be_one_line(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one line"):
        append_lines(tmp_path / "j.jsonl", ["a", "b\nc"])
    assert not (tmp_path / "j.jsonl").exists()
