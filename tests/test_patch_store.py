"""The scratch store a run's pair comparisons read patches from (audit A01-9)."""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest

from calico.errors import DiskSpaceError
from calico.patch_store import PatchStore, scratch_folders, size_text, sweep

if TYPE_CHECKING:
    from pathlib import Path


class Usage(NamedTuple):
    total: int
    used: int
    free: int


def patches(n: int, seed: int = 0) -> np.ndarray:
    """``n`` photos' patches: 4 patches of 8 dimensions each."""
    return np.random.default_rng(seed).normal(size=(n, 4, 8)).astype(np.float32)


def test_patches_put_are_read_back_and_a_photo_never_put_is_unreadable(tmp_path: Path) -> None:
    found = patches(2)
    with PatchStore(tmp_path, rows=3) as store:
        store.put(0, found[0])
        store.put(2, found[1])
        first, second, last = store.get(0), store.get(1), store.get(2)
        assert first is not None
        assert np.allclose(first, found[0], atol=1e-2)  # kept as float16
        assert second is None
        assert last is not None
        assert np.allclose(last, found[1], atol=1e-2)


def test_the_scratch_folder_is_named_by_process_and_removed_after(tmp_path: Path) -> None:
    with PatchStore(tmp_path, rows=1) as store:
        store.put(0, patches(1)[0])
        (folder,) = scratch_folders(tmp_path)
        assert folder.name.startswith(f"patches-{os.getpid()}-")
    assert scratch_folders(tmp_path) == []


def compare_then_fail(home: Path) -> None:
    with PatchStore(home, rows=1) as store:
        store.put(0, patches(1)[0])
        raise RuntimeError


def test_the_scratch_folder_is_removed_when_the_run_fails(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        compare_then_fail(tmp_path)
    assert scratch_folders(tmp_path) == []


def test_too_little_free_space_is_refused_saying_how_much_is_needed(tmp_path: Path) -> None:
    # 1,000 photos x 4 x 8 float16 = 64,000 bytes, plus a tenth to spare.
    short = Usage(total=10**9, used=10**9 - 60_000, free=60_000)
    with PatchStore(tmp_path, rows=1000, disk_usage=lambda _: short) as store:
        with pytest.raises(DiskSpaceError, match=r"needs 70 kB .* only 60 kB is free"):
            store.put(0, patches(1)[0])
        assert store.get(0) is None
    assert scratch_folders(tmp_path) == []


def test_enough_free_space_is_written(tmp_path: Path) -> None:
    roomy = Usage(total=10**9, used=0, free=71_000)
    with PatchStore(tmp_path, rows=1000, disk_usage=lambda _: roomy) as store:
        store.put(0, patches(1)[0])
        assert store.get(0) is not None


def test_a_folder_left_by_a_process_that_is_gone_is_swept(tmp_path: Path) -> None:
    stale = tmp_path / "patches-999999-abc"
    stale.mkdir()
    (stale / "patches.npy").write_bytes(b"x" * 10)
    mine = tmp_path / f"patches-{os.getpid()}-def"
    mine.mkdir()
    assert sweep(tmp_path, alive=lambda pid: pid == os.getpid()) == [stale]
    assert not stale.exists()
    assert mine.is_dir()  # in use by a process still running


def test_a_folder_named_by_an_older_calico_is_left_alone(tmp_path: Path) -> None:
    older = tmp_path / "patches-k2j3h4"  # tempfile's name: no process to check
    older.mkdir()
    assert sweep(tmp_path, alive=lambda _: False) == []
    assert older.is_dir()


def test_sweeping_a_home_that_does_not_exist_yet_does_nothing(tmp_path: Path) -> None:
    assert sweep(tmp_path / "nowhere") == []


def test_the_running_process_counts_as_alive(tmp_path: Path) -> None:
    mine = tmp_path / f"patches-{os.getpid()}-abc"
    mine.mkdir()
    assert sweep(tmp_path) == []
    shutil.rmtree(mine)


@pytest.mark.parametrize(
    ("size", "text"),
    [
        (0, "0 bytes"),
        (999, "999 bytes"),
        (70_400, "70 kB"),
        (1_500_000, "1.5 MB"),
        (3_200_000_000, "3.2 GB"),
    ],
)
def test_sizes_read_as_people_say_them(size: int, text: str) -> None:
    assert size_text(size) == text
