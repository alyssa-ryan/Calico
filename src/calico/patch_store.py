"""Scratch space for comparing photos with each other part by part (Recurring views).

Each photo's patches are read once per run into one file in the Calico home: 1.5 MB a
photo, so a large place needs several GB for the length of the run. The file is sparse,
so a disk that fills mid-run would crash the process on a page write rather than raise;
the space is checked before the file is made. A folder is named by the process that
made it, so one left behind by a Calico that was killed (``kill``, launchd, power loss:
nothing removes it then) is swept by the next one to start. Audit 28 October 2026, A01-9.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self

import numpy as np

from calico.errors import DiskSpaceError

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

PREFIX = "patches-"
HEADROOM = 1.1
"""Free space asked for beyond the file itself, so the run does not leave the disk full."""
_OWNED = re.compile(rf"^{PREFIX}(?P<pid>\d+)-")


class _Usage(Protocol):
    @property
    def free(self) -> int: ...


type DiskUsage = Callable[[Path], _Usage]
"""``shutil.disk_usage``, or a stand-in for it."""


class PatchStore:
    """Each photo's patches for one run's pair comparisons, by row. The file is made on the
    first ``put``, when the size of a photo's patches is known; a row never put reads as
    ``None`` (the photo could not be read). The folder is removed when the run ends, however
    it ends, short of the process being killed."""

    def __init__(self, home: Path, rows: int, *, disk_usage: DiskUsage = shutil.disk_usage) -> None:
        self._home = home
        self._rows = rows
        self._disk_usage = disk_usage
        self._folder: Path | None = None
        self._store: np.memmap[Any, np.dtype[np.float16]] | None = None
        self._written = np.zeros(rows, dtype=bool)

    def __enter__(self) -> Self:
        self._home.mkdir(parents=True, exist_ok=True)
        self._folder = Path(tempfile.mkdtemp(dir=self._home, prefix=f"{PREFIX}{os.getpid()}-"))
        return self

    def __exit__(self, *exc: object) -> None:
        self._store = None  # unmapped before its folder goes
        if self._folder is not None:
            shutil.rmtree(self._folder, ignore_errors=True)
            self._folder = None

    def put(self, row: int, patches: NDArray[np.floating[Any]]) -> None:
        if self._store is None:
            self._store = self._open(patches.shape)
        self._store[row] = patches
        self._written[row] = True

    def get(self, row: int) -> NDArray[np.float16] | None:
        if self._store is None or not self._written[row]:
            return None
        return np.asarray(self._store[row])

    def _open(self, shape: tuple[int, ...]) -> np.memmap[Any, np.dtype[np.float16]]:
        if self._folder is None:
            raise RuntimeError("PatchStore is used outside its with block.")
        need = round(self._rows * int(np.prod(shape)) * np.dtype(np.float16).itemsize * HEADROOM)
        free = self._disk_usage(self._home).free
        if free < need:
            raise DiskSpaceError(
                f"Comparing {self._rows} photos part by part needs {size_text(need)} of scratch "
                f"space in {self._home}, but only {size_text(free)} is free. Free some space on "
                "that disk, or choose fewer folders."
            )
        return np.lib.format.open_memmap(
            self._folder / "patches.npy",
            mode="w+",
            dtype=np.float16,
            shape=(self._rows, *shape),
        )


def scratch_folders(home: Path) -> list[Path]:
    """Every scratch folder in ``home``, in use or left behind."""
    if not home.is_dir():
        return []
    return sorted(p for p in home.glob(f"{PREFIX}*") if p.is_dir())


def owner(folder: Path) -> int | None:
    """The process that made ``folder``; None for a folder named by an older Calico."""
    found = _OWNED.match(folder.name)
    return int(found["pid"]) if found else None


def running(pid: int) -> bool:
    """Whether a process ``pid`` exists (one of another user's counts)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep(home: Path, alive: Callable[[int], bool] = running) -> list[Path]:
    """Remove the scratch folders whose process is gone, and return them. A folder an older
    Calico named (no process in its name) is left alone: that Calico may still be running."""
    removed = []
    for folder in scratch_folders(home):
        pid = owner(folder)
        if pid is None or alive(pid):
            continue
        shutil.rmtree(folder, ignore_errors=True)
        logger.info("removed scratch space a stopped Calico left behind: %s", folder)
        removed.append(folder)
    return removed


def folder_size(folder: Path) -> int:
    """Bytes the folder's files take on disk (the scratch file is sparse)."""
    total = 0
    for path in folder.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_blocks * 512
        except OSError:
            continue  # removed while counted
    return total


def size_text(size: int) -> str:
    """'70 kB', '1.5 MB', '3.2 GB': decimal units, as the Finder shows them."""
    if size < 1000:
        return f"{size} bytes"
    value, units = size / 1000, ["kB", "MB", "GB", "TB"]
    while value >= 1000 and len(units) > 1:
        value /= 1000
        units.pop(0)
    return f"{value:.2g} {units[0]}" if value < 10 else f"{value:.0f} {units[0]}"
