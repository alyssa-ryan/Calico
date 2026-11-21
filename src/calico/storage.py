"""Writing files that cannot be recomputed: ratings, labels, runs and trained models.

A crash, a full disk or a second Calico process (another ``calico serve``, the CLI beside
the web app) must never leave one of them half-written, replace one by accident, or lose
a change another process made at the same moment. Each helper covers one of those.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

_NO_LOCKS = frozenset({errno.ENOLCK, errno.ENOTSUP, errno.EOPNOTSUPP})
"""What a volume that cannot lock files (some network and FUSE mounts) answers."""


def write_atomic(path: Path, data: str | bytes) -> None:
    """Replace ``path`` with ``data``, so that a reader, or a crash at any moment, sees
    either the old file or the new one, never part of one."""
    tmp = _written_beside(path, data)
    try:
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _sync_folder(path.parent)


def create_exclusive(path: Path, data: str | bytes) -> None:
    """Write ``path``, which must not exist yet: ``FileExistsError`` otherwise, and nothing
    is replaced. Linking a finished file into place keeps the create atomic too."""
    tmp = _written_beside(path, data)
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise
    except OSError:  # a volume without hard links (exFAT): exclusive, written in place
        with path.open("xb") as fh:
            fh.write(tmp.read_bytes())
            _sync(fh.fileno())
    finally:
        tmp.unlink(missing_ok=True)
    _sync_folder(path.parent)


def append_lines(path: Path, lines: Sequence[str]) -> None:
    """Add ``lines`` to the end of ``path`` in one ``write`` (``O_APPEND``), flushed to disk:
    appends never overwrite one another, so no process can lose another's lines. A last line
    a crash left unfinished is closed first, so the first of these is never glued to it.
    Hold ``locked`` around it when two processes may append at once."""
    if any("\n" in line or "\r" in line for line in lines):
        raise ValueError("append_lines writes one line per item; an item holds a line break.")
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        size = os.fstat(fd).st_size
        torn = size > 0 and os.pread(fd, 1, size - 1) != b"\n"
        text = ("\n" if torn else "") + "".join(line + "\n" for line in lines)
        os.write(fd, text.encode("utf-8"))
        _sync(fd)
    finally:
        os.close(fd)


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold ``path`` as a lock across processes for the duration (``flock``), so a
    read-modify-write is never interleaved with another process's. On a volume that
    cannot lock, carry on unlocked and say so once: a lost race is better than a save that
    fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno not in _NO_LOCKS:
                raise
            _warn_unlocked(path)
            yield
            return
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


_warned: set[str] = set()


def _warn_unlocked(path: Path) -> None:
    if str(path.parent) not in _warned:
        _warned.add(str(path.parent))
        logger.warning(
            "%s cannot lock files, so two Calico processes changing it at once could lose "
            "a change. Keep the Calico home on this computer's own disk.",
            path.parent,
        )


def _written_beside(path: Path, data: str | bytes) -> Path:
    """``data`` in a uniquely named file in ``path``'s folder, flushed to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    raw = data.encode("utf-8") if isinstance(data, str) else data
    with tmp.open("xb") as fh:
        fh.write(raw)
        _sync(fh.fileno())
    return tmp


def _sync(fd: int) -> None:
    fh_sync = getattr(fcntl, "F_FULLFSYNC", None)  # macOS: fsync alone may stay in the drive
    if fh_sync is not None:
        with contextlib.suppress(OSError):
            fcntl.fcntl(fd, fh_sync)
            return
    os.fsync(fd)


def _sync_folder(folder: Path) -> None:
    """Make the rename itself durable; some volumes cannot open a folder, which is fine."""
    with contextlib.suppress(OSError):
        fd = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
