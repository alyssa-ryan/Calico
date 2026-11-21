"""``calico backup``: everything in the Calico home that cannot be recomputed, in one file.

The catalog, thumbnails and features can be rebuilt from the archive; your ratings, label
sets, saved runs and trained taste cannot (audit A10-12). A backup is one
``calico-backup-<date>-<time>.tar.gz`` holding:

* ``ratings/``, ``labels/`` (with ``references/`` and the trash), ``runs/`` (a 👍/👎 on a
  result needs its run's task) and ``models/``;
* ``calico.toml``, ``looks.toml``, ``taxonomies/``, ``holdout.toml`` and ``controls.toml`` (a
  hold-out or a pool drawn again would not be the same draw) and ``experiments.jsonl``
  (what was tried);
* the uploads that saved runs point to (uploads are pruned after a week);
* with ``with_catalog``, the catalog, copied with ``VACUUM INTO``, never as a file beside a
  running Calico (its WAL may hold what the file does not).

A ``manifest.json`` inside counts each part, so ``verify`` can tell a whole backup from a
cut-short one. The archive is written under a temporary name and linked into place when it
is complete, never over another; it may not be written inside the home or a collection
(rule 8). The newest backup is noted in ``<home>/last_backup.json`` for ``calico doctor``.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tarfile
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from calico import __version__
from calico.catalog.store import Catalog
from calico.errors import BackupError
from calico.paths import normalise
from calico.storage import write_atomic

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from calico.settings import Settings

PREFIX = "calico-backup-"
MANIFEST = "manifest.json"
NOTE = "last_backup.json"
FOLDERS = ("ratings", "labels", "runs", "models", "taxonomies")
FILES = ("calico.toml", "looks.toml", "holdout.toml", "controls.toml", "experiments.jsonl")
CATALOG = "catalog.sqlite"
_SKIPPED_SUFFIXES = (".lock", ".tmp")
"""Lock files and half-written temporaries: nothing a restore needs."""


@dataclass(frozen=True)
class Part:
    """One part of a backup: ``ratings``, ``labels``... or ``settings`` for the files at the
    top of the home."""

    name: str
    files: int
    bytes: int


@dataclass(frozen=True)
class Backup:
    path: Path
    parts: tuple[Part, ...]
    catalog: bool
    uploads_gone: int
    """Uploads saved runs point to that were pruned before this backup."""


@dataclass(frozen=True)
class Checked:
    """What a restore of a backup would bring back, and anything wrong with it."""

    path: Path
    created_at: datetime | None
    calico: str | None
    parts: tuple[Part, ...]
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class Noted:
    at: datetime
    path: Path


def backup(
    settings: Settings,
    dest: Path,
    *,
    with_catalog: bool = False,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Backup:
    """Write a backup of ``settings.home`` into the folder ``dest`` and note it."""
    home, folder = normalise(settings.home), normalise(dest)
    _refuse_inside(settings, home, folder)
    now = clock()
    files = dict(_kept(home))
    gone = 0
    for upload in _uploads(home, [c.root for c in settings.collections]):
        if upload.is_file():
            files[upload.relative_to(home).as_posix()] = upload
        else:
            gone += 1
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder, prefix=".calico-backup-") as scratch:
        if with_catalog and (home / CATALOG).is_file():
            copy = Path(scratch) / CATALOG
            with Catalog(home / CATALOG) as catalog:
                catalog.copy_to(copy)
            files[CATALOG] = copy
        parts = _parts({name: path.stat().st_size for name, path in files.items()})
        manifest = {
            "calico": __version__,
            "created_at": now.isoformat(),
            "home": str(home),
            "parts": {p.name: {"files": p.files, "bytes": p.bytes} for p in parts},
        }
        written = Path(scratch) / "backup.tar.gz"
        with written.open("wb") as fh:
            with tarfile.open(fileobj=fh, mode="w:gz") as tar:
                data = json.dumps(manifest, indent=1).encode()
                info = tarfile.TarInfo(MANIFEST)
                info.size, info.mtime = len(data), int(now.timestamp())
                tar.addfile(info, io.BytesIO(data))
                for name, path in sorted(files.items()):
                    tar.add(path, arcname=name, recursive=False)
            fh.flush()
            os.fsync(fh.fileno())
        path = _link_new(written, folder, f"{PREFIX}{now:%Y%m%d-%H%M%S}")
    write_atomic(home / NOTE, json.dumps({"at": now.isoformat(), "path": str(path)}, indent=1))
    return Backup(path=path, parts=parts, catalog=CATALOG in files, uploads_gone=gone)


def verify(path: Path) -> Checked:
    """Read every file in the backup at ``path``, and compare each part with its manifest.
    ``BackupError`` if the archive itself cannot be read."""
    sizes: dict[str, int] = {}
    manifest: dict[str, object] = {}
    problems = []
    try:
        with tarfile.open(path, mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    problems.append(f"{member.name} would be restored outside the home")
                    continue
                content = tar.extractfile(member)
                if content is None:
                    continue
                if member.name == MANIFEST:
                    manifest = json.loads(content.read())
                    continue
                read = 0
                while chunk := content.read(1 << 20):
                    read += len(chunk)
                sizes[member.name] = read
    except (OSError, tarfile.TarError, EOFError, ValueError) as exc:
        raise BackupError(
            f"{path} cannot be read as a Calico backup ({exc}). It may have been cut short "
            "while it was copied; use another backup, or make a new one: calico backup DEST"
        ) from exc
    parts = _parts(sizes)
    listed = manifest.get("parts")
    if not isinstance(listed, dict):
        problems.append(f"{MANIFEST} is missing, so the parts cannot be checked")
        listed = {}
    found = {p.name: p for p in parts}
    for name, want in sorted(listed.items()):
        have = found.get(name)
        files = have.files if have else 0
        if not isinstance(want, dict) or want.get("files") != files:
            expected = want.get("files") if isinstance(want, dict) else "?"
            problems.append(
                f"{name}: the manifest lists {expected} files, the backup holds {files}"
            )
    created = manifest.get("created_at")
    calico = manifest.get("calico")
    return Checked(
        path=path,
        created_at=datetime.fromisoformat(created) if isinstance(created, str) else None,
        calico=calico if isinstance(calico, str) else None,
        parts=parts,
        problems=tuple(problems),
    )


def last_backup(settings: Settings) -> Noted | None:
    """The newest backup Calico made of this home, or None (none, or a damaged note)."""
    try:
        data = json.loads((settings.home / NOTE).read_text(encoding="utf-8"))
        return Noted(at=datetime.fromisoformat(data["at"]), path=Path(data["path"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _refuse_inside(settings: Settings, home: Path, folder: Path) -> None:
    if folder.is_relative_to(home):
        raise BackupError(
            f"{folder} is inside the calico home ({home}): a backup there would be lost with "
            "it. Choose a folder outside it, ideally on another disk."
        )
    for collection in settings.collections:
        if folder.is_relative_to(normalise(collection.root)):
            raise BackupError(
                f"{folder} is inside the {collection.name!r} collection ({collection.root}), "
                "and Calico never writes into your archive. Choose a folder outside it."
            )


def _kept(home: Path) -> Iterator[tuple[str, Path]]:
    for name in FILES:
        if (home / name).is_file():
            yield name, home / name
    for top in FOLDERS:
        for path in sorted((home / top).rglob("*")) if (home / top).is_dir() else ():
            if (
                path.is_file()
                and not path.is_symlink()
                and not path.name.endswith(_SKIPPED_SUFFIXES)
            ):
                yield path.relative_to(home).as_posix(), path


def _uploads(home: Path, roots: Sequence[Path]) -> set[Path]:
    """The uploaded references saved runs point to, in this home. Read from the raw files, so
    a run another version saved still keeps its uploads; matched by the part after
    ``uploads/``, since a run saved before the home was copied or restored names the old
    home."""
    uploads = home / "uploads"
    found = set()
    for result in (home / "runs").glob("*/result.json"):
        with contextlib.suppress(OSError, ValueError, AttributeError, TypeError):
            request = json.loads(result.read_text(encoding="utf-8"))["request"]
            for ref in request.get("references", ()):
                parts = Path(ref).parts
                if "uploads" not in parts[:-1] or any(Path(ref).is_relative_to(r) for r in roots):
                    continue  # a photo from a collection, not an upload
                below = parts[len(parts) - parts[::-1].index("uploads") :]
                found.add(uploads.joinpath(*below))
    return found


def _parts(sizes: dict[str, int]) -> tuple[Part, ...]:
    files: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for name, size in sizes.items():
        part = (
            name.split("/", 1)[0] if "/" in name else ("catalog" if name == CATALOG else "settings")
        )
        files[part] += 1
        total[part] += size
    return tuple(Part(name, files[name], total[name]) for name in sorted(files))


def _link_new(written: Path, folder: Path, stem: str) -> Path:
    """Put ``written`` in ``folder`` under a new name: ``stem``, or ``stem-2``... if taken."""
    for n in range(1, 1000):
        path = folder / f"{stem}{'' if n == 1 else f'-{n}'}.tar.gz"
        try:
            os.link(written, path)
        except FileExistsError:
            continue
        except OSError:  # a volume without hard links (exFAT): copy, exclusively
            with path.open("xb") as out, written.open("rb") as src:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
        return path
    raise BackupError(f"{folder} already holds 999 backups named {stem}; choose another folder.")
