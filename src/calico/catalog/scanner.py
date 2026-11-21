"""Find the image files under a folder.

The scan only lists; it never opens a file. A RAW next to its rendered JPEG is the
same photograph, so the JPEG represents it. A lone RAW is a photo when RAW files can
be decoded (``imaging.raw_support``), and otherwise something the user should hear
about.
"""

from __future__ import annotations

import fnmatch
import os
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from calico.errors import FolderNotFoundError
from calico.paths import missing_folder

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp"}
)
RAW_SUFFIXES = frozenset(
    {
        ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".sr2", ".srf", ".dng", ".raf", ".orf",
        ".rw2", ".rwl", ".pef", ".srw", ".x3f", ".3fr", ".fff",
    }
)  # fmt: skip
"""What LibRaw reads: Canon, Nikon, Sony, Adobe, Fujifilm, Olympus, Panasonic, Leica,
Pentax, Samsung, Sigma and Hasselblad."""


@dataclass(frozen=True)
class ScanResult:
    images: tuple[Path, ...]
    raw_without_rendered: tuple[Path, ...]
    unreadable: tuple[Path, ...] = ()
    """Directories that could not be listed; their contents are unknown, not absent."""
    links: tuple[Path, ...] = ()
    """Symlinked image files, skipped: a link can point outside every collection."""
    linked_dirs: tuple[Path, ...] = ()
    """Symlinked folders, not entered, for the same reason."""
    unsupported: tuple[tuple[str, int], ...] = ()
    """How many files of each other extension (lower-cased) were not read."""


def nfc_path(path: Path) -> Path:
    """Unicode NFC form of ``path``.

    macOS hands back decomposed (NFD) names while typed input is composed, so
    without one canonical form the same file would be two catalog rows.
    """
    return Path(unicodedata.normalize("NFC", str(path)))


def scan(
    folder: Path,
    exclude: Sequence[str] = (),
    *,
    raw: bool = False,
    skip: Collection[Path] = (),
    exclude_for: Callable[[Path], Sequence[str]] | None = None,
) -> ScanResult:
    """Recursively list decodable images and RAW files that have no rendered sibling;
    with ``raw``, those RAW files are images too. Folders in ``skip`` (the calico home)
    are never entered. ``exclude_for`` gives each folder's own patterns (its collection's:
    a nested collection excludes what it chooses), instead of ``exclude`` for all."""
    if not folder.is_dir():
        raise FolderNotFoundError(missing_folder(folder))
    images: list[Path] = []
    raws: list[Path] = []
    links: list[Path] = []
    unreadable: list[Path] = []
    linked_dirs: list[Path] = []
    unsupported: Counter[str] = Counter()
    walked = _walk(folder, exclude_for or (lambda _: exclude), unreadable, skip, linked_dirs)
    for path in walked:
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            (links if path.is_symlink() else images).append(path)
        elif suffix in RAW_SUFFIXES:
            raws.append(path)
        else:
            unsupported[suffix or "(no extension)"] += 1
    rendered = {(p.parent, p.stem.lower()) for p in images}
    lone = [p for p in raws if (p.parent, p.stem.lower()) not in rendered]
    if raw:
        images.extend(p for p in lone if not p.is_symlink())
        links.extend(p for p in lone if p.is_symlink())
        lone = []
    return ScanResult(
        images=tuple(sorted(images)),
        raw_without_rendered=tuple(sorted(lone)),
        unreadable=tuple(sorted(unreadable)),
        links=tuple(sorted(links)),
        linked_dirs=tuple(sorted(linked_dirs)),
        unsupported=tuple(sorted(unsupported.items())),
    )


def _excluded(name: str, exclude: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in exclude)


def _walk(
    folder: Path,
    exclude_for: Callable[[Path], Sequence[str]],
    unreadable: list[Path],
    skip: Collection[Path],
    linked_dirs: list[Path],
) -> list[Path]:
    """NFC paths of the files under ``folder``; skips excluded names, skipped folders and
    symlinked dirs."""
    skipped = {nfc_path(p) for p in skip}

    def on_error(error: OSError) -> None:
        if error.filename is not None:
            unreadable.append(nfc_path(Path(str(error.filename))))

    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(folder, onerror=on_error, followlinks=False):
        base = Path(dirpath)
        exclude = exclude_for(nfc_path(base))
        dirnames[:] = [
            d for d in dirnames if not _excluded(d, exclude) and nfc_path(base / d) not in skipped
        ]
        linked_dirs.extend(nfc_path(base / d) for d in dirnames if (base / d).is_symlink())
        found.extend(nfc_path(base / name) for name in filenames if not _excluded(name, exclude))
    return found
