"""One spelling for every path Calico compares: the disk's own, in NFC.

A folder can reach Calico typed in other capitals (macOS volumes ignore case), in
decomposed Unicode (Finder hands out NFD) or through a link, while the catalog and the
collection roots keep one spelling. So every path from outside (a request, a setting, a
command line) is put into that spelling before it is compared with them.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path


def normalise(path: Path) -> Path:
    """``path`` with ``~`` and links resolved and each part spelt as the disk spells it, in
    NFC. A part the disk does not have (an unmounted drive, a folder not made yet) is kept
    as typed."""
    resolved = path.expanduser().resolve()
    spelt = Path(resolved.anchor)
    for part in resolved.parts[1:]:
        spelt = spelt / _on_disk(spelt, part)
    return Path(unicodedata.normalize("NFC", str(spelt)))


def _on_disk(folder: Path, part: str) -> str:
    """How the disk spells ``part`` in ``folder``: the entry whose name differs from it only
    in capitals or Unicode form and is the same file. On a volume that tells capitals
    apart, that is another file, so ``part`` is kept."""
    try:
        names = [entry.name for entry in folder.iterdir()]
    except OSError:
        return part
    if part in names:
        return part
    key = unicodedata.normalize("NFC", part).casefold()
    for name in names:
        if unicodedata.normalize("NFC", name).casefold() == key and _same(folder, part, name):
            return name
    return part


def _same(folder: Path, typed: str, name: str) -> bool:
    try:
        return (folder / typed).samefile(folder / name)
    except OSError:
        return False


def missing_folder(path: Path) -> str:
    """Why ``path`` cannot be read, said so it can be fixed: where a folder on the way is
    there under a name that differs only in spaces or capitals (a name ending in two
    spaces looks like the typed one), the path the disk has; otherwise that it does not
    exist or its drive is not mounted."""
    plain = f"{path} does not exist or is not mounted."
    found = Path(path.anchor)
    renamed: list[tuple[str, str]] = []
    for part in path.parts[1:]:
        if (found / part).exists():
            found = found / part
            continue
        near = _near(found, part)
        if near is None:
            return plain
        renamed.append((part, near))
        found = found / near
    if not renamed:
        return plain
    how = "; ".join(f"{name!r} has {_difference(typed, name)}" for typed, name in renamed)
    return (
        f"{path} does not exist, but {found} does: on disk, {how}. Use that path, in quotes "
        "so its spaces are kept."
    )


def _near(folder: Path, part: str) -> str | None:
    """The one entry of ``folder`` whose name is ``part`` but for spaces and capitals."""
    try:
        names = [entry.name for entry in folder.iterdir()]
    except OSError:
        return None
    key = _loose(part)
    near = [name for name in names if _loose(name) == key]
    return near[0] if len(near) == 1 else None


def _loose(name: str) -> str:
    return " ".join(unicodedata.normalize("NFC", name).split()).casefold()


def _difference(typed: str, name: str) -> str:
    """How the disk's ``name`` differs from the ``typed`` one, in words."""
    if typed.strip() != name.strip():
        return "other spaces or capitals"
    ends = []
    for where, strip in (("start", str.lstrip), ("end", str.rstrip)):
        extra = (len(name) - len(strip(name))) - (len(typed) - len(strip(typed)))
        if extra:
            spaces = f"space{'s' if abs(extra) != 1 else ''}"
            ends.append(
                f"{extra} {spaces} at the {where}"
                if extra > 0
                else f"{-extra} fewer {spaces} at the {where}"
            )
    return " and ".join(ends)
