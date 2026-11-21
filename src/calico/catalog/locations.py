"""Which place a photo belongs to, decided by its folder and the collection's rules."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from calico.domain import place_key
from calico.errors import NotInCollectionError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from calico.settings import CollectionConfig


def location_for(
    collection: CollectionConfig, path: Path, aliases: Mapping[str, str] | None = None
) -> str:
    """The location name for ``path``.

    Photos directly in the root take the collection's name. Otherwise the folder
    ``location_depth`` levels down names the place (clamped to the deepest
    folder available). ``location_ignore`` text is removed from it, then
    ``location_pattern`` refines it when it matches, so "Old Mill (flooded) -
    March 2021" can still be "Old Mill". With ``location_deepest``, the deepest folder
    below that one whose cleaned name is not empty and not a ``location_not_places``
    name is taken instead. Runs of spaces become one, and finally ``aliases`` (from
    settings) rename misspelt places.
    """
    if not collection.contains(path):
        raise NotInCollectionError(f"{path} is not under {collection.name} ({collection.root})")
    folders = path.relative_to(collection.root).parts[:-1]
    if not folders:
        return collection.name
    depth = min(collection.location_depth, len(folders))
    if collection.location_deepest:
        for folder in reversed(folders[depth:]):
            if not _ignored_away(collection, folder) and not _organises(collection, folder):
                return _renamed(_name(collection, folder), aliases or {})
    return _renamed(_name(collection, folders[depth - 1]), aliases or {})


def _name(collection: CollectionConfig, folder: str) -> str:
    """The folder name without ``location_ignore`` text (unless that leaves nothing),
    refined by ``location_pattern``, spaces tidied."""
    name = folder if _ignored_away(collection, folder) else _without_ignored(collection, folder)
    if collection.location_pattern:
        match = re.search(collection.location_pattern, name)
        group = match.group("location") if match else None
        if group and group.strip():
            name = group
    return " ".join(name.split())


def _without_ignored(collection: CollectionConfig, folder: str) -> str:
    name = folder
    for pattern in collection.location_ignore:
        name = re.sub(pattern, " ", name)
    return name


def _ignored_away(collection: CollectionConfig, folder: str) -> bool:
    return not _without_ignored(collection, folder).strip()


def _organises(collection: CollectionConfig, folder: str) -> bool:
    """Whether the folder only organises photos (``location_not_places``), by its own
    name or its name once cleaned."""
    names = {folder.strip(), _name(collection, folder)}
    return any(
        re.fullmatch(pattern, name, re.IGNORECASE)
        for pattern in collection.location_not_places
        for name in names
    )


def _renamed(location: str, aliases: Mapping[str, str]) -> str:
    key = place_key(location)
    for old, new in aliases.items():
        if place_key(old) == key:
            return new
    return location
