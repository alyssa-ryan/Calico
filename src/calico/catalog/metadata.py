"""Facts about a file (cheap, from the bytes) and about an image (from its EXIF)."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image

_CHUNK = 1024 * 1024
_EXIF_IFD = 0x8769
_DATETIME_TAGS = ((_EXIF_IFD, 0x9003), (_EXIF_IFD, 0x9004), (None, 0x0132))
_MAKE, _MODEL, _ORIENTATION = 0x010F, 0x0110, 0x0112
_ROTATED = frozenset({5, 6, 7, 8})
logger = logging.getLogger(__name__)

FIRST_PHOTOGRAPH = 1826
"""No capture date is older (Niépce, View from the Window at Le Gras)."""
_DATETIME_RE = re.compile(r"(\d{4})[:-](\d{2})[:-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})")
_OFFSET_TAGS = (0x9011, 0x9010)  # OffsetTimeOriginal, OffsetTime (Exif IFD)
_OFFSET_RE = re.compile(r"^([+-])(\d{2}):(\d{2})$")


@dataclass(frozen=True)
class FileFacts:
    size: int
    mtime: float
    content_key: str


@dataclass(frozen=True)
class ImageFacts:
    captured_at: datetime | None
    width: int
    height: int
    camera: str | None
    utc_offset: int | None = None
    """Minutes east of UTC, from the time zone the camera wrote, if it wrote one."""


def file_facts(path: Path) -> FileFacts:
    """Size, mtime and a content key: blake2b of the whole file.

    The file is read in full anyway to decode it, and hashing everything means
    two different photos can never share a key (and so silently share features),
    which sampling head and tail would allow for edits in the middle.
    """
    stat = path.stat()
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return FileFacts(size=stat.st_size, mtime=stat.st_mtime, content_key=digest.hexdigest())


def image_facts(image: Image.Image) -> ImageFacts:
    """Capture time, oriented size and camera. Bad EXIF degrades to ``None``, never raises."""
    width, height = image.size
    try:
        exif = image.getexif()
        if exif.get(_ORIENTATION) in _ROTATED:
            width, height = height, width
        return ImageFacts(_captured_at(exif), width, height, _camera(exif), _utc_offset(exif))
    except (OSError, ValueError, TypeError, SyntaxError, KeyError):
        return ImageFacts(None, width, height, None)


def _clean(value: object) -> str:
    return str(value).replace("\x00", "").strip() if value is not None else ""


def _captured_at(exif: Image.Exif) -> datetime | None:
    for ifd, tag in _DATETIME_TAGS:
        source = exif.get_ifd(ifd) if ifd is not None else exif
        parsed = _parse_datetime(_clean(source.get(tag)))
        if parsed is not None:
            return parsed
    return None


def _parse_datetime(text: str) -> datetime | None:
    """The date and time in an EXIF date tag, or None if it holds none or one no photo can
    have: before the first photograph (1826) or later than next year, which is a damaged
    tag. A historic scan keeps the original's date, which scanning software often records."""
    match = _DATETIME_RE.match(text)
    if match is None:
        return None
    try:
        year, month, day, hour, minute, second = (int(g) for g in match.groups())
        when = datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None
    if not FIRST_PHOTOGRAPH <= year <= datetime.now().year + 1:
        logger.warning("EXIF date %r is not a capture date; the photo counts as undated", text)
        return None
    return when


def _utc_offset(exif: Image.Exif) -> int | None:
    """The capture time's zone ("+01:00") as minutes east of UTC; phones write the zone
    they are in, many cameras the one they were set to, older ones none."""
    ifd = exif.get_ifd(_EXIF_IFD)
    for tag in _OFFSET_TAGS:
        match = _OFFSET_RE.match(_clean(ifd.get(tag)))
        if match:
            sign = -1 if match.group(1) == "-" else 1
            return sign * (int(match.group(2)) * 60 + int(match.group(3)))
    return None


def _camera(exif: Image.Exif) -> str | None:
    make, model = _clean(exif.get(_MAKE)), _clean(exif.get(_MODEL))
    if model and make and not model.lower().startswith(make.lower()):
        return f"{make} {model}"
    return model or make or None
