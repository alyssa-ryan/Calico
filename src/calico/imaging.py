"""Image decoding shared by indexing and reference images.

One decode path means a photo gets identical pixels whether it is catalogued or
supplied as a reference, so their features are comparable.

RAW files open through LibRaw (``rawpy``, the ``raw`` extra) as the JPEG the camera
embedded: its own rendering, with the camera, capture time and orientation LibRaw writes
into it, so the rest of Calico treats it as any JPEG. A Canon CR3 embeds one at full size.
"""

from __future__ import annotations

import importlib
from io import BytesIO
from typing import TYPE_CHECKING, Any

import numpy as np
import pillow_heif
from PIL import Image, ImageOps

from calico.catalog.scanner import RAW_SUFFIXES
from calico.errors import ImageDecodeError

if TYPE_CHECKING:
    from pathlib import Path

# Panoramas and stitched scans are legitimate photographs; Pillow's default cap
# (~89M pixels) warns on them. 200M still stops decompression bombs.
Image.MAX_IMAGE_PIXELS = 200_000_000

THUMBNAIL_QUALITY = 85

RAW_PREVIEW_MIN_SIDE = 2048
"""A RAW's embedded preview is used when its longest side is at least this (the web
viewer shows 2048 px; features read 1024). Smaller, the RAW is developed at half size."""

RAW_HINT = "uv sync --extra raw"

_DATETIME, _ORIENTATION = 0x0132, 0x0112

_heif_registered = False
TWELVE_BIT_MAX = 4095.0
"""The brightest value a 12-bit scan stored in 16 bits can hold."""


def _register_heif() -> None:
    """Register the HEIF opener once; repeated registration is wasted work."""
    global _heif_registered  # noqa: PLW0603
    if not _heif_registered:
        pillow_heif.register_heif_opener()
        _heif_registered = True


def open_image(path: Path) -> Image.Image:
    """Open ``path`` lazily. Pixels are not decoded until the image is used (a RAW file
    is read when opened)."""
    if path.suffix.lower() in RAW_SUFFIXES:
        return _open_raw(path)
    _register_heif()
    try:
        return Image.open(path)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError(f"{path}: {exc}") from exc


def raw_support() -> str | None:
    """Which LibRaw decodes RAW files, or ``None`` without the ``raw`` extra."""
    rawpy = _rawpy()
    if rawpy is None:
        return None
    return "LibRaw " + ".".join(str(n) for n in rawpy.libraw_version)


def _rawpy() -> Any | None:
    try:
        return importlib.import_module("rawpy")
    except ImportError:
        return None


def _open_raw(path: Path) -> Image.Image:
    """The RAW's embedded preview; a RAW without a large one is developed at half size
    (upright already), keeping the preview's EXIF, or dated by LibRaw without one."""
    rawpy = _rawpy()
    if rawpy is None:
        raise ImageDecodeError(f"{path}: RAW files need RAW support: {RAW_HINT}")
    try:
        with rawpy.imread(str(path)) as raw:
            preview = _preview(raw, rawpy)
            if preview is not None and max(preview.size) >= RAW_PREVIEW_MIN_SIDE:
                return preview
            developed = Image.fromarray(
                raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
            )
            exif = preview.getexif() if preview is not None else Image.Exif()
            stamp = raw.other.timestamp
            if _DATETIME not in exif and stamp is not None:
                exif[_DATETIME] = stamp.strftime("%Y:%m:%d %H:%M:%S")
            exif[_ORIENTATION] = 1
            developed.info["exif"] = exif.tobytes()
            return developed
    except (rawpy.LibRawError, OSError, ValueError) as exc:
        raise ImageDecodeError(f"{path}: {exc}") from exc


def _preview(raw: Any, rawpy: Any) -> Image.Image | None:
    """The JPEG the camera embedded, if any."""
    try:
        thumb = raw.extract_thumb()
    except rawpy.LibRawError:
        return None
    if thumb.format != rawpy.ThumbFormat.JPEG:
        return None
    return Image.open(BytesIO(thumb.data))


def prepare(image: Image.Image, max_side: int) -> Image.Image:
    """An oriented, RGB copy no larger than ``max_side`` on its longest side.

    Extractors rely on this contract, so none of them re-implements it.
    """
    try:
        oriented = ImageOps.exif_transpose(image)
        rgb = _eight_bit(oriented).convert("RGB")
        if max(rgb.size) > max_side:
            scale = max_side / max(rgb.size)
            size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
            rgb = rgb.resize(size, Image.Resampling.LANCZOS)
        rgb.load()
    except (OSError, ValueError, SyntaxError) as exc:
        raise ImageDecodeError(f"could not decode image: {exc}") from exc
    return rgb


def _eight_bit(image: Image.Image) -> Image.Image:
    """A greyscale image wider than 8 bits (16-bit and 12-bit scans, 32-bit integer, float)
    as 8-bit, by its real range: ``convert`` clips every value above 255, so such a scan
    would be read as white. Others are returned as they are."""
    if image.mode not in ("I", "F") and not image.mode.startswith("I;16"):
        return image
    values = np.asarray(image, dtype=np.float64)
    if image.mode.startswith("I;16"):
        # 12-bit scanners store 0-4095 in 16 bits; a true 16-bit scan reaches higher.
        top = 65535.0 if values.max() > TWELVE_BIT_MAX else TWELVE_BIT_MAX
        low = 0.0
    else:
        low, top = (float(v) for v in np.percentile(values, [0.1, 99.9]))
    scaled = (values - low) / max(top - low, 1e-12) * 255
    return Image.fromarray(np.clip(scaled, 0, 255).astype(np.uint8), mode="L")


def decode(path: Path, max_side: int) -> tuple[Image.Image, Image.Image]:
    """Open once; return (opened image for metadata, prepared image for features)."""
    image = open_image(path)
    try:
        return image, prepare(image, max_side)
    except ImageDecodeError as exc:
        image.close()
        raise ImageDecodeError(f"{path}: {exc}") from exc


def as_thumbnail(image: Image.Image, side: int) -> Image.Image:
    """``image`` as Calico's thumbnails are: scaled to ``side`` and saved as JPEG, so a
    photo that has no thumbnail (an uploaded reference) is compared like one that has."""
    with Image.open(BytesIO(thumbnail_jpeg(image, side))) as thumb:
        return thumb.convert("RGB")


def thumbnail_jpeg(image: Image.Image, side: int) -> bytes:
    """JPEG bytes of ``image`` scaled so its longest side is at most ``side``."""
    thumb = image.convert("RGB")
    thumb.thumbnail((side, side), Image.Resampling.LANCZOS)
    buf = BytesIO()
    thumb.save(buf, "JPEG", quality=THUMBNAIL_QUALITY)
    return buf.getvalue()
