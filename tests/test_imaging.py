from __future__ import annotations

import sys
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from calico import imaging
from calico.catalog.metadata import image_facts
from calico.errors import ImageDecodeError
from tests.factories import fake_rawpy, save_jpeg, scene, solid, with_exif

if TYPE_CHECKING:
    from pathlib import Path


def test_open_image_reads_jpeg(tmp_path: Path) -> None:
    path = save_jpeg(scene(1), tmp_path / "a.jpg")
    with imaging.open_image(path) as img:
        assert img.size == (320, 240)


def test_open_image_missing_file_is_decode_error(tmp_path: Path) -> None:
    with pytest.raises(ImageDecodeError, match=r"nope\.jpg"):
        imaging.open_image(tmp_path / "nope.jpg")


def test_open_image_garbage_is_decode_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image at all")
    with pytest.raises(ImageDecodeError, match=r"bad\.jpg"):
        imaging.open_image(bad)


def test_prepare_downscales_keeping_aspect() -> None:
    out = imaging.prepare(solid((1, 2, 3), (400, 200)), 100)
    assert out.size == (100, 50)
    assert out.mode == "RGB"


def test_prepare_never_upscales() -> None:
    assert imaging.prepare(solid((1, 2, 3), (40, 20)), 100).size == (40, 20)


def test_prepare_converts_to_rgb() -> None:
    assert imaging.prepare(Image.new("L", (10, 10)), 100).mode == "RGB"


def test_prepare_applies_exif_orientation(tmp_path: Path) -> None:
    path = save_jpeg(solid((9, 9, 9), (60, 30)), tmp_path / "r.jpg", orientation=6)
    img, prepared = imaging.decode(path, 1000)
    assert img.size == (60, 30)
    assert prepared.size == (30, 60)


def test_prepare_wraps_decode_errors(tmp_path: Path) -> None:
    buf = BytesIO()
    scene(2, (200, 200)).save(buf, "JPEG")
    path = tmp_path / "trunc.jpg"
    path.write_bytes(buf.getvalue()[: len(buf.getvalue()) // 2])
    with pytest.raises(ImageDecodeError):
        imaging.decode(path, 100)


def test_thumbnail_jpeg_bounds_and_format() -> None:
    data = imaging.thumbnail_jpeg(solid((200, 10, 10), (400, 100)), 80)
    with Image.open(BytesIO(data)) as thumb:
        assert thumb.format == "JPEG"
        assert thumb.size == (80, 20)


# -- RAW files --------------------------------------------------------------------------


def _raw_file(tmp_path: Path, name: str = "IMG_4982.CR3") -> Path:
    path = tmp_path / name
    path.write_bytes(b"raw bytes")
    return path


def test_a_raw_file_opens_as_its_preview_with_its_camera_date_and_orientation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview = with_exif(
        scene(1, (3000, 2000)),
        camera=("Canon", "EOS RP"),
        captured_at=datetime(2021, 8, 6, 10, 31, 58),
        orientation=8,  # held upright: a portrait
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy(preview))
    with imaging.open_image(_raw_file(tmp_path)) as image:
        facts = image_facts(image)
        assert (facts.camera, facts.captured_at) == (
            "Canon EOS RP",
            datetime(2021, 8, 6, 10, 31, 58),
        )
        assert imaging.prepare(image, 1024).size == (683, 1024)


def test_a_raw_with_only_a_small_preview_is_developed_keeping_its_camera_and_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = with_exif(
        scene(1, (160, 120)), camera=("DJI", "FC3411"), captured_at=datetime(2022, 1, 20, 15, 42)
    )
    developed = scene(1, (2736, 1824))  # already upright: developing applies the rotation
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy(small, developed=developed))
    with imaging.open_image(_raw_file(tmp_path, "DJI_0061.DNG")) as image:
        facts = image_facts(image)
        assert (facts.width, facts.height) == (2736, 1824)
        assert (facts.camera, facts.captured_at) == ("DJI FC3411", datetime(2022, 1, 20, 15, 42))


def test_a_raw_without_a_preview_is_developed_and_dated_from_its_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rawpy = fake_rawpy(developed=scene(2, (2000, 1500)), timestamp=datetime(2019, 3, 2, 8, 0))
    monkeypatch.setitem(sys.modules, "rawpy", rawpy)
    with imaging.open_image(_raw_file(tmp_path)) as image:
        facts = image_facts(image)
        assert (facts.width, facts.captured_at) == (2000, datetime(2019, 3, 2, 8, 0))


def test_without_raw_support_a_raw_file_says_how_to_add_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "rawpy", None)
    assert imaging.raw_support() is None
    with pytest.raises(ImageDecodeError, match=r"uv sync --extra raw"):
        imaging.open_image(_raw_file(tmp_path))


def test_raw_support_names_libraw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy())
    assert imaging.raw_support() == "LibRaw 0.22.1"


def _ramp(mode: str, top: float) -> Image.Image:
    import numpy as np  # noqa: PLC0415 - only these tests build arrays

    values = np.tile(np.linspace(0, top, 256), (64, 1))
    dtype = {"I;16": np.uint16, "I": np.int32, "F": np.float32}[mode]
    image = Image.fromarray(values.astype(dtype))
    return image if mode != "I;16" or image.mode == "I;16" else image.convert("I;16")


@pytest.mark.parametrize(
    ("mode", "top"),
    [("I;16", 65535), ("I;16", 4095), ("I", 1_000_000), ("F", 1.0)],
    ids=["16-bit", "12-bit in 16", "32-bit integer", "float"],
)
def test_a_wide_greyscale_scan_keeps_its_tones(mode: str, top: float) -> None:
    """Historic scans are often 16-bit; converting them straight to RGB clips every value
    above 255 to white."""
    import numpy as np  # noqa: PLC0415

    prepared = imaging.prepare(_ramp(mode, top), 512)
    pixels = np.asarray(prepared, dtype=np.float32)
    assert prepared.mode == "RGB"
    assert 100 < pixels.mean() < 155
    assert pixels.min() < 10
    assert pixels.max() > 245


def test_an_eight_bit_photo_is_prepared_as_before() -> None:
    image = scene(3)
    assert imaging.prepare(image, 512).tobytes() == image.convert("RGB").tobytes()
