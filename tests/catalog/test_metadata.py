from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from calico.catalog.metadata import file_facts, image_facts
from tests.factories import EXIF_IFD, EXIF_MAKE, EXIF_MODEL, save_jpeg, solid

if TYPE_CHECKING:
    from pathlib import Path


def test_content_key_depends_on_bytes_not_path(tmp_path: Path) -> None:
    a, b = tmp_path / "a.bin", tmp_path / "sub" / "b.bin"
    b.parent.mkdir()
    a.write_bytes(b"hello" * 100)
    b.write_bytes(b"hello" * 100)
    assert file_facts(a).content_key == file_facts(b).content_key
    assert file_facts(a).size == 500


def test_content_key_changes_with_size_head_or_tail(tmp_path: Path) -> None:
    big = bytes(range(256)) * 2000
    base = tmp_path / "base"
    base.write_bytes(big)
    key = file_facts(base).content_key
    variants = {
        "size": big + b"x",
        "head": b"Z" + big[1:],
        "tail": big[:-1] + b"Z",
    }
    for name, data in variants.items():
        path = tmp_path / name
        path.write_bytes(data)
        assert file_facts(path).content_key != key, name


def test_file_facts_reports_mtime(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"x")
    assert file_facts(path).mtime == path.stat().st_mtime


def test_image_facts_full_exif(tmp_path: Path) -> None:
    path = save_jpeg(
        solid((1, 2, 3), (60, 30)),
        tmp_path / "x.jpg",
        captured_at=datetime(2021, 3, 4, 5, 6, 7),
        camera=("Canon", "EOS R5"),
    )
    with Image.open(path) as img:
        facts = image_facts(img)
    assert facts.captured_at == datetime(2021, 3, 4, 5, 6, 7)
    assert (facts.width, facts.height) == (60, 30)
    assert facts.camera == "Canon EOS R5"


def test_image_facts_without_exif(tmp_path: Path) -> None:
    path = save_jpeg(solid((1, 2, 3), (60, 30)), tmp_path / "x.jpg")
    with Image.open(path) as img:
        facts = image_facts(img)
    assert facts.captured_at is None
    assert facts.camera is None


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_dimensions_swap_for_rotated_orientations(tmp_path: Path, orientation: int) -> None:
    path = save_jpeg(solid((1, 2, 3), (60, 30)), tmp_path / "x.jpg", orientation=orientation)
    with Image.open(path) as img:
        facts = image_facts(img)
    assert (facts.width, facts.height) == (30, 60)


def test_dimensions_unchanged_for_orientation_3(tmp_path: Path) -> None:
    path = save_jpeg(solid((1, 2, 3), (60, 30)), tmp_path / "x.jpg", orientation=3)
    with Image.open(path) as img:
        facts = image_facts(img)
    assert (facts.width, facts.height) == (60, 30)


def make(exif_ifd: dict[int, object], ifd0: dict[int, object] | None = None) -> Image.Image:
    img = Image.new("RGB", (10, 10))
    exif = Image.Exif()
    for tag, value in (ifd0 or {}).items():
        exif[tag] = value
    for tag, value in exif_ifd.items():
        exif.get_ifd(EXIF_IFD)[tag] = value
    img.info["exif"] = exif.tobytes()
    reopened = Image.new("RGB", (10, 10))
    reopened.getexif().load(img.info["exif"])
    return reopened


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2020:01:02 03:04:05", datetime(2020, 1, 2, 3, 4, 5)),
        ("2020-01-02 03:04:05", datetime(2020, 1, 2, 3, 4, 5)),
        ("2020:01:02 03:04:05\x00", datetime(2020, 1, 2, 3, 4, 5)),
        ("  2020:01:02 03:04:05.123 +01:00 ", datetime(2020, 1, 2, 3, 4, 5)),
        ("0000:00:00 00:00:00", None),
        ("garbage", None),
    ],
)
def test_datetime_parsing(value: str, expected: datetime | None) -> None:
    assert image_facts(make({0x9003: value})).captured_at == expected


def test_datetime_precedence_and_fallthrough() -> None:
    img = make(
        {0x9003: "0000:00:00 00:00:00", 0x9004: "2019:05:06 07:08:09"},
        {0x0132: "2018:01:01 01:01:01"},
    )
    assert image_facts(img).captured_at == datetime(2019, 5, 6, 7, 8, 9)
    only_ifd0 = make({}, {0x0132: "2018:01:01 01:01:01"})
    assert image_facts(only_ifd0).captured_at == datetime(2018, 1, 1, 1, 1, 1)
    original_wins = make({0x9003: "2020:01:01 00:00:00", 0x9004: "2019:01:01 00:00:00"})
    assert image_facts(original_wins).captured_at == datetime(2020, 1, 1)


@pytest.mark.parametrize(
    ("make_", "model", "expected"),
    [
        ("Canon", "EOS R5", "Canon EOS R5"),
        ("NIKON CORPORATION", "NIKON D850", "NIKON CORPORATION NIKON D850"),
        ("Apple", "Apple iPhone 12", "Apple iPhone 12"),
        ("  Sony\x00", " A7 \x00", "Sony A7"),
        ("Canon", None, "Canon"),
        (None, "X100", "X100"),
        (None, None, None),
        ("", " ", None),
    ],
)
def test_camera_name(make_: str | None, model: str | None, expected: str | None) -> None:
    ifd0: dict[int, object] = {}
    if make_ is not None:
        ifd0[EXIF_MAKE] = make_
    if model is not None:
        ifd0[EXIF_MODEL] = model
    assert image_facts(make({}, ifd0)).camera == expected


def test_bad_exif_never_raises() -> None:
    class Broken(Image.Image):
        def getexif(self) -> Image.Exif:
            raise OSError("corrupt")

    broken = Broken()
    broken._size = (7, 5)
    facts = image_facts(broken)
    assert facts.captured_at is None
    assert (facts.width, facts.height) == (7, 5)


def test_content_key_sees_changes_in_the_middle(tmp_path: Path) -> None:
    data = bytes(range(256)) * 2000
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(data)
    mid = len(data) // 2
    b.write_bytes(data[:mid] + b"Z" + data[mid + 1 :])
    assert file_facts(a).size == file_facts(b).size
    assert file_facts(a).content_key != file_facts(b).content_key


@pytest.mark.parametrize(
    ("tag", "minutes"), [("+01:00", 60), ("-04:00", -240), ("+05:30", 330), ("bad", None)]
)
def test_the_time_zone_is_read_as_minutes_from_utc(
    tmp_path: Path, tag: str, minutes: int | None
) -> None:
    path = save_jpeg(
        solid((9, 9, 9)),
        tmp_path / "a.jpg",
        captured_at=datetime(2026, 2, 26, 13, 2),
        utc_offset=tag,
    )
    with Image.open(path) as image:
        assert image_facts(image).utc_offset == minutes


def test_without_a_time_zone_tag_the_zone_is_unknown(tmp_path: Path) -> None:
    path = save_jpeg(solid((9, 9, 9)), tmp_path / "a.jpg", captured_at=datetime(2026, 2, 26))
    with Image.open(path) as image:
        assert image_facts(image).utc_offset is None


@pytest.mark.parametrize(
    "value", ["0001:01:01 00:30:00", "1700:06:01 12:00:00", "2999:01:01 00:00:00"]
)
def test_an_implausible_capture_date_counts_as_undated(value: str) -> None:
    """A damaged tag: kept, it would put the photo centuries away. No photograph is older
    than 1826, and none is taken next year."""
    assert image_facts(make({0x9003: value})).captured_at is None


def test_a_historic_scan_keeps_its_original_date() -> None:
    """Scanning software often records when the original was taken."""
    assert image_facts(make({0x9003: "1885:06:01 12:00:00"})).captured_at == datetime(
        1885, 6, 1, 12
    )
