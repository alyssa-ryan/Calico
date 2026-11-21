from __future__ import annotations

from datetime import datetime

from calico.selection.clock import on_one_clock
from tests.factories import photo

NOON = datetime(2026, 2, 26, 12, 0)


def test_photos_with_their_time_zone_are_put_on_utc() -> None:
    canon = photo(
        "canon.jpg", captured_at=datetime(2026, 2, 26, 8, 48), camera="Canon", utc_offset=-240
    )
    phone = photo(
        "phone.jpg", captured_at=datetime(2026, 2, 26, 13, 2), camera="iPhone", utc_offset=60
    )
    assert on_one_clock([canon, phone]) == [datetime(2026, 2, 26, 12, 48), NOON.replace(minute=2)]


def test_an_untagged_photo_takes_its_cameras_zone_in_that_folder() -> None:
    tagged = photo("a.jpg", captured_at=NOON, camera="Canon", utc_offset=-240)
    export = photo("b.jpg", captured_at=NOON, camera="Canon")
    assert on_one_clock([tagged, export])[1] == datetime(2026, 2, 26, 16, 0)


def test_an_untagged_camera_takes_the_folders_zone_when_the_folder_has_one() -> None:
    phone = photo("a.jpg", captured_at=NOON, camera="iPhone", utc_offset=60)
    fuji = photo("b.jpg", captured_at=NOON, camera="Fuji")
    assert on_one_clock([phone, fuji])[1] == datetime(2026, 2, 26, 11, 0)


def test_with_two_zones_in_a_folder_an_untagged_camera_keeps_its_own_time() -> None:
    canon = photo("a.jpg", captured_at=NOON, camera="Canon", utc_offset=-240)
    phone = photo("b.jpg", captured_at=NOON, camera="iPhone", utc_offset=60)
    fuji = photo("c.jpg", captured_at=NOON, camera="Fuji")
    assert on_one_clock([canon, phone, fuji])[2] == NOON


def test_zones_are_not_borrowed_from_another_folder() -> None:
    tagged = photo("a.jpg", captured_at=NOON, camera="Canon", utc_offset=-240, folder="/trip1")
    other = photo("b.jpg", captured_at=NOON, camera="Canon", folder="/trip2")
    assert on_one_clock([tagged, other])[1] == NOON


def test_undated_photos_stay_undated() -> None:
    assert on_one_clock([photo("a.jpg", camera="Canon", utc_offset=60)]) == [None]


def test_a_date_at_the_edge_of_the_calendar_keeps_its_own_clock() -> None:
    """Moving it to UTC would leave the calendar; it is not worth a crash."""
    edge = photo("edge.jpg", captured_at=datetime(1, 1, 1, 0, 30), camera="Canon", utc_offset=60)
    assert on_one_clock([edge]) == [datetime(1, 1, 1, 0, 30)]
