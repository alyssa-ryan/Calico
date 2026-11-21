from __future__ import annotations

from datetime import datetime

from calico.domain import Evidence
from calico.selection.picks import make_pick
from tests.factories import photo


def test_pick_copies_the_photo_identity() -> None:
    photos = [photo("a.jpg", captured_at=datetime(2024, 1, 2))]
    pick = make_pick(photos, 0, 0.7, ())
    assert (pick.path, pick.content_key, pick.location) == (
        photos[0].path,
        photos[0].content_key,
        photos[0].location,
    )
    assert pick.captured_at == datetime(2024, 1, 2)
    assert pick.score == 0.7


def test_alternates_become_paths_in_order_and_set_burst_size() -> None:
    photos = [photo(f"{i}.jpg") for i in range(4)]
    pick = make_pick(photos, 0, 0.5, (), alternates=[3, 1])
    assert pick.alternates == (photos[3].path, photos[1].path)
    assert pick.burst_size == 3


def test_evidence_is_kept() -> None:
    ev = (Evidence(signal="x", value=0.5, weight=1.0),)
    assert make_pick([photo()], 0, 0.5, ev).evidence == ev
