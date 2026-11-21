"""The frozen hold-out: drawn once, never tuned on (plan, Workstream 7: A10-3)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from calico.errors import HoldoutError
from calico.evaluation.holdout import HOLDOUT_SIZES, HoldOut, draw_holdout, near_places
from calico.holdout import read_holdout, write_holdout

if TYPE_CHECKING:
    from pathlib import Path

AT = datetime(2026, 10, 29, 16, tzinfo=UTC)
PLACES = [f"place {i}" for i in range(15)]
REFERENCES = [f"/archive/ref{i}.jpg" for i in range(19)]
SESSIONS = [f"2026101{i}-090000-stor" for i in range(10)]


def test_the_draw_takes_the_owners_sizes_repeatably_whatever_the_order_given() -> None:
    drawn = draw_holdout(PLACES, REFERENCES, SESSIONS, seed=7, at=AT)
    assert HOLDOUT_SIZES == {"places": 4, "references": 5, "story_sessions": 2}
    assert (len(drawn.places), len(drawn.references), len(drawn.story_sessions)) == (4, 5, 2)
    assert set(drawn.places) <= set(PLACES)
    again = draw_holdout(PLACES[::-1], REFERENCES[::-1], SESSIONS[::-1], seed=7, at=AT)
    assert again == drawn
    assert draw_holdout(PLACES, REFERENCES, SESSIONS, seed=8, at=AT) != drawn
    assert (drawn.seed, drawn.drawn_at) == (7, AT)


def test_never_more_than_a_third_is_held_out() -> None:
    drawn = draw_holdout(PLACES[:6], REFERENCES[:2], [], seed=7, at=AT)
    assert (len(drawn.places), len(drawn.references), drawn.story_sessions) == (2, 0, ())


def test_a_hold_out_is_written_once_and_read_back(tmp_path: Path) -> None:
    odd = ['Mill "old"', "Café \\ Ouest", "Kestrel 21 \u2013 Riverton"]  # quotes, accents, dashes
    drawn = draw_holdout(odd, [], [], seed=1, at=AT)
    assert drawn.places
    path = tmp_path / "holdout.toml"
    assert read_holdout(path) is None
    write_holdout(path, drawn)
    assert read_holdout(path) == drawn
    with pytest.raises(HoldoutError, match="already drawn"):
        write_holdout(path, drawn)


def test_a_damaged_hold_out_says_how_to_fix_it(tmp_path: Path) -> None:
    path = tmp_path / "holdout.toml"
    path.write_text("places = [", encoding="utf-8")
    with pytest.raises(HoldoutError, match="cannot be read"):
        read_holdout(path)


def test_what_is_held_out_is_asked_by_name() -> None:
    held = HoldOut(places=("Mill",), references=("/r.jpg",), story_sessions=("s",), seed=1,
                   drawn_at=AT)  # fmt: skip
    assert held.holds_place("Mill")
    assert not held.holds_place("Harbor")


def test_a_location_sharing_a_tenth_of_the_smaller_ones_files_is_near() -> None:
    # "Day 2" and "Day 1" of a trip, and the umbrella folder both are filed in again.
    sizes = {"Day 1": 100, "Day 2": 60, "Umbrella": 2000, "Elsewhere": 80, "Few": 40}
    shared = {"Day 1": 12, "Day 2": 8, "Umbrella": 15, "Elsewhere": 2, "Few": 4}
    assert near_places(100, shared, sizes) == {"Day 1", "Day 2", "Umbrella", "Few"}
    # 12 of the smaller (100) is 12%, Day 2 is 8 of 60 = 13%, the umbrella 15 of 100 = 15%, Few
    # is 4 of 40 = 10% (exactly enough), Elsewhere 2 files is under the least that counts.


def test_the_share_is_of_the_smaller_location_so_a_big_folder_does_not_hide_its_neighbours() -> (
    None
):
    # The place holds 50; a huge folder holds all of them: that is 100% of the smaller.
    assert near_places(50, {"Huge": 50}, {"Huge": 9466}) == {"Huge"}
    assert near_places(9466, {"Small": 50}, {"Small": 50}) == {"Small"}  # and the other way round


def test_a_few_files_in_common_are_not_a_trip() -> None:
    assert near_places(10, {"A": 2}, {"A": 2}) == frozenset()  # two files, even if all of one
    assert near_places(10, {}, {}) == frozenset()


def test_neighbours_are_one_hop_not_a_chain() -> None:
    # Tilden shares with Mixed and Mixed with Old Mill: Tilden's neighbours are Mixed only;
    # the hold-out of Tilden does not reach Old Mill.
    sizes = {"Mixed": 9400, "Old Mill": 8000}
    assert near_places(799, {"Mixed": 231}, sizes) == {"Mixed"}
