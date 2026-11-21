"""Control photos: a fixed pool of photos you rated before, asked again inside Compare Strongest
sessions without saying so, to measure how consistent you are (the noise ceiling the bar sits
under, and drift). The draw, the pick for a session, what a re-ask says, and the pool's file."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from calico.controls import Controls, read_controls, write_controls
from calico.errors import ControlsError
from calico.evaluation.controls import (
    CONTROL_PER_LOCATION,
    Ask,
    Candidate,
    Control,
    draw_controls,
    measure_retest,
    pick_controls,
)

if TYPE_CHECKING:
    from pathlib import Path

T0 = datetime(2026, 11, 11, 9, 0, tzinfo=UTC)


def candidates(n: int = 60, *, liked_every: int = 4, per_location: int = 1) -> list[Candidate]:
    """``n`` rated photos, three in four liked, ``per_location`` to a location."""
    return [
        Candidate(f"k{i:03d}", f"Location {i // per_location}", i % liked_every != 0)
        for i in range(n)
    ]


def test_the_pool_is_two_thirds_liked_and_repeats_for_a_seed_in_any_order() -> None:
    pool = candidates()
    drawn = draw_controls(pool, seed=7, size=12)
    assert len(drawn) == 12
    assert sum(c.liked for c in drawn) == 8  # two thirds
    assert drawn == draw_controls(list(reversed(pool)), seed=7, size=12)
    assert drawn != draw_controls(pool, seed=8, size=12)
    assert [c.key for c in drawn] == sorted(c.key for c in drawn)


def test_a_location_gives_the_pool_at_most_a_few_photos() -> None:
    pool = candidates(40, per_location=10)  # four locations, ten photos each
    drawn = draw_controls(pool, seed=1, size=30)
    by_location: dict[str, int] = {}
    for c in drawn:
        location = next(p.location for p in pool if p.key == c.key)
        by_location[location] = by_location.get(location, 0) + 1
    assert max(by_location.values()) == CONTROL_PER_LOCATION
    assert len(drawn) == CONTROL_PER_LOCATION * 4  # all there is, under that limit


def test_a_class_with_too_few_photos_gives_what_there_is() -> None:
    pool = [Candidate("a", "A", True), Candidate("b", "B", True), Candidate("c", "C", False)]
    drawn = draw_controls(pool, seed=1, size=9)
    assert {c.key: c.liked for c in drawn} == {"a": True, "b": True, "c": False}
    assert draw_controls([], seed=1) == ()


def test_each_session_asks_the_photos_asked_least_and_never_one_it_shows() -> None:
    pool = [f"c{i}" for i in range(6)]
    asked = {"c0": 3, "c1": 1, "c2": 0, "c3": 0, "c4": 2, "c5": 0}
    picked = pick_controls(pool, asked, {"c3"}, k=3, seed=5)
    assert picked == sorted(picked, key=lambda k: (asked[k], picked.index(k)))  # least asked first
    assert set(picked) == {"c2", "c5", "c1"}  # c3 is shown in the session
    assert picked == pick_controls(list(reversed(pool)), asked, {"c3"}, k=3, seed=5)
    assert len(pick_controls(pool, {}, set(), k=10, seed=5)) == 6  # no more than the pool


def ask(key: str, liked: bool, hours: float, session: str = "s1") -> Ask:
    return Ask(key=key, at=T0 + timedelta(hours=hours), liked=liked, session=session)


def test_a_re_ask_is_read_against_the_baseline_and_against_the_ask_before() -> None:
    baseline = {"a": True, "b": True, "c": False, "d": False}
    asks = [
        ask("a", True, 1, "s1"),
        ask("b", False, 1, "s1"),  # liked, then not: a flip
        ask("c", False, 1, "s1"),
        ask("a", True, 30, "s2"),
        ask("b", False, 30, "s2"),
        ask("d", True, 30, "s2"),  # not liked, then liked
    ]
    got = measure_retest(baseline, asks)
    assert (got.asked, got.photos, got.sessions) == (6, 4, 2)
    assert (got.against_baseline.kept, got.against_baseline.n) == (3, 6)
    assert (got.from_liked.kept, got.from_liked.n) == (2, 4)  # a, a kept; b, b flipped
    assert (got.from_disliked.kept, got.from_disliked.n) == (1, 2)  # c kept; d flipped
    assert (got.between_asks.kept, got.between_asks.n) == (2, 2)  # a again, b again
    assert [(s.session, s.liked, s.n) for s in got.by_session] == [("s1", 1, 3), ("s2", 2, 3)]


def test_agreement_gives_the_ceiling_a_perfect_ranker_could_reach_if_the_noise_is_symmetric() -> (
    None
):
    baseline = {f"k{i}": True for i in range(100)}
    asks = [ask(f"k{i}", i >= 13, 1) for i in range(100)]  # 87 of 100 kept
    got = measure_retest(baseline, asks)
    assert got.ceiling is not None
    e = (1 - math.sqrt(2 * 0.87 - 1)) / 2  # 2e(1-e) = 1 - agreement
    assert got.ceiling.value == pytest.approx(1 - e)
    assert got.ceiling.value == pytest.approx(0.93, abs=0.005)
    assert 0.87 < got.ceiling.lo < got.ceiling.value < got.ceiling.hi < 0.97  # about 100 photos


def test_without_enough_agreement_or_asks_there_is_no_ceiling() -> None:
    assert measure_retest({"a": True}, []).ceiling is None
    flipped = measure_retest({"a": True, "b": False}, [ask("a", False, 1), ask("b", True, 1)])
    assert flipped.ceiling is None  # it agrees less than chance: no symmetric noise says that
    assert measure_retest({"a": True}, [ask("zz", True, 1)]).asked == 0  # not a control


def test_the_pools_file_is_written_once_and_read_back(tmp_path: Path) -> None:
    path = tmp_path / "controls.toml"
    assert read_controls(path) is None
    drawn = Controls(photos=(Control("k1", True), Control("k2", False)), seed=11, drawn_at=T0)
    write_controls(path, drawn)
    assert read_controls(path) == drawn
    with pytest.raises(ControlsError, match="already drawn"):
        write_controls(path, drawn)


def test_a_damaged_pool_says_how_to_restore_it(tmp_path: Path) -> None:
    path = tmp_path / "controls.toml"
    path.write_text("this is [not toml")
    with pytest.raises(ControlsError, match="Restore"):
        read_controls(path)
