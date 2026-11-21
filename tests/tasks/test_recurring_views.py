from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from calico.tasks.base import ReferenceUse
from calico.tasks.recurring_views import RecurringViewsTask
from tests.factories import context, photo, unit, view

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from calico.domain import Group, Photo
    from calico.selection.context import FeatureView, TaskContext

D0 = datetime(2023, 3, 1, 9, 0)
TASK = RecurringViewsTask()

# Directions in "view embedder" space: photos of one view point the same way.
HALL, ROOM, YARD = (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)
WOBBLE = np.asarray([0.0, 0.0, 0.0, 1.0])
"""An axis no view points along: how each photo of a view differs from the others."""


class Shot:
    """One photo for a test: which view it shows, when, and where. Photos of one view
    point the same way, each a little off (``wobble``, along ``WOBBLE``), so no two
    are copies of one frame unless a test makes them so (``wobble=0`` for both). Each
    has a camera unless a test strips it (``camera=None``), as an edited export does."""

    def __init__(
        self,
        name: str,
        direction: tuple[float, float, float, float],
        when: datetime | None,
        *,
        location: str = "Old Mill",
        wobble: float | None = None,
        camera: str | None = "Canon EOS RP",
        utc_offset: int | None = None,
    ) -> None:
        self.name, self.direction, self.when, self.location = name, direction, when, location
        self.wobble, self.camera, self.utc_offset = wobble, camera, utc_offset


def pool(shots: Sequence[Shot]) -> tuple[list[Photo], FeatureView]:
    photos = [
        photo(
            s.name,
            location=s.location,
            captured_at=s.when,
            camera=s.camera,
            utc_offset=s.utc_offset,
        )
        for s in shots
    ]
    vectors = [
        np.asarray(s.direction) + (0.02 * (i + 1) if s.wobble is None else s.wobble) * WOBBLE
        for i, s in enumerate(shots)
    ]
    return photos, view(unit(np.stack(vectors)))


def same_direction(shots: Sequence[Shot]) -> Callable[[int, int], float]:
    """Photos of one view share 0.5 of their details; any other two share 0.05."""
    return lambda a, b: 0.5 if shots[a].direction == shots[b].direction else 0.05


def run(
    shots: Sequence[Shot],
    share: Callable[[int, int], float] | None = None,
    *,
    options: dict[str, Any] | None = None,
    compare: bool = True,
    **kw: Any,
) -> tuple[TaskContext, list[Group]]:
    photos, v = pool(shots)
    ctx = context(photos, v, task="recurring_views", **kw)
    opts = TASK.parse_options(options or {})
    if compare:
        pairs = TASK.patch_pairs(ctx, opts)
        assert pairs is not None
        rate = share or same_direction(shots)
        shares = np.asarray([rate(int(a), int(b)) for a, b in pairs], dtype=np.float32)
        ctx = dataclasses.replace(ctx, compared_pairs=pairs, pair_shares=shares)
    return ctx, TASK.run(ctx, opts)


def names(group: Group) -> list[str]:
    return [p.path.name for p in group.picks]


def visit(day: int, hour: float = 0.0) -> datetime:
    return D0 + timedelta(days=day, hours=hour)


def test_it_needs_no_reference_and_names_itself() -> None:
    assert TASK.reference_use is ReferenceUse.UNUSED
    assert TASK.name == "recurring_views"


def test_photos_are_compared_only_with_look_alikes_shot_hours_apart_at_the_same_place() -> None:
    shots = [
        Shot("hall-1.jpg", HALL, visit(0)),
        Shot("hall-2.jpg", HALL, visit(30)),
        Shot("room-1.jpg", ROOM, visit(0)),
        Shot("room-2.jpg", ROOM, visit(0, 0.5)),  # the same visit: not a repeat
        Shot("yard-1.jpg", YARD, visit(5)),
        Shot("hall-elsewhere.jpg", HALL, visit(60), location="Harbor"),
    ]
    photos, v = pool(shots)
    pairs = TASK.patch_pairs(context(photos, v), TASK.parse_options({}))
    assert pairs is not None
    assert {tuple(sorted((photos[a].path.name, photos[b].path.name))) for a, b in pairs} == {
        ("hall-1.jpg", "hall-2.jpg")
    }


def test_a_view_shot_on_two_occasions_is_one_group_with_a_photo_from_each() -> None:
    shots = [
        Shot("a.jpg", HALL, visit(0)),
        Shot("b.jpg", HALL, visit(0, 0.01)),
        Shot("c.jpg", HALL, visit(400)),
        Shot("d.jpg", ROOM, visit(3)),
    ]
    _, groups = run(shots, lambda a, b: 0.5 if {a, b} <= {0, 1, 2} and a != b else 0.0)
    (group,) = groups
    assert "Old Mill" in group.title
    assert "2 occasions" in group.title
    assert [p.path.name for p in group.picks][1] == "c.jpg"  # oldest first
    first = group.picks[0]
    assert {first.path.name, *(a.name for a in first.alternates)} == {"a.jpg", "b.jpg"}
    assert first.burst_size == 2
    assert all(e.signal == "view" for e in first.evidence)
    assert "occasion 1 of 2" in first.evidence[0].detail


def test_the_most_typical_frame_of_an_occasion_is_shown() -> None:
    shots = [
        Shot("odd.jpg", HALL, visit(0)),
        Shot("typical.jpg", HALL, visit(0, 0.01)),
        Shot("later.jpg", HALL, visit(10)),
    ]
    rates = {(0, 1): 0.3, (0, 2): 0.2, (1, 2): 0.6}
    _, (group,) = run(shots, lambda a, b: rates[(min(a, b), max(a, b))])
    assert names(group) == ["typical.jpg", "later.jpg"]


def test_a_morning_and_an_evening_visit_on_one_day_are_two_occasions() -> None:
    shots = [Shot("morning.jpg", HALL, visit(0, 0)), Shot("evening.jpg", HALL, visit(0, 9))]
    _, (group,) = run(shots)
    assert names(group) == ["morning.jpg", "evening.jpg"]


def test_the_gap_between_occasions_is_an_option() -> None:
    shots = [Shot("morning.jpg", HALL, visit(0, 0)), Shot("evening.jpg", HALL, visit(0, 9))]
    ctx, groups = run(shots, options={"occasion_gap_hours": 12})
    assert groups == []
    assert any("on 2 or more occasions" in n for n in ctx.trace.notes)


def test_look_alike_rooms_stay_separate_views() -> None:
    shots = [
        Shot("hall-1.jpg", HALL, visit(0)),
        Shot("hall-2.jpg", HALL, visit(9)),
        Shot("hall-twin-1.jpg", HALL, visit(1)),
        Shot("hall-twin-2.jpg", HALL, visit(8)),
    ]
    twin = {2, 3}

    def share(a: int, b: int) -> float:
        return 0.5 if (a in twin) == (b in twin) else 0.1

    _, groups = run(shots, share)
    assert sorted(sorted(names(g)) for g in groups) == [
        ["hall-1.jpg", "hall-2.jpg"],
        ["hall-twin-1.jpg", "hall-twin-2.jpg"],
    ]
    # Two views at one place are told apart by number.
    assert [g.title.split(":")[0] for g in groups] == ["Old Mill, view 1", "Old Mill, view 2"]


def test_views_with_the_most_occasions_come_first_and_the_limit_counts_views() -> None:
    shots = [
        *(Shot(f"room-{i}.jpg", ROOM, visit(10 * i)) for i in range(2)),
        *(Shot(f"hall-{i}.jpg", HALL, visit(10 * i)) for i in range(3)),
        *(Shot(f"yard-{i}.jpg", YARD, visit(10 * i)) for i in range(2)),
    ]
    ctx, groups = run(shots, limit=2)
    assert [g.title.split(":")[1].strip().split(",")[0] for g in groups] == [
        "3 occasions",
        "2 occasions",
    ]
    assert any("1 more view" in n for n in ctx.trace.notes)


def test_views_seen_fewer_times_than_asked_are_left_out() -> None:
    shots = [Shot(f"hall-{i}.jpg", HALL, visit(10 * i)) for i in range(2)]
    _, groups = run(shots, options={"min_occasions": 3})
    assert groups == []


def test_a_frame_saved_again_later_is_not_a_second_occasion() -> None:
    shots = [
        Shot("frame.jpg", HALL, visit(0), wobble=0.0),
        Shot("frame-export.jpg", HALL, visit(200), wobble=0.0),
    ]
    ctx, groups = run(shots)
    assert groups == []
    assert any("saved twice" in n and "once" in n for n in ctx.trace.notes)


def test_edits_exported_without_camera_data_are_not_more_occasions() -> None:
    # One frame, its original from the camera and
    # two edits whose exports lost the camera and carry later dates, 3 and 27 hours on.
    shots = [
        Shot("_MG_5850.JPG", HALL, visit(0)),
        Shot("MGIQE9612.jpg", HALL, visit(0, 3), camera=None),
        Shot("0A4BAFCE.jpg", HALL, visit(1, 3), camera=None),
    ]
    ctx, groups = run(shots, lambda a, b: 0.62)
    assert groups == []
    assert any("2 photos with no camera" in n for n in ctx.trace.notes)
    edit = ctx.trace.columns["edit_of"]
    assert np.isnan(edit[0])
    assert edit[1:].tolist() == [pytest.approx(0.62)] * 2


def test_an_edit_is_shown_with_the_occasion_of_its_original() -> None:
    shots = [
        Shot("original.jpg", HALL, visit(0)),
        Shot("edit.jpg", HALL, visit(30), camera=None),
        Shot("return.jpg", HALL, visit(400)),
    ]
    shares = {(0, 1): 0.62, (0, 2): 0.3, (1, 2): 0.3}
    _, (group,) = run(shots, lambda a, b: shares[(min(a, b), max(a, b))])
    assert "2 occasions" in group.title
    first = group.picks[0]
    assert {first.path.name, *(a.name for a in first.alternates)} == {"original.jpg", "edit.jpg"}


def test_a_return_without_camera_data_still_counts_when_it_is_its_own_shot() -> None:
    shots = [Shot("canon.jpg", HALL, visit(0)), Shot("phone.jpg", HALL, visit(400), camera=None)]
    ctx, (group,) = run(shots, lambda a, b: 0.45)
    assert names(group) == ["canon.jpg", "phone.jpg"]
    assert np.isnan(ctx.trace.columns["edit_of"]).all()


def test_one_visit_with_two_cameras_on_different_clocks_is_one_occasion() -> None:
    # One visit: the Canon kept its home zone (-04:00), the phone moved to the
    # destination's (+01:00). By their clocks five hours apart; in UTC, together.
    shots = [
        Shot("_MG_9209.JPG", HALL, D0.replace(hour=8, minute=48), utc_offset=-240),
        Shot(
            "IMG_1977.HEIC",
            HALL,
            D0.replace(hour=13, minute=2),
            camera="Apple iPhone 16",
            utc_offset=60,
        ),
    ]
    _, groups = run(shots)
    assert groups == []


def test_a_frame_and_its_export_without_a_time_zone_stay_one_occasion() -> None:
    # RAW/IMG_9312.CR3 carries the Canon's zone (-04:00) and
    # JPG/IMG_9312.JPG, in the next folder, does not; both read 14:13 on the camera.
    raw = Shot("IMG_9312.CR3", HALL, D0.replace(hour=14, minute=13), utc_offset=-240)
    jpg = Shot("IMG_9312.JPG", HALL, D0.replace(hour=14, minute=13), location="Old Mill")
    shots = [raw, jpg, Shot("later.jpg", HALL, visit(200), utc_offset=-240)]
    photos, v = pool(shots)
    photos[1] = dataclasses.replace(photos[1], path=photos[1].path.parent / "JPG" / "IMG_9312.JPG")
    ctx = context(photos, v, task="recurring_views")
    opts = TASK.parse_options({})
    pairs = TASK.patch_pairs(ctx, opts)
    assert pairs is not None
    shares = np.full(len(pairs), 0.5, dtype=np.float32)
    ctx = dataclasses.replace(ctx, compared_pairs=pairs, pair_shares=shares)
    (group,) = TASK.run(ctx, opts)
    assert "2 occasions" in group.title


def test_undated_photos_are_left_out_and_counted() -> None:
    shots = [
        Shot("a.jpg", HALL, visit(0)),
        Shot("b.jpg", HALL, visit(50)),
        Shot("scan.jpg", HALL, None),
    ]
    ctx, (group,) = run(shots)
    assert "scan.jpg" not in names(group)
    assert any("1 photo has no capture time" in n for n in ctx.trace.notes)


def test_without_part_by_part_comparison_it_says_so_and_finds_nothing() -> None:
    shots = [Shot("a.jpg", HALL, visit(0)), Shot("b.jpg", HALL, visit(50))]
    ctx, groups = run(shots, compare=False)
    assert groups == []
    assert any("part by part" in n for n in ctx.trace.notes)


def test_every_compared_photo_records_its_view_and_closest_match() -> None:
    shots = [
        Shot("a.jpg", HALL, visit(0)),
        Shot("b.jpg", HALL, visit(50)),
        Shot("lonely.jpg", ROOM, visit(3)),
    ]
    ctx, _ = run(shots)
    view_no, closest = ctx.trace.columns["view"], ctx.trace.columns["closest_share"]
    assert view_no.tolist()[:2] == [1.0, 1.0]
    assert np.isnan(view_no[2])  # never compared
    assert np.isnan(closest[2])
    assert closest[0] == pytest.approx(0.5)


def test_the_description_chooses_between_frames_of_an_occasion() -> None:
    shots = [
        Shot("plain.jpg", HALL, visit(0)),
        Shot("foggy.jpg", HALL, visit(0, 0.01), wobble=0.2),
        Shot("later.jpg", HALL, visit(10)),
    ]
    _, (group,) = run(shots, criteria=tuple(WOBBLE), describe_weight=1.0)
    assert names(group)[0] == "foggy.jpg"


def test_a_place_with_too_many_look_alikes_compares_each_photo_with_its_closest() -> None:
    """Comparing every pair of a 600-photo "Misc" folder would take hours; say so, cap it."""
    shots = [Shot(f"{i:03}.jpg", HALL, visit(i)) for i in range(60)]
    photos, v = pool(shots)
    ctx = context(photos, v, task="recurring_views")
    every = TASK.patch_pairs(context(photos, v, task="recurring_views"), TASK.parse_options({}))
    assert every is not None
    pairs = TASK.patch_pairs(ctx, TASK.parse_options({"max_pairs_per_place": 100}))
    assert pairs is not None
    assert 0 < len(pairs) <= 100 + len(shots) < len(every)  # each keeps its closest few
    assert {(int(a), int(b)) for a, b in pairs} <= {(int(a), int(b)) for a, b in every}
    notes = " ".join(ctx.trace.notes)
    assert f"{len(every)} look-alike pairs, more than max_pairs_per_place (100)" in notes


def test_the_pairs_to_compare_and_how_long_they_take_are_said_first() -> None:
    shots = [Shot(f"{i}.jpg", HALL, visit(i)) for i in range(4)]
    photos, v = pool(shots)
    ctx = context(photos, v, task="recurring_views")
    TASK.patch_pairs(ctx, TASK.parse_options({}))
    (note,) = [n for n in ctx.trace.notes if "to compare part by part" in n]
    assert note.startswith("6 pairs of look-alike photos")


def test_planning_many_places_takes_time_in_proportion_to_the_photos() -> None:
    """The camera clock of every photo was rebuilt for every place: 18 s at the archive.
    Twice the photos at twice the places took four times as long; now about twice."""
    import time  # noqa: PLC0415

    def planning(count: int) -> float:
        shots = [Shot(f"{i}.jpg", HALL, visit(i % 7), location=f"P{i // 4}") for i in range(count)]
        photos, v = pool(shots)
        ctx = context(photos, v, task="recurring_views")
        best = float("inf")
        for _ in range(2):
            started = time.perf_counter()
            TASK.patch_pairs(ctx, TASK.parse_options({}))
            best = min(best, time.perf_counter() - started)
        return best

    assert planning(24_000) / planning(12_000) < 3.0


def test_an_edit_does_not_date_its_view_by_its_export() -> None:
    """A camera-less edit takes its original's date for occasions; its own export date
    must not appear in the title or as the shown photo's date either."""
    shots = [
        Shot("a.jpg", HALL, datetime(2023, 3, 1, 9)),
        Shot("0edit.jpg", HALL, datetime(2026, 9, 1, 9), camera=None),
        Shot("b.jpg", HALL, datetime(2023, 4, 1, 9)),
    ]
    shares = {(0, 1): 0.62, (0, 2): 0.3, (1, 2): 0.3}
    _, (group,) = run(shots, lambda a, b: shares[(min(a, b), max(a, b))])
    assert "2 occasions, Mar 2023 to Apr 2023" in group.title
    assert [p.path.name for p in group.picks] == ["a.jpg", "b.jpg"]


def test_photos_left_out_of_the_occasions_say_why() -> None:
    shots = [
        Shot("a.jpg", HALL, visit(0)),
        Shot("b.jpg", HALL, visit(9)),
        Shot("undated.jpg", HALL, None),
        Shot("again.jpg", HALL, visit(20), wobble=0.0),
        Shot("copy.jpg", HALL, visit(21), wobble=0.0),
    ]
    ctx, _ = run(shots)
    dropped = ctx.trace.columns["dropped"].tolist()
    assert dropped[2] == 1.0  # undated
    assert dropped[4] == 2.0  # a copy of again.jpg, counted once
    assert np.isnan(dropped[0])
