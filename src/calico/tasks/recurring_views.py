"""``recurring_views``: the views you keep coming back to, found without a reference.

At each place, photos that look alike as a whole (the view embedder, DINOv2) and were
taken hours apart are compared with each other part by part (the engine measures each
pair's share of patches that are mutual best matches, ``selection.patch_match``, and
remembers it). Photos are grouped into views by average linkage of those shares, and a
view's photos are split into occasions wherever two in a row are more than
``occasion_gap_hours`` apart (``selection.view_groups``). Views shot on at least
``min_occasions`` occasions come back, most occasions first; each shows one photo per
occasion, oldest first, with the occasion's other frames as alternates. A frame saved
twice counts once, and an edit exported without camera data is dated by its original
(``selection.copies``), since its own date is when it was exported. Times are compared
by the cameras' clocks and on one clock (``selection.clock``, UTC where the camera wrote
its time zone): photos close on either are one occasion, so a zone can join a visit but
never split one.

Measured on four places (a train yard, a school, a factory and a 480-photo site): 64 of 69
labelled repeats were grouped with their reference, 2 of 219 photos labelled "not the same
view" were pulled in, and of 229 groups reviewed by eye 213 were one view (often from
more than one angle, which is welcome) and 8 more were similar views worth seeing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field

from calico.domain import Group
from calico.selection.clock import on_one_clock
from calico.selection.copies import EDIT_SHARE, copy_of, edited_from
from calico.selection.picks import make_pick
from calico.selection.scoring import Signal, evidence_for, with_criteria
from calico.selection.view_groups import (
    CANDIDATE_SIMILARITY,
    GROUP_SHARE,
    group_views,
    seconds,
    split_occasions,
)
from calico.tasks.base import ReferenceUse, Task, TaskOptions

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime

    from numpy.typing import NDArray

    from calico.domain import Pick
    from calico.selection.context import TaskContext

OCCASION_GAP_HOURS = 2.0
"""Photos of one view more than this far apart are separate occasions. Of the labelled same
views, re-shoots on one walk-through were all within an hour of their reference, same-day
returns 2-3 and 6-12 hours later, and none 1-2 hours."""

_BLOCK = 1024
PAIR_SECONDS = 0.004
"""About how long comparing one pair part by part takes (DINOv2 patches on the CPU)."""


class RecurringViewsOptions(TaskOptions):
    candidate_similarity: float = Field(
        default=CANDIDATE_SIMILARITY,
        ge=0,
        le=1,
        description="Only photos this alike as a whole are compared detail by detail",
    )
    occasion_gap_hours: float = Field(
        default=OCCASION_GAP_HOURS,
        gt=0,
        le=24 * 30,
        description="Photos of a view taken more than this many hours apart are separate occasions",
    )
    group_share: float = Field(
        default=GROUP_SHARE,
        ge=0,
        le=1,
        description="How much of their details the photos of one view share, on average",
    )
    min_occasions: int = Field(
        default=2,
        ge=2,
        le=100,
        description="Show views photographed on at least this many occasions",
    )
    max_pairs_per_place: int = Field(
        default=200_000,
        ge=1,
        description="At a place with more look-alike pairs than this (about 13 minutes of "
        "comparing), each photo is compared only with its closest look-alikes",
    )


@dataclass(frozen=True)
class _Place:
    """The photos at one place worth comparing, and the pairs to compare. ``alike`` is
    how many look-alike pairs there were, when that was more than could be compared."""

    location: str
    members: list[int]
    """Photos worth comparing here (indices into ``ctx.photos``), in path order."""
    pairs: NDArray[np.intp]
    """Pairs of ``members`` to compare (rows of two indices into ``ctx.photos``)."""
    alike: int | None = None


@dataclass(frozen=True)
class _Plan:
    places: list[_Place]
    undated: int
    copies: int
    utc: list[datetime | None]
    """Each photo's capture time on one clock (``selection.clock``), indexed like
    ``ctx.photos``; ``captured_at`` is the camera's clock."""
    dropped: NDArray[np.float32]
    """Why each photo was left out before comparing: 1 undated, 2 a copy counted once,
    NaN when it was not."""


@dataclass(frozen=True)
class _View:
    location: str
    occasions: list[list[int]]
    """Indices into ``ctx.photos``, one list per occasion, oldest first."""
    when: dict[int, datetime]
    """Each member's capture time on its camera's clock, an edit's being its original's."""

    @property
    def size(self) -> int:
        return sum(len(o) for o in self.occasions)


class RecurringViewsTask(Task[RecurringViewsOptions]):
    """One result = one view; its picks are one photo per occasion, oldest first, and
    ``Group.score`` is their mean score."""

    name = "recurring_views"
    title = "Views you keep coming back to"
    description = (
        "Every view you photographed on two or more occasions, place by place, "
        "most occasions first."
    )
    reference_use = ReferenceUse.UNUSED
    default_limit = 50
    default_describe_weight = 0.3
    """Low: the description only chooses which frame of an occasion is shown."""
    options_model = RecurringViewsOptions

    def patch_pairs(
        self, ctx: TaskContext, options: RecurringViewsOptions
    ) -> NDArray[np.intp] | None:
        """Within each place, every pair of photos that look alike as a whole, among those
        with a look-alike taken more than an occasion's gap away."""
        places = _plan(ctx, options).places
        if not places:
            return np.zeros((0, 2), dtype=np.intp)
        pairs = np.concatenate([p.pairs for p in places])
        for place in places:
            if place.alike is not None:
                ctx.trace.note(
                    f"{place.location} has {place.alike} look-alike pairs, more than "
                    f"max_pairs_per_place ({options.max_pairs_per_place}), so each photo was "
                    "compared only with its closest look-alikes. Raise max_pairs_per_place "
                    "to compare more, at about 4 ms a pair."
                )
        ctx.trace.note(
            f"{len(pairs)} pairs of look-alike photos to compare part by part, at "
            f"{len(places)} place{'s' if len(places) != 1 else ''} "
            f"(about {_duration(len(pairs) * PAIR_SECONDS)} the first time)."
        )
        return pairs

    def run(self, ctx: TaskContext, options: RecurringViewsOptions) -> list[Group]:
        trace = ctx.trace
        plan = _plan(ctx, options)
        _plan_notes(ctx, plan, options)
        if ctx.compared_pairs is None or ctx.pair_shares is None:
            trace.note(
                "Photos could not be compared part by part, so no views were grouped. "
                'Set view_embedder = "dinov2" in calico.toml.'
            )
            return []
        known = {
            (min(int(a), int(b)), max(int(a), int(b))): float(s)
            for (a, b), s in zip(ctx.compared_pairs, ctx.pair_shares, strict=True)
        }
        n = len(ctx.photos)
        closest = np.full(n, np.nan, dtype=np.float32)
        edit_of = np.full(n, np.nan, dtype=np.float32)
        views: list[_View] = []
        with trace.stage("views", sum(len(p.members) for p in plan.places)) as stage:
            for place in plan.places:
                views.extend(
                    _views_at(
                        ctx,
                        options,
                        place,
                        known,
                        utc=plan.utc,
                        closest=closest,
                        edit_of=edit_of,
                    )
                )
            stage.count_out = len(views)
        trace.record("closest_share", closest)
        trace.record("edit_of", edit_of)
        _edit_note(ctx, int(np.isfinite(edit_of).sum()))
        views.sort(
            key=lambda v: (-len(v.occasions), -v.size, str(ctx.photos[v.occasions[0][0]].path))
        )
        if not views:
            compared = sum(len(p.members) for p in plan.places)
            trace.note(
                f"No view was photographed on {options.min_occasions} or more occasions "
                f"({compared} photos compared detail by detail). Lower group_share to join "
                "views that share less."
            )
            return []
        shown = views[: ctx.limit]
        if len(views) > len(shown):
            more = len(views) - len(shown)
            trace.note(
                f"{more} more view{'s' if more != 1 else ''} came back on "
                f"{options.min_occasions} or more occasions; raise How many to see them."
            )
        return self._groups(ctx, shown, known)

    def _groups(
        self, ctx: TaskContext, views: list[_View], known: dict[tuple[int, int], float]
    ) -> list[Group]:
        n, photos = len(ctx.photos), list(ctx.photos)
        view_no = np.full(n, np.nan, dtype=np.float32)
        typical_of = np.full(n, np.nan, dtype=np.float32)
        per_place = Counter(v.location for v in views)
        numbered: Counter[str] = Counter()
        groups = []
        for rank, view in enumerate(views, start=1):
            members = [i for o in view.occasions for i in o]
            typical = _typicality(members, known)
            view_no[members], typical_of[members] = rank, typical
            signals, pct = with_criteria(
                ctx.criteria_scores,
                members,
                [Signal("view", typical, 1.0, "how typical of its view")],
                self.describe_weight(ctx),
            )
            score = sum((s.weight * s.values for s in signals), np.zeros_like(typical))
            local = {i: k for k, i in enumerate(members)}
            picks: list[Pick] = []
            for nth, occasion in enumerate(view.occasions, start=1):
                # A photo with camera data first: an edit's own date is its export's.
                ranked = sorted(
                    occasion,
                    key=lambda i: (
                        photos[i].camera is None,
                        -float(score[local[i]]),
                        str(photos[i].path),
                    ),
                )
                k = local[ranked[0]]
                details = {
                    "view": f"occasion {nth} of {len(view.occasions)} · shares "
                    f"{typical[k]:.0%} of its details with the rest of this view, on average"
                }
                if pct is not None:
                    details["criteria"] = f"matches criteria {float(pct[k]):.2f}"
                picks.append(
                    make_pick(
                        photos,
                        ranked[0],
                        float(score[k]),
                        evidence_for(k, signals, details),
                        alternates=ranked[1:],
                    )
                )
            numbered[view.location] += 1
            number = numbered[view.location] if per_place[view.location] > 1 else None
            title = _title(ctx, view, number)
            groups.append(
                Group(
                    title=title, score=float(np.mean([p.score for p in picks])), picks=tuple(picks)
                )
            )
        ctx.trace.record("view", view_no)
        ctx.trace.record("typicality", typical_of)
        return groups


def _plan(ctx: TaskContext, options: RecurringViewsOptions) -> _Plan:
    """Which photos, at which places, are worth comparing part by part, and in which pairs.

    Undated photos cannot be placed on an occasion; a frame saved twice is kept once (its
    copy may carry the date it was exported, which would look like another occasion)."""
    vectors = ctx.view_vectors()
    utc = on_one_clock(ctx.photos)
    gap = options.occasion_gap_hours * 3600
    by_place: dict[str, list[int]] = {}
    undated = 0
    dropped = np.full(len(ctx.photos), np.nan, dtype=np.float32)
    for i, p in enumerate(ctx.photos):
        if p.captured_at is None:
            undated += 1
            dropped[i] = 1.0
        else:
            by_place.setdefault(p.location, []).append(i)
    places, copies = [], 0
    local = [p.captured_at for p in ctx.photos]  # once: per place, it cost O(places x photos)
    for location in sorted(by_place):
        members = sorted(by_place[location], key=lambda i: str(ctx.photos[i].path))
        first = copy_of(vectors[members])
        kept = [i for k, i in enumerate(members) if first[k] == k]
        dropped[[i for k, i in enumerate(members) if first[k] != k]] = 2.0
        copies += len(members) - len(kept)
        place = _place(
            location,
            kept,
            vectors=vectors,
            local=local,
            utc=utc,
            gap=gap,
            similarity=options.candidate_similarity,
            max_pairs=options.max_pairs_per_place,
        )
        if place is not None:
            places.append(place)
    return _Plan(places=places, undated=undated, copies=copies, utc=utc, dropped=dropped)


def _place(
    location: str,
    members: list[int],
    *,
    vectors: NDArray[np.float32],
    local: list[datetime | None],
    utc: list[datetime | None],
    gap: float,
    similarity: float,
    max_pairs: int,
) -> _Place | None:
    """The photos here with a look-alike more than ``gap`` seconds away on both clocks (the
    camera's and UTC), and every pair of them alike enough to compare. Compared block by
    block: a large place has thousands."""
    if len(members) < 2:
        return None
    raw = vectors[members]
    unit = (raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    on_camera = np.asarray([seconds(_dated(local[i])) for i in members])
    on_utc = np.asarray([seconds(_dated(utc[i])) for i in members])
    worth = np.zeros(len(members), dtype=bool)
    for start in range(0, len(members), _BLOCK):
        rows = slice(start, start + _BLOCK)
        alike = unit[rows] @ unit.T >= similarity
        apart = (np.abs(on_camera[rows, None] - on_camera[None, :]) > gap) & (
            np.abs(on_utc[rows, None] - on_utc[None, :]) > gap
        )
        worth[rows] = (alike & apart).any(axis=1)
    chosen = np.flatnonzero(worth)
    if len(chosen) < 2:
        return None
    index = np.asarray(members, dtype=np.intp)[chosen]
    a, b, count = _alike_pairs(unit[chosen], similarity, max_pairs)
    return _Place(location, index.tolist(), np.stack([index[a], index[b]], axis=1), count)


def _alike_pairs(
    unit: NDArray[np.float32], similarity: float, max_pairs: int
) -> tuple[NDArray[np.intp], NDArray[np.intp], int | None]:
    """Every pair (a < b) of rows at least ``similarity`` alike, block by block so the
    k x k matrix is never held. Above ``max_pairs``, each row keeps only its closest
    look-alikes (enough for ``max_pairs`` in all), and the full count is returned too."""
    k = len(unit)

    def blocks() -> Iterator[tuple[int, NDArray[np.float32]]]:
        for start in range(0, k, _BLOCK):
            block = unit[start : start + _BLOCK] @ unit.T
            rows = np.arange(start, min(start + _BLOCK, k))
            block[np.arange(k)[None, :] <= rows[:, None]] = -np.inf  # each pair once, a < b
            yield start, block

    total = sum(int((block >= similarity).sum()) for _, block in blocks())
    keep = None if total <= max_pairs else max(1, max_pairs // k)
    found_a, found_b = [], []
    for start, block in blocks():
        closest = block
        if keep is not None and keep < k:
            top = np.argpartition(-block, keep - 1, axis=1)[:, :keep]
            mask = np.zeros_like(block, dtype=bool)
            np.put_along_axis(mask, top, values=True, axis=1)
            closest = np.where(mask, block, -np.inf)
        r, c = np.nonzero(closest >= similarity)
        found_a.append(r + start)
        found_b.append(c)
    a = np.concatenate(found_a).astype(np.intp)
    b = np.concatenate(found_b).astype(np.intp)
    return a, b, (total if keep is not None else None)


def _views_at(
    ctx: TaskContext,
    options: RecurringViewsOptions,
    place: _Place,
    known: dict[tuple[int, int], float],
    *,
    utc: list[datetime | None],
    closest: NDArray[np.float32],
    edit_of: NDArray[np.float32],
) -> list[_View]:
    """The views at one place shot on enough occasions; ``closest`` gets each compared
    photo's best share with any other, and ``edit_of`` each edit's share with the photo
    whose date it took, for ``calico runs explain``."""
    local = {i: k for k, i in enumerate(place.members)}
    rows, shares = [], []
    for a, b in place.pairs:
        share = known.get((min(int(a), int(b)), max(int(a), int(b))), np.nan)
        rows.append((local[int(a)], local[int(b)]))
        shares.append(share)
        if np.isfinite(share):
            for i in (int(a), int(b)):
                closest[i] = share if np.isnan(closest[i]) else max(closest[i], share)
    pairs = np.asarray(rows, dtype=np.intp).reshape(-1, 2)
    gap = options.occasion_gap_hours * 3600
    views = []
    for group in group_views(
        len(place.members),
        pairs,
        np.asarray(shares, dtype=np.float32),
        min_share=options.group_share,
    ):
        members = [place.members[k] for k in group]
        on_camera, on_utc = _dates(ctx, members, known, utc, edit_of)
        split = split_occasions(on_camera, gap, utc=on_utc)
        occasions = [[members[k] for k in o] for o in split]
        if len(occasions) >= options.min_occasions:
            when = dict(zip(members, on_camera, strict=True))
            views.append(_View(place.location, occasions, when))
    return views


def _dates(
    ctx: TaskContext,
    members: list[int],
    known: dict[tuple[int, int], float],
    utc: list[datetime | None],
    edit_of: NDArray[np.float32],
) -> tuple[list[datetime], list[datetime]]:
    """Each member's capture time by its camera's clock and in UTC, an edit exported
    without camera data taking its original's (``selection.copies.edited_from``), so it
    joins that occasion."""
    photos = ctx.photos
    rows, shares = [], []
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            share = known.get((min(members[a], members[b]), max(members[a], members[b])))
            if share is not None:
                rows.append((a, b))
                shares.append(share)
    source = edited_from(
        [photos[i].camera is not None for i in members],
        np.asarray(rows, dtype=np.intp).reshape(-1, 2),
        np.asarray(shares, dtype=np.float32),
    )
    for k, j in enumerate(source.tolist()):
        if j != k:
            i = members[k]
            edit_of[i] = max(s for (a, b), s in zip(rows, shares, strict=True) if k in (a, b))
    dated = [members[j] for j in source.tolist()]
    return [_dated(photos[i].captured_at) for i in dated], [_dated(utc[i]) for i in dated]


def _edit_note(ctx: TaskContext, edits: int) -> None:
    if edits:
        one = edits == 1
        its = "its" if one else "their"
        ctx.trace.note(
            f"{edits} photo{'' if one else 's'} with no camera recorded "
            f"{'shares' if one else 'share'} {EDIT_SHARE:.0%} or more of {its} details with "
            "another frame (an edit whose export lost its camera and date), so "
            f"{'it was' if one else 'they were'} placed on that frame's occasion."
        )


def _typicality(members: list[int], known: dict[tuple[int, int], float]) -> NDArray[np.float32]:
    """Each member's mean share with the others (never compared or unreadable: nothing)."""
    out = np.zeros(len(members), dtype=np.float32)
    for k, i in enumerate(members):
        others = [known.get((min(i, j), max(i, j)), 0.0) for j in members if j != i]
        out[k] = float(np.mean(np.nan_to_num(others, nan=0.0)))
    return out


def _plan_notes(ctx: TaskContext, plan: _Plan, options: RecurringViewsOptions) -> None:
    trace = ctx.trace
    trace.record("dropped", plan.dropped)
    if plan.undated:
        one = plan.undated == 1
        trace.note(
            f"{plan.undated} photo{'' if one else 's'} {'has' if one else 'have'} no capture "
            f"time, so {'it' if one else 'they'} could not be placed on an occasion and "
            f"{'was' if one else 'were'} left out."
        )
    if plan.copies:
        one = plan.copies == 1
        trace.note(
            f"{plan.copies} {'copy' if one else 'copies'} of a frame saved twice (a re-export, "
            f"a HEIC also kept as JPEG) {'was' if one else 'were'} counted once."
        )
    with trace.stage("candidates", len(ctx.photos)) as stage:
        stage.count_out = sum(len(p.members) for p in plan.places)
    if not plan.places:
        trace.note(
            f"No two photos taken more than {options.occasion_gap_hours:g} hours apart at one "
            "place look alike enough to compare."
        )


def _dated(t: datetime | None) -> datetime:
    if t is None:  # only dated photos are planned
        raise ValueError("an undated photo reached the occasions")
    return t


def _title(ctx: TaskContext, view: _View, number: int | None) -> str:
    """The place, numbered when several of its views are shown, then how often and when:
    from the earliest to the latest capture time, an edit's being its original's."""
    first, last = min(view.when.values()), max(view.when.values())
    span = f"{first:%b %Y}"
    if f"{last:%b %Y}" != span:
        span = f"{span} to {last:%b %Y}"
    place = view.location if number is None else f"{view.location}, view {number}"
    return f"{place}: {len(view.occasions)} occasions, {span}"


def _duration(seconds: float) -> str:
    if seconds < 90:
        n = max(1, round(seconds))
        return f"{n} second{'s' if n != 1 else ''}"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)} minutes"
    return f"{seconds / 3600:.1f} hours"
