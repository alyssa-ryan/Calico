"""Control photos: a fixed pool of photos you rated before, asked again inside Compare Strongest
sessions without saying so. Pure; no I/O.

The bar (90% of a taste's first 30 liked) sits under how consistent you are: a photo that
deserves a like does not always get one, and a perfect ranker could not beat that. A first
estimate from 30 photos answered twice put the ceiling between 83% and 93%, which the bar sits
inside; about 250 independent photos would tell above 90% from below (10 November 2026 audit,
BR-5, BR-6). The pool also separates your drift from a location's difficulty: the same photos
asked over weeks cannot get harder. Each session asks ``CONTROLS_PER_SESSION`` of the pool, the
least asked first, so no photo is asked twice in a few days and none is remembered from the last
session. Their answers are kept apart: they never train Yours nor count in any share.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from datetime import datetime

CONTROL_POOL = 40
"""Photos in the pool: enough that the ceiling does not rest on a handful of photos (ten asked
every session would give ten photos, however many ratings pile up)."""

CONTROLS_PER_SESSION = 10
"""Controls added to each Compare Strongest session: a sixth more photos to rate, and about 250
re-asks in 25 sessions."""

CONTROL_LIKED_SHARE = 2 / 3
"""Share of the pool liked when drawn: what a good ranker's first picks are mostly made of, with
enough dislikes to see the other direction (a like becoming a dislike was 4 of 4 flips seen)."""

CONTROL_PER_LOCATION = 2
"""Most photos one location gives the pool, so it does not rest on a few places."""

Z95 = 1.959963984540054


@dataclass(frozen=True)
class Candidate:
    """A rated photo the pool may be drawn from."""

    key: str
    location: str
    liked: bool


@dataclass(frozen=True)
class Control:
    """A photo in the pool, and how it was rated when the pool was drawn."""

    key: str
    liked: bool


def draw_controls(
    candidates: Sequence[Candidate], *, seed: int, size: int = CONTROL_POOL
) -> tuple[Control, ...]:
    """``size`` photos at random, ``CONTROL_LIKED_SHARE`` of them liked, at most
    ``CONTROL_PER_LOCATION`` from one location; what there is when a class has too few. The same
    for the same candidates and seed in any order."""
    wanted = {True: round(size * CONTROL_LIKED_SHARE)}
    wanted[False] = size - wanted[True]
    per_location: Counter[str] = Counter()
    chosen: list[Control] = []
    for liked in (True, False):
        pool = sorted((c for c in candidates if c.liked == liked), key=lambda c: c.key)
        taken = 0
        for i in np.random.default_rng([seed, int(liked)]).permutation(len(pool)):
            c = pool[int(i)]
            if taken < wanted[liked] and per_location[c.location] < CONTROL_PER_LOCATION:
                per_location[c.location] += 1
                chosen.append(Control(c.key, liked))
                taken += 1
    return tuple(sorted(chosen, key=lambda c: c.key))


def pick_controls(
    pool: Sequence[str],
    asked: Mapping[str, int],
    avoid: Collection[str],
    *,
    k: int,
    seed: int,
) -> list[str]:
    """The ``k`` photos of ``pool`` to ask in a session: those asked least so far, ties settled
    by the seed, never one in ``avoid`` (a photo the session shows already)."""
    chance = dict(zip(sorted(pool), np.random.default_rng(seed).random(len(pool)), strict=True))
    ranked = sorted(
        (key for key in pool if key not in avoid),
        key=lambda key: (asked.get(key, 0), float(chance[key]), key),
    )
    return ranked[:k]


@dataclass(frozen=True)
class Ask:
    """One clear answer (Like or Don't like) to a control photo."""

    key: str
    at: datetime
    liked: bool
    session: str


@dataclass(frozen=True)
class Share:
    """``kept`` of ``n``, with the 95% Wilson interval."""

    kept: int
    n: int

    @property
    def value(self) -> float:
        return self.kept / self.n if self.n else math.nan

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.kept, self.n)


@dataclass(frozen=True)
class Ceiling:
    """A share with its 95% interval."""

    value: float
    lo: float
    hi: float


@dataclass(frozen=True)
class SessionRate:
    """How many of the controls one session asked were liked, when."""

    session: str
    at: datetime
    liked: int
    n: int


@dataclass(frozen=True)
class Retest:
    """What asking the same photos again says about how consistent you are."""

    asked: int
    photos: int
    sessions: int
    against_baseline: Share
    """Asks that agree with how the photo was rated when the pool was drawn."""
    from_liked: Share
    """Of photos liked at the draw: asked again and liked again. What a good ranker's first
    picks face."""
    from_disliked: Share
    between_asks: Share
    """Consecutive asks of one photo that agree: what asking is like once the draw is past."""
    by_session: tuple[SessionRate, ...]
    """Each session's controls liked, oldest first: the same photos, so a fall is a change in
    your standard, not a harder location."""
    ceiling: Ceiling | None
    """The share liked a perfect ranker's first picks could reach if the noise were symmetric
    (``2e(1-e) = 1 - agreement``, ceiling ``1 - e``), from ``against_baseline``."""


def measure_retest(baseline: Mapping[str, bool], asks: Sequence[Ask]) -> Retest:
    """``asks`` against how each photo was rated when the pool was drawn (``baseline``);
    answers to a photo not in it are not controls and are left out."""
    mine = sorted((a for a in asks if a.key in baseline), key=lambda a: a.at)
    kept = Counter(a.liked == baseline[a.key] for a in mine)
    liked_then = [a for a in mine if baseline[a.key]]
    disliked_then = [a for a in mine if not baseline[a.key]]
    by_photo: dict[str, list[Ask]] = defaultdict(list)
    for a in mine:
        by_photo[a.key].append(a)
    pairs = [x.liked == y.liked for run in by_photo.values() for x, y in pairwise(run)]
    sessions: dict[str, list[Ask]] = defaultdict(list)
    for a in mine:
        sessions[a.session].append(a)
    against = Share(kept[True], len(mine))
    return Retest(
        asked=len(mine),
        photos=len(by_photo),
        sessions=len(sessions),
        against_baseline=against,
        from_liked=Share(sum(a.liked for a in liked_then), len(liked_then)),
        from_disliked=Share(sum(not a.liked for a in disliked_then), len(disliked_then)),
        between_asks=Share(sum(pairs), len(pairs)),
        by_session=tuple(
            SessionRate(s, run[0].at, sum(a.liked for a in run), len(run))
            for s, run in sorted(sessions.items(), key=lambda kv: kv[1][0].at)
        ),
        ceiling=_ceiling(against),
    )


def _ceiling(agreement: Share) -> Ceiling | None:
    if not agreement.n or agreement.value <= 0.5:
        return None  # no symmetric noise agrees with a photo less than chance
    lo, hi = agreement.interval
    return Ceiling(_perfect(agreement.value), _perfect(lo), _perfect(hi))


def _perfect(agreement: float) -> float:
    """1 - e, where 2e(1-e) = 1 - agreement (about 0.5 at an agreement of 0.5)."""
    return 1 - (1 - math.sqrt(max(0.0, 2 * agreement - 1))) / 2


def wilson(kept: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval for ``kept`` of ``n``."""
    if not n:
        return (math.nan, math.nan)
    p = kept / n
    scale = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / scale
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return (centre - half, centre + half)
