"""``triptych``: three photographs that tell a story together.

A triptych is an anchor plus the pair of companions that maximise strength and
connection. "Connection" is user-chosen (shared theme, mood, surface, palette,
place, or complementary shot kinds) and measured pairwise, so the same machinery
serves "three moods of one room" and "establishing + detail + portrait".

One result = one triptych; ``Group.score`` is the story score (strength and
connection together), not the mean of its picks' strengths.

Connections reward alike photos, so a share of the connection side goes to
*variety* (``selection.variety``): three frames of one moment are not a story.
It counts when an anchor's companions are shortlisted too, or the shortlist would
hold only the anchor's look-alikes.

The defaults come from stories you finished yourself (``calico eval
finished-stories``): theirs are as varied as chance in subject and view (three
photos 0.62 alike, where the task's own were 0.84) but share the kind of place.
So variety starts falling at 0.70, place type is a default connection, and 1,000
candidates, not 150, let a story use photos that are not among a large place's
very strongest (half the finished stories needed one). 1,000 rather than 500 told 86%
of them, not 79% (General), and its stories worked as often as 500's in seven rating
sessions at large places (3 November 2026); below about 900 photos the two are alike.

Why only some connections are percentile-ranked: raw embedding, label-overlap
and palette similarities have no absolute scale (it depends on the embedder and
the collection), so ``theme``, ``mood``, ``surface``, ``place_type`` and ``color`` become
"how alike compared with every other pair here". ``location`` is categorical
(1 same, 0 different, 0.5 unknown) and the kind connections are overlaps of
probability distributions; both already mean the same thing everywhere, and
ranking them would turn "same place" into an arbitrary percentile.

Three opt-in ideas from your rated stories and the 28 you finished (26 October 2026).
*Anchors by kind* (``anchor_kinds``): anchors take turns among kinds of shot, since
yours are wide shots or portraits and Calico's were all its strongest photos, nearly
all wide interiors; a story session rates it against today's way. *The strongest
connection* (``connect_by``), since yours each lean on one, and *variety by repetition*
(``variety_by``), pairs showing one space at one kind of shot, since three alike photos
of one room is what you rejected: both made your finished stories beat a one-photo
swap less often (0.71 → 0.66 each, 0.61 together), so they stay choices to measure
again once the features see what you mean (the palette misses E's colour grading).
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from calico.domain import Evidence, Group, Photo, place_key
from calico.errors import CapabilityError
from calico.models.base import LabelModel, rollup
from calico.selection.anchors import SCALES, anchor_prior, expected_scale
from calico.selection.bursts import group_bursts, representatives
from calico.selection.clock import on_one_clock
from calico.selection.context import FeatureView
from calico.selection.picks import make_pick, scatter
from calico.selection.scoring import (
    bhattacharyya,
    histogram_intersection,
    pairwise_percentile,
)
from calico.selection.strength import (
    AESTHETIC_WEIGHT,
    QUALITY_WEIGHT,
    finite_or_zero,
    photo_strength,
)
from calico.selection.variety import variety
from calico.tasks.base import ReferenceUse, Task, TaskOptions
from calico.tasks.same_view import MIN_SIMILARITY

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping
    from pathlib import Path

    from numpy.typing import ArrayLike

    from calico.selection.context import TaskContext

Matrix = NDArray[np.float32]
Triple = tuple[int, int, int]
HISTOGRAM_BLOCK = 64
NOT_CONNECTIONS = frozenset({"strength", "variety", "arrangement", "fresh", "anchor", "distinct"})
"""A story's evidence that is not one of its connections (told by roles, its closest pair
and how the third photo belongs are its connections)."""
ROLE_WEIGHTS: Mapping[str, float] = {
    "bond": 0.35,
    "belong": 0.15,
    "arrangement": 0.15,
    "fresh": 0.15,
    "anchor": 0.20,
}
"""A story told by roles: its closest pair (``bond``), how the third photo connects with
them (``belong``), an echo or a change of scale (``arrangement``), not three shots of one
space at one scale (``fresh``), and an anchor that shows a person or animal or is the
widest of the three (``anchor``). Set before measuring. On your 28 finished stories
(General), they beat a one-photo swap 0.745 of the time, against 0.726 by pairs (16 better,
12 worse; V, all wide, worst at 0.74 to 0.49). Without each role in turn: bond 0.690,
anchor 0.721, belong 0.730, arrangement 0.738, fresh 0.749. Swaps cannot see ``fresh``
(your stories are almost never three of one space); rated stories can (46% of those that
did not work, 33% of those that did). A first version that asked for a change of scale
in every story, and counted any pair of one space at one scale against it, scored 0.710:
your echoes are often of one scale, and AA's pair is one space seen twice."""


class Connection(StrEnum):
    THEME = "theme"
    LOCATION = "location"
    PLACE_TYPE = "place_type"
    MOOD = "mood"
    SURFACE = "surface"
    COLOR = "color"
    KIND_MIX = "kind_mix"
    KIND_SAME = "kind_same"


class TriptychOptions(TaskOptions):
    connections: tuple[Connection, ...] = Field(
        default=(Connection.THEME, Connection.PLACE_TYPE, Connection.MOOD, Connection.COLOR),
        min_length=1,
        description="What should tie the three photos together",
    )
    strength_weight: float = Field(
        default=0.4, ge=0, le=1, description="Strong photos (1) vs. a strong connection (0)"
    )
    candidates: int = Field(
        default=1000, ge=3, le=2000, description="How many of the strongest photos to consider"
    )
    companions: int = Field(
        default=25, ge=2, le=200, description="Best-connected partners tried for each anchor"
    )
    max_similarity: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="Members more alike than this are near-duplicates and never share a triptych",
    )
    too_alike: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Three photos more alike than this on average are one moment, not a story",
    )
    variety_weight: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description="How much variety counts against how well the three connect",
    )
    max_uses: int = Field(
        default=1, ge=1, le=10, description="How many triptychs one photo may appear in"
    )
    spend_alike: float = Field(
        default=0,
        ge=0,
        le=1,
        description=(
            "A photo used in a story also uses up the photos at its place that are this alike "
            "to it by the view embedder, and its near-duplicates (0: only the photo itself)"
        ),
    )
    burst_window_seconds: float = Field(
        default=120, ge=0, le=3600, description="Frames this close in time may be one burst"
    )
    burst_similarity: float = Field(
        default=0.9, ge=0, le=1, description="Frames this alike (cosine) in a window are one burst"
    )
    connect_by: Literal["average", "strongest"] = Field(
        default="average",
        description="A story's connection: the mean of the chosen ones, or the strongest one",
    )
    variety_by: Literal["likeness", "repetition"] = Field(
        default="likeness",
        description=(
            "Variety as the three being unlike overall, or as no two of them showing one "
            "space at one kind of shot"
        ),
    )
    same_space: float = Field(
        default=MIN_SIMILARITY,
        ge=0,
        le=1,
        description="View-embedder similarity from which two photos may show one space",
    )
    """Same view over time's line for "possibly the same view" (calibrated on 175
    labelled photos): a room seen again from about the same place. Of 137 rated stories,
    55 had all three pairs this alike at one kind of shot (43 rejected); none of the 28
    you finished did."""
    anchor_kinds: tuple[str, ...] = Field(
        default=(),
        description="Kinds of shot that take turns anchoring stories; none: the strongest photos",
    )
    anchor_by: Literal["strength", "likely"] = Field(
        default="strength",
        description=(
            "Which photos anchor stories first: the strongest, or the likeliest anchors "
            "(strength with a person or animal or the widest view, as roles try them)"
        ),
    )
    """Your anchors are not the strongest photos (3 November 2026, ``calico eval anchors``,
    69 could anchor and 291 could not at 6 places): general strength found them at 0.68,
    the anchor prior (``selection.anchors.anchor_prior``) at 0.81, +0.14 [0.07, 0.22], a
    wide view alone 0.81, Yours's strength 0.82. Told by roles, anchors were already tried
    by the prior, but roles' stories worked less often (34% against 46%), so this tries
    them by the prior today's way, a change of anchors alone. Off until rated sessions
    say (``evaluation.STORY_VARIANT``)."""
    anchors_tried: int | None = Field(
        default=None,
        ge=1,
        le=2000,
        description="How many candidates are tried as anchors; none: three per story (at least 10)",
    )
    """At a large place three per story is 18 of 1,000 candidates. Those reached 4 of your 28
    finished stories' anchors by strength and 7 by the anchor prior (General, 5 November
    2026), so the prior's better order barely changes who is tried. Trying more costs time
    in proportion (each anchor is paired with its ``companions``). Off until ``calico eval
    finished-stories --variant anchors_tried=…`` and a timed search say how many."""
    distinct_weight: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description=(
            "How much it counts that no two of the three are one moment (alike by the view "
            "embedder, shot minutes apart), of the whole score"
        ),
    )
    """Your stories are three different images (30 October 2026): the same room from
    another angle is a carousel, and different rooms of one depth and geometry want
    variety too. Two photos of one look beside an anchor (your E, I and N) are the
    exception, which you pull by hand. Of 365 rated session stories, those whose closest
    pair was under 0.80 alike by the view embedder worked 43-46% of the time, 0.85-0.90
    27%, 0.90 and up 18%; 174 had a pair at 0.85 or more. 5 of the 28 you finished have
    one (I at 0.98), so it weighs against such a pair rather than forbidding it. It counts
    in the companions' shortlist too, or that would offer only the anchor's look-alikes.
    At 0.15 (31 October 2026, ``docs/decisions/0004``): rated stories +0.04 [-0.03, 0.11]
    against today's (15 sessions, 342 stories), finished stories over swaps 0.74 to 0.74
    (-0.00 [-0.02, 0.01]), only E and N lower. 0.3 was no better on rated stories and cost
    E more."""
    distinct_from: float = Field(
        default=0.80,
        ge=0,
        le=1,
        description="View-embedder likeness of a story's closest pair from which it counts against",
    )
    distinct_until: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="View-embedder likeness at which the closest pair is one view: distinct is 0",
    )
    distinct_within: float = Field(
        default=600,
        ge=0,
        le=31_536_000,
        description=(
            "Seconds apart within which two alike photos are one moment, which distinct "
            "counts against; 0: whenever they were shot"
        ),
    )
    """One moment, not an echo (31 October 2026). Alike by the view embedder alone,
    your echoes (X, J, ZZ, 0.85-0.87, shot hours to years apart) could not be told from
    Calico's re-angled rooms, and distinct cost your finished stories against swaps
    (0.74 to 0.68 at weight 0.15). Of rated session stories with a pair 0.80 or more
    alike, those shot within 10 minutes worked 15 of 105 times, the rest 45 of 143. Of
    your finished stories only E (29 s) and N (4.5 min) are one moment. Undated photos
    are not one moment. Chosen from that same data, so the hold-out has the last word."""
    tell_by: Literal["pairs", "roles"] = Field(
        default="pairs",
        description=(
            "A story as the mean of its three pairs, or by roles: a close pair, an anchor "
            "with a person or animal or the widest view, and an echo or a change of scale"
        ),
    )
    """Your stories have roles (27 October 2026): one close pair, the anchor apart from
    its outer photos in space and scale, often with a person in it. By pairs, every pair
    of Calico's stories is alike and all three are one scale in two of three."""

    @model_validator(mode="after")
    def _lines_fall_somewhere(self) -> TriptychOptions:
        """Variety falls from ``too_alike`` to 0 at ``max_similarity``, and distinct from
        ``distinct_from`` to ``distinct_until``: with the first at or above the second there
        is nothing to fall over, and either was 1 for every story (audit A03-3)."""
        for start, end in (("too_alike", "max_similarity"), ("distinct_from", "distinct_until")):
            if getattr(self, start) >= getattr(self, end):
                raise ValueError(
                    f"{start} ({getattr(self, start):g}) must be below {end} "
                    f"({getattr(self, end):g}): it is where the penalty starts, and {end} "
                    "where it is whole."
                )
        return self


@dataclass(frozen=True)
class _Conn:
    """One connection: a pairwise matrix in [0, 1] and a way to say what links a triple."""

    name: Connection
    matrix: Matrix
    describe: Callable[[Triple], str]


@dataclass(frozen=True)
class _Rows:
    """The working set: pool photos plus any reference that is not in the pool."""

    photos: list[Photo]
    pool_index: list[int | None]  # index into ctx.photos, or None for an external reference
    strength: NDArray[np.float32]
    is_reference: NDArray[np.bool_]
    features: FeatureView
    shortlisted: NDArray[np.bool_]
    """The row's burst is among the strongest ``candidates`` on its own strength (not only
    there as a reference or because it was asked for)."""


@dataclass(frozen=True)
class _Roles:
    """What telling by roles reads of the working set: each pair's connection (the mean of
    the chosen ones), each photo's scale (None without the scale model), how surely it
    shows a person or animal (None without the subject model), and which pairs repeat one
    space at one scale (None without the view embedder's features)."""

    pairs: Matrix
    scale: NDArray[np.float32] | None
    living: NDArray[np.float32] | None
    repeats: NDArray[np.bool_] | None

    def prior(self, strength: NDArray[np.float32]) -> NDArray[np.float32]:
        """How likely each photo is to anchor (``selection.anchors.anchor_prior``)."""
        return anchor_prior(strength, scale=self.scale, living=self.living)

    @property
    def weights(self) -> dict[str, float]:
        """``ROLE_WEIGHTS`` over the roles that can be measured, summing to 1: ``fresh``
        needs the view features and the scale model, ``anchor`` a subject or scale model.
        A role that cannot be told is left out, not counted as filled."""
        measured = {
            k: w
            for k, w in ROLE_WEIGHTS.items()
            if (k != "fresh" or self.repeats is not None)
            and (k != "anchor" or self.living is not None or self.scale is not None)
        }
        total = sum(measured.values())
        return {k: w / total for k, w in measured.items()}

    def parts_with(self, a: int, comp: NDArray[np.intp]) -> dict[str, NDArray[np.float32]]:
        """Each role's value for every story (x, a, y) with x, y from ``comp``."""
        p = self.pairs
        xa, ay = p[comp, a][:, None], p[a, comp][None, :]
        xy = p[np.ix_(comp, comp)]
        links = np.stack(np.broadcast_arrays(xa, ay, xy))
        bond = links.max(axis=0)
        echo = np.clip(0.5 + 2 * (xy - np.maximum(xa, ay)), 0, 1)
        arrangement, widest = echo, np.zeros_like(bond)
        if self.scale is not None:
            sx, sy, sa = self.scale[comp][:, None], self.scale[comp][None, :], self.scale[a]
            spread = np.maximum(np.maximum(sx, sy), sa) - np.minimum(np.minimum(sx, sy), sa)
            arrangement = np.maximum(echo, np.minimum(1.0, spread))
            widest = np.clip(1 - np.maximum(0.0, sa - np.minimum(sx, sy)), 0, 1)
        fresh = np.ones_like(bond)
        if self.repeats is not None:
            r = self.repeats
            count = r[comp, a][:, None].astype(int) + r[a, comp][None, :] + r[np.ix_(comp, comp)]
            fresh = (count < 2).astype(np.float32)
        living = float(self.living[a]) if self.living is not None else 0.0
        return {
            "bond": bond,
            "belong": (links.sum(axis=0) - bond) / 2,
            "arrangement": arrangement,
            "fresh": fresh,
            "anchor": np.maximum(living, widest),
        }

    def parts(self, triple: Triple) -> dict[str, float]:
        """The measured roles' values for one story."""
        x, a, y = triple
        found = self.parts_with(a, np.array([x, y]))
        return {k: float(found[k][0, 1]) for k in self.weights}

    def value_with(self, a: int, comp: NDArray[np.intp]) -> NDArray[np.float32]:
        found = self.parts_with(a, comp)
        total = np.zeros_like(found["bond"])
        for k, w in self.weights.items():
            total += w * found[k]
        return total.astype(np.float32)


@dataclass(frozen=True)
class _Scores:
    """How stories over the working set are scored, in one place so that the search, a
    story's evidence and ``StoryScorer`` cannot disagree. ``links`` stacks the connections'
    pairwise matrices; ``repeats`` (variety by repetition) is how surely each pair shows
    one space at one kind of shot, or None when variety is likeness."""

    conns: list[_Conn]
    links: NDArray[np.float32]
    similarity: Matrix
    repeats: Matrix | None
    options: TriptychOptions
    roles: _Roles | None = None
    """Set when stories are told by roles (``tell_by``)."""
    likeness: Matrix | None = None
    """How alike each pair is by the view embedder (cosine), for pairs shot within
    ``distinct_within`` of each other (0 for the rest): what ``distinct`` reads. None
    without the view embedder's features."""

    @property
    def distinct_share(self) -> float:
        """``distinct_weight`` of the whole score, or 0 when it cannot be measured (noted).
        Taken from strength and the rest alike, so stories it rates the same keep their
        order: taken from the connections alone, it let a stronger photo win where it
        changed nothing (your H and S, 31 October 2026)."""
        return self.options.distinct_weight if self.likeness is not None else 0.0

    def _distinct(self, closest: ArrayLike) -> Matrix:
        return variety(
            closest,
            too_alike=self.options.distinct_from,
            near_duplicate=self.options.distinct_until,
        )

    def distinct_with(self, a: int, comp: NDArray[np.intp]) -> Matrix:
        """Distinct for every story (x, a, y), x and y from ``comp``: by its closest pair."""
        m = self.likeness
        if m is None:
            return np.ones((len(comp), len(comp)), dtype=np.float32)
        xa, ay = m[comp, a][:, None], m[a, comp][None, :]
        return self._distinct(np.maximum(np.maximum(xa, ay), m[np.ix_(comp, comp)]))

    def closest_pair(self, triple: Triple) -> tuple[int, int, float]:
        """The two of the story most alike by the view embedder, and how alike (0 without
        the view embedder's features)."""
        x, a, y = triple
        m = self.likeness
        if m is None:
            return x, a, 0.0
        i, j = max(((x, a), (a, y), (x, y)), key=lambda p: float(m[p]))
        return i, j, float(m[i, j])

    def distinct_of(self, triple: Triple) -> float:
        if self.likeness is None:
            return 1.0
        return float(self._distinct(self.closest_pair(triple)[2]))

    @property
    def strongest(self) -> bool:
        return self.options.connect_by == "strongest"

    def partners(self, a: int, pool: NDArray[np.intp]) -> NDArray[np.float32]:
        """How well each of ``pool`` would sit beside anchor ``a``, to shortlist them. By
        roles, half of it is how well the photo pairs with any other: an echo's two outer
        photos need not be like the anchor."""
        d = self.distinct_share
        partner = self._partners(a, pool)
        if d and self.likeness is not None:
            partner = (1 - d) * partner + d * self._distinct(self.likeness[a, pool])
        return np.asarray(partner, dtype=np.float32)

    def _partners(self, a: int, pool: NDArray[np.intp]) -> NDArray[np.float32]:
        if self.roles is not None:
            among = self.roles.pairs[np.ix_(pool, pool)].copy()
            np.fill_diagonal(among, -np.inf)
            best = among.max(axis=1) if len(pool) > 1 else np.zeros(len(pool), np.float32)
            return np.asarray(0.5 * self.roles.pairs[a, pool] + 0.5 * best, dtype=np.float32)
        to_anchor = self.links[:, a, pool]
        link = to_anchor.max(axis=0) if self.strongest else to_anchor.mean(axis=0)
        v = self.options.variety_weight
        if self.repeats is None:
            varied = _variety(self.similarity[a, pool], self.options)
        else:
            varied = 1 - self.repeats[a, pool]
        return ((1 - v) * link + v * varied).astype(np.float32)

    def pairs_with(
        self, a: int, comp: NDArray[np.intp]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Connection and variety of every story (x, a, y) with x, y from ``comp``."""
        per_link = _triple_means(self.links, a, comp)
        connection = per_link.max(axis=0) if self.strongest else per_link.mean(axis=0)
        if self.repeats is None:
            varied = _variety(_triple_means(self.similarity, a, comp), self.options)
        else:
            varied = 1 - _triple_means(self.repeats, a, comp)
        return connection, varied

    def values_with(self, a: int, comp: NDArray[np.intp]) -> NDArray[np.float32]:
        """The side of every story (x, a, y) that is not strength, x and y from ``comp``."""
        if self.roles is not None:
            side = self.roles.value_with(a, comp)
        else:
            v = self.options.variety_weight
            connection, varied = self.pairs_with(a, comp)
            side = ((1 - v) * connection + v * varied).astype(np.float32)
        return side

    def links_of(self, triple: Triple) -> list[float]:
        """Each connection's value for one story: the mean over its three pairs."""
        return [_mean_over_pairs(c.matrix, triple) for c in self.conns]

    def connection_of(self, triple: Triple) -> float:
        values = self.links_of(triple)
        return max(values) if self.strongest else float(np.mean(values))

    def variety_of(self, triple: Triple) -> float:
        if self.repeats is None:
            return float(_variety(_mean_over_pairs(self.similarity, triple), self.options))
        return 1 - _mean_over_pairs(self.repeats, triple)

    def variety_detail(self, triple: Triple) -> str:
        if self.repeats is None:
            alike = _mean_over_pairs(self.similarity, triple)
            return (
                f"the three are {alike:.2f} alike on average; "
                f"from {self.options.too_alike:g} they read as one moment"
            )
        x, a, y = triple
        repeated = sum(self.repeats[i, j] >= 0.5 for i, j in ((x, a), (a, y), (x, y)))
        if not repeated:
            return "no two show one space at one kind of shot"
        return f"{repeated} of 3 pairs show one space at one kind of shot"

    def score(self, strength: float, triple: Triple) -> float:
        w, v = self.options.strength_weight, self.options.variety_weight
        if self.roles is not None:
            parts = self.roles.parts(triple)
            side = sum(weight * parts[k] for k, weight in self.roles.weights.items())
        else:
            side = (1 - v) * self.connection_of(triple) + v * self.variety_of(triple)
        total = w * strength + (1 - w) * side
        d = self.distinct_share
        return total if not d else (1 - d) * total + d * self.distinct_of(triple)


@dataclass(frozen=True)
class StoryScorer:
    """Any three photos of a working set scored as the task scores a story, to measure
    stories made elsewhere (``calico eval finished-stories``). Rows are working-set rows."""

    keys: tuple[str, ...]
    shortlisted: NDArray[np.bool_]
    strength: NDArray[np.float32]
    scores: _Scores
    anchored: NDArray[np.bool_]
    """The rows the task would try as anchors (without an anchor photo), as ``run`` does."""

    def row(self, key: str) -> int:
        return self.keys.index(key)

    def allowed(self, a: int, b: int, c: int) -> bool:
        """Whether the task could tell this story: no two of the three near-duplicates."""
        cut, sim = self.scores.options.max_similarity, self.scores.similarity
        return all(sim[i, j] < cut for i, j in ((a, b), (a, c), (b, c)))

    def variety_of(self, a: int, b: int, c: int) -> float:
        return self.scores.variety_of((a, b, c))

    def score(self, a: int, b: int, c: int) -> float:
        return self.scores.score(float(np.mean(self.strength[[a, b, c]])), (a, b, c))

    def roles_of(self, a: int, b: int, c: int) -> dict[str, float]:
        """Each role's value for the story, b its anchor; empty unless told by roles."""
        roles = self.scores.roles
        return {} if roles is None else roles.parts((a, b, c))

    def signals(self, a: int, b: int, c: int) -> dict[str, float]:
        """Every signal the task can read of the story, b its anchor, whichever it was told
        with: its mean strength, each connection, variety by likeness and, told by roles,
        each role that can be measured. So a scorer can be learned from your answers over
        all of them (``evaluation.learn_stories``)."""
        triple = (a, b, c)
        found = {"strength": float(np.mean(self.strength[[a, b, c]]))}
        links = self.scores.links_of(triple)
        found |= {conn.name.value: v for conn, v in zip(self.scores.conns, links, strict=True)}
        found["variety"] = float(self.scores.variety_of(triple))
        if self.scores.likeness is not None:
            found["distinct"] = self.scores.distinct_of(triple)
        return found | self.roles_of(a, b, c)


class TriptychTask(Task[TriptychOptions]):
    name = "triptych"
    title = "Three-photo stories"
    description = "Sets of three photographs that connect through theme, mood, colour or place."
    reference_use = ReferenceUse.OPTIONAL
    default_limit = 6
    default_describe_weight = 1 / 3
    """Of each photo's strength; connections between the three are scored apart."""
    reference_label = "Anchor photo"
    reference_help = (
        "Your photo goes in the middle; Calico finds pairs of photos that tell a story with it."
    )
    uses_taste = True
    options_model = TriptychOptions

    def run(self, ctx: TaskContext, options: TriptychOptions) -> list[Group]:
        trace = ctx.trace
        strength = _strength(ctx, self.describe_weight(ctx))
        trace.record("strength", strength)

        rows, alternates = self._working_set(ctx, options, strength)
        if len(rows.photos) < 3:
            trace.note("Fewer than three usable photos, so no triptych can be made.")
            return []

        scores = _scores(ctx, options, rows)

        anchors, by_kind = self._anchors(ctx, rows, options, scores.roles)
        uses: Counter[int] = Counter()
        alike = _spent_with(ctx, options, rows, scores)
        taken: Counter[int] = Counter()
        made = {"before": 0, "strict": alike is not None}

        def offer(anchor: int) -> tuple[float, int, int, int] | None:
            """The anchor's best pair among photos that still have uses left."""
            held = taken if made["strict"] else Counter[int]()
            spent = np.array(
                [uses[r] + held[r] >= options.max_uses for r in range(len(rows.photos))]
            )
            if spent[anchor] and not ctx.references:
                return None
            return _best_pair(anchor, rows, scores=scores, spent=spent)

        def build(anchor: int, x: int, y: int, number: int) -> Group:
            used = (x, y) if ctx.references else (anchor, x, y)
            uses.update(used)
            if alike is not None:
                also = np.flatnonzero(alike[list(used)].any(axis=0))
                taken.update(int(r) for r in also if r not in used)
            return _group(
                ctx,
                rows=rows,
                alternates=alternates,
                scores=scores,
                triple=(x, anchor, y),
                number=made["before"] + number,
                anchored=bool(ctx.references),
            )

        def tell(limit: int) -> list[Group]:
            if ctx.references or by_kind:
                return _round_robin(anchors, limit, offer, build)
            return _best_first(anchors, limit, rows, offer, build)

        with trace.stage("tell stories", len(anchors)) as stage:
            groups = tell(ctx.limit)
            if made["strict"] and len(groups) < ctx.limit:
                made["before"], made["strict"] = len(groups), False
                more = tell(ctx.limit - len(groups))
                groups += more
                if more:
                    trace.note(
                        f"spend_alike: {len(more)} of {len(groups)} stories use a photo that "
                        "looks like one in an earlier story, because nothing else was left "
                        "to make a story from. Ask for fewer stories to avoid them."
                    )
            stage.count_out = len(groups)
        counts = np.asarray([uses[r] for r in range(len(rows.photos))], dtype=np.float32)
        _record(ctx, rows, "uses", counts)
        if not groups:
            trace.note("No set of three distinct photos met the constraints.")
        return groups

    def scorer(
        self, ctx: TaskContext, options: TriptychOptions, include: Collection[str]
    ) -> StoryScorer:
        """Score stories as this task would, over its working set plus the ``include``d
        photos (by content key), which are scored even if too weak to be candidates."""
        strength = _strength(ctx, self.describe_weight(ctx))
        rows, _ = self._working_set(ctx, options, strength, also=include)
        scores = _scores(ctx, options, rows)
        anchored = np.zeros(len(rows.photos), dtype=bool)
        anchored[self._anchors(ctx, rows, options, scores.roles)[0]] = True
        return StoryScorer(
            keys=tuple(p.content_key for p in rows.photos),
            shortlisted=rows.shortlisted,
            strength=rows.strength,
            scores=scores,
            anchored=anchored,
        )

    @staticmethod
    def _working_set(
        ctx: TaskContext,
        options: TriptychOptions,
        strength: NDArray[np.float32],
        also: Collection[str] = (),
    ) -> tuple[_Rows, dict[int, list[int]]]:
        """Collapse bursts, keep the strongest ``candidates``, then add the anchors (and
        ``also``, photos a measurement wants scored)."""
        trace, photos = ctx.trace, ctx.photos
        ref_keys = set(ctx.reference_keys)
        with trace.stage("collapse bursts", len(photos)) as stage:
            bursts = group_bursts(
                photos,
                ctx.view.embeddings,
                window_seconds=options.burst_window_seconds,
                min_similarity=options.burst_similarity,
            )
            reps = representatives(bursts, strength)
            stage.count_out = len(reps)
        alternates = dict(reps)
        # A reference's own burst must not supply its companions.
        blocked = {
            rep
            for (rep, _), burst in zip(reps, bursts, strict=True)
            if any(photos[i].content_key in ref_keys for i in burst)
        }
        ranked = sorted(
            (rep for rep, _ in reps if rep not in blocked),
            key=lambda i: (-float(strength[i]), photos[i].path),
        )[: options.candidates]
        in_pool = {p.content_key: i for i, p in enumerate(photos)}
        pool_refs = [in_pool[k] for k in ctx.reference_keys if k in in_pool]
        asked = [in_pool[k] for k in also if k in in_pool]
        chosen = sorted({*ranked, *pool_refs, *asked}, key=lambda i: photos[i].path)
        burst_of = {i: rep for (rep, _), burst in zip(reps, bursts, strict=True) for i in burst}
        strongest = set(ranked)
        trace.record("in_working_set", scatter(len(photos), chosen, np.ones(len(chosen))))
        with trace.stage("working set", len(reps)) as stage:
            stage.count_out = len(chosen)

        row_photos = [photos[i] for i in chosen]
        pool_index: list[int | None] = list(chosen)
        views = [ctx.view.take(chosen)]
        row_strength = [float(strength[i]) for i in chosen]
        # An uploaded reference has no model scores of its own; the working set's median is
        # neutral, where a perfect 1.0 would inflate every story built around it.
        neutral = float(np.median(row_strength)) if row_strength else 0.5
        external = [
            r for r, k in enumerate(ctx.reference_keys) if k not in in_pool and ctx.references
        ]
        if external and ctx.references is not None:
            views.append(ctx.references.take(external))
            for r in external:
                row_photos.append(_reference_photo(ctx.reference_paths[r], ctx.reference_keys[r]))
                pool_index.append(None)
                row_strength.append(neutral)
        is_ref = np.array([p.content_key in ref_keys for p in row_photos], dtype=bool)
        shortlisted = np.array(
            [i is not None and burst_of[i] in strongest for i in pool_index], dtype=bool
        )
        rows = _Rows(
            photos=row_photos,
            pool_index=pool_index,
            strength=np.asarray(row_strength, dtype=np.float32),
            is_reference=is_ref,
            features=_stack(views),
            shortlisted=shortlisted,
        )
        return rows, alternates

    @staticmethod
    def _anchors(
        ctx: TaskContext, rows: _Rows, options: TriptychOptions, roles: _Roles | None = None
    ) -> tuple[list[int], bool]:
        """The anchors in the order they are tried, and whether they take turns by kind
        of shot (``anchor_kinds``): the strongest of each kind named, then the next; as
        many as ``anchors_tried``, or three per story asked for (at least 10). Told
        by roles, or with ``anchor_by="likely"``, the likeliest anchors (``anchor_prior``)
        instead of the strongest. Only candidates anchor: a photo a measurement asked to
        score (``scorer(include=)``) is not one."""
        if ctx.references is not None:
            for name in ("anchor_kinds", "anchor_by", "anchors_tried"):
                if getattr(options, name) != TriptychOptions.model_fields[name].default:
                    ctx.trace.note(
                        f"{name} is ignored: the stories are built around your anchor photo."
                    )
            by_key = {p.content_key: r for r, p in enumerate(rows.photos)}
            return [by_key[k] for k in ctx.reference_keys], False
        rank = _anchor_rank(ctx, options, rows, roles)
        order = sorted(
            (r for r in range(len(rows.photos)) if rows.shortlisted[r]),
            key=lambda r: (-float(rank[r]), r),
        )
        count = options.anchors_tried or max(3 * ctx.limit, 10)
        if not options.anchor_kinds:
            return order[:count], False
        found = _label_probs(
            ctx,
            "shot_kind",
            rows.features,
            consequence="the strongest photos anchor stories, whatever anchor_kinds says",
        )
        if found is None:
            return order[:count], False
        labels, probs = found
        for kind in options.anchor_kinds:
            if kind not in labels:
                ctx.trace.note(
                    f"{kind!r} is not a kind of shot ({', '.join(labels)}), so it anchors no story."
                )
        top = np.argmax(probs, axis=1)
        queues = [
            [r for r in order if labels[top[r]] == kind]
            for kind in dict.fromkeys(options.anchor_kinds)
            if kind in labels
        ]
        turns = [r for queue in itertools.zip_longest(*queues) for r in queue if r is not None]
        if not turns:
            ctx.trace.note(
                "No photo here is of a kind anchor_kinds names, so the strongest photos "
                "anchor stories instead."
            )
            return order[:count], False
        return turns[:count], True


type Found = tuple[float, int, int, int]


def _round_robin(
    anchors: list[int],
    limit: int,
    offer: Callable[[int], Found | None],
    build: Callable[[int, int, int, int], Group],
) -> list[Group]:
    """Given anchors every story is built around one: take turns, best pair first.

    Companions are spent as stories are made (``max_uses``), so each anchor's next
    story uses different photos. Stops at ``limit`` or when nobody can tell another.
    """
    groups: list[Group] = []
    active = list(anchors)
    while active and len(groups) < limit:
        for anchor in list(active):
            if len(groups) >= limit:
                break
            found = offer(anchor)
            if found is None:
                active.remove(anchor)
                continue
            groups.append(build(anchor, found[2], found[3], len(groups) + 1))
    return groups


def _best_first(
    anchors: list[int],
    limit: int,
    rows: _Rows,
    offer: Callable[[int], Found | None],
    build: Callable[[int, int, int, int], Group],
) -> list[Group]:
    """No given anchor: accept the highest-scoring stories, any photo as anchor.

    A story whose companion was spent meanwhile is re-searched for that anchor's
    next best pair; an accepted anchor is offered again if reuse allows.
    """
    heap: list[tuple[float, Path, int, int, int]] = []

    def push(anchor: int) -> None:
        if (found := offer(anchor)) is not None:
            heapq.heappush(heap, (-found[0], rows.photos[anchor].path, anchor, found[2], found[3]))

    for anchor in anchors:
        push(anchor)
    groups: list[Group] = []
    while heap and len(groups) < limit:
        _, _, anchor, x, y = heapq.heappop(heap)
        current = offer(anchor)
        if current is None:
            continue
        if (current[2], current[3]) != (x, y):
            push(anchor)  # photos were spent since this was queued: re-queue the next best
            continue
        groups.append(build(anchor, x, y, len(groups) + 1))
        push(anchor)
    return groups


def _reference_photo(path: Path, key: str) -> Photo:
    """A reference outside the catalog, as a row of the working set (location unknown)."""
    return Photo(
        path=path, content_key=key, collection="", location="", captured_at=None, width=0, height=0
    )


def _stack(views: list[FeatureView]) -> FeatureView:
    """Concatenate views, keeping only the feature matrices every part has."""
    shared = set(views[0].features)
    for v in views[1:]:
        shared &= set(v.features)
    return FeatureView(
        embeddings=np.vstack([v.embeddings for v in views]),
        features={k: np.vstack([v.features[k] for v in views]) for k in sorted(shared)},
        columns=views[0].columns,
    )


def _strength(ctx: TaskContext, describe_weight: float) -> NDArray[np.float32]:
    """Pool-wide strength with the shared convention."""
    scores, _ = photo_strength(
        ctx,
        range(len(ctx.photos)),
        quality_weight=QUALITY_WEIGHT,
        aesthetic_weight=AESTHETIC_WEIGHT,
        criteria_share=describe_weight,
    )
    return scores


def _label_probs(
    ctx: TaskContext,
    name: str,
    features: FeatureView,
    consequence: str = "that connection is dropped",
) -> tuple[tuple[str, ...], Matrix] | None:
    try:
        model = ctx.models.get(name)
    except CapabilityError as exc:
        ctx.trace.note(f"The {name!r} model is unavailable, so {consequence}: {exc}")
        return None
    if not isinstance(model, LabelModel):
        ctx.trace.note(f"The {name!r} model does not predict labels, so {consequence}.")
        return None
    return model.labels, finite_or_zero(ctx.trace, name, model.predict_proba(features))


def _scores(ctx: TaskContext, options: TriptychOptions, rows: _Rows) -> _Scores:
    """The chosen connections, the rows' cosine similarity and, for variety by
    repetition, which pairs repeat one space at one kind of shot."""
    conns = _connections(ctx, options, rows)
    emb = rows.features.embeddings
    links = np.stack([c.matrix for c in conns]).astype(np.float32)
    roles = options.tell_by == "roles"
    if roles:
        _say_what_roles_ignore(ctx, options)
    return _Scores(
        conns=conns,
        links=links,
        similarity=emb @ emb.T,
        repeats=(
            _repeats(ctx, options, rows)
            if options.variety_by == "repetition" and not roles
            else None
        ),
        options=options,
        roles=_roles(ctx, options, rows, links) if options.tell_by == "roles" else None,
        likeness=_likeness(ctx, options, rows),
    )


def _spent_with(
    ctx: TaskContext, options: TriptychOptions, rows: _Rows, scores: _Scores
) -> NDArray[np.bool_] | None:
    """Which photos a used photo takes with it (``spend_alike``), or None when off or
    when the view embedder's features are missing (noted)."""
    if not options.spend_alike:
        return None
    lacking = "spend_alike (a used photo also uses up its look-alikes; ignored here)"
    by_view = _view_similarity(ctx, rows, lacking) if ctx.view_feature else None
    if by_view is None:
        if not ctx.view_feature:
            ctx.trace.note(
                "spend_alike needs a view embedder, and none is set (view_embedder in "
                "calico.toml), so only the photo itself is used up."
            )
        return None
    places = np.array([p.location for p in rows.photos])
    same_place = places[:, None] == places[None, :]
    return np.asarray(
        (same_place & (by_view >= options.spend_alike))
        | (scores.similarity >= options.max_similarity)
    )


def _view_similarity(ctx: TaskContext, rows: _Rows, lacking: str | None) -> Matrix | None:
    """How alike each pair is by the view embedder (cosine), or None, with a note saying
    what is ``lacking`` (when given) without the view embedder's features."""
    name = ctx.view_feature
    vectors = rows.features.features.get(name) if name else rows.features.embeddings
    if vectors is None:
        if lacking is not None:
            ctx.trace.note(
                f"{lacking} needs the view embedder's features ({name!r}), which these photos "
                "lack. Index them with the view embedder (calico index) to use it."
            )
        return None
    unit_rows = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return (unit_rows @ unit_rows.T).astype(np.float32)


def _same_space(
    ctx: TaskContext, options: TriptychOptions, rows: _Rows, lacking: str
) -> NDArray[np.bool_] | None:
    """Which pairs may show one space (the view embedder, from ``same_space``), or None
    with a note saying what is ``lacking`` without the view embedder's features."""
    similarity = _view_similarity(ctx, rows, lacking)
    return None if similarity is None else np.asarray(similarity >= options.same_space)


def _likeness(ctx: TaskContext, options: TriptychOptions, rows: _Rows) -> Matrix | None:
    """What ``distinct`` reads: view-embedder likeness of pairs shot within
    ``distinct_within`` of each other (on one clock; undated photos never are). Noted as
    lacking only when it counts."""
    lacking = (
        "Distinct (no two photos of one view or one look; its weight goes to the rest)"
        if options.distinct_weight > 0
        else None
    )
    similarity = _view_similarity(ctx, rows, lacking)
    if similarity is None or not options.distinct_within:
        return similarity
    epoch = datetime(1970, 1, 1)
    seconds = np.array(
        [math.nan if t is None else (t - epoch).total_seconds() for t in on_one_clock(rows.photos)]
    )
    moment = np.abs(seconds[:, None] - seconds[None, :]) <= options.distinct_within
    return np.where(moment, similarity, 0.0).astype(np.float32)


def _distinct_detail(scores: _Scores, rows: _Rows, triple: Triple) -> str:
    """Which two of the story are most alike as one moment, or that none are."""
    options = scores.options
    within = options.distinct_within
    when = f" shot within {within / 60:g} minutes" if within else ""
    i, j, alike_by_view = scores.closest_pair(triple)
    if within and alike_by_view < options.distinct_from:
        return f"no two of the three were{when} of each other and look alike"
    return (
        f"closest pair{when}: {rows.photos[i].path.name} and {rows.photos[j].path.name}, "
        f"{alike_by_view:.2f} alike by the view embedder; from {options.distinct_from:g} two "
        f"read as one look, at {options.distinct_until:g} as one view"
    )


def _anchor_rank(
    ctx: TaskContext, options: TriptychOptions, rows: _Rows, roles: _Roles | None
) -> NDArray[np.float32]:
    """What orders the anchors: the anchor prior told by roles or with ``anchor_by=
    "likely"``, recorded so ``calico runs explain`` can say why a photo anchored; else
    strength. Without scale or subject the prior has nothing to add, so strength."""
    if roles is not None:
        return roles.prior(rows.strength)
    if options.anchor_by == "strength":
        return rows.strength
    scale, living = _scale_and_living(
        ctx,
        rows,
        without_scale="anchors are tried without how wide each photo is",
        without_living="anchors are tried without whether a person or animal is in them",
    )
    if scale is None and living is None:
        ctx.trace.note(
            "anchor_by='likely' needs the 'scale' or 'subject' vocabulary, so the strongest "
            "photos anchor stories instead."
        )
        return rows.strength
    prior = anchor_prior(rows.strength, scale=scale, living=living)
    _record(ctx, rows, "anchor_prior", prior)
    return prior


def _scale_and_living(
    ctx: TaskContext, rows: _Rows, *, without_scale: str, without_living: str
) -> tuple[NDArray[np.float32] | None, NDArray[np.float32] | None]:
    """Each photo's scale (0 aerial ... 4 macro) and how surely it shows a person or
    animal, each traced, or None (noted, with what is lost) where its vocabulary is
    missing."""
    scale = living = None
    found = _label_probs(ctx, "scale", rows.features, consequence=without_scale)
    if found is not None:
        scale = expected_scale(*found)
        if scale is None:
            names = ", ".join(SCALES)
            ctx.trace.note(f"The 'scale' model does not name {names}, so it is not used.")
        else:
            _record(ctx, rows, "scale", scale)
    found = _label_probs(ctx, "subject", rows.features, consequence=without_living)
    if found is not None and "living" in found[0]:
        living = found[1][:, found[0].index("living")].astype(np.float32)
        _record(ctx, rows, "living", living)
    elif found is not None:
        ctx.trace.note("The 'subject' model has no 'living' label, so it is not used.")
    return scale, living


def _roles(ctx: TaskContext, options: TriptychOptions, rows: _Rows, links: Matrix) -> _Roles:
    """What telling by roles reads of the working set, noting each part it must go
    without; each photo's scale and person-or-animal are traced."""
    scale, living = _scale_and_living(
        ctx,
        rows,
        without_scale="no story is arranged by a change of scale",
        without_living="no anchor counts for a person or animal",
    )
    same = _same_space(
        ctx, options, rows, "Holding back three shots of one space when telling by roles"
    )
    one_scale = None if scale is None else np.abs(scale[:, None] - scale[None, :]) < 0.5
    repeats = None if same is None or one_scale is None else same & one_scale
    lost = []
    if repeats is None:
        lost.append("'fresh' (three shots of one space at one scale)")
    if living is None and scale is None:
        lost.append("'anchor' (a person or animal, or the widest view)")
    if lost:
        ctx.trace.note(
            f"Telling by roles: {' and '.join(lost)} not measured, so the other roles share "
            "its weight."
        )
    return _Roles(pairs=links.mean(axis=0), scale=scale, living=living, repeats=repeats)


def _record(ctx: TaskContext, rows: _Rows, name: str, values: NDArray[np.float32]) -> None:
    """Trace a per-row signal over the pool, so ``calico runs explain`` can show it."""
    at = [(i, float(values[r])) for r, i in enumerate(rows.pool_index) if i is not None]
    found = np.array([v for _, v in at])
    ctx.trace.record(name, scatter(len(ctx.photos), [i for i, _ in at], found))


ROLES_IGNORE = ("connect_by", "variety_by", "variety_weight", "too_alike")
"""Options of telling by pairs that a story told by roles does not read."""


def _say_what_roles_ignore(ctx: TaskContext, options: TriptychOptions) -> None:
    fields = TriptychOptions.model_fields
    unused = [k for k in ROLES_IGNORE if getattr(options, k) != fields[k].default]
    if unused:
        ctx.trace.note(
            f"Telling by roles scores a story by its roles, so {', '.join(unused)} "
            f"{'does' if len(unused) == 1 else 'do'} not apply."
        )


def _repeats(ctx: TaskContext, options: TriptychOptions, rows: _Rows) -> Matrix | None:
    """How surely each pair is one space (the view embedder, from ``same_space``) seen
    at one kind of shot (how much their ``shot_kind`` labels overlap). A person or a
    detail in the room is another kind of shot, so it does not repeat the room."""
    same = _same_space(
        ctx,
        options,
        rows,
        "Variety by repetition (the three are judged by how alike they are instead)",
    )
    if same is None:
        return None
    same_space = same.astype(np.float32)
    found = _label_probs(
        ctx,
        "shot_kind",
        rows.features,
        consequence="repetition counts one space alone, whatever the kind of shot",
    )
    if found is None:
        return same_space
    _, probs = found
    return (same_space * bhattacharyya(probs, probs)).astype(np.float32)


def _connections(ctx: TaskContext, options: TriptychOptions, rows: _Rows) -> list[_Conn]:
    """Build the chosen connection matrices, dropping (with a note) any that cannot run."""
    built: list[_Conn] = []
    for name in dict.fromkeys(options.connections):
        conn = _build(ctx, name, rows)
        if conn is not None:
            built.append(conn)
    if not built:
        ctx.trace.note("None of the chosen connections is available; using 'theme' instead.")
        built.append(_theme(ctx, rows))
    return built


def _build(ctx: TaskContext, name: Connection, rows: _Rows) -> _Conn | None:
    match name:
        case Connection.THEME:
            return _theme(ctx, rows)
        case Connection.LOCATION:
            return _location(rows)
        case Connection.PLACE_TYPE | Connection.MOOD | Connection.SURFACE:
            return _distribution(ctx, name, rows)
        case Connection.COLOR:
            return _color(ctx, rows)
        case Connection.KIND_MIX | Connection.KIND_SAME:
            return _kind(ctx, name, rows)


def _finite(ctx: TaskContext, what: str, matrix: NDArray[np.floating]) -> NDArray[np.float32]:
    """``matrix`` with each row that is not all finite numbers set to 0, saying how many:
    such a photo connects to nothing, so it can never be in a story (audit A03-16)."""
    bad = ~np.isfinite(matrix).all(axis=1)
    if not bad.any():
        return matrix.astype(np.float32, copy=False)
    count = int(bad.sum())
    ctx.trace.note(
        f"{count} photo{'' if count == 1 else 's'} {'has' if count == 1 else 'have'} {what} "
        "that is not finite, so it connects to nothing. Index the folders again "
        "(calico index FOLDER)."
    )
    return np.where(bad[:, None], np.float32(0.0), matrix).astype(np.float32)


def _theme(ctx: TaskContext, rows: _Rows) -> _Conn:
    emb = _finite(ctx, "an embedding", rows.features.embeddings)
    return _Conn(
        Connection.THEME, pairwise_percentile(emb @ emb.T), lambda _: "similar subject matter"
    )


def _distribution(ctx: TaskContext, name: Connection, rows: _Rows) -> _Conn | None:
    """Place type, mood or surface: how much two photos' label distributions overlap."""
    found = _label_probs(ctx, name.value, rows.features)
    if found is None:
        return None
    labels, probs = found
    m = pairwise_percentile(bhattacharyya(probs, probs))
    word = "place type" if name is Connection.PLACE_TYPE else name.value
    return _Conn(name, m, lambda t: f"similar {word}: {_shared_label(labels, probs, t)}")


def _color(ctx: TaskContext, rows: _Rows) -> _Conn | None:
    palette = rows.features.features.get("palette")
    if palette is None:
        ctx.trace.note("No palette features are available, so 'color' is dropped.")
        return None
    m = pairwise_percentile(_intersection(_finite(ctx, "a palette", palette)))
    return _Conn(Connection.COLOR, m, lambda _: "similar palette")


def _kind(ctx: TaskContext, name: Connection, rows: _Rows) -> _Conn | None:
    """Shot kind at the coarse level: same kind, or its complement for a varied set."""
    found = _label_probs(ctx, "shot_kind", rows.features)
    if found is None:
        return None
    labels, probs = rollup(found[0], found[1], 1)
    same = bhattacharyya(probs, probs)
    if name is Connection.KIND_SAME:
        return _Conn(name, same, lambda t: f"same kind: {_shared_label(labels, probs, t)}")
    return _Conn(name, 1 - same, lambda t: f"complementary kinds: {_kinds(labels, probs, t)}")


def _location(rows: _Rows) -> _Conn:
    """1 for the same known place, 0 for two known different places, 0.5 if unsure."""
    loc = np.array([place_key(p.location) for p in rows.photos])
    known = loc != ""
    both = known[:, None] & known[None, :]
    m = np.where(both, (loc[:, None] == loc[None, :]).astype(np.float32), 0.5).astype(np.float32)

    def describe(t: Triple) -> str:
        places = {place_key(rows.photos[r].location) for r in t}
        if len(places) == 1 and "" not in places:
            return f"same location: {rows.photos[t[0]].location}"
        return "different or unknown locations"

    return _Conn(Connection.LOCATION, m, describe)


def _intersection(palette: Matrix) -> Matrix:
    """All-pairs histogram intersection in row blocks, to bound peak memory."""
    out = np.empty((len(palette), len(palette)), dtype=np.float32)
    for start in range(0, len(palette), HISTOGRAM_BLOCK):
        block = palette[start : start + HISTOGRAM_BLOCK]
        out[start : start + HISTOGRAM_BLOCK] = histogram_intersection(block, palette)
    return out


def _shared_label(labels: tuple[str, ...], probs: Matrix, triple: Triple) -> str:
    """The label all three photos agree on most."""
    joint = np.prod(probs[list(triple)], axis=0)
    return labels[int(np.argmax(joint))]


def _kinds(labels: tuple[str, ...], probs: Matrix, triple: Triple) -> str:
    return " + ".join(labels[int(np.argmax(probs[r]))] for r in triple)


def _best_pair(
    a: int,
    rows: _Rows,
    *,
    scores: _Scores,
    spent: NDArray[np.bool_],
) -> tuple[float, int, int, int] | None:
    """The best (score, anchor, x, y) for anchor ``a`` among photos not ``spent``, or None."""
    options, sim = scores.options, scores.similarity
    eligible = ~rows.is_reference & ~spent & (sim[a] < options.max_similarity)
    eligible[a] = False
    pool = np.flatnonzero(eligible)
    if len(pool) < 2:
        return None
    partner = scores.partners(a, pool)
    top = np.lexsort((pool, -partner))[: options.companions]
    comp = np.sort(pool[top])
    s = rows.strength[comp]
    w = options.strength_weight
    total = w * (rows.strength[a] + s[:, None] + s[None, :]) / 3 + (1 - w) * scores.values_with(
        a, comp
    )
    if d := scores.distinct_share:
        total = (1 - d) * total + d * scores.distinct_with(a, comp)
    valid = np.triu(sim[np.ix_(comp, comp)] < options.max_similarity, k=1) & np.isfinite(total)
    if not valid.any():
        return None
    i, j = divmod(int(np.argmax(np.where(valid, total, -np.inf))), len(comp))
    return float(total[i, j]), a, int(comp[i]), int(comp[j])


def _triple_means(m: NDArray[np.float32], a: int, comp: NDArray[np.intp]) -> NDArray[np.float32]:
    """For every (x, y) of ``comp``, the mean of ``m`` over the pairs of (x, a, y); ``m``
    is one pairwise matrix or a stack of them (leading axis kept)."""
    ax = m[..., a, comp]
    total = ax[..., :, None] + ax[..., None, :] + m[..., comp[:, None], comp[None, :]]
    return (total / 3).astype(np.float32)


def _mean_over_pairs(m: Matrix, triple: Triple) -> float:
    x, a, y = triple
    return float(np.mean([m[a, x], m[a, y], m[x, y]]))


def _variety(mean_similarity: ArrayLike, options: TriptychOptions) -> Matrix:
    return variety(
        mean_similarity, too_alike=options.too_alike, near_duplicate=options.max_similarity
    )


def _connection_evidence(scores: _Scores, triple: Triple) -> list[Evidence]:
    """Every connection sharing the connection side, or (``connect_by="strongest"``) the
    strongest alone, saying which it was chosen from."""
    options = scores.options
    side = (1 - options.strength_weight) * (1 - options.variety_weight)
    side *= 1 - scores.distinct_share
    values = scores.links_of(triple)
    if not scores.strongest:
        return [
            Evidence(
                signal=c.name.value,
                value=value,
                weight=side / len(scores.conns),
                detail=c.describe(triple),
            )
            for c, value in zip(scores.conns, values, strict=True)
        ]
    best = int(np.argmax(values))
    chosen = scores.conns[best]
    names = ", ".join(c.name.value for c in scores.conns)
    return [
        Evidence(
            signal=chosen.name.value,
            value=values[best],
            weight=side,
            detail=f"{chosen.describe(triple)} (strongest of {names})",
        )
    ]


def _role_evidence(roles: _Roles, rows: _Rows, triple: Triple, *, side: float) -> list[Evidence]:
    """Each role of a story told by roles, saying what filled it."""
    x, a, y = triple
    parts = roles.parts(triple)
    name = [p.path.name for p in rows.photos]
    pairs = {(x, a): roles.pairs[x, a], (a, y): roles.pairs[a, y], (x, y): roles.pairs[x, y]}
    i, j = max(pairs, key=lambda pair: float(pairs[pair]))
    arrangement = "neither an echo nor a change of scale"
    if (i, j) == (x, y):
        arrangement = "an echo: the outer two are the closest pair"
    elif roles.scale is not None and parts["arrangement"] >= 0.5:
        found = roles.scale[[x, a, y]]
        lo, hi = SCALES[round(float(found.min()))], SCALES[round(float(found.max()))]
        arrangement = f"a change of scale, from {lo} to {hi}"
    if roles.living is not None and roles.living[a] >= 0.5:
        anchor = "the anchor shows a person or animal"
    elif roles.scale is not None and parts.get("anchor", 0) >= 0.5:
        anchor = "the anchor is the widest of the three"
    elif roles.scale is None:
        anchor = "the anchor shows no person or animal (its width is not measured)"
    elif roles.living is None:
        anchor = "the anchor is not the widest (a person or animal is not looked for)"
    else:
        anchor = "the anchor shows no person or animal and is not the widest"
    details = {
        "bond": f"closest pair: {name[i]} and {name[j]}",
        "belong": "how the third photo connects with them",
        "arrangement": arrangement,
        "fresh": (
            "not three shots of one space at one scale"
            if parts.get("fresh", 1)
            else "three shots of one space at one scale"
        ),
        "anchor": anchor,
    }
    return [
        Evidence(signal=k, value=parts[k], weight=side * w, detail=details[k])
        for k, w in roles.weights.items()
    ]


def _group(
    ctx: TaskContext,
    *,
    rows: _Rows,
    alternates: dict[int, list[int]],
    scores: _Scores,
    triple: Triple,
    number: int,
    anchored: bool,
) -> Group:
    """One triptych, picks ordered (x, anchor, y); evidence sums to the score."""
    _, a, _ = triple
    options, conns = scores.options, scores.conns
    w, v, d = options.strength_weight, options.variety_weight, scores.distinct_share
    mean_strength = float(np.mean(rows.strength[list(triple)]))
    evidence = [
        Evidence(signal="strength", value=mean_strength, weight=w * (1 - d), detail="mean strength")
    ]
    if scores.roles is not None:
        evidence += _role_evidence(scores.roles, rows, triple, side=(1 - w) * (1 - d))
    else:
        evidence += _connection_evidence(scores, triple)
    if v > 0 and scores.roles is None:
        evidence.append(
            Evidence(
                signal="variety",
                value=scores.variety_of(triple),
                weight=(1 - w) * (1 - d) * v,
                detail=scores.variety_detail(triple),
            )
        )
    if d:
        evidence.append(
            Evidence(
                signal="distinct",
                value=scores.distinct_of(triple),
                weight=d,
                detail=_distinct_detail(scores, rows, triple),
            )
        )
    picks = []
    for r in triple:
        s = float(rows.strength[r])
        detail = (
            "photo strength" if rows.pool_index[r] is not None else "reference (median strength)"
        )
        own = (Evidence(signal="strength", value=s, weight=1.0, detail=detail),)
        idx = rows.pool_index[r]
        if idx is None:
            picks.append(make_pick(rows.photos, r, s, own))
        else:
            picks.append(make_pick(ctx.photos, idx, s, own, alternates.get(idx, [])))
    return Group(
        title=(
            f"Story {number} with {rows.photos[a].path.name}"
            if anchored
            else f"Story {number}: " + " · ".join(c.name.value for c in conns)
        ),
        score=sum(e.contribution for e in evidence),
        picks=tuple(picks),
        evidence=tuple(evidence),
    )
