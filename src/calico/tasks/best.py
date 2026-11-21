"""``best``: the strongest photographs in the pool.

Strength is a weighted mix of technical quality, aesthetic strength and match
to the criteria text. Bursts collapse to their best frame, then MMR spreads the
result so it is not twelve views of one doorway.

MMR rewards being unlike what is already picked, and at a place full of rooms a
museum placard is as unlike as it gets: at one site, diversity 0.5 put a book cover
second although it was among the weakest photos. So diversity only chooses among
the strongest ``diversity_reach`` times as many photos as were asked for.

**One pick per view.** Strongest is a shortlist of portfolio-worthy candidates, as many as
the request asks for, so a view shot on two occasions should take one place in it, not two
(owner's choice, 5 November 2026). Bursts only join frames seconds apart, and MMR
discourages a repeat without preventing one. So the strongest photos that look alike as a
whole are compared part by part (``patch_pairs``; the engine remembers every pair, as for
recurring views) and grouped into views the way recurring views groups them; each view's
strongest frame is the pick and its other frames are alternates.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field

from calico.domain import Group, Marker, Pick
from calico.errors import CapabilityError, InvalidRequestError
from calico.models.base import LabelModel, rollup
from calico.models.taste import TasteModel
from calico.selection.bursts import group_bursts, representatives
from calico.selection.context import Trace
from calico.selection.diversity import mmr
from calico.selection.picks import make_pick, ordinal, scatter
from calico.selection.scoring import evidence_for, percentile_rank
from calico.selection.strength import AESTHETIC_WEIGHT, QUALITY_WEIGHT, photo_strength
from calico.selection.view_groups import CANDIDATE_SIMILARITY, GROUP_SHARE, group_views
from calico.tasks.base import ReferenceUse, Task, TaskOptions

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from calico.selection.context import TaskContext
    from calico.selection.scoring import Signal

_BLOCK = 1024
"""Rows of whole-photo similarity computed at a time, so a wide reach never holds every pair."""

VIEW_REACH = 2
"""Views are looked for among the strongest ``VIEW_REACH`` x ``diversity_reach`` times as
many photos as were asked for. Once a view's other frames fold into its pick, weaker photos
move up into diversity's reach, and they must have been compared too. Twice the reach (about
240 photos for a first page of 40) costs about a minute the first time at a place, and
nothing after: every pair compared is remembered."""


class BestOptions(TaskOptions):
    kind: str | None = Field(
        default=None,
        description="Only this kind of image, e.g. 'portrait' or 'establishing:interior'",
    )
    min_kind_probability: float = Field(
        default=0.4, ge=0, le=1, description="How sure the kind model must be to keep a photo"
    )
    quality_weight: float = Field(
        default=QUALITY_WEIGHT,
        ge=0,
        le=1,
        description="Weight of technical quality (sharpness, exposure)",
    )
    aesthetic_weight: float = Field(
        default=AESTHETIC_WEIGHT, ge=0, le=1, description="Weight of aesthetic strength"
    )
    diversity: float = Field(
        default=0.3, ge=0, le=1, description="0 = pure ranking, 1 = maximally varied results"
    )
    diversity_reach: float = Field(
        default=3,
        ge=1,
        le=100,
        description="Diversity chooses only among the strongest (this many times the number "
        "asked for) photos, so it never brings in a weak one",
    )
    burst_window_seconds: float = Field(
        default=120, ge=0, le=3600, description="Frames this close in time may be one burst"
    )
    burst_similarity: float = Field(
        default=0.9, ge=0, le=1, description="Frames this alike (cosine) in a window are one burst"
    )
    one_per_view: bool = Field(
        default=True,
        description="Show one photo of each view: its other frames, from any occasion, become "
        "alternates",
    )
    view_similarity: float = Field(
        default=CANDIDATE_SIMILARITY,
        ge=0,
        le=1,
        description="Only photos this alike as a whole are compared detail by detail, to "
        "find one view",
    )
    view_share: float = Field(
        default=GROUP_SHARE,
        ge=0,
        le=1,
        description="How much of their details the photos of one view share, on average",
    )
    marker_probability: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Yours draws the line above the first pick you are less likely than this "
        "to like, so the list can be read down to where it stops being good",
    )


class BestTask(Task[BestOptions]):
    """One result = one photo; ``Group.score`` is the mean of its picks' scores."""

    name = "best"
    title = "Strongest photographs"
    description = "The strongest photographs in the chosen folders, one per view and varied."
    reference_use = ReferenceUse.UNUSED
    default_limit = 24
    default_describe_weight = 0.5
    uses_taste = True
    options_model = BestOptions

    def patch_pairs(self, ctx: TaskContext, options: BestOptions) -> NDArray[np.intp] | None:
        """Pairs among the strongest photos (``VIEW_REACH``) that look alike as a whole
        at one location, to compare part by part so that each view is shown once."""
        if not options.one_per_view:
            return None
        quiet = dataclasses.replace(ctx, trace=Trace())  # run records these steps itself
        ranked = self._ranked(quiet, options)
        if ranked is None:
            return np.zeros((0, 2), dtype=np.intp)
        candidates, scores, _ = ranked
        reps = self._bursts(quiet, options, candidates, scores)
        reach = math.ceil(VIEW_REACH * options.diversity_reach * ctx.limit)
        places: dict[str, list[int]] = {}
        for r, _ in _strongest(reps, scores, reach):
            places.setdefault(ctx.photos[candidates[r]].location, []).append(candidates[r])
        vectors = ctx.view_vectors()
        found = [
            _alike_pairs(vectors, members, options.view_similarity) for members in places.values()
        ]
        return np.concatenate([np.zeros((0, 2), dtype=np.intp), *found])

    def run(self, ctx: TaskContext, options: BestOptions) -> list[Group]:
        n = len(ctx.photos)
        trace = ctx.trace
        ranked = self._ranked(ctx, options)
        if ranked is None:
            return []
        candidates, scores, signals = ranked
        percentiles = {s.name: percentile_rank(s.values) for s in signals}
        for s in signals:
            trace.record(s.name, scatter(n, candidates, s.values))
        trace.record("score", scatter(n, candidates, scores))
        taste = _taste(ctx)
        likely = _likely(ctx, taste, candidates)
        if likely is not None:
            trace.record("probability", scatter(n, candidates, likely))

        photos = [ctx.photos[i] for i in candidates]
        with trace.stage("collapse bursts", len(candidates)) as stage:
            reps = self._bursts(ctx, options, candidates, scores)
            stage.count_out = len(reps)
        lead = np.full(len(candidates), np.nan, dtype=np.float32)
        for rep, others in reps:
            if others:  # in a burst: 1 for the frame shown, 0 for the frames behind it
                lead[rep], lead[others] = 1.0, 0.0
        trace.record("burst_lead", scatter(n, candidates, lead))
        rank = np.full(len(candidates), np.nan, dtype=np.float32)
        by_strength = sorted((r for r, _ in reps), key=lambda r: (-float(scores[r]), r))
        rank[by_strength] = np.arange(1, len(by_strength) + 1)
        trace.record("strength_rank", scatter(n, candidates, rank))

        if options.one_per_view:
            with trace.stage("one per view", len(reps)) as stage:
                reps = _one_per_view(ctx, options, candidates, scores, reps)
                stage.count_out = len(reps)
        sub = ctx.view.take(candidates)
        with trace.stage("within reach", len(reps)) as stage:
            reps = _strongest(reps, scores, math.ceil(options.diversity_reach * ctx.limit))
            stage.count_out = len(reps)
        rep_idx = np.array([r for r, _ in reps], dtype=np.intp)
        with trace.stage("diversify", len(reps)) as stage:
            order = mmr(
                scores[rep_idx],
                sub.embeddings[rep_idx],
                ctx.limit,
                diversity=options.diversity,
            )
            stage.count_out = len(order)
        if len(order) < ctx.limit:
            collapsed = len(candidates) > len(reps)
            once = (
                "frames of one burst or one view" if options.one_per_view else "frames of one burst"
            )
            trace.note(
                f"Only {len(order)} distinct moments here, fewer than the {ctx.limit} asked for"
                + (f": {once} count once." if collapsed else ".")
                + " Choose more folders to see more."
            )
        # Variety chooses the set, but the list is read top down: in MMR's greedy order a later
        # pick can be stronger than an earlier one, so "until it stops being good" would be
        # scrolling down an order of variety (10 November 2026 audit, B-4). Ties break by path.
        order = sorted(order, key=lambda o: (-float(scores[reps[o][0]]), reps[o][0]))

        learned = frozenset() if taste is None else frozenset(taste.weights.places)
        picks = []
        for o in order:
            local, others = reps[o]
            detail = {
                s.name: f"{s.describe} {ordinal(float(percentiles[s.name][local]))} percentile"
                for s in signals
            }
            # Yours reads its own training photos too sure: a place it learned has no probability.
            sure = likely is not None and photos[local].location not in learned
            picks.append(
                make_pick(
                    photos,
                    local,
                    float(scores[local]),
                    evidence_for(local, signals, detail),
                    alternates=others,
                    probability=float(likely[local]) if sure and likely is not None else None,
                )
            )
        if likely is not None and any(p.probability is None for p in picks):
            trace.note(
                f"{sum(p.probability is None for p in picks)} picks are at places your taste "
                "learned from, where it reads too sure: they show no probability."
            )
        return [
            Group(
                title=self._title(ctx, options, self.describe_weight(ctx)),
                score=float(np.mean([p.score for p in picks])),
                picks=tuple(picks),
                marker=self._marker(ctx, options, picks),
            )
        ]

    def _marker(self, ctx: TaskContext, options: BestOptions, picks: list[Pick]) -> Marker | None:
        """Where the list stops being good: above the first pick you are less likely than
        ``marker_probability`` to like (past the last when none is), or None when no pick has a
        probability. A description reorders the list by what it matches, so then there is no
        line to draw."""
        known = [p.probability for p in picks]
        if all(p is None for p in known):
            return None
        if ctx.criteria_scores is not None and self.describe_weight(ctx) > 0:
            ctx.trace.note(
                "A description reorders the list, so no line is drawn where it stops being good."
            )
            return None
        line = options.marker_probability
        below = [i for i, p in enumerate(known) if p is not None and p < line]
        return Marker(before=below[0] if below else len(picks), probability=line)

    def _ranked(
        self, ctx: TaskContext, options: BestOptions
    ) -> tuple[list[int], NDArray[np.float32], list[Signal]] | None:
        """The photos Strongest chooses among (in path order) and their strength, or None
        (noted) when there are none."""
        excluded = set(ctx.reference_keys)
        # Path order makes "lower index" mean "lower path" in every tie below.
        candidates = sorted(
            (i for i, p in enumerate(ctx.photos) if p.content_key not in excluded),
            key=lambda i: ctx.photos[i].path,
        )
        with ctx.trace.stage("candidates", len(ctx.photos)) as stage:
            stage.count_out = len(candidates)
        if not candidates:
            ctx.trace.note(
                "No photos to choose from (the pool is empty or only holds the reference)."
            )
            return None

        if options.kind is not None:
            candidates = self._filter_kind(ctx, options, options.kind, candidates)
            if not candidates:
                ctx.trace.note(
                    f"No photo is confidently {options.kind!r} "
                    f"(min_kind_probability={options.min_kind_probability})."
                )
                return None

        scores, signals = photo_strength(
            ctx,
            candidates,
            quality_weight=options.quality_weight,
            aesthetic_weight=options.aesthetic_weight,
            criteria_share=self.describe_weight(ctx),
        )
        if not signals:
            if options.quality_weight + options.aesthetic_weight == 0:
                raise InvalidRequestError(
                    "quality_weight and aesthetic_weight are both zero, so nothing ranks the "
                    "photos. Set either above zero, or describe what to look for."
                )
            raise CapabilityError("No scoring model is available for 'best'.")
        if options.quality_weight + options.aesthetic_weight == 0:
            ctx.trace.note(
                "quality_weight and aesthetic_weight are both zero, so photos are ranked by "
                "the description alone."
            )
        return candidates, scores, signals

    @staticmethod
    def _bursts(
        ctx: TaskContext, options: BestOptions, candidates: list[int], scores: NDArray[np.float32]
    ) -> list[tuple[int, list[int]]]:
        """Each burst's strongest frame (positions in ``candidates``) and its other frames."""
        bursts = group_bursts(
            [ctx.photos[i] for i in candidates],
            ctx.view.take(candidates).embeddings,
            window_seconds=options.burst_window_seconds,
            min_similarity=options.burst_similarity,
        )
        return representatives(bursts, scores)

    @staticmethod
    def _filter_kind(
        ctx: TaskContext, options: BestOptions, kind: str, candidates: list[int]
    ) -> list[int]:
        """Keep photos the kind model is confident about, at the depth of ``kind``."""
        model = ctx.models.get("shot_kind")  # CapabilityError: an explicit filter can't degrade
        if not isinstance(model, LabelModel):
            raise CapabilityError("model 'shot_kind' does not predict labels")
        depth = kind.count(":") + 1
        probs = model.predict_proba(ctx.view.take(candidates))
        labels, rolled = rollup(model.labels, probs, depth)
        if kind not in labels:
            raise InvalidRequestError(
                f"Unknown kind {kind!r}. Valid kinds: {', '.join(sorted(labels))}. "
                "Use a shorter name (e.g. 'establishing') for the coarse kind."
            )
        p = rolled[:, labels.index(kind)]
        ctx.trace.record("kind_probability", scatter(len(ctx.photos), candidates, p))
        with ctx.trace.stage(f"kind {kind}", len(candidates)) as stage:
            kept = [
                c
                for c, keep in zip(candidates, p >= options.min_kind_probability, strict=True)
                if keep
            ]
            stage.count_out = len(kept)
        return kept

    @staticmethod
    def _title(ctx: TaskContext, options: BestOptions, describe_weight: float) -> str:
        parts = ["Strongest", options.kind or "", "photographs"]
        title = " ".join(p for p in parts if p)
        if ctx.criteria_scores is not None and describe_weight > 0:
            text = ctx.request.criteria.strip()
            if text:
                title += f" matching “{text}”"
            else:
                title += (
                    " like your examples" if ctx.request.examples else " matching your criteria"
                )
        return title


def _taste(ctx: TaskContext) -> TasteModel | None:
    """Yours, when it is what ranks the aesthetic."""
    try:
        model = ctx.models.get("aesthetic")
    except CapabilityError:
        return None
    return model if isinstance(model, TasteModel) else None


def _likely(
    ctx: TaskContext, taste: TasteModel | None, candidates: list[int]
) -> NDArray[np.float32] | None:
    """How likely you are to like each candidate (``models.calibration``), or None: General
    ranks, or Yours has not been calibrated (said, with how to)."""
    if taste is None:
        return None
    calibration = taste.weights.calibration
    if calibration is None:
        ctx.trace.note(
            "Yours has not been calibrated, so picks show no probability. It is read when your "
            "taste is trained, from the photos Strongest showed first at places Yours had not "
            "learned: run calico train taste"
        )
        return None
    strong = taste.predict_proba(ctx.view.take(candidates))[:, taste.labels.index("strong")]
    return calibration.probability(strong).astype(np.float32)


def _strongest(
    reps: list[tuple[int, list[int]]], scores: NDArray[np.float32], reach: int
) -> list[tuple[int, list[int]]]:
    """The ``reach`` strongest burst representatives, still in path order, so ties in MMR
    still go to the lower path."""
    order = np.argsort(-scores[[r for r, _ in reps]], kind="stable")[:reach]
    return [reps[i] for i in sorted(order)]


def _one_per_view(
    ctx: TaskContext,
    options: BestOptions,
    candidates: list[int],
    scores: NDArray[np.float32],
    reps: list[tuple[int, list[int]]],
) -> list[tuple[int, list[int]]]:
    """``reps`` with each view's weaker bursts folded into its strongest, as alternates
    (strongest first, each followed by its own burst frames). Records ``view_lead``: 1 for
    the frame shown for a view, 0 for the frames folded behind it."""
    if ctx.compared_pairs is None or ctx.pair_shares is None:
        ctx.trace.note(
            "Photos could not be compared part by part, so a view shot on several occasions "
            'may be shown more than once. Set view_embedder = "dinov2" in calico.toml.'
        )
        return reps
    at = {candidates[r]: k for k, (r, _) in enumerate(reps)}
    rows, shares = [], []
    for (a, b), share in zip(ctx.compared_pairs, ctx.pair_shares, strict=True):
        if int(a) in at and int(b) in at:
            rows.append((at[int(a)], at[int(b)]))
            shares.append(float(share))
    groups = group_views(
        len(reps),
        np.asarray(rows, dtype=np.intp).reshape(-1, 2),
        np.asarray(shares, dtype=np.float32),
        min_share=options.view_share,
    )
    lead = np.full(len(candidates), np.nan, dtype=np.float32)
    behind: dict[int, list[int]] = {}
    for group in groups:
        first, *rest = sorted(group, key=lambda k: (-float(scores[reps[k][0]]), k))
        behind[first] = [i for k in rest for i in (reps[k][0], *reps[k][1])]
        lead[reps[first][0]], lead[behind[first]] = 1.0, 0.0
    ctx.trace.record("view_lead", scatter(len(ctx.photos), candidates, lead))
    folded = {k for group in groups for k in group} - behind.keys()
    if folded:
        ctx.trace.note(
            f"{len(folded)} photo{'s' if len(folded) != 1 else ''} of a view already shown "
            f"{'were' if len(folded) != 1 else 'was'} kept as alternates of its strongest "
            "frame (one pick per view)."
        )
    return [
        (r, [*others, *behind.get(k, [])]) for k, (r, others) in enumerate(reps) if k not in folded
    ]


def _alike_pairs(
    vectors: NDArray[np.float32], members: list[int], similarity: float
) -> NDArray[np.intp]:
    """Pairs of ``members`` (rows of ``vectors``) at least ``similarity`` alike, each once."""
    index = np.asarray(members, dtype=np.intp)
    mine = vectors[index]
    found = []
    for start in range(0, len(index), _BLOCK):
        rows, cols = np.nonzero(mine[start : start + _BLOCK] @ mine.T >= similarity)
        rows += start
        later = rows < cols
        found.append(np.stack([index[rows[later]], index[cols[later]]], axis=1))
    return np.concatenate([np.zeros((0, 2), dtype=np.intp), *found])
