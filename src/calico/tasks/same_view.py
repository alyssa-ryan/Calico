"""``same_view``: the same place, over time.

Photos are compared with the reference part by part: a shortlist of those most alike
as a whole (the view embedder, DINOv2) is handed to the engine, which measures each
one's share of patches that are mutual best matches with the reference
(``selection.patch_match``). Two tiers come back:

* **Same view** (share ≥ ``sure_share``): oldest first, because the point is to see
  change, not a ranking. Every such photo, not one per day.
* **Possibly the same view** (share ≥ ``possible_share``): the same place, perhaps
  re-aimed, closest first, to look over. A single line could not serve every site:
  look-alike rooms at one place share as much as true repeats at another.

When photos cannot be compared part by part (a view embedder without patches), the
whole-photo match of before is used: "same subject" (embedding cosine) mixed with
"same camera angle" (gradient-layout cosine), kept from ``min_similarity``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, model_validator

from calico.domain import Group
from calico.errors import InvalidRequestError
from calico.selection.bursts import group_bursts, representatives
from calico.selection.picks import make_pick, scatter
from calico.selection.scoring import Signal, evidence_for, with_criteria
from calico.selection.view_match import best_cosine, view_match
from calico.tasks.base import ReferenceUse, Task, TaskOptions

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from calico.domain import Evidence, Photo, Pick
    from calico.selection.context import FeatureView, TaskContext, Trace

NEAR_MISS = 0.10


MIN_SIMILARITY = 0.75
"""Calibrated on 4 labelled references (two large sites and two drone sites;
175 photos, 55 same views; `calico eval same-view`). The old 0.80 never kept a wrong
photo but missed a third of the same views (recall 67%, 27% at one site); 0.75 (the
best balance, 0.747, rounded) keeps 78%, and 98% of what it keeps is one. Match
separated same views well at every site (AUC 0.89-1.00): the misses were the line, not
the scoring. Two sites still keep only about half at 0.75 (a factory and a drone site:
their same views run down to 0.72); more labelled references would show whether that
is the sites or the line."""


SURE_SHARE = 0.25
"""Patch share (``selection.patch_match``) from which a photo is surely the same view.
Set on 8 labelled references just above the highest look-alike, then tested on 7 fresh
ones: 35 of 36 same views kept, 2 look-alikes; over all 15, 75% kept and 97% right
(``calico eval same-view --patches``)."""

POSSIBLE_SHARE = 0.18
"""From here a photo is possibly the same view. With the sure tier it holds 117 of the
118 labelled same views; a real search shows 2-56 possible photos (0.15 holds all 118
but up to 250). Chosen on those same labels."""

POSSIBLE_LIMIT = 60
"""The most possible matches shown: enough for every labelled site, few enough to look
over."""

SHORTLIST_SIMILARITY = 0.75
"""Whole-photo similarity (the view embedder) from which a photo is compared part by
part. No labelled same view scored below 0.84 (15 references, 118 same views)."""

SHORTLIST_SIZE = 600
"""The most photos compared part by part (~0.1 s each): above the ~480 photos
of the largest labelled place at 0.80."""


class SameViewOptions(TaskOptions):
    sure_share: float = Field(
        default=SURE_SHARE,
        ge=0,
        le=1,
        description="Share of the reference's details a photo must have to be the same view",
    )
    possible_share: float = Field(
        default=POSSIBLE_SHARE,
        ge=0,
        le=1,
        description="Share from which a photo is shown as possibly the same view",
    )
    possible_limit: int = Field(
        default=POSSIBLE_LIMIT, ge=0, le=1000, description="Most possible matches shown"
    )
    shortlist_similarity: float = Field(
        default=SHORTLIST_SIMILARITY,
        ge=0,
        le=1,
        description="Only photos this alike as a whole are compared detail by detail",
    )
    shortlist_size: int = Field(
        default=SHORTLIST_SIZE,
        ge=1,
        le=10000,
        description="Most photos compared detail by detail (about 0.1 s each)",
    )
    layout_weight: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Whole-photo matching only: 0 = same subject, 1 = same camera angle",
    )
    min_similarity: float = Field(
        default=MIN_SIMILARITY,
        ge=0,
        le=1,
        description="Whole-photo matching only: how close a match must be",
    )
    collapse_bursts: bool = Field(
        default=False, description="Show only the best frame of each burst"
    )
    burst_window_seconds: float = Field(
        default=120, ge=0, le=3600, description="Frames this close in time may be one burst"
    )
    burst_similarity: float = Field(
        default=0.9, ge=0, le=1, description="Frames this alike (cosine) in a window are one burst"
    )

    @model_validator(mode="after")
    def _possible_is_looser(self) -> SameViewOptions:
        if self.possible_share > self.sure_share:
            raise ValueError(
                f"possible_share ({self.possible_share:g}) must not be above sure_share "
                f"({self.sure_share:g}): possible matches are the looser tier."
            )
        return self


class SameViewTask(Task[SameViewOptions]):
    """One result = one photo; ``Group.score`` is the mean of its picks' scores."""

    name = "same_view"
    title = "Same view over time"
    description = "Every photo of the same view as your reference, oldest to newest."
    reference_use = ReferenceUse.REQUIRED
    default_limit = 200
    default_describe_weight = 0.3
    """Low: the description reorders photos that already match the reference."""
    reference_label = "Reference photo"
    reference_help = (
        "The view you want to see again — every closely matching photo comes back, oldest first."
    )
    options_model = SameViewOptions

    def patch_shortlist(
        self, ctx: TaskContext, options: SameViewOptions
    ) -> NDArray[np.intp] | None:
        """The photos most alike as a whole (the view embedder), best first, to compare
        part by part."""
        if ctx.references is None or len(ctx.references) == 0:
            return None
        cand = _candidates(ctx)
        if not cand:
            return np.zeros(0, dtype=np.intp)
        mine, theirs = _view_vectors(ctx, ctx.references, cand)
        similarity = best_cosine(mine, theirs)
        alike = np.flatnonzero(similarity >= options.shortlist_similarity)
        ranked = [cand[k] for k in sorted(alike, key=lambda k: (-float(similarity[k]), int(k)))]
        if not ranked:
            ctx.trace.note(
                "No photo is alike enough as a whole to compare detail by detail: the closest "
                f"is {float(similarity.max()):.2f}, below shortlist_similarity "
                f"({options.shortlist_similarity:.2f}). Lower shortlist_similarity to compare "
                "it."
            )
        if len(ranked) > options.shortlist_size:
            size = options.shortlist_size
            last = float(similarity[cand.index(ranked[size - 1])])
            ctx.trace.note(
                f"The {size} photos most alike as a whole (down to {last:.2f}) were compared "
                f"detail by detail; {len(ranked) - size} less alike ones were not. Raise "
                "shortlist_size to include them (about 0.1 s each)."
            )
        return np.asarray(ranked[: options.shortlist_size], dtype=np.intp)

    def run(self, ctx: TaskContext, options: SameViewOptions) -> list[Group]:
        refs = ctx.references
        if refs is None or len(refs) == 0:
            raise InvalidRequestError(
                "same_view needs a reference photo: pass one with --reference (or upload it)."
            )
        cand = _candidates(ctx)
        with ctx.trace.stage("candidates", len(ctx.photos)) as stage:
            stage.count_out = len(cand)
        if not cand:
            ctx.trace.note("No photos to compare with the reference (the pool is empty).")
            return []
        if ctx.patch_scores is None:
            return self._whole_photos(ctx, options, cand, refs)
        return self._by_patches(ctx, options, cand, ctx.patch_scores)

    def _by_patches(
        self,
        ctx: TaskContext,
        options: SameViewOptions,
        cand: list[int],
        patch_scores: NDArray[np.float32],
    ) -> list[Group]:
        """Two tiers from each photo's share of the reference's patches."""
        n, trace = len(ctx.photos), ctx.trace
        compared = np.isfinite(patch_scores[cand])
        share = np.nan_to_num(patch_scores[cand], nan=0.0).astype(np.float32)
        # NaN for a photo never compared, so runs explain does not show it sharing nothing.
        trace.record("patch_share", scatter(n, cand, np.where(compared, share, np.nan)))
        signals, pct = with_criteria(
            ctx.criteria_scores,
            cand,
            [Signal("patches", share, 1.0, "of the reference's details")],
            self.describe_weight(ctx),
        )
        score = sum((sig.weight * sig.values for sig in signals), np.zeros_like(share))
        trace.record("score", scatter(n, cand, score))
        with trace.stage("same view", len(cand)) as stage:
            sure = np.flatnonzero(share >= options.sure_share)
            stage.count_out = len(sure)
        with trace.stage("possibly the same view", len(cand)) as stage:
            maybe = np.flatnonzero((share >= options.possible_share) & (share < options.sure_share))
            stage.count_out = len(maybe)
        if len(sure) == 0 and len(maybe) == 0 and compared.any():
            trace.note(
                f"No photo shares enough of the reference's details: the closest shares "
                f"{float(share.max()):.0%}. Lower possible_share (now "
                f"{options.possible_share:.0%}) to see it."
            )

        def evidence(k: int) -> tuple[Evidence, ...]:
            details = {"patches": f"shares {share[k]:.0%} of the reference's details"}
            if pct is not None:
                details["criteria"] = f"matches criteria {float(pct[k]):.2f}"
            return evidence_for(k, signals, details)

        picks = _timeline(ctx, options, cand, keep=sure, score=score, evidence=evidence)
        groups = [Group(title=_title(ctx), score=_mean(picks), picks=picks)]
        if len(maybe):
            ranked = [int(k) for k in sorted(maybe, key=lambda k: (-float(share[k]), int(k)))]
            shown = ranked[: options.possible_limit]
            if len(ranked) > len(shown):
                trace.note(
                    f"{len(ranked) - len(shown)} more photos are possibly the same view; "
                    "raise possible_limit to see them."
                )
            photos = [ctx.photos[i] for i in cand]
            possible = tuple(make_pick(photos, k, float(score[k]), evidence(k)) for k in shown)
            groups.append(Group(title=POSSIBLE_TITLE, score=_mean(possible), picks=possible))
        return groups

    def _whole_photos(
        self, ctx: TaskContext, options: SameViewOptions, cand: list[int], refs: FeatureView
    ) -> list[Group]:
        """One tier from the whole-photo match, when photos cannot be compared by patches."""
        n, trace = len(ctx.photos), ctx.trace
        sub = ctx.view.take(cand)
        sc = _score(ctx, options, refs, sub, cand, describe_weight=self.describe_weight(ctx))
        semantic, layout, match, score = sc.semantic, sc.layout, sc.match, sc.score
        trace.record("semantic", scatter(n, cand, semantic))
        # Unknown is not zero: without layout features the column is NaN.
        known = scatter(n, cand, layout)
        trace.record("layout", known if sc.layout_known else np.full(n, np.nan))
        trace.record("match", scatter(n, cand, match))
        trace.record("score", scatter(n, cand, score))

        z = (match - match.mean()) / (match.std() or 1.0)
        with trace.stage("min similarity", len(cand)) as stage:
            keep = np.flatnonzero(match >= options.min_similarity)
            stage.count_out = len(keep)
        _tuning_notes(trace, match, keep, options.min_similarity)
        if len(keep) == 0:
            return []

        def evidence(k: int) -> tuple[Evidence, ...]:
            details = {
                "semantic": f"cosine {semantic[k]:.2f} to the closest reference",
                "layout": f"cosine {layout[k]:.2f} to the closest reference",
                "criteria": f"matches criteria {float(sc.criteria_pct[k]):.2f}"
                if sc.criteria_pct is not None
                else "",
            }
            stands_out = f" · match stands out: z={z[k]:.1f}"
            return evidence_for(
                k, sc.signals, {name: text + stands_out for name, text in details.items()}
            )

        picks = _timeline(ctx, options, cand, keep=keep, score=score, evidence=evidence)
        return [Group(title=_title(ctx), score=_mean(picks), picks=picks)]


POSSIBLE_TITLE = "Possibly the same view"


def _candidates(ctx: TaskContext) -> list[int]:
    """Every photo but the references; path order turns "lower index" into "lower path"
    for every tie."""
    excluded = set(ctx.reference_keys)
    return sorted(
        (i for i, p in enumerate(ctx.photos) if p.content_key not in excluded),
        key=lambda i: ctx.photos[i].path,
    )


def _view_vectors(
    ctx: TaskContext, refs: FeatureView, cand: list[int]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """The view embedder's vectors for ``cand`` and the references, or the main embedding's
    when it is the view embedder (or was not computed)."""
    name = ctx.view_feature
    if name and name in ctx.view.features and name in refs.features:
        return ctx.view.features[name][cand], refs.features[name]
    return ctx.view.embeddings[cand], refs.embeddings


def _timeline(
    ctx: TaskContext,
    options: SameViewOptions,
    cand: list[int],
    *,
    keep: NDArray[np.intp],
    score: NDArray[np.float32],
    evidence: Callable[[int], tuple[Evidence, ...]],
) -> tuple[Pick, ...]:
    """The kept photos (``keep`` indexes ``cand``), bursts collapsed if asked, the best up
    to the limit, oldest first."""
    if len(keep) == 0:
        return ()
    trace = ctx.trace
    photos = [ctx.photos[cand[k]] for k in keep]
    reps: list[tuple[int, list[int]]]
    if options.collapse_bursts:
        with trace.stage("collapse bursts", len(keep)) as stage:
            bursts = group_bursts(
                photos,
                ctx.view.embeddings[[cand[k] for k in keep]],
                window_seconds=options.burst_window_seconds,
                min_similarity=options.burst_similarity,
            )
            reps = representatives(bursts, score[keep])
            stage.count_out = len(reps)
    else:
        reps = [(i, []) for i in range(len(keep))]
    kept = score[keep]  # once, not per comparison: 60,000 kept photos took 2 s (A03-14)
    chosen = sorted(reps, key=lambda r: (-float(kept[r[0]]), r[0]))[: ctx.limit]
    if len(reps) > len(chosen):
        trace.note(f"{len(reps) - len(chosen)} more photos matched; raise How many to see them.")
    chosen.sort(key=lambda r: _chronological(photos, r[0]))
    return tuple(
        make_pick(
            photos, local, float(score[keep[local]]), evidence(int(keep[local])), alternates=others
        )
        for local, others in chosen
    )


def _mean(picks: tuple[Pick, ...]) -> float:
    return float(np.mean([p.score for p in picks])) if picks else 0.0


@dataclass(frozen=True)
class _Scores:
    semantic: NDArray[np.float32]
    layout: NDArray[np.float32]
    match: NDArray[np.float32]
    score: NDArray[np.float32]
    signals: list[Signal]
    criteria_pct: NDArray[np.float32] | None
    layout_known: bool


def _score(
    ctx: TaskContext,
    options: SameViewOptions,
    refs: FeatureView,
    sub: FeatureView,
    cand: list[int],
    *,
    describe_weight: float,
) -> _Scores:
    """Match to the references, optionally blended with the criteria text.

    ``match`` is pure similarity to the reference (what ``min_similarity`` and
    the z-score talk about); ``score`` is what ranks, and adds criteria.
    """
    vm = view_match(sub, refs, options.layout_weight, ctx.trace)
    semantic, layout, match, layout_known = vm.semantic, vm.layout, vm.match, vm.layout_known
    w = options.layout_weight if layout_known else 0.0
    signals, pct = with_criteria(
        ctx.criteria_scores,
        cand,
        [Signal("semantic", semantic, 1 - w, "subject"), Signal("layout", layout, w, "angle")],
        describe_weight,
    )
    signals = [s for s in signals if s.weight > 0]
    score = sum((s.weight * s.values for s in signals), np.zeros_like(match))
    return _Scores(semantic, layout, match, score.astype(np.float32), signals, pct, layout_known)


def _tuning_notes(
    trace: Trace, match: NDArray[np.float32], keep: NDArray[np.intp], low: float
) -> None:
    """Help the user tune ``min_similarity``: how close the nearest miss was."""
    if len(keep) == 0:
        trace.note(
            f"No photo matched at {low:.2f} or above; the closest scored {float(match.max()):.2f}. "
            "Lower min_similarity to see it."
        )
        return
    near = int(((match >= low - NEAR_MISS) & (match < low)).sum())
    if near:
        noun = "photo" if near == 1 else "photos"
        trace.note(
            f"{near} more {noun} scored within {NEAR_MISS:.2f} below {low:.2f}. "
            "Lower min_similarity to include them; to find where it belongs for your photos, "
            "use Label for calibration (calico eval new-view)."
        )


def _chronological(photos: list[Photo], i: int) -> tuple[bool, float, str]:
    p = photos[i]
    stamp = p.captured_at.timestamp() if p.captured_at is not None else 0.0
    return (p.captured_at is None, stamp, str(p.path))


def _title(ctx: TaskContext) -> str:
    if len(ctx.reference_paths) == 1:
        return f"Same view as {ctx.reference_paths[0].name}"
    return f"Same view as {len(ctx.reference_paths)} references"
