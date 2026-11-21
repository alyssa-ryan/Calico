"""Measuring rankings against your labels. Pure functions; no I/O.

Labels are 0 (doesn't match), 1 (partly) and 2 (matches). The metrics answer
the questions the "Describe it" work turns on:

* ``precision_at`` / ``ndcg_at`` -- are the first photos shown right? nDCG also
  rewards putting the best matches first. "Partly" earns half credit.
* ``auc`` -- does a true match outscore a non-match, anywhere in the list? That
  is what an "only matching" filter needs, beyond a good top ten.
* ``best_threshold`` -- the score that best separates matches from the rest.
* ``clear_calls`` -- when a model says a photo clearly matches, is it right?
* ``measure_taste`` -- does Yours rank your ratings better than General, over the whole
  range and among the photos each would show first?
* ``measure_compare`` -- in a blind Compare Strongest session, whose first picks did you
  like more?
* ``measure_stories`` -- do stories you liked outscore the ones you rejected?
* ``FinishedStory`` -- would the story task tell a story you finished yourself?

Describing by example is measured by taking some labelled matches as the
examples (``choose_examples``) and scoring the rest, leaving out photos shot
moments from an example (``apart_from``): a burst sibling would flatter them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np

from calico.criteria import CriteriaScores, blend_examples, example_similarity
from calico.evaluation.stats import (
    RESAMPLES,
    Compared,
    Interval,
    compare,
    macro_average,
    place_bootstrap,
)
from calico.models.taste import (
    TASTE_L2,
    familiarity,
    fit_by_scale,
    fit_general,
    fit_pairs,
    fit_taste,
    general_probability,
    lean_on_general,
    scaled_probability,
    taste_probability,
)
from calico.selection.copies import copies_of, copy_of
from calico.selection.scoring import percentile_rank
from calico.selection.variety import variety

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
    from datetime import datetime
    from pathlib import Path

    from numpy.typing import NDArray

    from calico.domain import Evidence
    from calico.features.base import Embedder
    from calico.labels import StoryWay
    from calico.models.taste import TasteWeights
    from calico.selection.context import FeatureView

MATCH = 2


def _gains(ranked_labels: Sequence[int]) -> list[float]:
    return [label / MATCH for label in ranked_labels]


def precision_at(ranked_labels: Sequence[int], k: int) -> float:
    """Mean credit of the first ``k`` (or all, if fewer): 1 a match, 0.5 partly, 0 not."""
    top = _gains(ranked_labels[:k])
    return sum(top) / len(top) if top else math.nan


def ndcg_at(ranked_labels: Sequence[int], k: int) -> float:
    """Discounted gain of the first ``k`` against the best possible order; NaN with no match."""

    def dcg(gains: Sequence[float]) -> float:
        return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))

    gains = _gains(ranked_labels)
    ideal = dcg(sorted(gains, reverse=True))
    return dcg(gains) / ideal if ideal > 0 else math.nan


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """P(a match outscores a non-match); ties count half, "partly" is left out."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    pos, neg = s[y == MATCH], s[y == 0]
    if not len(pos) or not len(neg):
        return math.nan
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


@dataclass(frozen=True)
class Threshold:
    threshold: float
    f1: float
    precision: float
    recall: float


def best_threshold(scores: Sequence[float], labels: Sequence[int]) -> Threshold:
    """The cut (score >= threshold means "matches") with the best F1; "partly" is left out."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    keep = y != 1
    s, positive = s[keep], y[keep] == MATCH
    best = Threshold(math.nan, 0.0, 0.0, 0.0)
    for cut in sorted(set(s.tolist()), reverse=True):
        chosen = s >= cut
        hits = int((chosen & positive).sum())
        if not hits:
            continue
        precision, recall = hits / int(chosen.sum()), hits / int(positive.sum())
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best.f1:
            best = Threshold(cut, f1, precision, recall)
    return best


@dataclass(frozen=True)
class SearchScore:
    """How one scorer did on one labelled search."""

    labelled: int
    matches: int
    precision_at_10: float
    ndcg_at_10: float
    auc: float
    threshold: Threshold


@dataclass(frozen=True)
class ClearCalls:
    """The photos a model called clear matches, by how you labelled them."""

    matches: int
    partly: int
    wrong: int

    @property
    def called(self) -> int:
        return self.matches + self.partly + self.wrong

    def __add__(self, other: ClearCalls) -> ClearCalls:
        return ClearCalls(
            self.matches + other.matches, self.partly + other.partly, self.wrong + other.wrong
        )


def clear_calls(clear: Mapping[str, bool], labels: Mapping[str, int]) -> ClearCalls:
    called = [labels[k] for k in labels if clear[k]]
    return ClearCalls(matches=called.count(MATCH), partly=called.count(1), wrong=called.count(0))


@dataclass(frozen=True)
class Evaluated:
    """One scorer on one labelled search; ``missing`` labelled photos are no longer catalogued.

    ``clear`` is None when the model cannot say what clearly matches.
    """

    set_id: str
    text: str
    scorer: str
    score: SearchScore
    missing: int
    clear: ClearCalls | None = None
    scores: Mapping[str, float] = field(default_factory=dict)
    """Each labelled photo's score, for ``criteria_intervals``."""
    labels: Mapping[str, int] = field(default_factory=dict)


def score_search(scores: Mapping[str, float], labels: Mapping[str, int]) -> SearchScore:
    """Rank the labelled photos by ``scores`` (ties by key) and measure the order.

    Only labelled photos count: the sample stands in for the whole folder. A
    search with nothing to find has no precision (it is not 0): what it tests is
    whether scores stay low, which ``auc`` and thresholds cannot say either.
    """
    keys = sorted(labels, key=lambda k: (-scores[k], k))
    ranked = [labels[k] for k in keys]
    values = [scores[k] for k in keys]
    return SearchScore(
        labelled=len(keys),
        matches=sum(label == MATCH for label in ranked),
        precision_at_10=precision_at(ranked, 10) if any(ranked) else math.nan,
        ndcg_at_10=ndcg_at(ranked, 10),
        auc=auc(values, ranked),
        threshold=best_threshold(values, ranked),
    )


TASTE_SAMPLE = 60
"""Photos in one taste-rating session: about ten minutes with the keyboard."""


@dataclass(frozen=True)
class Drawn:
    """A photo drawn for a rating session: its key, the stratum it was drawn from, and the
    chance any photo still unrated there had of being drawn, so evaluation can weight the
    sample back to the whole place."""

    key: str
    stratum: str
    inclusion: float


TASTE_RANDOM = 15
"""Of a session drawn by value (``sample_by_value``), the photos drawn at random from the
whole place, every one alike: what keeps the whole-range AUC fair as the rest of each
session goes to the top."""

VALUE_QUOTAS: tuple[tuple[str, int], ...] = (
    ("first: General only", 15),
    ("first: Yours only", 15),
    ("first: both", 5),
    ("top quarter", 10),
)
"""The rest of a session drawn by value, in turn; turns a stratum cannot fill pass to the
next. You judge Strongest by its first picks, and the photos only one template shows first
decide which is better, so they take most turns; photos both show first cannot tell the
templates apart, so they take few."""

VALUE_DEPTH = 40
"""How far into each template's first picks a session by value looks: the deepest first
picks ``calico eval taste`` reports (``TASTE_FIRST``), which you judge Strongest by."""

VALUE_STRATA = frozenset({name for name, _ in VALUE_QUOTAS} | {"rest"})
"""Strata drawn for their value, not over the whole range: the whole-range AUC leaves
their photos out (``TastePhoto.whole_range``), every other measure reads them."""

GIVEN = "given"
"""The stratum of a photo in a session made from a list (``Sessions.label_list``): it was
given, not drawn, and whoever wrote the list chose it by some ranking. So it is read as a
photo drawn for its value is: learned from, and left out of the whole-range measures."""


def sample_by_value(
    ranked: Sequence[str],
    general_first: Sequence[str],
    yours_first: Sequence[str],
    seed: int,
    *,
    exclude: Collection[str] = frozenset(),
) -> list[Drawn]:
    """Which photos to rate for your taste, shuffled: most where a rating tells the most.

    ``ranked`` is the place's photos best first by today's general strength;
    ``general_first`` and ``yours_first`` are what each template shows first there (as
    keys in ``ranked``; ``yours_first`` is empty without a trained taste). Sessions until
    31 October 2026 drew the same number from each quarter of the range (strata "quarter
    1" to "quarter 4"), 15 photos from the top quarter, so at a place of 400 bursts about 6
    of General's first 40, and whether Yours's first picks beat General's stayed unknown
    (30 October 2026, the four hold-out places: 29 of General's first 20 rated, 22 of
    Yours's).

    ``TASTE_RANDOM`` photos are drawn first, at random from the whole place; then the
    ``VALUE_QUOTAS`` in turn from what is left, the top quarter by general strength
    after the first picks; turns nothing can fill go to the rest of the place ("rest").
    Each photo records its stratum and its chance of being drawn there. Photos in
    ``exclude`` (already rated) are never drawn; a place with no more left than a session
    (``TASTE_SAMPLE``: the random photos and the quotas) is taken whole, and shuffled like
    any other: listed best first, it would show the rater General's ranking (10 November
    2026 audit, B-8)."""
    pool = [k for k in ranked if k not in exclude]
    rng = np.random.default_rng(seed)
    if len(pool) <= TASTE_RANDOM + sum(quota for _, quota in VALUE_QUOTAS):
        return [Drawn(pool[int(i)], "whole place", 1.0) for i in rng.permutation(len(pool))]
    random = [pool[int(i)] for i in sorted(rng.choice(len(pool), TASTE_RANDOM, replace=False))]
    chosen = [Drawn(k, "random", TASTE_RANDOM / len(pool)) for k in random]
    left = set(pool) - set(random)
    general, yours = set(general_first) & left, set(yours_first) & left
    first = general | yours
    top = (set(ranked[: len(ranked) // 4]) & left) - first  # the place's, not what is left
    strata = {
        "first: General only": general - yours,
        "first: Yours only": yours - general,
        "first: both": general & yours,
        "top quarter": top,
    }
    owed = 0
    for name, quota in VALUE_QUOTAS:
        members = [k for k in pool if k in strata[name]]
        n = min(len(members), quota + owed)
        owed += quota - n
        chosen += _draw(members, n, name, rng)
    rest = [k for k in pool if k in left - first - top]
    chosen += _draw(rest, min(len(rest), owed), "rest", rng)
    order = rng.permutation(len(chosen))
    return [chosen[i] for i in order]


def _draw(members: Sequence[str], n: int, stratum: str, rng: np.random.Generator) -> list[Drawn]:
    """``n`` of ``members`` at random, each recorded with its chance of being drawn."""
    if not n:
        return []
    picked = sorted(rng.choice(len(members), n, replace=False))
    return [Drawn(members[int(i)], stratum, n / len(members)) for i in picked]


ANCHOR_SAMPLE = 60
"""Photos in a session asking whether each could anchor a story, as many as a taste
session."""


def sample_for_anchors(
    ranked: Sequence[tuple[str, str]],
    seed: int,
    *,
    size: int = ANCHOR_SAMPLE,
    exclude: Collection[str] = frozenset(),
) -> list[Drawn]:
    """Which photos to rate as possible anchors, shuffled.

    ``ranked`` is (content key, kind) best first by today's general strength; a kind is a
    photo's scale and whether a person or animal is in it. Strength alone reached 2 of
    your 28 anchors (27 October 2026), so each kind takes its turn (one with too few
    photos gives its turns up), and within a kind the photos are spread from strongest to
    weakest. Shuffled, so the order hints at neither. Photos in ``exclude`` (already
    rated) are never drawn; each kind is a stratum.
    """
    kinds: dict[str, list[str]] = {}
    for key, kind in ranked:
        if key not in exclude:
            kinds.setdefault(kind, []).append(key)
    share = dict.fromkeys(kinds, 0)
    left = min(size, sum(len(keys) for keys in kinds.values()))
    while left:
        for kind, keys in kinds.items():
            if left and share[kind] < len(keys):
                share[kind] += 1
                left -= 1
    chosen = []
    for kind, keys in kinds.items():
        if share[kind]:
            at = np.round(np.linspace(0, len(keys) - 1, share[kind])).astype(int)
            chosen += [Drawn(keys[i], kind or "any", share[kind] / len(keys)) for i in at]
    order = np.random.default_rng(seed).permutation(len(chosen))
    return [chosen[i] for i in order]


@dataclass(frozen=True)
class AnchorRating:
    """A photo rated as a possible anchor, at its place, with each signal's value there
    (a percentile or a probability); a signal Calico could not compute is absent."""

    place: str
    could: bool
    signals: Mapping[str, float]


@dataclass(frozen=True)
class AnchorSignal:
    signal: str
    auc: float
    pairs: int


@dataclass(frozen=True)
class AnchorReport:
    could: int
    could_not: int
    places: int
    signals: list[AnchorSignal]
    measured: Signals | None = None
    """Each signal's AUC with its interval by place, and each against general strength."""
    skipped: list[str] = field(default_factory=list)
    """Sessions left out because their folders hold no catalogued photos now."""


ANCHOR_BASELINE = "general strength"


def measure_anchors(rows: Sequence[AnchorRating], *, resamples: int = RESAMPLES) -> AnchorReport:
    """How often a photo you could anchor a story with outscores one you could not, at the
    same place, by each signal: whether anchor quality is something Calico already
    measures (strength, a person or animal, a wide view) or needs a model of its own."""
    places: dict[str, list[Scored]] = {}
    for row in rows:
        places.setdefault(row.place, []).append(Scored(liked=row.could, scores=row.signals))
    signals = []
    for name in dict.fromkeys(k for row in rows for k in row.signals):
        wins, pairs = _signal_pairs(list(places.values()), name)
        signals.append(
            AnchorSignal(signal=name, auc=wins / pairs if pairs else math.nan, pairs=int(pairs))
        )
    baseline = ANCHOR_BASELINE if any(s.signal == ANCHOR_BASELINE for s in signals) else None
    return AnchorReport(
        could=sum(r.could for r in rows),
        could_not=sum(not r.could for r in rows),
        places=len(places),
        signals=signals,
        measured=measure_signals(places, baseline=baseline, resamples=resamples),
    )


@dataclass(frozen=True)
class Scored:
    """A rated photo at its place: liked (or could anchor) or not, and each signal's score
    there. A signal Calico could not compute for it is absent."""

    liked: bool
    scores: Mapping[str, float]


@dataclass(frozen=True)
class Signals:
    """How often a liked photo outscores a disliked one at the same place, by each signal:
    every place's pairs pooled (``auc``, with its interval by place), each place counting
    once (``macro``), and each signal minus the ``baseline``, paired (``versus``)."""

    auc: Mapping[str, Interval]
    macro: Mapping[str, float]
    versus: Mapping[str, Interval]
    baseline: str | None


def measure_signals(
    by_place: Mapping[str, Sequence[Scored]],
    *,
    baseline: str | None = None,
    resamples: int = RESAMPLES,
    unit: str = "place",
) -> Signals:
    names = list(dict.fromkeys(k for photos in by_place.values() for p in photos for k in p.scores))
    versus = {
        name: compare(
            _pooled_signal(name), _pooled_signal(baseline), by_place, n=resamples, unit=unit
        )
        for name in names
        if baseline is not None and name != baseline
    }
    return Signals(
        auc={
            name: versus[name].a
            if name in versus
            else place_bootstrap(by_place, _pooled_signal(name), n=resamples, unit=unit)
            for name in names
        },
        macro={name: macro_average(by_place, partial(_signal_auc, name=name)) for name in names},
        versus={name: measured.difference for name, measured in versus.items()},
        baseline=baseline,
    )


def _signal_pairs(places: Sequence[Sequence[Scored]], name: str) -> tuple[float, float]:
    """Liked-over-disliked wins (ties half) and pairs, within each place, all added up."""
    wins = pairs = 0.0
    for photos in places:
        yes = np.array([p.scores[name] for p in photos if p.liked and name in p.scores])
        no = np.array([p.scores[name] for p in photos if not p.liked and name in p.scores])
        if yes.size and no.size:
            wins += float((yes[:, None] > no[None, :]).sum())
            wins += 0.5 * float((yes[:, None] == no[None, :]).sum())
            pairs += yes.size * no.size
    return wins, pairs


def _signal_auc(photos: Sequence[Scored], name: str) -> float:
    wins, pairs = _signal_pairs([photos], name)
    return wins / pairs if pairs else math.nan


def _pooled_signal(name: str) -> Callable[[Sequence[Sequence[Scored]]], float]:
    def pooled(places: Sequence[Sequence[Scored]]) -> float:
        wins, pairs = _signal_pairs(places, name)
        return wins / pairs if pairs else math.nan

    return pooled


def sample_for_labelling(
    ranked: Sequence[str],
    seed: int,
    *,
    top: int = 36,
    near: int = 12,
    rest: int = 12,
    near_until: int = 300,
) -> list[str]:
    """Which photos to label, shuffled.

    Labelling only the current top results would show where the ranking is
    wrong at the top, never what it misses. So: the ``top`` best-ranked, a
    random ``near`` from just below them (to ``near_until``), and a random
    ``rest`` from everything else. Shuffled, so the order cannot hint at the
    ranking. A pool no bigger than the sample is taken whole.
    """
    rng = np.random.default_rng(seed)
    if len(ranked) <= top + near + rest:
        chosen = list(ranked)
    else:
        below, beyond = list(ranked[top:near_until]), list(ranked[near_until:])
        near_n = min(near, len(below))
        rest_n = min(rest + near - near_n, len(beyond))
        chosen = list(ranked[:top])
        chosen += [below[i] for i in sorted(rng.choice(len(below), near_n, replace=False))]
        chosen += [beyond[i] for i in sorted(rng.choice(len(beyond), rest_n, replace=False))]
    order = rng.permutation(len(chosen))
    return [chosen[i] for i in order]


def cut_at(scores: Sequence[float], labels: Sequence[int], cut: float) -> Threshold:
    """Precision and recall when ``score >= cut`` means "matches"; "partly" is left out."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    keep = y != 1
    chosen, positive = s[keep] >= cut, y[keep] == MATCH
    hits = int((chosen & positive).sum())
    precision = hits / int(chosen.sum()) if chosen.any() else math.nan
    recall = hits / int(positive.sum()) if positive.any() else math.nan
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return Threshold(cut, f1, precision, recall)


SAME_VIEW_BANDS: tuple[tuple[float, int], ...] = (
    (0.85, 15),
    (0.80, 15),
    (0.70, 20),
    (0.55, 10),
    (-math.inf, 5),
)
"""How many photos to label from each band of match to the reference, by its lower edge:
most around the default (0.75 since calibration; 0.80 before), enough above and below to
see where same views stop."""


OUTSIDE_SHORTLIST = -0.5
"""Where photos just outside same view's shortlist sit in ``PATCH_SAMPLE_BANDS``: alike as
a whole (``NEAR_SHORTLIST`` or more) but not enough to be compared part by part. Every
other photo sits in the last band."""

NEAR_SHORTLIST = 0.60
"""Whole-photo similarity from which a photo left off the shortlist may still be drawn,
to show whether the shortlist (0.75) leaves same views out."""

PATCH_SAMPLE_BANDS: tuple[tuple[float, int], ...] = (
    (0.25, 15),
    (0.18, 15),
    (0.12, 12),
    (0.06, 8),
    (0.0, 4),
    (OUTSIDE_SHORTLIST, 4),
    (-math.inf, 2),
)
"""How many photos to label from each band of patch share, by its lower edge: both tiers
(sure from 0.25, possible from 0.18), what lies just below them, a few photos just
outside the shortlist, and two from anywhere (plainly not the view, so every set has
something to tell apart). Drawn by what the task uses, so a repeat the whole-photo match
ranks low is still shown: at one site, five of six repeats were not in a sample drawn
by the whole-photo match."""


def sample_by_match(
    keys: Sequence[str],
    match: Sequence[float],
    seed: int,
    bands: Sequence[tuple[float, int]] = SAME_VIEW_BANDS,
) -> list[str]:
    """Which photos to label for a same-view search: up to each band's count, drawn at
    random within the band (all of a thinner one), shuffled together."""
    rng = np.random.default_rng(seed)
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    chosen: list[str] = []
    upper = math.inf
    for lower, count in bands:
        band = [keys[i] for i in order if lower <= match[i] < upper]
        picked = sorted(rng.choice(len(band), min(count, len(band)), replace=False))
        chosen += [band[i] for i in picked]
        upper = lower
    return [chosen[i] for i in rng.permutation(len(chosen))]


def choose_examples(keys: Sequence[str], k: int, seed: int) -> list[str]:
    """``k`` of ``keys`` (all, if fewer), the same for the same keys and seed in any order."""
    pool = sorted(keys)
    if len(pool) <= k:
        return pool
    rng = np.random.default_rng(seed)
    return [pool[i] for i in sorted(rng.choice(len(pool), k, replace=False))]


def apart_from(
    taken: Mapping[str, datetime | None], examples: Sequence[datetime | None], seconds: float
) -> set[str]:
    """Keys not shot within ``seconds`` of any example. Undated photos are kept: nothing
    says they are close."""
    times = [t for t in examples if t is not None]
    return {
        key
        for key, when in taken.items()
        if when is None or all(abs((when - t).total_seconds()) > seconds for t in times)
    }


@dataclass(frozen=True)
class Averaged:
    """A search's scores averaged over several draws of examples."""

    labelled: float
    precision_at_10: float
    ndcg_at_10: float
    auc: float


def average(scores: Sequence[SearchScore]) -> Averaged:
    """Means over draws; a metric undefined on a draw (NaN) is left out of its mean."""

    def mean(values: Sequence[float]) -> float:
        defined = [v for v in values if not math.isnan(v)]
        return sum(defined) / len(defined) if defined else math.nan

    return Averaged(
        labelled=mean([s.labelled for s in scores]),
        precision_at_10=mean([s.precision_at_10 for s in scores]),
        ndcg_at_10=mean([s.ndcg_at_10 for s in scores]),
        auc=mean([s.auc for s in scores]),
    )


EXAMPLE_WEIGHTS = (0.25, 0.5, 1.0, 2.0)
"""How much the examples count beside the text, measured. Text cosines spread about
0.1-0.35 and photo-to-photo cosines about 0.3-0.9, so these run from "text leads" to
"examples lead"."""

BURST_SECONDS = 120.0
"""Photos shot this close to an example are left out of its search's measurement."""

type Protocol = Literal["same search", "other site"]


@dataclass(frozen=True)
class LabelledSearch:
    """One labelled search as arrays: its photos still catalogued (``keys``, aligned with
    ``view``), when each was taken (on one clock: ``selection.clock``), their labels, the
    text's scores for them, and what tells a frame saved twice (``frames``, the vectors
    ``selection.copies`` was measured on)."""

    set_id: str
    text: str
    folders: Sequence[Path]
    keys: Sequence[str]
    taken: Sequence[datetime | None]
    labels: Mapping[str, int]
    view: FeatureView
    text_scores: CriteriaScores | None
    frames: NDArray[np.float32]

    @property
    def site(self) -> str:
        """The first folder's name, and how many more: enough to tell searches apart."""
        more = len(self.folders) - 1
        return self.folders[0].name.strip() + (f" +{more}" if more else "")

    def apart_from(self, other: LabelledSearch) -> bool:
        """No folder of one is, or is inside, a folder of the other: no photo is in both."""
        return not any(
            a.is_relative_to(b) or b.is_relative_to(a) for a in self.folders for b in other.folders
        )

    @property
    def matches(self) -> list[str]:
        return [k for k in self.keys if self.labels.get(k) == MATCH]

    def rows(self, keys: Sequence[str]) -> FeatureView:
        index = {k: i for i, k in enumerate(self.keys)}
        return self.view.take([index[k] for k in keys])


@dataclass(frozen=True)
class ExampleEvaluated:
    """One way of reading a search with examples, averaged over the draws of examples.

    ``source`` is the search the examples came from: the same one ("same search", the
    rest of it is scored) or another site's with the same words ("other site").
    """

    set_id: str
    text: str
    site: str
    protocol: Protocol
    source: str
    variant: str
    score: Averaged
    copies: float = 0.0
    """Labelled photos per draw left out as an example saved again (the same file, or a
    copy of its frame): scored, a copy of an example is a match for free (audit A05-8)."""
    nearby: float = 0.0
    """... and as shot within ``BURST_SECONDS`` of one (same search only)."""


def measure_examples(
    searches: Sequence[LabelledSearch],
    embedder: Embedder,
    *,
    k: int,
    draws: int,
    weights: Sequence[float],
) -> list[ExampleEvaluated]:
    """Each search read by its text alone, by text plus examples at each weight, and by
    the examples alone, with ``k`` labelled matches as the examples, ``draws`` times.

    Same search: the examples are left out of what is scored, with anything shot within
    ``BURST_SECONDS`` of one. Other site: examples from a search with the same words
    in other folders. Either way, an example saved again (the same file in another
    folder, or a copy of its frame) is never scored; each row counts what was left out.
    """
    rows: list[ExampleEvaluated] = []
    for search in searches:
        sources: list[tuple[Protocol, LabelledSearch]] = []
        if len(search.matches) > k:
            sources.append(("same search", search))
        sources += [
            ("other site", other)
            for other in searches
            if other.set_id != search.set_id
            and _same_words(other.text, search.text)
            and other.apart_from(search)
            and len(other.matches) >= k
        ]
        for protocol, source in sources:
            drawn = [
                _variants(search, source, protocol, embedder=embedder, k=k, draw=d, weights=weights)
                for d in range(draws)
            ]
            rows.extend(
                ExampleEvaluated(
                    set_id=search.set_id,
                    text=search.text,
                    site=search.site,
                    protocol=protocol,
                    source=source.set_id,
                    variant=variant,
                    score=average([scored[variant] for scored, _ in drawn]),
                    copies=sum(left.copies for _, left in drawn) / draws,
                    nearby=sum(left.nearby for _, left in drawn) / draws,
                )
                for variant in drawn[0][0]
            )
    return rows


def example_intervals(
    rows: Sequence[ExampleEvaluated], *, resamples: int = RESAMPLES
) -> dict[tuple[Protocol, str], tuple[Interval, Interval | None]]:
    """Each reading's AUC averaged over searches with its interval, and its difference
    from text alone, search by search, for each protocol. Each search is drawn whole: its
    AUC is already an average over draws of examples, so the photos within it are not
    drawn again, and the range is, if anything, too narrow."""
    out: dict[tuple[Protocol, str], tuple[Interval, Interval | None]] = {}
    for protocol in dict.fromkeys(r.protocol for r in rows):
        found: dict[str, dict[str, float]] = {}
        for r in rows:
            if r.protocol == protocol:
                found.setdefault(f"{r.set_id} {r.source}", {})[r.variant] = r.score.auc
        searches = {key: [values] for key, values in found.items()}

        def mean(variant: str) -> Callable[[Sequence[Sequence[Mapping[str, float]]]], float]:
            def averaged(drawn: Sequence[Sequence[Mapping[str, float]]]) -> float:
                values = [v for (s,) in drawn if not math.isnan(v := s.get(variant, math.nan))]
                return sum(values) / len(values) if values else math.nan

            return averaged

        for variant in dict.fromkeys(r.variant for r in rows if r.protocol == protocol):
            if variant == "text" or not any("text" in v for v in found.values()):
                out[protocol, variant] = (
                    place_bootstrap(searches, mean(variant), n=resamples, unit="search"),
                    None,
                )
                continue
            both = compare(mean(variant), mean("text"), searches, n=resamples, unit="search")
            out[protocol, variant] = (both.a, both.difference)
    return out


def _variants(
    search: LabelledSearch,
    source: LabelledSearch,
    protocol: Protocol,
    *,
    embedder: Embedder,
    k: int,
    draw: int,
    weights: Sequence[float],
) -> tuple[dict[str, SearchScore], _LeftOut]:
    """Every reading scored on one set of labelled photos (the same for each, so they
    compare), and what was left out to keep the examples from being scored."""
    seed = _seed(search.set_id, source.set_id, str(draw))
    examples = choose_examples(source.matches, k, seed)
    index = {key: i for i, key in enumerate(source.keys)}
    copied = copies_of(search.frames, source.frames[[index[e] for e in examples]])
    shown = set(examples) if protocol == "same search" else set()
    copies = {
        key
        for i, key in enumerate(search.keys)
        if key not in shown and (key in examples or bool(copied[i]))
    }
    labels = {key: v for key, v in search.labels.items() if key not in shown | copies}
    nearby = 0
    if protocol == "same search":
        when = dict(zip(search.keys, search.taken, strict=True))
        kept = apart_from(
            {key: when[key] for key in labels}, [when[e] for e in examples], BURST_SECONDS
        )
        nearby = len(labels) - len(kept)
        labels = {key: labels[key] for key in kept}
    similarity = example_similarity(embedder, search.view, source.rows(examples))
    readings = dict(_readings(search.text_scores, similarity, weights))
    scored = {
        name: score_search(dict(zip(search.keys, values.scores.tolist(), strict=True)), labels)
        for name, values in readings.items()
    }
    return scored, _LeftOut(copies=len(copies), nearby=nearby)


@dataclass(frozen=True)
class _LeftOut:
    copies: int
    nearby: int


def _readings(
    text: CriteriaScores | None, similarity: NDArray[np.float32], weights: Sequence[float]
) -> Iterator[tuple[str, CriteriaScores]]:
    if text is not None:
        yield "text", text
        for weight in weights:
            yield f"text+examples×{weight:g}", blend_examples(text, similarity, weight)
    yield "examples", blend_examples(None, similarity, 0.0)


def _same_words(a: str, b: str) -> bool:
    return " ".join(a.casefold().split()) == " ".join(b.casefold().split())


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.blake2b("\n".join(parts).encode(), digest_size=8).digest())


@dataclass(frozen=True)
class LabelledView:
    """One labelled same-view search: each labelled photo's match to the reference."""

    set_id: str
    text: str
    match: Mapping[str, float]
    labels: Mapping[str, int]


@dataclass(frozen=True)
class ViewEvaluated:
    """How well match separates same views on one search, and what a cut would keep."""

    set_id: str
    text: str
    labelled: int
    same: int
    auc: float
    at_default: Threshold
    best: Threshold
    at_pooled: Threshold
    """At the cut that is best over every search together."""
    held_out: Threshold
    """At the cut that is best over the *other* searches: what a threshold chosen without
    this view would do for it. NaN with nothing else to choose on."""


@dataclass(frozen=True)
class ViewReport:
    """Every search, and the cut that is best over all of them together (``pooled``): one
    ``min_similarity`` has to serve every view. ``skipped`` sets lost their reference."""

    rows: list[ViewEvaluated]
    default: float
    pooled: Threshold
    pooled_at_default: Threshold
    skipped: list[str]
    copies: int = 0
    """Labelled photos counted with another, as one frame saved twice (``count_copies_once``)."""
    embedder: str = ""
    """What gave "same subject"."""
    layout_weight: float = math.nan
    """How much "same camera angle" counted."""
    auc: Interval | None = None
    """The AUC averaged over searches, with its interval: references, then the photos
    labelled against each, drawn again."""
    recall: Interval | None = None
    """All searches together at the default cut: of the same views, how many it keeps."""
    precision: Interval | None = None
    """... and of what it keeps, how much is the same view."""
    searches: tuple[LabelledView, ...] = ()
    """What was measured, for comparing ways of matching (``view_differences``)."""
    versus: Interval | None = None
    """The averaged AUC minus the first way's, photo by photo, paired."""


def count_copies_once(
    search: LabelledView, keys: Sequence[str], vectors: NDArray[np.float32]
) -> tuple[LabelledView, int]:
    """``search`` with each frame once, and how many copies were folded in.

    A frame saved twice is one view, not two, so counting both would flatter recall
    wherever a shoot was exported twice. Frames of a burst stay apart: each is a shot.
    ``keys`` name the rows of ``vectors``; the first copy in that order stays, with the
    highest label any copy got (a frame called the same view once is one).
    """
    first = copy_of(vectors)
    labels: dict[str, int] = {}
    for i, key in enumerate(keys):
        head = keys[int(first[i])]
        labels[head] = max(labels.get(head, 0), search.labels[key])
    match = {k: search.match[k] for k in labels}
    return dataclasses.replace(search, match=match, labels=labels), len(keys) - len(labels)


def measure_views(
    searches: Sequence[LabelledView], default: float, *, resamples: int = RESAMPLES
) -> ViewReport:
    """2 (same view) is a match, 0 is not; 1 (same place, re-aimed) is left out."""
    scored = []
    for search in searches:
        keys = sorted(search.labels)
        scored.append(([search.match[k] for k in keys], [search.labels[k] for k in keys]))
    pooled_scores = [v for scores, _ in scored for v in scores]
    pooled_labels = [y for _, labels in scored for y in labels]
    pooled = best_threshold(pooled_scores, pooled_labels)
    rows = []
    for i, (search, (scores, labels)) in enumerate(zip(searches, scored, strict=True)):
        others_scores = [v for j, (s, _) in enumerate(scored) if j != i for v in s]
        others_labels = [y for j, (_, ys) in enumerate(scored) if j != i for y in ys]
        chosen = best_threshold(others_scores, others_labels).threshold
        held_out = (
            Threshold(math.nan, 0.0, math.nan, math.nan)
            if math.isnan(chosen)
            else cut_at(scores, labels, chosen)
        )
        rows.append(
            ViewEvaluated(
                set_id=search.set_id,
                text=search.text,
                labelled=len(scores),
                same=labels.count(MATCH),
                auc=auc(scores, labels),
                at_default=cut_at(scores, labels, default),
                best=best_threshold(scores, labels),
                at_pooled=cut_at(scores, labels, pooled.threshold),
                held_out=held_out,
            )
        )
    by_search = {
        search.set_id: [Labelled(search.labels[k], {"match": search.match[k]}) for k in keys]
        for search, keys in ((search, sorted(search.labels)) for search in searches)
    }

    def at_default(name: str) -> Callable[[Sequence[Sequence[Labelled]]], float]:
        def measured(groups: Sequence[Sequence[Labelled]]) -> float:
            photos = [p for g in groups for p in g]
            cut = cut_at([p.scores["match"] for p in photos], [p.label for p in photos], default)
            return float(getattr(cut, name))

        return measured

    return ViewReport(
        rows=rows,
        default=default,
        pooled=pooled,
        pooled_at_default=cut_at(pooled_scores, pooled_labels, default),
        skipped=[],
        auc=place_bootstrap(by_search, _mean_auc("match"), n=resamples, unit=VIEW_UNIT),
        recall=place_bootstrap(by_search, at_default("recall"), n=resamples, unit=VIEW_UNIT),
        precision=place_bootstrap(by_search, at_default("precision"), n=resamples, unit=VIEW_UNIT),
        searches=tuple(searches),
    )


VIEW_UNIT = "reference"


def view_differences(
    reports: Sequence[ViewReport], *, resamples: int = RESAMPLES
) -> list[ViewReport]:
    """``reports`` (ways of matching the same labelled searches), each after the first with
    its averaged AUC minus the first's, on the photos both measured."""
    if not reports:
        return []
    first = {s.set_id: s for s in reports[0].searches}
    out = [reports[0]]
    for report in reports[1:]:
        by_search = {}
        for search in report.searches:
            base = first.get(search.set_id)
            if base is None:
                continue
            keys = sorted(set(search.labels) & set(base.labels))
            by_search[search.set_id] = [
                Labelled(base.labels[k], {"this": search.match[k], "first": base.match[k]})
                for k in keys
            ]
        measured = compare(
            _mean_auc("this"), _mean_auc("first"), by_search, n=resamples, unit=VIEW_UNIT
        )
        out.append(dataclasses.replace(report, versus=measured.difference))
    return out


GEOMETRY_BAND = (0.12, 0.30)
"""Patch shares where the share alone is least sure (audit A10-7): from below the possible
tier (0.18) to above the sure one (0.25). Look-alike rooms at one place share as much here
as true repeats at another, so this is where matches in place could tell them apart. Taken
from the audit, not measured: it is where to look, not a cut. Move it with the tiers."""


@dataclass(frozen=True)
class ViewGeometry:
    """Today's patch share and the share of matches in place (``consistent_shares``), each
    second report paired against the first: over every labelled photo, and over the photos
    whose share is in ``band`` (``in_band`` of them)."""

    whole: list[ViewReport]
    within_band: list[ViewReport]
    band: tuple[float, float]
    in_band: int


def measure_view_geometry(
    share: Sequence[LabelledView],
    consistent: Sequence[LabelledView],
    *,
    sure: float,
    band: tuple[float, float] = GEOMETRY_BAND,
    resamples: int = RESAMPLES,
) -> ViewGeometry:
    """``share`` and ``consistent`` measure the same labelled searches, photo for photo.
    Both are judged from the ``sure`` cut, though only the share was chosen for it: the
    best cut and the cut held out from each search are what to compare. The band is chosen
    by the share, the photo's place in today's tiers, so both are judged on the same photos."""
    lo, hi = band

    def within(search: LabelledView, by: LabelledView) -> LabelledView:
        keys = [k for k in search.labels if lo <= by.match[k] < hi]
        return dataclasses.replace(
            search,
            match={k: search.match[k] for k in keys},
            labels={k: search.labels[k] for k in keys},
        )

    pairs = [(s, c) for s, c in zip(share, consistent, strict=True)]
    banded = [(within(s, s), within(c, s)) for s, c in pairs]
    banded = [(s, c) for s, c in banded if s.labels]

    def compared(
        today: Sequence[LabelledView], in_place: Sequence[LabelledView]
    ) -> list[ViewReport]:
        reports = [
            dataclasses.replace(measure_views(views, sure, resamples=resamples), embedder=name)
            for name, views in (("share", today), ("in place", in_place))
        ]
        return view_differences(reports, resamples=resamples)

    return ViewGeometry(
        whole=compared(share, consistent),
        within_band=compared([s for s, _ in banded], [c for _, c in banded]),
        band=band,
        in_band=sum(len(s.labels) for s, _ in banded),
    )


@dataclass(frozen=True)
class Labelled:
    """A labelled photo in a search (2 matches, 1 partly, 0 not) and each reading's score."""

    label: int
    scores: Mapping[str, float]


@dataclass(frozen=True)
class SearchIntervals:
    """Each reading's AUC averaged over searches, with its interval (the searches, then the
    labelled photos within them, drawn again), and each minus the ``baseline``'s, paired."""

    auc: Mapping[str, Interval]
    versus: Mapping[str, Interval]
    baseline: str | None


def _mean_auc(name: str) -> Callable[[Sequence[Sequence[Labelled]]], float]:
    """The AUC of each search, averaged over the searches where it is defined."""

    def mean(searches: Sequence[Sequence[Labelled]]) -> float:
        found = [
            auc([p.scores[name] for p in photos], [p.label for p in photos]) for photos in searches
        ]
        defined = [v for v in found if not math.isnan(v)]
        return sum(defined) / len(defined) if defined else math.nan

    return mean


def criteria_intervals(rows: Sequence[Evaluated], *, resamples: int = RESAMPLES) -> SearchIntervals:
    """Each scorer's AUC over your labelled searches, and each minus the first scorer's, on
    the same photos."""
    by_search: dict[str, dict[str, dict[str, float]]] = {}
    labels: dict[str, Mapping[str, int]] = {}
    for row in rows:
        labels[row.set_id] = row.labels
        found = by_search.setdefault(row.set_id, {})
        for key in row.labels:
            found.setdefault(key, {})[row.scorer] = row.scores[key]
    names = list(dict.fromkeys(r.scorer for r in rows))
    baseline = names[0] if names else None
    photos = {
        set_id: [Labelled(labels[set_id][k], scores) for k, scores in sorted(found.items())]
        for set_id, found in by_search.items()
    }
    versus = {
        name: compare(_mean_auc(name), _mean_auc(baseline), photos, n=resamples, unit="search")
        for name in names
        if baseline is not None and name != baseline
    }
    return SearchIntervals(
        auc={
            name: versus[name].a
            if name in versus
            else place_bootstrap(photos, _mean_auc(name), n=resamples, unit="search")
            for name in names
        },
        versus={name: measured.difference for name, measured in versus.items()},
        baseline=baseline,
    )


@dataclass(frozen=True)
class LocationRatings:
    """How well each signal separates the photos you liked from those you did not, at one
    location (scored against that location's whole pool, as a request would)."""

    location: str
    rated: int
    liked: int
    disliked: int
    auc: Mapping[str, float]


@dataclass(frozen=True)
class RatingsReport:
    """``rated`` photos were found in the catalog, ``missing`` were not; ``one_sided``
    locations had only likes or only dislikes, so nothing could be measured there."""

    rows: list[LocationRatings]
    rated: int
    missing: int
    one_sided: int
    measured: Signals | None = None
    """Each signal's AUC with its interval by place, and each against strength."""


def pair_weighted_auc(rows: Sequence[LocationRatings], name: str) -> float:
    """The AUC over every like/dislike pair within a location, all locations together:
    each location counts by its pairs, so one with a single dislike counts little."""
    pairs = [(r.auc[name], r.liked * r.disliked) for r in rows if name in r.auc]
    pairs = [(value, n) for value, n in pairs if not math.isnan(value)]
    total = sum(n for _, n in pairs)
    return sum(value * n for value, n in pairs) / total if total else math.nan


@dataclass(frozen=True)
class TastePhoto:
    """One rated photo, for measuring taste. ``from_session``: rated in a taste session (a
    sample of the whole range, so it can be judged); otherwise 👍/👎 on a result (only
    trained on: results are what Calico already ranked high). ``general`` is today's
    aesthetic p("strong"), ``quality`` today's technical quality."""

    location: str
    liked: bool
    from_session: bool
    vector: NDArray[np.float32]
    general: float
    quality: float
    content_key: str
    same_frame: str = ""
    """The content key of the rated photo it is a copy of (``selection.copies``), if any:
    a frame saved twice has two keys."""
    same_place: str = ""
    """The name its place shares with names that probably mean the same place (as
    ``calico locations`` groups them), if any."""
    candidate_quality: float | None = None
    """Its quality with a candidate technical measure (``calico eval taste --technical``),
    when one is measured."""
    candidate_general: float | None = None
    """General's aesthetic for it from a candidate (``calico eval taste --aesthetic``), on
    any scale where higher looks better, when one is measured."""
    candidate_score: float | None = None
    """One number for it from a file (``calico eval taste --scores``), on any scale where
    higher looks better, when one is measured. Photos with one are the photos judged then;
    the others still teach Yours."""
    place_mean: NDArray[np.float32] | None = None
    """The mean ``vector`` over every catalogued photo at its place, rated or not, for a fit
    centred on each place (``TASTE_FITS``); None where the place has no name."""
    candidate_vector: NDArray[np.float32] | None = None
    """Its vector in a candidate embedding (``calico eval taste --yours-embedder``), for Yours
    learned on that embedding (``TasteFit`` "embedding"), when one is measured."""
    whole_range: bool = True
    """False for a photo drawn for its value (``VALUE_STRATA``), or shown first by General or
    Yours in Compare Strongest: each was chosen by a taste's ranking, so judging a candidate
    on it favours that taste (owner's decision, 7 November 2026). It counts among the first
    picks and is learned from, but the AUCs over the whole range and the top share leave it
    out. Read apart, they are ``TasteReport.chosen`` (11 November 2026, with the owner's
    leave)."""
    collection: str = ""
    """The collection it is catalogued in, so another fit can be judged collection by
    collection: a fit for trips must not cost the places Yours already knows."""
    scale: str = ""
    """Its scale (aerial, wide, medium, close, macro: the built-in ``scale`` vocabulary), for
    a fit by scale (``TASTE_FITS``); "" where it was not read."""

    @property
    def frame(self) -> str:
        return self.same_frame or self.content_key

    @property
    def place(self) -> str:
        return self.same_place or self.location


@dataclass(frozen=True)
class Apart:
    """Rated photos that tell nothing about some places, and those left out because they
    could: ``copies`` of a photo there, and photos at a place grouped with one there
    (content keys)."""

    photos: list[TastePhoto]
    copies: set[str]
    same_place: set[str]


def taste_apart(photos: Sequence[TastePhoto], places: Collection[str]) -> Apart:
    """``photos`` not at ``places``, nor at a place whose name probably means one of them,
    nor a copy of a photo at one. Held out by name alone, a judged photo saved twice, or a
    trip filed under two names, would be learned from as another place's: with an
    embedding, a copy with the same label is a near-perfect predictor (audit A05-4)."""
    at = [p for p in photos if p.location in places]
    groups = {p.place for p in at} | set(places)
    frames = {p.frame for p in at}
    kept, copies, grouped = [], set(), set()
    for p in photos:
        if p.location in places:
            continue
        if p.place in groups:
            grouped.add(p.content_key)
        elif p.frame in frames:
            copies.add(p.content_key)
        else:
            kept.append(p)
    return Apart(kept, copies, grouped)


TASTE_FIRST = (20, 40)
"""How many of each template's first picks at a place are judged: every Strongest pick
should be portfolio worthy, and the first are in every shortlist whatever its size, so a
result is judged by its picks, not by how it orders the weakest."""

type FirstPicksFn = Callable[[str, "TasteWeights | None", int], Sequence[str]]
"""The content keys of the first ``k`` photos a template would show at a place, in order:
General's when the taste is ``None``, else Yours's with that taste."""


@dataclass(frozen=True)
class FirstPicks:
    """Among a template's first picks at a place: how many you rated in a session (Like or
    Don't like), and how many of those you liked. Taste sessions sample the whole range
    evenly, so few rated photos fall among the first picks: read ``share`` with ``rated``."""

    rated: int
    liked: int

    @property
    def share(self) -> float:
        return self.liked / self.rated if self.rated else math.nan


@dataclass(frozen=True)
class TasteRow:
    """One place, judged by a model trained on every other place. AUC: how often a photo
    you liked outscores one you did not, by the aesthetic alone and by strength.
    ``general_first`` and ``yours_first``: the photos you rated among each template's
    first picks, one per ``TasteReport.first_at``."""

    location: str
    rated: int
    liked: int
    disliked: int
    general_aesthetic: float
    yours_aesthetic: float
    general_strength: float
    yours_strength: float
    general_first: tuple[FirstPicks, ...] = ()
    yours_first: tuple[FirstPicks, ...] = ()
    for_value: int = 0
    """Of ``rated``, the photos a taste's ranking chose (drawn for their value, or shown in
    Compare Strongest; ``TastePhoto.whole_range``): in the first picks, not in the AUCs."""
    fitted_strength: float = math.nan
    """By strength, Yours fitted another way (``TasteReport.fit``), when one is measured."""
    leaned: float = math.nan
    """With ``fit="leaning"``: General's mean share of the mix here (0 = Yours alone)."""


@dataclass(frozen=True)
class Forecast:
    """One judged photo as a model that never learned its place scored it: p(you like it),
    what you said, and whether a taste's ranking chose it (a first pick, not a sample of the
    whole range). What a probability is calibrated on (``selection.calibration``)."""

    yours: float
    liked: bool
    first_pick: bool
    place: str


@dataclass(frozen=True)
class ScoreReading:
    """A file's scores in one population of photos, paired with what they are measured
    against: the scores (``a``) against General's strength (``b``), and Yours plus the scores
    (``a``) against Yours (``b``), each over the like/dislike pairs at each location."""

    versus_general: Compared
    blend_versus_yours: Compared


@dataclass(frozen=True)
class ScoreSide:
    """One population at one location: its likes and dislikes, and how often a like outscores
    a dislike by the file's scores, by General's strength and by Yours's (NaN with no pair)."""

    liked: int
    disliked: int
    scores: float
    general: float
    yours: float


@dataclass(frozen=True)
class ScoreRow:
    """One judged location, so a reading can be traced to the locations that carry it."""

    location: str
    whole: ScoreSide
    chosen: ScoreSide


@dataclass(frozen=True)
class ScoreReport:
    """A file of scores measured as a candidate (``TastePhoto.candidate_score``), apart for the
    whole range and for the photos a taste chose to show. ``scored`` photos were judged."""

    scored: int
    whole: ScoreReading
    chosen: ScoreReading
    name: str = ""
    """The file and a digest of its numbers (``evaluation.scores.read_scores``)."""
    rows: tuple[ScoreRow, ...] = ()


@dataclass(frozen=True)
class TasteReport:
    """Every judged place, and each AUC over all of them (each place counts by its
    like/dislike pairs). ``one_sided`` places had only likes or only dislikes.
    ``general_first`` and ``yours_first`` add up every place's first picks (each rated
    photo counts once), one per ``first_at``."""

    rows: list[TasteRow]
    general_aesthetic: float
    yours_aesthetic: float
    general_strength: float
    yours_strength: float
    trained_on: int
    one_sided: list[str]
    feature: str
    l2: float
    first_at: tuple[int, ...] = ()
    general_first: tuple[FirstPicks, ...] = ()
    yours_first: tuple[FirstPicks, ...] = ()
    counted: str = ""
    """Which answers gave each photo its label (``reviews.TASTE_RULE``)."""
    notes: tuple[str, ...] = ()
    """What the measurement could not do as asked, such as General falling back to a
    constant where its aesthetic cannot run (rule 5)."""
    unsure: int = 0
    """Photos left out: every answer to them was not sure."""
    disagree: int = 0
    """Photos liked in one use case and not in another (the newest answer decided)."""
    strength: Compared | None = None
    """The strength AUCs with their intervals by place: ``a`` is Yours, ``b`` General, and
    the difference Yours minus General, paired."""
    aesthetic: Compared | None = None
    first: tuple[Compared, ...] = ()
    """The first-picks shares (liked of rated), one per ``first_at``, as ``strength``."""
    macro: Mapping[str, float] = field(default_factory=dict)
    """Each AUC averaged over places, each counting once, beside the pooled one."""
    copies_left_out: int = 0
    """Rated photos never learned from where a copy of them was judged (``taste_apart``)."""
    same_place_left_out: int = 0
    """... and those at a place grouped with the judged one."""
    same_places: tuple[tuple[str, ...], ...] = ()
    """The place names grouped as probably one place, among those rated."""
    quality_weight: float = math.nan
    """Quality's share of strength, as measured here."""
    candidate: Compared | None = None
    """General's strength with the candidate quality (``a``) against today's (``b``), and
    the difference, paired: whether the candidate measure ranks your ratings better."""
    technical: str = ""
    """The candidate technical measure (``technical@3``), when one was measured."""
    candidate_general: Compared | None = None
    """General's strength with a candidate aesthetic (``a``) against today's (``b``), and
    the difference, paired: whether the candidate ranks your ratings better than today's
    prompts, as Strongest would mix it with quality."""
    candidate_general_alone: Compared | None = None
    """... and by the aesthetic alone, where quality cannot dilute the difference."""
    aesthetic_candidate: str = ""
    """The candidate aesthetic (``laion_aesthetic@...``), when one was measured."""
    general_with_quality: Compared | None = None
    """General's strength (``a``) against its aesthetic alone (``b``), and the difference,
    paired: whether mixing in quality at ``quality_weight`` ranks your ratings better. The
    two AUCs' own intervals overlap too much to say; the paired one is narrower."""
    yours_with_quality: Compared | None = None
    """... and the same for Yours."""
    unread: int = 0
    """Rated photos the candidate could not read, left out of every measure here."""
    fit: str = ""
    """Another way of fitting Yours (``TASTE_FITS``), when one was measured."""
    fitted: Compared | None = None
    """Yours fitted that way at ``l2`` (``a``) against today's as it is used, at
    ``TASTE_L2`` (``b``), by strength, and the difference, paired: whether the other fit
    ranks your ratings better at places it never saw."""
    uncentred: tuple[str, ...] = ()
    """Places with no mean to centre on (no name), learned from as they are."""
    fitted_by_collection: tuple[tuple[str, Compared], ...] = ()
    """``fitted`` again over each collection's places alone, when the judged places span more
    than one: whether a fit that helps on trips costs the places Yours already knows."""
    yours_candidate: str = ""
    """With ``fit`` "embedding": the candidate embedding Yours was learned on (its key)."""
    candidate_general_by_collection: tuple[tuple[str, Compared], ...] = ()
    """``candidate_general`` the same way: whether a General that helps on trips costs the
    abandoned places."""
    general_blend: bool = False
    """Whether the candidate General was today's General and the candidate averaged by rank
    within each place, half each (``measure_taste(general_blend=True)``)."""
    offset_on_candidate: bool = False
    """Whether ``fit="offset"`` read the candidate General rather than today's: Yours on top
    of the General being tried."""
    top_candidate_general: Compared | None = None
    """The candidate General (``a``) against today's (``b``), by the share you liked among
    the top ``TOP_SHARE`` of each place's rated photos by strength, and the difference."""
    top_fitted: Compared | None = None
    """... and Yours fitted another way (``a``) against today's (``b``), the same way."""
    liked_rank_by_scale: tuple[tuple[str, float, float], ...] = ()
    """With ``fit="by_scale"``: for each scale, where the photos you liked rank at their
    place on average (0 the weakest, 1 the strongest), by today's Yours and by the fit."""
    forecasts: tuple[Forecast, ...] = ()
    """Every judged photo's p(you like it) from the model that never learned its place, in
    place order: what Yours's probability is read from (``train_taste``, from the first picks)."""
    pooled: int = 0
    """How many places had their strength scored over their whole pool, as Strongest ranks it
    (``Pool``); the rest, among their rated photos alone, which is all a candidate can be."""
    chosen: Compared | None = None
    """Yours (``a``) against General (``b``) by strength among the photos a taste chose to
    show, apart from the whole range: where ``strength`` leaves them out. A taste's ranking
    chose them, so one that chose some has less to rank them by than one that chose none: this
    shows how each ranks contenders, not which chooses better. None where none was rated."""
    scores: ScoreReport | None = None
    """With a file of scores (``TastePhoto.candidate_score``): measured beside General and
    Yours, among the photos the file scores."""

    @property
    def yours_wins(self) -> bool:
        """Whether Yours may ship: it ranks your ratings better than General by strength
        (what Strongest and Stories rank by), at places it never learned from, by more than
        chance (the paired difference's interval lies above 0) at ``WIN_PLACES`` or more."""
        return (
            self.strength is not None
            and self.strength.difference.places >= WIN_PLACES
            and self.strength.difference.lo > 0
        )


type TasteFit = Literal["centred", "offset", "leaning", "by_scale", "top_pairs", "embedding"]
TASTE_FITS: tuple[TasteFit, ...] = ("centred", "offset", "leaning", "by_scale", "top_pairs")
"""Other ways of fitting Yours, each measured against today's (``measure_taste(fit=...)``).
``centred``: each photo's vector minus its place's mean (audit A10-4). You judge photos
against others at the same place, but places sit apart in every embedding and you like
more at some than at others, so a taste fitted across places partly learns "this kind of
site". Centred, only the differences within a place are left to learn. Yours is linear, so
centring when scoring would move every photo at a place alike and change no order there:
it is the fit that changes. Measured on 542 ratings at 11 places (29 October 2026), it
ranked worse on every embedding (CLIP -0.03 [-0.07, -0.00] at quality weight 0.35, SigLIP
-0.05 [-0.09, -0.01]); on 975 at 20 places (30 October 2026), CLIP -0.02 [-0.04, -0.01]
at 0.35: what sets your places apart tells likes within a place too, so today's fit stays.

``offset``: General's view is one more input, not held back (``fit_taste(general=)``), so
Yours keeps what General gets right and learns only what it misses. Unlike centring, it
throws nothing away. With it, ``l2`` also sets how much the embedding counts beside
General, so it is measured at more than one. Measured on 542 ratings at 11 places (29
October 2026), on CLIP: -0.07 [-0.12, -0.01] at l2 10, -0.02 [-0.05, 0.02] at 100, +0.00
[-0.03, 0.03] at 1000. General is zero-shot on the same CLIP embedding, so its logit is
nearly a direction Yours can already learn: the offset adds no new view, and today's fit
stays.

``leaning``: Yours leans on General where its ratings hold few photos like the one scored
(audit plan S6). Yours is fitted as today; General is calibrated on the same ratings
(``models.taste.fit_general``); their log-odds are mixed by how familiar each photo is
(``models.taste.familiarity``). Unlike the offset, the mix differs photo by photo: Yours is
learned mostly from abandoned places and ranks worse than General on trips, where it has
seen little. Where every photo is as familiar as the typical rated one, it is today's Yours.
Kept only if trips improve and the abandoned places hold (``fitted_by_collection``).
Measured on 1,611 ratings at 32 places (7 November 2026), CLIP: travel +0.03 [-0.11, 0.11]
at 9 places, explore -0.01 [-0.04, 0.02] at 18. Travel moved up but cannot be shown on so
few places, so today's fit stays; it is worth running again when more trips are rated.

``by_scale``: how much you like each scale (``models.taste.scale_offsets``) is learned apart
from what makes a photo strong within its scale (``models.taste.fit_by_scale``). Fitted as
today, Yours learned close shots, which you like least, as a direction that ranks every
detail down: liked macros at 0.28 of their place against General's 0.55 (7 November 2026).
Kept only if it is shown better overall and not shown worse at abandoned places.
Measured on 1,848 ratings at 37 places (7 November 2026), CLIP: -0.008 [-0.030, 0.010]
overall, abandoned places -0.021 [-0.056, 0.004]. Liked details rose (macros 0.28 -> 0.39)
but weak ones with them, and liked medium shots fell, so today's fit stays.

``top_pairs``: Yours learned from pairs among the first picks alone (``models.taste.fit_pairs``):
a photo you liked against one you did not, at the same place, both chosen by a taste's
ranking (``TastePhoto.whole_range`` false: Compare Strongest and the first-picks strata).
The bar counts weak photos among the first picks, and these pairs are exactly that choice,
where a fit on every rating also learns the throwaways a place holds. Within a place the
place's level cancels, so the fit cannot learn that you like more at some places: nothing
within one place, but a search across several ranks each place as if all were liked alike
(owner's decision to note, 8 November 2026). Judged on the other rated photos, so no pair
is ever judged. Unlike the pairwise loss skipped on 28 October 2026, over every pair at a
place, it learns only from the top.

``embedding`` (not a ``--fit``: ``calico eval taste --yours-embedder``): Yours fitted as
today, on another embedding's vectors (``TastePhoto.candidate_vector``), such as CLIP
ViT-L/14 against today's B/16. Paired with today's Yours on the same photos, so the
difference has its own interval (roadmap, "ViT-L/14, planned")."""


TOP_SHARE = 0.25
"""The top of a place for judging first picks among rated photos: its strongest quarter by
strength. Strongest's bar is about what it shows first, and the whole-range AUC counts
the order of weak photos as much (decision 3, 7 November 2026). A candidate General cannot
run Strongest here, so its first picks are judged among the photos you rated."""

WIN_PLACES = 5
"""Fewer judged places than this cannot show Yours beats General, however clear: with each
place both a test and, for the others, training, a few places say little about the next."""


@dataclass(frozen=True)
class _Judged:
    """One photo judged at its place, as the intervals resample it."""

    liked: bool
    general_aesthetic: float
    yours_aesthetic: float
    general_strength: float
    yours_strength: float
    general_first: tuple[bool, ...]
    """Whether General would show it among its first ``k``, for each ``first_at``."""
    yours_first: tuple[bool, ...]
    candidate_strength: float = math.nan
    candidate_general_aesthetic: float = math.nan
    candidate_general_strength: float = math.nan
    fitted_strength: float = math.nan
    today_strength: float = math.nan
    """Today's Yours as it is used (``TASTE_L2``), which another fit is measured against."""
    score: float = math.nan
    """The photo's number from a file (``TastePhoto.candidate_score``)."""
    blend: float = math.nan
    """Yours plus that number: the mean of their z-scores at the place (``mean_z``)."""
    whole_range: bool = True
    """Read by the AUCs (``TastePhoto.whole_range``); the first picks read every photo. Not
    a photo only thumbed on a result, judged when a file scores it: a taste ranked those."""


@dataclass(frozen=True)
class _PlaceApart:
    """One judged place: its rated photos (``here``), the ratings that tell nothing about it
    (``train``), and Yours fitted on those."""

    location: str
    here: list[TastePhoto]
    train: list[TastePhoto]
    weights: TasteWeights


@dataclass(frozen=True)
class _Judging:
    places: list[_PlaceApart]
    one_sided: list[str]
    copies: set[str]
    same_place: set[str]
    learnable: Apart


def _places_apart(
    photos: Sequence[TastePhoto],
    *,
    feature: str,
    extractor: str,
    l2: float,
    judged: Collection[str] | None,
    untrained: Collection[str],
    scored_only: bool = False,
) -> _Judging:
    """Each place rated in a session (or each of ``judged``), with Yours fitted on every
    rating that tells nothing about it: the one way a taste is judged at places it never
    learned from. Places with only likes or only dislikes, here or in what is left to learn
    from, are ``one_sided``; photos at ``untrained`` places are never learned from. With
    ``scored_only`` (a file of scores), every photo with a score is judged, rated in a session
    or thumbed on a result, and no other: the rest still teach."""

    def judgeable(p: TastePhoto) -> bool:
        return p.candidate_score is not None if scored_only else p.from_session

    learnable = taste_apart(photos, untrained)
    copies, grouped = set(learnable.copies), set(learnable.same_place)
    places = {p.location for p in photos if judgeable(p)}
    out, one_sided = [], []
    for location in sorted(places if judged is None else places & set(judged)):
        here = [p for p in photos if judgeable(p) and p.location == location]
        apart = taste_apart(learnable.photos, [location])
        train = apart.photos
        liked = sum(p.liked for p in here)
        if liked in {0, len(here)} or len({p.liked for p in train}) < 2:
            one_sided.append(location)
            continue
        copies |= apart.copies
        grouped |= apart.same_place
        weights = fit_taste(
            np.stack([p.vector for p in train]),
            np.asarray([p.liked for p in train], dtype=np.int8),
            feature=feature,
            extractor=extractor,
            l2=l2,
        )
        out.append(_PlaceApart(location, here, train, weights))
    return _Judging(out, one_sided, copies, grouped, learnable)


def measure_taste(
    photos: Sequence[TastePhoto],
    *,
    feature: str,
    extractor: str = "",
    quality_weight: float,
    l2: float = TASTE_L2,
    first_picks: FirstPicksFn | None = None,
    at: Sequence[int] = TASTE_FIRST,
    resamples: int = RESAMPLES,
    judged: Collection[str] | None = None,
    untrained: Collection[str] = (),
    fit: TasteFit | None = None,
    general_blend: bool = False,
    pools: Mapping[str, Pool] | None = None,
) -> TasteReport:
    """Would the Yours template rank your ratings better than the General one, at places
    it never learned from? Each place rated in a session is judged by a model trained on
    every other rating. Strength mixes technical quality and the aesthetic as Strongest
    does (``quality_weight``): over each place's whole ``pool`` where one is given (Yours's
    aesthetic from the weights that never saw the place, applied to the pool), which is what
    a search ranks; among the place's rated photos alone where there is none, and whenever a
    candidate or a fit is measured, which is read for those photos only and is compared with
    today's the same way.

    With ``first_picks``, also: of the photos each template would show first at the place
    (the first ``k`` for each of ``at``), how many you rated, and liked.

    Each measure over all places comes with its interval by place (``stats``), from
    ``resamples`` resamples of the judged photos: each place's model is kept, not refitted.

    ``judged`` limits which places are judged (default: every one rated in a session);
    photos at ``untrained`` places are never learned from (the hold-out). ``fit`` also
    measures Yours fitted another way (``TASTE_FITS``) against today's, paired. Photos
    with a ``candidate_quality`` or ``candidate_general`` also measure that candidate
    against today's General, paired; with ``general_blend``, the candidate General is
    today's and the candidate averaged by rank within each place, half each, untuned. With
    a candidate General, an offset Yours (``fit="offset"``) reads it rather than today's."""
    first_at = tuple(at) if first_picks is not None else ()
    if general_blend:
        photos = _blended(photos)
    measured_general = bool(photos) and all(p.candidate_general is not None for p in photos)
    scored = any(p.candidate_score is not None for p in photos)
    plain = (
        fit is None
        and not general_blend
        and not measured_general
        and not scored
        and not any(p.candidate_quality is not None for p in photos)
    )
    pooled_places = 0
    centred = _centring(photos) if fit == "centred" else None
    _require_fit_inputs(photos, fit)
    liked_ranks: dict[str, tuple[list[float], list[float]]] = {}
    rows = []
    judged_at: dict[str, list[_Judged]] = {}
    leaned: dict[str, float] = {}
    collection_of: dict[str, str] = {}
    judging = _places_apart(
        photos,
        feature=feature,
        extractor=extractor,
        l2=l2,
        judged=judged,
        untrained=untrained,
        scored_only=scored,
    )
    learnable, one_sided = judging.learnable, judging.one_sided
    copies, grouped = judging.copies, judging.same_place
    for place in judging.places:
        location, here, train, weights = place.location, place.here, place.train, place.weights
        liked = sum(p.liked for p in here)
        collection_of[location] = min(p.collection for p in here)
        yours = taste_probability(weights, np.stack([p.vector for p in here]))
        general = np.asarray([p.general for p in here], dtype=np.float32)
        quality = percentile_rank(np.asarray([p.quality for p in here], dtype=np.float32))
        verdicts = {p.content_key: p.liked for p in here}
        general_shown, yours_shown = (
            ((), ())
            if first_picks is None
            else (
                _shown(first_picks, location, None, first_at),
                _shown(first_picks, location, weights, first_at),
            )
        )
        general_strength, yours_strength, over_pool = _place_strengths(
            here,
            pools.get(location) if plain and pools else None,
            general=general,
            yours=yours,
            weights=weights,
            quality=quality,
            quality_weight=quality_weight,
        )
        pooled_places += over_pool
        scores = {
            "general_aesthetic": [float(v) for v in general],
            "yours_aesthetic": [float(v) for v in yours],
            "general_strength": general_strength,
            "yours_strength": yours_strength,
        }
        if all(p.candidate_quality is not None for p in here):
            candidate = np.asarray([p.candidate_quality for p in here], dtype=np.float32)
            scores["candidate_strength"] = _strength(
                percentile_rank(candidate), general, quality_weight
            )
        if all(p.candidate_general is not None for p in here):
            other = np.asarray([p.candidate_general for p in here], dtype=np.float32)
            scores["candidate_general_aesthetic"] = [float(v) for v in other]
            scores["candidate_general_strength"] = _strength(quality, other, quality_weight)
        if scored:
            scores |= _file_scores(here, scores["yours_strength"])
        if fit is not None:
            scores["today_strength"] = (
                scores["yours_strength"]
                if l2 == TASTE_L2
                else _strength(quality, _today(train, here, feature, extractor), quality_weight)
            )
            fitted, leaned_on = _fitted(
                fit,
                train,
                here,
                centred,
                feature=feature,
                extractor=extractor,
                l2=l2,
                on_candidate=measured_general,
            )
            scores["fitted_strength"] = _strength(quality, fitted, quality_weight)
            if leaned_on is not None:
                leaned[location] = float(np.mean(1 - leaned_on))
            if fit == "by_scale":
                _add_liked_ranks(liked_ranks, here, scores)
        judged_at[location] = [
            _Judged(
                liked=p.liked,
                general_aesthetic=scores["general_aesthetic"][i],
                yours_aesthetic=scores["yours_aesthetic"][i],
                general_strength=scores["general_strength"][i],
                yours_strength=scores["yours_strength"][i],
                general_first=tuple(p.content_key in shown for shown in general_shown),
                yours_first=tuple(p.content_key in shown for shown in yours_shown),
                candidate_strength=scores.get("candidate_strength", [math.nan] * len(here))[i],
                candidate_general_aesthetic=scores.get(
                    "candidate_general_aesthetic", [math.nan] * len(here)
                )[i],
                candidate_general_strength=scores.get(
                    "candidate_general_strength", [math.nan] * len(here)
                )[i],
                fitted_strength=scores.get("fitted_strength", [math.nan] * len(here))[i],
                today_strength=scores.get("today_strength", [math.nan] * len(here))[i],
                score=scores.get("score", [math.nan] * len(here))[i],
                blend=scores.get("blend", [math.nan] * len(here))[i],
                whole_range=p.whole_range and p.from_session,
            )
            for i, p in enumerate(here)
        ]
        rows.append(
            TasteRow(
                location=location,
                rated=len(here),
                liked=liked,
                disliked=len(here) - liked,
                general_aesthetic=_place_auc(judged_at[location], "general_aesthetic"),
                yours_aesthetic=_place_auc(judged_at[location], "yours_aesthetic"),
                general_strength=_place_auc(judged_at[location], "general_strength"),
                yours_strength=_place_auc(judged_at[location], "yours_strength"),
                general_first=_first(general_shown, verdicts),
                yours_first=_first(yours_shown, verdicts),
                for_value=sum(not (p.whole_range and p.from_session) for p in here),
                fitted_strength=_place_auc(judged_at[location], "fitted_strength")
                if fit is not None
                else math.nan,
                leaned=leaned.get(location, math.nan),
            )
        )

    def added(name: str) -> tuple[FirstPicks, ...]:
        return tuple(
            FirstPicks(
                rated=sum(getattr(r, name)[i].rated for r in rows),
                liked=sum(getattr(r, name)[i].liked for r in rows),
            )
            for i in range(len(first_at))
        )

    return TasteReport(
        rows=rows,
        general_aesthetic=_pooled(judged_at, "general_aesthetic"),
        yours_aesthetic=_pooled(judged_at, "yours_aesthetic"),
        general_strength=_pooled(judged_at, "general_strength"),
        yours_strength=_pooled(judged_at, "yours_strength"),
        trained_on=len(learnable.photos),
        one_sided=one_sided,
        feature=feature,
        l2=l2,
        pooled=pooled_places,
        chosen=_chosen(judged_at, resamples),
        scores=_scores_report(judged_at, resamples) if scored else None,
        first_at=first_at,
        general_first=added("general_first"),
        yours_first=added("yours_first"),
        strength=compare(
            _pooled_auc("yours_strength"), _pooled_auc("general_strength"), judged_at, n=resamples
        ),
        aesthetic=compare(
            _pooled_auc("yours_aesthetic"),
            _pooled_auc("general_aesthetic"),
            judged_at,
            n=resamples,
        ),
        first=tuple(
            compare(
                _first_share("yours_first", i),
                _first_share("general_first", i),
                judged_at,
                n=resamples,
            )
            for i in range(len(first_at))
        ),
        macro={name: macro_average(judged_at, _at_place(name)) for name in _TASTE_AUCS},
        forecasts=tuple(
            Forecast(j.yours_aesthetic, j.liked, not j.whole_range, location)
            for location in sorted(judged_at)
            for j in judged_at[location]
        ),
        copies_left_out=len(copies),
        same_place_left_out=len(grouped),
        same_places=_same_places(photos),
        quality_weight=quality_weight,
        general_with_quality=compare(
            _pooled_auc("general_strength"),
            _pooled_auc("general_aesthetic"),
            judged_at,
            n=resamples,
        ),
        yours_with_quality=compare(
            _pooled_auc("yours_strength"), _pooled_auc("yours_aesthetic"), judged_at, n=resamples
        ),
        candidate=compare(
            _pooled_auc("candidate_strength"),
            _pooled_auc("general_strength"),
            judged_at,
            n=resamples,
        )
        if photos and all(p.candidate_quality is not None for p in photos)
        else None,
        candidate_general=compare(
            _pooled_auc("candidate_general_strength"),
            _pooled_auc("general_strength"),
            judged_at,
            n=resamples,
        )
        if measured_general
        else None,
        candidate_general_alone=compare(
            _pooled_auc("candidate_general_aesthetic"),
            _pooled_auc("general_aesthetic"),
            judged_at,
            n=resamples,
        )
        if measured_general
        else None,
        fit=fit or "",
        fitted=None
        if fit is None
        else compare(
            _pooled_auc("fitted_strength"), _pooled_auc("today_strength"), judged_at, n=resamples
        ),
        uncentred=tuple(sorted({p.location for p in photos if p.place_mean is None}))
        if fit == "centred"
        else (),
        fitted_by_collection=()
        if fit is None
        else _by_collection(judged_at, collection_of, resamples),
        candidate_general_by_collection=_by_collection(
            judged_at,
            collection_of,
            resamples,
            measured="candidate_general_strength",
            today="general_strength",
        )
        if measured_general
        else (),
        general_blend=general_blend,
        offset_on_candidate=fit == "offset" and measured_general,
        top_candidate_general=compare(
            _top_share("candidate_general_strength"),
            _top_share("general_strength"),
            judged_at,
            n=resamples,
        )
        if measured_general
        else None,
        top_fitted=None
        if fit is None
        else compare(
            _top_share("fitted_strength"), _top_share("today_strength"), judged_at, n=resamples
        ),
        liked_rank_by_scale=tuple(
            (scale, float(np.mean(today)), float(np.mean(fitted)))
            for scale, (today, fitted) in sorted(liked_ranks.items(), key=_scale_order)
        ),
    )


def _blended(photos: Sequence[TastePhoto]) -> list[TastePhoto]:
    """Each photo with its candidate General replaced by the mean of its rank among its
    place's rated photos by today's General and by the candidate: two views, half each."""
    if not all(p.candidate_general is not None for p in photos):
        raise ValueError("Blending General with a candidate needs the candidate for every photo.")
    blended: dict[str, float] = {}
    for location in {p.location for p in photos}:
        here = [p for p in photos if p.location == location]
        today = percentile_rank(np.asarray([p.general for p in here], dtype=np.float64))
        other = percentile_rank(np.asarray([p.candidate_general for p in here], dtype=np.float64))
        for i, p in enumerate(here):
            blended[p.content_key] = float((today[i] + other[i]) / 2)
    return [dataclasses.replace(p, candidate_general=blended[p.content_key]) for p in photos]


def _top_share(name: str) -> Callable[[Sequence[Sequence[_Judged]]], float]:
    """Liked of the top ``TOP_SHARE`` of each place's photos (over the whole range, as the
    AUC reads them) by ``name``, every place together."""

    def share(places: Sequence[Sequence[_Judged]]) -> float:
        liked = shown = 0
        for photos in places:
            whole = [p for p in photos if p.whole_range]
            top = (
                sorted(whole, key=lambda p: -getattr(p, name))[
                    : max(1, math.ceil(TOP_SHARE * len(whole)))
                ]
                if whole
                else []
            )
            liked += sum(p.liked for p in top)
            shown += len(top)
        return liked / shown if shown else math.nan

    return share


def _add_liked_ranks(
    ranks: dict[str, tuple[list[float], list[float]]],
    here: Sequence[TastePhoto],
    scores: Mapping[str, list[float]],
) -> None:
    """Each liked photo's rank at its place by today's Yours and by the fit, by its scale."""
    today = percentile_rank(np.asarray(scores["today_strength"], dtype=np.float32))
    fitted = percentile_rank(np.asarray(scores["fitted_strength"], dtype=np.float32))
    for i, p in enumerate(here):
        if p.liked:
            before, after = ranks.setdefault(p.scale, ([], []))
            before.append(float(today[i]))
            after.append(float(fitted[i]))


_SCALES = ("aerial", "wide", "medium", "close", "macro")


def _scale_order(item: tuple[str, object]) -> tuple[int, str]:
    """Farthest to nearest, as the vocabulary lists them; any other name after, by name."""
    name = item[0]
    return (_SCALES.index(name) if name in _SCALES else len(_SCALES), name)


def _by_collection(
    judged_at: Mapping[str, Sequence[_Judged]],
    collection_of: Mapping[str, str],
    resamples: int,
    *,
    measured: str = "fitted_strength",
    today: str = "today_strength",
) -> tuple[tuple[str, Compared], ...]:
    """A candidate (by default another fit) against today's, over each collection's judged
    places alone; nothing when they all share one, since the overall difference says it."""
    names = sorted({collection_of[location] for location in judged_at})
    if len(names) < 2:
        return ()
    candidate, current = _pooled_auc(measured), _pooled_auc(today)
    return tuple(
        (
            name,
            compare(
                candidate,
                current,
                {loc: judged for loc, judged in judged_at.items() if collection_of[loc] == name},
                n=resamples,
            ),
        )
        for name in names
    )


DISAGREEMENT_GAP = 0.25
"""How far apart two tastes must rank a rated photo, as shares of its place's rated photos,
to count as disagreeing about it: a quarter of the place, so near-ties are left out."""

DISAGREEMENT_COUNT = 24
"""How many of each kind of disagreement are shown: a contact sheet's worth, enough to see
a pattern and few enough to look at each."""


@dataclass(frozen=True)
class Disagreement:
    """A rated photo two tastes rank far apart at its place, judged by a Yours that never
    learned from that place. ``general`` and ``yours``: its strength under each, as a rank
    among the place's rated photos (0 the weakest, 1 the strongest)."""

    content_key: str
    location: str
    liked: bool
    general: float
    yours: float


@dataclass(frozen=True)
class Disagreements:
    """The rated photos General and Yours rank furthest apart, by whom you sided with, widest
    first (audit plan S5). ``counts``: how many there are of each, in field order, beyond
    the ``DISAGREEMENT_COUNT`` or so shown."""

    general_missed: tuple[Disagreement, ...]
    """Liked, and Yours ranked it far above General: what General misses."""
    general_wrong: tuple[Disagreement, ...]
    """Not liked, and General ranked it far above Yours: General's picks that are not
    portfolio worthy."""
    yours_missed: tuple[Disagreement, ...]
    """Liked, and General ranked it far above Yours."""
    yours_wrong: tuple[Disagreement, ...]
    """Not liked, and Yours ranked it far above General."""
    counts: tuple[int, int, int, int]
    places: int
    rated: int
    one_sided: tuple[str, ...] = ()


def taste_disagreements(
    photos: Sequence[TastePhoto],
    *,
    feature: str,
    extractor: str = "",
    quality_weight: float,
    l2: float = TASTE_L2,
    untrained: Collection[str] = (),
    count: int = DISAGREEMENT_COUNT,
    pools: Mapping[str, Pool] | None = None,
) -> Disagreements:
    """Where General and Yours disagree most about photos you rated, and which one you sided
    with: to name what Yours sees that General misses, and what General picks that is not
    portfolio worthy. Each place is judged as ``measure_taste`` judges it, by a Yours fitted
    on every other rating, with strength mixed as Strongest mixes it (``quality_weight``),
    over the place's whole ``pool`` where one is given."""
    judging = _places_apart(
        photos, feature=feature, extractor=extractor, l2=l2, judged=None, untrained=untrained
    )
    found: list[Disagreement] = []
    for place in judging.places:
        here = place.here
        yours = taste_probability(place.weights, np.stack([p.vector for p in here]))
        general = np.asarray([p.general for p in here], dtype=np.float32)
        quality = percentile_rank(np.asarray([p.quality for p in here], dtype=np.float32))
        general_strength, yours_strength, _ = _place_strengths(
            here,
            pools.get(place.location) if pools else None,
            general=general,
            yours=yours,
            weights=place.weights,
            quality=quality,
            quality_weight=quality_weight,
        )
        general_rank = percentile_rank(np.asarray(general_strength, dtype=np.float32))
        yours_rank = percentile_rank(np.asarray(yours_strength, dtype=np.float32))
        found += [
            Disagreement(
                p.content_key, place.location, p.liked, float(general_rank[i]), float(yours_rank[i])
            )
            for i, p in enumerate(here)
        ]

    def widest(liked: bool, yours_higher: bool) -> list[Disagreement]:
        sign = 1.0 if yours_higher else -1.0
        kept = [
            d
            for d in found
            if d.liked == liked and sign * (d.yours - d.general) >= DISAGREEMENT_GAP
        ]
        return sorted(kept, key=lambda d: (-sign * (d.yours - d.general), d.content_key))

    groups = (widest(True, True), widest(False, False), widest(True, False), widest(False, True))
    return Disagreements(
        general_missed=tuple(groups[0][:count]),
        general_wrong=tuple(groups[1][:count]),
        yours_missed=tuple(groups[2][:count]),
        yours_wrong=tuple(groups[3][:count]),
        counts=(len(groups[0]), len(groups[1]), len(groups[2]), len(groups[3])),
        places=len(judging.places),
        rated=len(found),
        one_sided=tuple(judging.one_sided),
    )


def _today(
    train: Sequence[TastePhoto], here: Sequence[TastePhoto], feature: str, extractor: str
) -> NDArray[np.float32]:
    """Today's Yours as it is used (``TASTE_L2``), whatever ``l2`` the other fit is tried at:
    an offset needs a lighter ``l2``, and today's at that ``l2`` would be a weaker bar."""
    weights = fit_taste(
        np.stack([p.vector for p in train]),
        np.asarray([p.liked for p in train], dtype=np.int8),
        feature=feature,
        extractor=extractor,
    )
    return taste_probability(weights, np.stack([p.vector for p in here]))


def _fitted(
    fit: TasteFit,
    train: Sequence[TastePhoto],
    here: Sequence[TastePhoto],
    centred: Callable[[TastePhoto], NDArray[np.float32]] | None,
    *,
    feature: str,
    extractor: str,
    l2: float,
    on_candidate: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]:
    """Yours fitted ``fit``'s way on ``train``, as p(you like it) for each of ``here``, and
    for ``leaning`` how familiar each of ``here`` is (``models.taste.familiarity``)."""
    labels = np.asarray([p.liked for p in train], dtype=np.int8)
    if fit == "centred" and centred is not None:
        weights = fit_taste(
            np.stack([centred(p) for p in train]),
            labels,
            feature=feature,
            extractor=extractor,
            l2=l2,
        )
        return taste_probability(weights, np.stack([centred(p) for p in here])), None
    if fit == "top_pairs":
        pairs = _top_pairs(train)
        weights = fit_pairs(
            np.stack([p.vector for p in train]),
            pairs,
            feature=feature,
            extractor=extractor,
            l2=l2,
        )
        return taste_probability(weights, np.stack([p.vector for p in here])), None
    if fit == "embedding":
        weights = fit_taste(
            np.stack([_candidate_vector(p) for p in train]),
            labels,
            feature=feature,
            extractor=extractor,
            l2=l2,
        )
        return taste_probability(weights, np.stack([_candidate_vector(p) for p in here])), None
    if fit == "by_scale":
        scaled = fit_by_scale(
            np.stack([p.vector for p in train]),
            labels,
            [p.scale for p in train],
            feature=feature,
            extractor=extractor,
            l2=l2,
        )
        return scaled_probability(
            scaled, np.stack([p.vector for p in here]), [p.scale for p in here]
        ), None
    if fit == "leaning":
        rated, vectors = np.stack([p.vector for p in train]), np.stack([p.vector for p in here])
        weights = fit_taste(rated, labels, feature=feature, extractor=extractor, l2=l2)
        calibration = fit_general(np.asarray([p.general for p in train]), labels)
        familiar = familiarity(rated, [p.place for p in train], vectors)
        general = general_probability(calibration, np.asarray([p.general for p in here]))
        return (
            lean_on_general(taste_probability(weights, vectors), general, familiar),
            familiar,
        )

    def view(p: TastePhoto) -> float:
        """The General an offset reads: the candidate's where one is measured."""
        if on_candidate and p.candidate_general is not None:
            return p.candidate_general
        return p.general

    seen_train = np.asarray([view(p) for p in train], dtype=np.float64)
    seen_here = np.asarray([view(p) for p in here], dtype=np.float64)
    if on_candidate:  # a candidate on any scale (LAION's 1 to 10): its rank, as a probability
        ranks = np.asarray(
            percentile_rank(np.concatenate([seen_train, seen_here])), dtype=np.float64
        )
        seen_train, seen_here = ranks[: len(train)], ranks[len(train) :]
    weights = fit_taste(
        np.stack([p.vector for p in train]),
        labels,
        feature=feature,
        extractor=extractor,
        l2=l2,
        general=seen_train,
    )
    return taste_probability(weights, np.stack([p.vector for p in here]), general=seen_here), None


def _top_pairs(photos: Sequence[TastePhoto]) -> NDArray[np.intp]:
    """(liked, disliked) indices into ``photos``: both at one place and both chosen by a
    taste's ranking (``whole_range`` false)."""
    by_place: dict[str, tuple[list[int], list[int]]] = {}
    for i, p in enumerate(photos):
        if not p.whole_range:
            by_place.setdefault(p.place, ([], []))[0 if p.liked else 1].append(i)
    found = [(a, b) for liked, disliked in by_place.values() for a in liked for b in disliked]
    return np.asarray(found, dtype=np.intp).reshape(-1, 2)


def _candidate_vector(photo: TastePhoto) -> NDArray[np.float32]:
    if photo.candidate_vector is None:  # measure_taste checks every photo has one
        raise ValueError("Fitting Yours on a candidate embedding needs every rated photo's.")
    return photo.candidate_vector


def _centring(photos: Sequence[TastePhoto]) -> Callable[[TastePhoto], NDArray[np.float32]]:
    """Each photo's vector minus its place's mean; at a place with none, minus the mean of
    every rated photo, so it keeps its place's offset as it would uncentred."""
    overall = np.mean([p.vector for p in photos], axis=0) if photos else 0.0

    def centred(p: TastePhoto) -> NDArray[np.float32]:
        mean = overall if p.place_mean is None else p.place_mean
        return np.asarray(p.vector - mean, dtype=np.float32)

    return centred


def _same_places(photos: Sequence[TastePhoto]) -> tuple[tuple[str, ...], ...]:
    names: dict[str, set[str]] = {}
    for p in photos:
        names.setdefault(p.place, set()).add(p.location)
    return tuple(sorted(tuple(sorted(group)) for group in names.values() if len(group) > 1))


_TASTE_AUCS = ("general_aesthetic", "yours_aesthetic", "general_strength", "yours_strength")


def _place_auc(photos: Sequence[_Judged], name: str, *, chosen: bool = False) -> float:
    """The AUC over the whole range: photos a taste chose (drawn for their value, shown in
    Compare Strongest) are left out; with ``chosen``, it is over those alone."""
    here = [p for p in photos if p.whole_range != chosen]
    return auc([getattr(p, name) for p in here], [MATCH if p.liked else 0 for p in here])


def _pooled_auc(
    name: str, *, chosen: bool = False
) -> Callable[[Sequence[Sequence[_Judged]]], float]:
    """The AUC at each place, each counting by its like/dislike pairs, as the report pools."""

    def pooled(places: Sequence[Sequence[_Judged]]) -> float:
        total = weight = 0.0
        for photos in places:
            here = [p for p in photos if p.whole_range != chosen]
            liked = sum(p.liked for p in here)
            pairs = liked * (len(here) - liked)
            if pairs:
                total += _place_auc(here, name, chosen=chosen) * pairs
                weight += pairs
        return total / weight if weight else math.nan

    return pooled


def _file_scores(here: Sequence[TastePhoto], yours: Sequence[float]) -> dict[str, list[float]]:
    """A place's numbers from a file, and Yours plus them (``mean_z``)."""
    given = [float(p.candidate_score or 0.0) for p in here]
    return {"score": given, "blend": mean_z(yours, given)}


def _pooled(judged_at: Mapping[str, Sequence[_Judged]], name: str) -> float:
    return _pooled_auc(name)(list(judged_at.values()))


def _at_place(name: str) -> Callable[[Sequence[_Judged]], float]:
    return lambda photos: _place_auc(photos, name)


def _population(
    judged_at: Mapping[str, Sequence[_Judged]], *, chosen: bool
) -> dict[str, list[_Judged]]:
    """Each place's photos in one population (the whole range, or the photos a taste chose),
    where it holds both a like and a dislike: a place with only one has no pair to count, and
    is no unit of the range."""
    out = {}
    for location, photos in judged_at.items():
        part = [p for p in photos if p.whole_range != chosen]
        if 0 < sum(p.liked for p in part) < len(part):
            out[location] = part
    return out


def _chosen(judged_at: Mapping[str, Sequence[_Judged]], resamples: int) -> Compared | None:
    """Yours against General by strength among the photos a taste chose to show."""
    apart = _population(judged_at, chosen=True)
    if not apart:
        return None
    return compare(
        _pooled_auc("yours_strength", chosen=True),
        _pooled_auc("general_strength", chosen=True),
        apart,
        n=resamples,
    )


def _reading(
    judged_at: Mapping[str, Sequence[_Judged]], resamples: int, *, chosen: bool
) -> ScoreReading:
    apart = _population(judged_at, chosen=chosen)
    return ScoreReading(
        versus_general=compare(
            _pooled_auc("score", chosen=chosen),
            _pooled_auc("general_strength", chosen=chosen),
            apart,
            n=resamples,
        ),
        blend_versus_yours=compare(
            _pooled_auc("blend", chosen=chosen),
            _pooled_auc("yours_strength", chosen=chosen),
            apart,
            n=resamples,
        ),
    )


def _side(photos: Sequence[_Judged], *, chosen: bool) -> ScoreSide:
    here = [p for p in photos if p.whole_range != chosen]
    liked = sum(p.liked for p in here)
    return ScoreSide(
        liked=liked,
        disliked=len(here) - liked,
        scores=_place_auc(here, "score", chosen=chosen),
        general=_place_auc(here, "general_strength", chosen=chosen),
        yours=_place_auc(here, "yours_strength", chosen=chosen),
    )


def _scores_report(judged_at: Mapping[str, Sequence[_Judged]], resamples: int) -> ScoreReport:
    return ScoreReport(
        scored=sum(len(photos) for photos in judged_at.values()),
        whole=_reading(judged_at, resamples, chosen=False),
        chosen=_reading(judged_at, resamples, chosen=True),
        rows=tuple(
            ScoreRow(location, _side(photos, chosen=False), _side(photos, chosen=True))
            for location, photos in sorted(judged_at.items())
        ),
    )


def mean_z(*columns: Sequence[float]) -> list[float]:
    """The mean, photo by photo, of each column's distance from its own mean in standard
    deviations: two views of the same photos given equal say, whatever their scales. A column
    that is the same for every photo says nothing and counts as zero, taking its share."""
    z = []
    for column in columns:
        x = np.asarray(column, dtype=np.float64)
        z.append(np.zeros_like(x) if np.ptp(x) == 0 else (x - x.mean()) / x.std())
    return [float(v) for v in np.mean(z, axis=0)]


def _first_share(name: str, i: int) -> Callable[[Sequence[Sequence[_Judged]]], float]:
    """Liked of rated among a template's first picks, every place together."""

    def share(places: Sequence[Sequence[_Judged]]) -> float:
        shown = [p.liked for photos in places for p in photos if getattr(p, name)[i]]
        return sum(shown) / len(shown) if shown else math.nan

    return share


def _shown(
    first_picks: FirstPicksFn, location: str, taste: TasteWeights | None, at: Sequence[int]
) -> tuple[frozenset[str], ...]:
    """A template's first ``k`` at ``location``, for each of ``at``."""
    return tuple(frozenset(first_picks(location, taste, k)) for k in at)


def _first(shown: Sequence[frozenset[str]], verdicts: Mapping[str, bool]) -> tuple[FirstPicks, ...]:
    """The photos you rated at a place among a template's first ``k``, for each ``k``."""
    counted = []
    for keys in shown:
        found = [liked for key, liked in verdicts.items() if key in keys]
        counted.append(FirstPicks(rated=len(found), liked=sum(found)))
    return tuple(counted)


@dataclass(frozen=True)
class Pool:
    """Everything catalogued at a place: what Strongest ranks when you search there. Strength is
    scored over it, not among the rated photos alone: a rated photo sits near the top of its
    pool's aesthetic, and the percentile among the rated photos stretches those over 0 to 1,
    drowning the quality that orders them in a search (10 November 2026 audit, B-7: General's
    first-pick AUC read 0.667 and ships as 0.638)."""

    keys: tuple[str, ...]
    """The content keys, in the order of the arrays."""
    vectors: NDArray[np.float32]
    """The embedding Yours reads."""
    general: NDArray[np.float32]
    """General's aesthetic, p("strong")."""
    quality: NDArray[np.float32]
    """Technical quality as Strongest scores it: its sharpness ranked over this pool."""


def shipped_strength(
    quality: NDArray[np.float32], aesthetic: NDArray[np.float32], quality_weight: float
) -> NDArray[np.float32]:
    """Strength over a pool as Strongest mixes it (``selection.strength.photo_strength``):
    quality as scored, and the aesthetic's percentile over the pool, by ``quality_weight``."""
    mixed = quality_weight * np.clip(quality, 0, 1) + (1 - quality_weight) * percentile_rank(
        aesthetic
    )
    return mixed.astype(np.float32)


def _over_pool(
    here: Sequence[TastePhoto],
    pool: Pool | None,
    aesthetic: NDArray[np.float32],
    quality_weight: float,
) -> list[float] | None:
    """The strength of each photo of ``here`` over its ``pool``, the aesthetic scored by
    ``aesthetic`` for every photo in it; None where there is no pool, or it lacks one of them."""
    if pool is None:
        return None
    index = {key: i for i, key in enumerate(pool.keys)}
    if any(p.content_key not in index for p in here):
        return None
    at = [index[p.content_key] for p in here]
    return [float(v) for v in shipped_strength(pool.quality, aesthetic, quality_weight)[at]]


def _require_fit_inputs(photos: Sequence[TastePhoto], fit: TasteFit | None) -> None:
    """Refuse a way of fitting Yours whose inputs the photos lack, saying which."""
    if fit == "by_scale" and not all(p.scale for p in photos):
        raise ValueError("Fitting Yours by scale needs every rated photo's scale.")
    if fit == "top_pairs" and not len(_top_pairs(photos)):
        raise ValueError(
            "Fitting Yours on pairs among the first picks needs a liked and a disliked photo "
            "a taste chose to show at one place: rate a Compare Strongest session "
            "(calico eval new-compare)."
        )
    if fit == "embedding" and not all(p.candidate_vector is not None for p in photos):
        raise ValueError("Fitting Yours on a candidate embedding needs every rated photo's.")


def _place_strengths(
    here: Sequence[TastePhoto],
    pool: Pool | None,
    *,
    general: NDArray[np.float32],
    yours: NDArray[np.float32],
    weights: TasteWeights,
    quality: NDArray[np.float32],
    quality_weight: float,
) -> tuple[list[float], list[float], bool]:
    """General's and Yours's strength for each photo of ``here``: over ``pool`` where it holds
    every one of them (Yours from ``weights`` applied to the whole pool), else among the rated
    photos alone (``quality`` is their percentile). Whether the pool was used comes last."""
    if pool is not None:
        over_general = _over_pool(here, pool, pool.general, quality_weight)
        over_yours = _over_pool(
            here, pool, taste_probability(weights, pool.vectors), quality_weight
        )
        if over_general is not None and over_yours is not None:
            return over_general, over_yours, True
    return (
        _strength(quality, general, quality_weight),
        _strength(quality, yours, quality_weight),
        False,
    )


def _strength(
    quality: NDArray[np.float32], aesthetic: NDArray[np.float32], quality_weight: float
) -> list[float]:
    """Strength among a place's rated photos alone, from quality percentiles and an
    aesthetic: all a candidate, read for those photos, can be scored by."""
    mixed = quality_weight * quality + (1 - quality_weight) * percentile_rank(aesthetic)
    return [float(v) for v in mixed]


COMPARE_PICKS = 30
"""How many of each template's first picks a Compare Strongest session shows: a sample of
the shortlist (which may be any size), two overlapping 30s rated in a sitting."""

BAR_AT = 30
"""Where the bar is read: the liked share of each list's first 30 (roadmap, "Strongest's
bar"). A session that shows more is read at 60 as well (``PRECISION_AT``), never in the bar's
own figure: 12 November 2026, a session of 60 had counted twice as many photos as one of 30."""


@dataclass(frozen=True)
class PhotoToCompare:
    """A photo for a Compare Strongest session, and each way that showed it: (the way's
    index, the photo's rank among that way's picks)."""

    content_key: str
    shown_by: tuple[tuple[int, int], ...]


def photos_to_compare(shown: Sequence[Sequence[str]], seed: int) -> list[PhotoToCompare]:
    """Every photo any way ``shown`` (per way, its picks' content keys in order), once,
    credited to each way that showed it. Shuffled, so neither the order nor a run of one
    template's picks can say who chose a photo."""
    credited: dict[str, list[tuple[int, int]]] = {}
    for way, keys in enumerate(shown):
        for rank, key in enumerate(keys):
            credited.setdefault(key, []).append((way, rank))
    chosen = [PhotoToCompare(key, tuple(ways)) for key, ways in credited.items()]
    order = np.random.default_rng(seed).permutation(len(chosen))
    return [chosen[i] for i in order]


@dataclass(frozen=True)
class ComparedPhoto:
    """A photo you rated Like or Don't like in a Compare Strongest session, and the
    templates that showed it."""

    liked: bool
    shown_by: frozenset[str]
    ranks: tuple[tuple[str, int], ...] = ()
    """Where each template that showed it had it in its list (0 = first)."""
    marker: int | None = None
    """How many picks of Yours's list in its session come before the line it drew where the
    list stops being good (``Marker.before``), when it drew one."""
    bursts: int | None = None
    """How many bursts the pool its session searched held (the ``collapse bursts`` stage of
    Yours's run), when that run can be read: the size the bar's place rule is about."""

    def rank_in(self, template: str) -> int | None:
        return next((rank for shown, rank in self.ranks if shown == template), None)


@dataclass(frozen=True)
class CompareRow:
    """One session: of each template's picks, how many you rated and liked. Photos both
    chose count for both; ``general_only`` and ``yours_only`` are where they differ."""

    place: str
    rated: int
    general: FirstPicks
    yours: FirstPicks
    general_only: FirstPicks
    yours_only: FirstPicks
    both: FirstPicks
    bursts: int | None = None


@dataclass(frozen=True)
class DropRanks:
    """Where, in each session, the liked share of a list so far first fell below
    ``RUNNING_LINE`` (once ``RUNNING_AFTER`` photos were rated): the list position (1 = the
    first pick) of the photo that took it under, or None where it never did within the list.
    One per session with enough photos rated."""

    positions: tuple[int | None, ...]

    @property
    def fell(self) -> list[int]:
        return [p for p in self.positions if p is not None]

    @property
    def median(self) -> float:
        """Of the sessions where it fell; NaN where none did."""
        return float(np.median(self.fell)) if self.fell else math.nan


@dataclass(frozen=True)
class MarkerCheck:
    """Whether the line Yours drew where its list stops being good sits where the liked share
    falls: the picks above it (``above``) against those from it down (``below``), liked of
    rated, and the difference, paired by place. ``at`` is where each session drew it (the
    picks before it) and ``fell`` where that session's running share first fell below
    ``RUNNING_LINE`` (None: it did not, or too few were rated)."""

    above: FirstPicks
    below: FirstPicks
    difference: Compared
    at: tuple[int, ...]
    fell: tuple[int | None, ...]


@dataclass(frozen=True)
class CompareReport:
    """Every session, and every rated photo added up (each counts once). ``empty``
    sessions have no Like or Don't like yet."""

    rows: list[CompareRow]
    general: FirstPicks
    yours: FirstPicks
    general_only: FirstPicks
    yours_only: FirstPicks
    empty: list[str]
    in_sample: CompareReport | None = None
    """Sessions whose taste had learned from their place (or cannot say), measured apart."""
    cross_fitted: CompareReport | None = None
    """Sessions whose Yours was fitted without their place (``TasteSeen.cross_fit``), measured
    apart: the place is fresh to that taste, which is not the saved one, so it is neither
    counted with the sessions the saved taste had not learned nor judged in-sample."""
    small: CompareReport | None = None
    """Sessions whose pool held fewer bursts than ``min_bursts``, measured apart: a place
    that small has few photos worth narrowing, so its first 30 is much of the place."""
    shown: Compared | None = None
    """Liked of rated among each template's first ``BAR_AT`` picks with intervals by place:
    ``a`` is Yours, ``b`` General, the difference Yours minus General, paired."""
    only: Compared | None = None
    """The same, among the photos only one of them chose."""
    macro: Mapping[str, float] = field(default_factory=dict)
    """Each template's share averaged over places, each counting once."""
    curve: Mapping[str, tuple[FirstPicks, ...]] = field(default_factory=dict)
    """For each template whose picks carry their place in its list: liked of rated in each ten
    of it (``CURVE_STEP``), the first ten first. Where a list reads strongest first the share
    falls down it; a list in the order variety chose it (before 11 November 2026) is flat."""
    fall: Mapping[str, Interval] = field(default_factory=dict)
    """... and how far the liked share falls from the first ten to the last, by place."""
    precision: Mapping[int, Compared] = field(default_factory=dict)
    """The liked share of each list's first ``k`` (``PRECISION_AT``) with an interval by place:
    Yours (``a``) against General (``b``) and the difference, paired, for each ``k`` the
    lists reach (a list of 30 has no ``k`` of 60)."""
    drop: Mapping[str, DropRanks] = field(default_factory=dict)
    """For each template, where its liked share so far first fell below ``RUNNING_LINE``."""
    marker: MarkerCheck | None = None
    """Where Yours drew its line, checked against the ratings; None when no session's list
    drew one."""


CURVE_STEP = 10
"""Each list is read in tens: the liked share of its first ten picks, the next ten, and so on.
Where the owner stops scrolling is one number a search could record; until it does, this says
how fast the list stops being good (10 November 2026 audit, SG-8)."""

PRECISION_AT = (10, 20, 30, 60)
"""How far down a list the liked share is read as one number: the owner takes 5 to 25 of the
first 30 to 60 into the edit (a take rate from 8% to 83%), so a bar at 30 alone answers another
question than the one the list is used for (10 November 2026 audit, section 14, item 5)."""

RUNNING_LINE = 0.8
"""The liked share so far under which a list has stopped being good: what the marker's line
(``BestOptions.marker_probability``) is meant to say."""

RUNNING_AFTER = 10
"""Rated photos before the share so far is read: one disliked first pick is not a list that
stopped being good."""


def _in_the_bar(photo: ComparedPhoto) -> ComparedPhoto | None:
    """``photo`` as the bar reads it: credited only to the templates that had it among their
    first ``BAR_AT``, and None when none did. A photo that carries no place in a list (a
    session from before they were saved) is read as it was."""
    if not photo.ranks:
        return photo
    within = frozenset(template for template, rank in photo.ranks if rank < BAR_AT)
    return dataclasses.replace(photo, shown_by=within) if within else None


def _bursts_of(photos: Sequence[ComparedPhoto]) -> int | None:
    """How many bursts a session's pool held, as its photos carry it; None when none does."""
    return next((p.bursts for p in photos if p.bursts is not None), None)


def measure_compare(
    sessions: Sequence[tuple[str, Sequence[ComparedPhoto]]],
    *,
    resamples: int = RESAMPLES,
    min_bursts: int | None = None,
) -> CompareReport:
    """Whose first picks you liked more, session by session (named by their place).
    Intervals resample places: two sessions at one place are drawn together. The figures
    that make the bar (the table, the shares, the differences) read each list's first
    ``BAR_AT`` picks; the curve, the precision at 10 to 60, where the share falls and the
    marker read the whole list. With ``min_bursts``, sessions whose pool held fewer bursts
    are measured apart (``small``); one whose size is unknown stays in the report."""
    if min_bursts is not None:
        large: list[tuple[str, Sequence[ComparedPhoto]]] = []
        small: list[tuple[str, Sequence[ComparedPhoto]]] = []
        for session in sessions:
            held = _bursts_of(session[1])
            (small if held is not None and held < min_bursts else large).append(session)
        return dataclasses.replace(
            measure_compare(large, resamples=resamples),
            small=measure_compare(small, resamples=resamples) if small else None,
        )

    def counted(photos: Sequence[ComparedPhoto]) -> FirstPicks:
        return FirstPicks(rated=len(photos), liked=sum(p.liked for p in photos))

    def picks(photos: Sequence[ComparedPhoto], shown: set[str]) -> FirstPicks:
        return counted([p for p in photos if p.shown_by == shown])

    bar = [
        (place, [read for p in photos if (read := _in_the_bar(p)) is not None])
        for place, photos in sessions
    ]
    rows, empty = [], []
    for place, photos in bar:
        if not photos:
            empty.append(place)
            continue
        rows.append(
            CompareRow(
                place=place,
                rated=len(photos),
                general=counted([p for p in photos if "general" in p.shown_by]),
                yours=counted([p for p in photos if "yours" in p.shown_by]),
                general_only=picks(photos, {"general"}),
                yours_only=picks(photos, {"yours"}),
                both=picks(photos, {"general", "yours"}),
                bursts=_bursts_of(photos),
            )
        )

    def added(name: str) -> FirstPicks:
        return FirstPicks(
            rated=sum(getattr(r, name).rated for r in rows),
            liked=sum(getattr(r, name).liked for r in rows),
        )

    by_place: dict[str, list[ComparedPhoto]] = {}
    for place, whole in sessions:
        by_place.setdefault(place, []).extend(whole)
    by_bar: dict[str, list[ComparedPhoto]] = {}
    for place, first in bar:
        by_bar.setdefault(place, []).extend(first)
    curve: dict[str, tuple[FirstPicks, ...]] = {}
    fall: dict[str, Interval] = {}
    for template in ("general", "yours"):
        tens = [
            (rank // CURVE_STEP, p)
            for _, photos in sessions
            for p in photos
            if (rank := p.rank_in(template)) is not None
        ]
        if not tens:
            continue
        last = max(ten for ten, _ in tens)
        curve[template] = tuple(
            counted([p for ten, p in tens if ten == k]) for k in range(last + 1)
        )
        if last > 0 and curve[template][0].rated and curve[template][last].rated:
            fall[template] = compare(
                _liked_share(_in_ten(template, 0)),
                _liked_share(_in_ten(template, last)),
                by_place,
                n=resamples,
            ).difference
    return CompareReport(
        rows=rows,
        general=added("general"),
        yours=added("yours"),
        general_only=added("general_only"),
        yours_only=added("yours_only"),
        empty=empty,
        shown=compare(
            _liked_share(lambda p: "yours" in p.shown_by),
            _liked_share(lambda p: "general" in p.shown_by),
            by_bar,
            n=resamples,
        ),
        only=compare(
            _liked_share(lambda p: p.shown_by == {"yours"}),
            _liked_share(lambda p: p.shown_by == {"general"}),
            by_bar,
            n=resamples,
        ),
        macro={
            "general": macro_average(by_bar, lambda photos: _share_of(photos, "general")),
            "yours": macro_average(by_bar, lambda photos: _share_of(photos, "yours")),
        },
        curve=curve,
        fall=fall,
        precision=_precision(by_place, resamples),
        drop=_drops(sessions),
        marker=_marker_check(sessions, by_place, resamples),
    )


def _within(template: str, k: int) -> Callable[[ComparedPhoto], bool]:
    """Whether ``template`` had the photo among its first ``k``."""

    def chosen(photo: ComparedPhoto) -> bool:
        rank = photo.rank_in(template)
        return rank is not None and rank < k

    return chosen


def _precision(
    by_place: Mapping[str, Sequence[ComparedPhoto]], resamples: int
) -> dict[int, Compared]:
    """Yours against General in the liked share of each list's first ``k``, for each ``k`` of
    ``PRECISION_AT``, read at the places where both lists reach it (``_reaches``): a place
    whose lists stop at 30 adds nothing to the share at 60 (12 November 2026: it had, so the
    figure at 60 was the share of everything shown)."""
    found = {}
    for k in PRECISION_AT:
        reaching = {place: photos for place, photos in by_place.items() if _reaches(photos, k)}
        if reaching:
            found[k] = compare(
                _liked_share(_within("yours", k)),
                _liked_share(_within("general", k)),
                reaching,
                n=resamples,
            )
    return found


def _reaches(photos: Sequence[ComparedPhoto], k: int) -> bool:
    """Whether both lists of a place reach ``k``: each has a rated photo among the ``CURVE_STEP``
    ranks before it, so a list of 30 does not reach 60 however few it was asked for."""
    return all(
        any(
            (rank := p.rank_in(template)) is not None and k - CURVE_STEP <= rank < k for p in photos
        )
        for template in ("general", "yours")
    )


def _running_fall(photos: Sequence[ComparedPhoto], template: str) -> int | None:
    """The list position of the photo that first took the liked share so far under
    ``RUNNING_LINE``, counting from the ``RUNNING_AFTER``th rated photo; None if none did."""
    ranked = sorted(
        (p for p in photos if p.rank_in(template) is not None),
        key=lambda p: p.rank_in(template) or 0,
    )
    liked = 0
    for n, photo in enumerate(ranked, start=1):
        liked += photo.liked
        if n >= RUNNING_AFTER and liked / n < RUNNING_LINE:
            return (photo.rank_in(template) or 0) + 1
    return None


def _drops(sessions: Sequence[tuple[str, Sequence[ComparedPhoto]]]) -> dict[str, DropRanks]:
    """Where each template's running liked share first fell, session by session, for the
    sessions where enough of its list was rated."""
    drops = {}
    for template in ("general", "yours"):
        rated = [
            photos
            for _, photos in sessions
            if sum(p.rank_in(template) is not None for p in photos) >= RUNNING_AFTER
        ]
        if rated:
            drops[template] = DropRanks(tuple(_running_fall(photos, template) for photos in rated))
    return drops


def _marker_check(
    sessions: Sequence[tuple[str, Sequence[ComparedPhoto]]],
    by_place: Mapping[str, Sequence[ComparedPhoto]],
    resamples: int,
) -> MarkerCheck | None:
    """Whether the picks above Yours's line were liked more than those from it down."""

    def above(photo: ComparedPhoto) -> bool:
        rank = photo.rank_in("yours")
        return photo.marker is not None and rank is not None and rank < photo.marker

    def below(photo: ComparedPhoto) -> bool:
        rank = photo.rank_in("yours")
        return photo.marker is not None and rank is not None and rank >= photo.marker

    marked = [
        (next(p.marker for p in photos if p.marker is not None), photos)
        for _, photos in sessions
        if any(p.marker is not None for p in photos)
    ]
    if not marked:
        return None
    every = [p for _, photos in marked for p in photos]
    return MarkerCheck(
        above=FirstPicks(
            rated=sum(above(p) for p in every), liked=sum(above(p) and p.liked for p in every)
        ),
        below=FirstPicks(
            rated=sum(below(p) for p in every), liked=sum(below(p) and p.liked for p in every)
        ),
        difference=compare(_liked_share(above), _liked_share(below), by_place, n=resamples),
        at=tuple(at for at, _ in marked),
        fell=tuple(_running_fall(photos, "yours") for _, photos in marked),
    )


def _share_of(photos: Sequence[ComparedPhoto], template: str) -> float:
    shown = [p.liked for p in photos if template in p.shown_by]
    return sum(shown) / len(shown) if shown else math.nan


def _in_ten(template: str, ten: int) -> Callable[[ComparedPhoto], bool]:
    """Whether ``template`` had the photo in that ten of its list."""

    def chosen(photo: ComparedPhoto) -> bool:
        rank = photo.rank_in(template)
        return rank is not None and rank // CURVE_STEP == ten

    return chosen


def _liked_share(
    chosen: Callable[[ComparedPhoto], bool],
) -> Callable[[Sequence[Sequence[ComparedPhoto]]], float]:
    """Liked of rated among the ``chosen`` photos, every place together."""

    def share(places: Sequence[Sequence[ComparedPhoto]]) -> float:
        found = [p.liked for photos in places for p in photos if chosen(p)]
        return sum(found) / len(found) if found else math.nan

    return share


type StoryVerdict = Literal["liked", "rejected", "mixed", "unsure"]
type StoryArmKey = tuple[str, bool, str]
"""How a session told a story: the template, whether variety counted, and the way
(``labels.StoryArm.way``)."""

STORY_TOO_ALIKE = (0.7, 0.75, 0.8, 0.85, 0.9)
"""Where variety starts to fall, tried by ``calico eval stories``: liked stories' photos
averaged 0.79 by CLIP, rejected sets of look-alikes 0.85-0.93; the task's default is
0.70, from the stories you finished yourself (0.62)."""
STORY_VARIETY_WEIGHTS = (0.25, 0.5)


@dataclass(frozen=True)
class RatedStory:
    """One story you rated: photo by photo on a result (👍 fits, 👎 doesn't fit the
    story), or whole in a rating session (``stated``). ``vectors`` are its three photos'
    embeddings (unit rows), ``marks`` +1, -1 (doesn't fit) or 0 (unrated) for each;
    ``evidence`` is what the task saved with it, whose sum is the score it gave the story
    then (strength, each connection or role, variety, distinct), so it can be re-scored
    without re-running the search. ``run_id`` is what its stories are compared within: its
    run, or its whole session (one pool, told several ways), and ``arms`` are the ways a
    session told it."""

    run_id: str
    vectors: NDArray[np.float32]
    marks: tuple[int, ...]
    evidence: tuple[Evidence, ...]
    stated: StoryVerdict | None = None
    arms: tuple[StoryArmKey, ...] = ()

    @property
    def verdict(self) -> StoryVerdict:
        """As rated whole, if it was. Otherwise rejected when two or more photos did not
        fit: around an anchor, that is both companions. One that did not fit makes it
        mixed: the story was nearly right."""
        if self.stated is not None:
            return self.stated
        misfits = sum(m < 0 for m in self.marks)
        return "liked" if misfits == 0 else "mixed" if misfits == 1 else "rejected"

    @property
    def mean_similarity(self) -> float:
        sim = self.vectors @ self.vectors.T
        return float(np.mean(sim[np.triu_indices(len(sim), k=1)]))


@dataclass(frozen=True)
class VarietyRow:
    """Rated stories re-scored with variety counting ``weight`` of the connection side,
    falling from ``too_alike`` (``None``: no variety, everything else as the task saved it).
    ``auc``: how often a liked story outscores a rejected one from the same run or session."""

    too_alike: float | None
    weight: float
    auc: float
    interval: Interval | None = None
    """``auc`` with its interval, resampling runs (or sessions), then stories in them."""
    versus: Interval | None = None
    """This row's AUC minus no variety's, paired; None for no variety itself."""


@dataclass(frozen=True)
class ArmRow:
    """Session stories told one way (template, variety on or off, today's way, arranged or
    by roles), by how you rated them; a story told several ways counts for each."""

    template: str
    variety: bool
    way: str
    works: int
    unsure: int
    doesnt: int
    works_share: Interval | None = None
    """Works of works and doesn't (not sure is left out), resampling sessions."""
    versus_today: Interval | None = None
    """This way's share minus today's way's with variety (same template), paired."""


@dataclass(frozen=True)
class StoryReport:
    """``doesnt_fit`` photos were the only one in their story that did not fit;
    ``odd_one_out`` of them were the least alike of their three (a story with one
    stranger, rather than three twins). ``missing`` stories could not be measured: a
    photo's embedding is gone."""

    rows: list[VarietyRow]
    liked: int
    rejected: int
    mixed: int
    runs: int
    doesnt_fit: int
    odd_one_out: int
    missing: int = 0
    unsure: int = 0
    arms: list[ArmRow] = field(default_factory=list)


@dataclass(frozen=True)
class StoryReports:
    """Stories rated whole in sessions, and 👍/👎 given on story results (the older way):
    two kinds of answer, measured apart and never pooled (audit A05-10)."""

    sessions: StoryReport
    results: StoryReport


def measure_stories(
    stories: Sequence[RatedStory],
    variants: Sequence[tuple[float, float]],
    *,
    near_duplicate: float,
    resamples: int = RESAMPLES,
) -> StoryReport:
    """Would variety (``variants``: too alike, weight) rank the stories you liked above
    the ones you rejected? No variety first: each story as the task saved it (strength, its
    connections or roles, distinct) with variety taken out, the same for every story whether
    or not it was told with variety (10 November 2026 audit, B-2: not a score re-added without
    ``distinct``). Scores are compared within a run or session (the connections are percentiles
    of its pool) and each counts by its pairs.

    These are stories already shown, so this measures their order, not which new stories
    variety would bring in; a session's stories told without variety stand in for those."""
    tried = [(None, 0.0), *variants]
    by_run: dict[str, list[Scored]] = {}
    for run in sorted({s.run_id for s in stories}):
        by_run[run] = [
            Scored(
                liked=s.verdict == "liked",
                scores={
                    str(i): _story_score(s, too_alike, weight, near_duplicate)
                    for i, (too_alike, weight) in enumerate(tried)
                },
            )
            for s in stories
            if s.run_id == run and s.verdict in {"liked", "rejected"}
        ]
    measured = measure_signals(by_run, baseline="0", resamples=resamples, unit=STORY_UNIT)
    rows = [
        VarietyRow(
            too_alike=too_alike,
            weight=weight,
            auc=_pooled_signal(str(i))(list(by_run.values())),
            interval=measured.auc.get(str(i)),
            versus=measured.versus.get(str(i)),
        )
        for i, (too_alike, weight) in enumerate(tried)
    ]

    misfits = [(s, s.marks.index(-1)) for s in stories if sum(m < 0 for m in s.marks) == 1]
    counts = Counter((arm, s.verdict) for s in stories for arm in s.arms)
    arms = sorted({arm for s in stories for arm in s.arms})
    told: dict[str, list[tuple[StoryArmKey, bool]]] = {}
    for s in stories:
        if s.verdict in {"liked", "rejected"}:
            told.setdefault(s.run_id, []).extend((arm, s.verdict == "liked") for arm in s.arms)
    shares = {arm: _arm_shares(arm, told, arms, resamples) for arm in arms}
    return StoryReport(
        rows=rows,
        liked=sum(s.verdict == "liked" for s in stories),
        rejected=sum(s.verdict == "rejected" for s in stories),
        mixed=sum(s.verdict == "mixed" for s in stories),
        runs=len({s.run_id for s in stories}),
        doesnt_fit=len(misfits),
        odd_one_out=sum(_least_alike(s) == i for s, i in misfits),
        unsure=sum(s.verdict == "unsure" for s in stories),
        arms=[
            ArmRow(
                template=arm[0],
                variety=arm[1],
                way=arm[2],
                works=counts[arm, "liked"],
                unsure=counts[arm, "unsure"],
                doesnt=counts[arm, "rejected"],
                works_share=shares[arm][0],
                versus_today=shares[arm][1],
            )
            for arm in arms
        ],
    )


STORY_UNIT = "run or session"
"""What story intervals resample: stories are compared within their run or session."""


def _arm_shares(
    arm: StoryArmKey,
    told: Mapping[str, Sequence[tuple[StoryArmKey, bool]]],
    arms: Sequence[StoryArmKey],
    resamples: int,
) -> tuple[Interval, Interval | None]:
    """The share of ``arm``'s stories that work, and its difference from today's way with
    variety, under the same template.

    Only the sessions that told ``arm`` count: a way rated at a few places is compared
    with today's at those places, not with today's everywhere it was ever rated."""

    def share(way: StoryArmKey) -> Callable[[Sequence[Sequence[tuple[StoryArmKey, bool]]]], float]:
        def works(sessions: Sequence[Sequence[tuple[StoryArmKey, bool]]]) -> float:
            found = [ok for rated in sessions for told_by, ok in rated if told_by == way]
            return sum(found) / len(found) if found else math.nan

        return works

    sessions = {run: rated for run, rated in told.items() if any(a == arm for a, _ in rated)}
    today = (arm[0], True, "today")
    if arm == today or today not in arms:
        return place_bootstrap(sessions, share(arm), n=resamples, unit="session"), None
    measured = compare(share(arm), share(today), sessions, n=resamples, unit="session")
    return measured.a, measured.difference


STORY_SESSION_STORIES = 6
"""Stories told each way in a rating session: four ways (two templates, today's way and
the variant under test) make up to 24 stories, 72 photos, about as many as a taste
session's 60."""

CANDIDATE_SESSION_STORIES = 12
"""Stories told each way in a rating session that sets a named candidate beside today's
way under one template: two ways make up to 24 stories, as four ways of
``STORY_SESSION_STORIES`` do."""

STORY_VARIANT: tuple[StoryWay, Mapping[str, object]] = (
    "likely_anchors",
    {"anchor_by": "likely"},
)
"""The way a rating session tells stories beside today's, and the story options it takes:
one change at a time, rated against today's at the same places (``calico eval stories``).
To test another way, name it in ``labels.StoryWay`` and set it here.

From 4 November 2026, anchors tried by the anchor prior (``TriptychOptions.anchor_by``):
it found the anchors you rated at 0.81 against strength's 0.68. Before it, 1,000
candidates rather than 500 (``deeper``, rated in seven sessions at large places on 3
November, since below about 900 photos it tells the same stories as 500): the finished
stories it could tell rose from 79% to 86% (General), at 9.3 s a search against 9.1 s, and
its stories worked 19 of 42 (General) and 20 of 42 (Yours), +0.02 [-0.21, 0.27] and +0.10
[-0.14, 0.36] against today's in the same sessions. The default since 5 November."""

ROLE_STORIES: Mapping[str, object] = {"tell_by": "roles"}
"""Stories told by roles (``TriptychOptions.tell_by``), rated against today's in sessions
from 28 October to 2 November 2026: 36 of 107 worked, against 48 of 105 today's way, in
the nine sessions told with ``distinct``. Still how ``learned_stories`` reads every
signal. Your finished stories beat a one-photo swap more often told by roles (General 0.73
to 0.75, Yours 0.70 to 0.72). Before it, sessions told stories *arranged*,
anchors taking turns by kind of shot (wide exteriors, wide interiors, portraits): 17 of
60 worked, against 22 of 60 today's way, at five places (27 October)."""


@dataclass(frozen=True)
class StoryToRate:
    """A story to rate: ``told`` is (way, story) of its first telling, ``arms`` every way
    that told it."""

    told: tuple[int, int]
    arms: tuple[StoryArmKey, ...]


def stories_to_rate(
    told: Sequence[tuple[StoryArmKey, Sequence[tuple[str, ...]]]], seed: int
) -> list[StoryToRate]:
    """Each story ``told`` (per way, its stories' content keys), once, shuffled.

    The same three photos told two ways (in any order) are one story to rate, credited to
    both. Shuffled, so neither the order nor a run of look-alike sets can say which way
    told a story."""
    first: dict[frozenset[str], tuple[int, int]] = {}
    arms: dict[frozenset[str], list[StoryArmKey]] = {}
    for w, (arm, stories) in enumerate(told):
        for i, keys in enumerate(stories):
            story = frozenset(keys)
            first.setdefault(story, (w, i))
            if arm not in arms.setdefault(story, []):
                arms[story].append(arm)
    chosen = [StoryToRate(told=first[s], arms=tuple(arms[s])) for s in first]
    order = np.random.default_rng(seed).permutation(len(chosen))
    return [chosen[i] for i in order]


_OUTSIDE_THE_SIDE = frozenset({"strength", "distinct"})
"""A story's evidence that is not its side: everything else (each connection or role, and
variety) shares what strength and distinct leave."""


def _story_score(
    story: RatedStory, too_alike: float | None, weight: float, near_duplicate: float
) -> float:
    """The story's score as the task adds it up, with variety counting ``weight`` of its side,
    falling from ``too_alike`` (``None``: no variety). The side is what strength and distinct
    leave (``Evidence.weight`` says how much): its connections or roles, then variety. Only
    it changes; strength and distinct stay as they were saved, so counting variety the way a
    story was told gives back the score it recorded, and the way any other options would
    (``TriptychTask.scorer``, checked in the tests)."""
    kept = [e for e in story.evidence if e.signal in _OUTSIDE_THE_SIDE]
    side = [e for e in story.evidence if e.signal not in _OUTSIDE_THE_SIDE]
    connected = [e for e in side if e.signal != "variety"]
    weights = sum(e.weight for e in connected)
    connection = sum(e.contribution for e in connected) / weights if weights else 0.0
    varied = (
        0.0
        if too_alike is None
        else float(
            variety(story.mean_similarity, too_alike=too_alike, near_duplicate=near_duplicate)
        )
    )
    w = 0.0 if too_alike is None else weight
    share = sum(e.weight for e in side)
    return sum(e.contribution for e in kept) + share * ((1 - w) * connection + w * varied)


def _least_alike(story: RatedStory) -> int:
    """The photo least like the other two (ties to the first)."""
    sim = story.vectors @ story.vectors.T
    np.fill_diagonal(sim, 0.0)
    return int(np.argmin(sim.sum(axis=1)))


def swap_wins(score: float, swaps: Sequence[float]) -> float:
    """How often a story outscores the stories made by swapping one of its photos (ties
    count half); NaN when nothing could be swapped in."""
    if not swaps:
        return math.nan
    beaten = sum(score > s for s in swaps) + 0.5 * sum(score == s for s in swaps)
    return beaten / len(swaps)


@dataclass(frozen=True)
class FinishedStory:
    """A story you finished, scored by the story task. ``swaps`` holds, for the left, middle
    and right photo, the scores of the stories made by putting each of the task's candidates
    in its place; ``shortlisted`` says which of the three photos are candidates on their own
    strength (the task never tells a story from a photo that is not), and ``anchor_tried``
    whether its middle photo is among the anchors the task tries: the swaps cannot judge
    anchors, this can say whether yours are reached."""

    name: str
    score: float
    swaps: tuple[tuple[float, ...], ...]
    shortlisted: tuple[bool, ...]
    anchor_tried: bool = False

    @property
    def wins(self) -> tuple[float, ...]:
        return tuple(swap_wins(self.score, side) for side in self.swaps)

    @property
    def win(self) -> float:
        """Mean over the sides that had anything to swap in."""
        found = [w for w in self.wins if not math.isnan(w)]
        return float(np.mean(found)) if found else math.nan


@dataclass(frozen=True)
class FinishedReport:
    """Finished stories under one template and one set of story options (``variant``).
    ``missing`` stories have a photo the catalog no longer has."""

    template: str
    variant: str
    stories: list[FinishedStory]
    missing: list[str]
    win_interval: Interval | None = None
    """``mean_win`` with its interval, resampling the stories (``finished_intervals``)."""
    versus: Interval | None = None
    """``mean_win`` minus the first report's, story by story, paired."""

    @property
    def mean_win(self) -> float:
        found = [s.win for s in self.stories if not math.isnan(s.win)]
        return float(np.mean(found)) if found else math.nan

    @property
    def shortlisted(self) -> float:
        """The share of the stories' photos that are candidates on their own strength."""
        flags = [f for s in self.stories for f in s.shortlisted]
        return sum(flags) / len(flags) if flags else math.nan

    @property
    def anchors_tried(self) -> float:
        """The share of stories whose middle photo the task would try as an anchor."""
        if not self.stories:
            return math.nan
        return sum(s.anchor_tried for s in self.stories) / len(self.stories)

    @property
    def all_three_shortlisted(self) -> float:
        """The share of stories the task could tell at all: all three photos candidates."""
        if not self.stories:
            return math.nan
        return sum(all(s.shortlisted) for s in self.stories) / len(self.stories)


def finished_intervals(
    reports: Sequence[FinishedReport], *, resamples: int = RESAMPLES
) -> list[FinishedReport]:
    """``reports`` with their intervals. Each finished story is drawn as a whole (it is
    one answer); every report after the first is compared with the first, story by story."""
    wins: dict[str, dict[int, float]] = {}
    for i, report in enumerate(reports):
        for story in report.stories:
            wins.setdefault(story.name, {})[i] = story.win
    by_story = {name: [found] for name, found in wins.items()}  # one item: drawn whole

    def mean_win(i: int) -> Callable[[Sequence[Sequence[Mapping[int, float]]]], float]:
        def mean(stories: Sequence[Sequence[Mapping[int, float]]]) -> float:
            found = [w for (won,) in stories if not math.isnan(w := won.get(i, math.nan))]
            return float(np.mean(found)) if found else math.nan

        return mean

    measured = []
    for i, report in enumerate(reports):
        if i == 0:
            interval = place_bootstrap(by_story, mean_win(0), n=resamples, unit="story")
            measured.append(dataclasses.replace(report, win_interval=interval))
            continue
        both = compare(mean_win(i), mean_win(0), by_story, n=resamples, unit="story")
        measured.append(dataclasses.replace(report, win_interval=both.a, versus=both.difference))
    return measured


STORY_L2 = 1.0
"""How strongly a learned story scorer's weights are pulled toward zero (on standardised
signals). Set before measuring: a dozen signals against a few hundred answers need far
less holding back than Yours's hundreds of dimensions (``TASTE_L2``)."""

TODAY, LEARNED, LEARNED_ALONE = "today", "learned", "learned, sessions only"
"""What ``learn_stories`` judges each session's stories by: today's way, the scorer learned
from every other place's answers and finished stories, and from the answers alone."""
LEARNED_SWAPS = "learned, with swaps"
"""The scorer learned from them and from swapped stories too (``SignalledStory.swapped``):
an opt-in variant (owner's Q4, 30 October 2026), judged on your answers alone and adopted
only if its range lies above today's."""

STORY_SWAPS_PER_SIDE = 1
"""Swapped stories made from each finished story, at each of its three places: one, so a
finished story is outweighed three to one by what it was chosen over. Set before
measuring."""


@dataclass(frozen=True)
class SignalledStory:
    """A story you judged, with every signal the story task reads of it
    (``StoryScorer.signals``). ``session`` is the story session it was rated in, or None for
    a story you finished yourself: learned from as one that works, never judged. ``places``
    are where its pool came from. ``today`` is its score told today's way."""

    session: str | None
    places: frozenset[str]
    signals: Mapping[str, float]
    works: bool
    today: float = math.nan
    swapped: bool = False
    """Made by swapping one photo of a finished story for another candidate at its places:
    learned from as a story that doesn't work, by ``LEARNED_SWAPS`` alone, never judged."""
    variants: Mapping[str, float] = field(default_factory=dict)
    """Its score told today's way with other options (by their label), each judged
    against ``today``."""


@dataclass(frozen=True)
class LearnedSession:
    """One story session judged: how often a story that works outscores one that doesn't,
    by each way of scoring (absent when there was nothing of both kinds to learn from)."""

    session: str
    name: str
    works: int
    doesnt: int
    auc: Mapping[str, float]


@dataclass(frozen=True)
class LearnedStories:
    """Stories scored by weights learned from your answers, against today's way.
    ``weights`` are fitted on every story (standardised signals: each says how much a
    spread of that signal counts); ``left_out`` signals were not measured for every story."""

    signals: tuple[str, ...]
    weights: Mapping[str, float]
    sessions: list[LearnedSession]
    auc: Mapping[str, Interval]
    versus: Mapping[str, Interval]
    finished: int
    left_out: tuple[str, ...]
    l2: float
    variants: tuple[str, ...] = ()
    """Today's way with other options, judged beside it (``SignalledStory.variants``)."""
    swaps: int = 0
    """Swapped stories learned from by ``LEARNED_SWAPS`` (none unless asked for)."""
    missing: int = 0
    """Stories that could not be measured: a photo gone, or without features."""
    unsure: int = 0
    """Stories rated not sure, left out."""
    notes: tuple[str, ...] = ()
    """What could not be read (a role without its model), as the task noted it."""


def learned_apart(stories: Sequence[SignalledStory], session: str) -> list[int]:
    """Which ``stories`` may be learned from to judge ``session``: none from it, and none
    whose pool shares a place with it (another session there, or a story you finished
    there), so no answer about its photos is learned from."""
    places = frozenset().union(*(s.places for s in stories if s.session == session))
    return [i for i, s in enumerate(stories) if s.session != session and not s.places & places]


def learn_stories(
    stories: Sequence[SignalledStory],
    *,
    names: Mapping[str, str] | None = None,
    l2: float = STORY_L2,
    resamples: int = RESAMPLES,
) -> LearnedStories:
    """Would a regularised logistic over the story signals, learned from your answers, rank
    the stories that work above those that don't better than today's way? Each session is
    judged by a scorer learned apart from its places (``learned_apart``), its stories
    compared within it, and the difference from today paired by session (audit A10-14).
    ``names`` are the sessions' names to show."""
    seen = [list(s.signals) for s in stories]
    common = [
        k
        for k in dict.fromkeys(k for keys in seen for k in keys)
        if all(k in keys for keys in seen)
    ]
    left_out = tuple(k for k in dict.fromkeys(k for keys in seen for k in keys) if k not in common)
    x = np.asarray([[s.signals[k] for k in common] for s in stories], dtype=np.float64)
    y = np.asarray([s.works for s in stories], dtype=np.int8)
    swapped = np.asarray([s.swapped for s in stories], dtype=bool)
    finished = np.asarray([s.session is None for s in stories], dtype=bool) & ~swapped

    def fitted(rows: Sequence[int]) -> TasteWeights | None:
        if len({int(y[i]) for i in rows}) < 2:
            return None
        return fit_taste(x[list(rows)], y[list(rows)], feature="story signals", l2=l2)

    by_session: dict[str, list[Scored]] = {}
    for session in sorted({s.session for s in stories if s.session is not None}):
        judged = [i for i, s in enumerate(stories) if s.session == session]
        apart = learned_apart(stories, session)
        answers = [i for i in apart if not swapped[i]]
        scorers = {
            LEARNED: fitted(answers),
            LEARNED_ALONE: fitted([i for i in answers if not finished[i]]),
        }
        if swapped.any():
            scorers[LEARNED_SWAPS] = fitted(apart)
        scores: dict[str, NDArray[np.float32]] = {
            name: taste_probability(w, x[judged]) for name, w in scorers.items() if w is not None
        }
        by_session[session] = [
            Scored(
                liked=bool(y[i]),
                scores={TODAY: stories[i].today, **stories[i].variants}
                | {k: float(v[n]) for k, v in scores.items()},
            )
            for n, i in enumerate(judged)
        ]
    measured = measure_signals(by_session, baseline=TODAY, resamples=resamples, unit="session")
    everything = fitted(np.flatnonzero(~swapped).tolist())
    return LearnedStories(
        signals=tuple(common),
        weights={}
        if everything is None
        else dict(zip(common, map(float, everything.coef), strict=True)),
        sessions=[
            LearnedSession(
                session=session,
                name=(names or {}).get(session, session),
                works=sum(s.liked for s in scored),
                doesnt=sum(not s.liked for s in scored),
                auc={k: _pooled_signal(k)([scored]) for k in scored[0].scores},
            )
            for session, scored in by_session.items()
        ],
        auc=measured.auc,
        versus=measured.versus,
        finished=int(finished.sum()),
        left_out=left_out,
        l2=l2,
        variants=tuple(dict.fromkeys(k for s in stories for k in s.variants)),
        swaps=int(swapped.sum()),
    )
