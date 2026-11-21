"""``taste``: your taste, learned from your ratings, for the Yours template.

A regularised logistic regression on one image embedding Calico already stores for
every photo, fitted to Like / Don't like ratings ("Not sure" is left out). It answers
with the same labels as the zero-shot ``aesthetic`` taxonomy ("weak", "strong"), so
``selection.strength`` ranks it exactly as it ranks the general aesthetic, and choosing
the Yours template only changes which model answers to "aesthetic".

Pure: arrays in, weights or probabilities out. Saving and loading the weights is
``calico.trained``'s job.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from calico.models.calibration import Calibration
    from calico.selection.context import FeatureView

TASTE_L2 = 1000.0
"""How strongly the weights are pulled toward zero (on standardised features): a few
hundred ratings against hundreds of dimensions would otherwise be learned by heart.
Measured with ``calico eval taste --l2`` on 853 ratings over 14 places (Oct 2026): on
CLIP, strength AUC rose from 0.75 at l2 1 to 0.83 at 300 to 3000 and fell past that;
1000 is the middle of that plateau."""


@dataclass(frozen=True)
class TasteWeights:
    """What a fit learned, and from what. ``extractor`` is the ``FeatureView.features``
    entry it reads ("" = the main embedding); ``feature`` its exact version, so weights
    are never applied to another model's vectors."""

    feature: str
    extractor: str
    mean: NDArray[np.float32]
    scale: NDArray[np.float32]
    coef: NDArray[np.float32]
    bias: float
    rated: int
    liked: int
    places: tuple[str, ...] = ()
    """The places it learned from: a Compare Strongest session at one of them is in-sample.
    Empty for a taste trained before 29 October 2026, which did not record them. Not part of
    ``version``: it changes nothing a search scores."""
    general_coef: float = 0.0
    """How much General's view counts, with General as an offset (``fit_taste(general=)``);
    0 for a taste that does not read it."""
    general_mean: float = 0.0
    general_scale: float = 1.0
    """How General's logit was standardised for the fit."""
    calibration: Calibration | None = None
    """How often you liked what this taste gave each probability, at places it never learned
    (``models.calibration``, read when it was trained). None for a taste trained before 11
    November 2026, or without enough ratings to say. Not part of ``version``: it changes
    nothing a search scores."""

    @property
    def version(self) -> str:
        digest = hashlib.blake2b(digest_size=6)
        digest.update(self.feature.encode())
        for array in (self.mean, self.scale, self.coef):
            digest.update(np.ascontiguousarray(array, dtype=np.float32).tobytes())
        digest.update(np.float32(self.bias).tobytes())
        if self.general_coef:  # a taste without General keeps the version it always had
            for value in (self.general_coef, self.general_mean, self.general_scale):
                digest.update(np.float32(value).tobytes())
        return digest.hexdigest()


def fit_taste(
    x: NDArray[np.floating],
    y: NDArray[np.integer],
    *,
    feature: str,
    extractor: str = "",
    l2: float = TASTE_L2,
    general: NDArray[np.floating] | None = None,
) -> TasteWeights:
    """Weights that tell rows of ``x`` you liked (``y`` 1) from those you did not (0).

    Each class counts as much as the other in total: with few dislikes, "like
    everything" would otherwise look nearly right.

    With ``general`` (General's p("strong") per row), General is an offset: its logit is
    one more input that is not held back, so the fit leans on General as far as it tells
    your likes and the embedding, held back by ``l2``, learns what General misses. Its
    weight is learned rather than fixed at 1, since General's probabilities come from a
    zero-shot softmax and are not on the scale of your likes. Here ``l2`` sets the mix as
    well: alone, shrinking the weights changes no order, but beside an offset that is not
    held back, a heavy ``l2`` leaves the embedding too little say."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(x).all() or (general is not None and not np.isfinite(general).all()):
        # One NaN gave all-zero weights: a constant saved as if it were a taste (A04-10).
        raise ValueError("Learning a taste needs features that are all finite numbers.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Learning a taste needs labels of 1 for a like and 0 for a dislike.")
    liked = int(y.sum())
    if liked == 0 or liked == len(y):
        raise ValueError("Learning a taste needs both likes and dislikes.")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - mean) / scale
    g_mean, g_scale, g = 0.0, 1.0, np.zeros(len(y))
    if general is not None:
        g_raw = _logit(general)
        g_mean, g_scale = float(g_raw.mean()), float(g_raw.std())
        g_scale = g_scale if g_scale > 1e-8 else 1.0
        g = (g_raw - g_mean) / g_scale
    weight = np.where(y == 1, len(y) / (2 * liked), len(y) / (2 * (len(y) - liked)))

    def loss(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        coef, general_coef, bias = params[:-2], params[-2], params[-1]
        logits = z @ coef + general_coef * g + bias
        p = expit(logits)
        value = np.sum(weight * (np.logaddexp(0.0, logits) - y * logits))
        value += 0.5 * l2 * float(coef @ coef)
        residual = weight * (p - y)
        grad = np.concatenate([z.T @ residual + l2 * coef, [residual @ g, residual.sum()]])
        return float(value), grad

    start = np.zeros(z.shape[1] + 2)
    found = minimize(loss, start, jac=True, method="L-BFGS-B")
    params = found.x
    if not (found.success or np.isfinite(params).all()):
        raise ValueError(f"Learning a taste did not converge: {found.message}")
    return TasteWeights(
        feature=feature,
        extractor=extractor,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        coef=params[:-2].astype(np.float32),
        bias=float(params[-1]),
        rated=len(y),
        liked=liked,
        general_coef=float(params[-2]) if general is not None else 0.0,
        general_mean=g_mean,
        general_scale=g_scale,
    )


def fit_pairs(
    x: NDArray[np.floating],
    pairs: NDArray[np.integer],
    *,
    feature: str,
    extractor: str = "",
    l2: float = TASTE_L2,
) -> TasteWeights:
    """Weights that rank the liked row of each pair above its disliked one: ``pairs`` holds
    (liked, disliked) row indices into ``x``, both from one place. A pairwise logistic fit
    on their differences, with no bias, since only order within a pair is learned.

    Within a place both rows share its level, so it cancels: the fit learns what tells a
    like from a dislike there, which is what Strongest ranks, and cannot learn that you
    like more at some places than others. In a search across several places that matters:
    each place's photos are ranked as if every place were liked alike.

    Each pair counts the same (as the pooled AUC counts places by their pairs), and the
    pairs together weigh as many as the photos they come from, so ``l2`` holds the weights
    back as hard as it does in ``fit_taste``, however many pairs a photo is in."""
    x = np.asarray(x, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
    if not len(pairs):
        raise ValueError("Learning from pairs needs at least one pair of a like and a dislike.")
    rows = np.unique(pairs)
    if not np.isfinite(x[rows]).all():
        raise ValueError("Learning a taste needs features that are all finite numbers.")
    mean, scale = x[rows].mean(axis=0), x[rows].std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - mean) / scale
    d = z[pairs[:, 0]] - z[pairs[:, 1]]
    weight = len(rows) / len(pairs)

    def loss(coef: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        s = d @ coef
        value = weight * float(np.logaddexp(0.0, -s).sum()) + 0.5 * l2 * float(coef @ coef)
        grad = -weight * (d.T @ expit(-s)) + l2 * coef
        return value, grad

    found = minimize(loss, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B")
    if not (found.success or np.isfinite(found.x).all()):
        raise ValueError(f"Learning a taste did not converge: {found.message}")
    return TasteWeights(
        feature=feature,
        extractor=extractor,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        coef=found.x.astype(np.float32),
        bias=0.0,
        rated=len(rows),
        liked=len(np.unique(pairs[:, 0])),
    )


def _logit(p: NDArray[np.floating]) -> NDArray[np.float64]:
    clipped = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.asarray(np.log(clipped / (1 - clipped)))


class TasteModel:
    """Your taste as p("strong"), from the embedding the weights were fitted on."""

    name = "taste"
    description = "Your taste, learned from your ratings"
    labels: tuple[str, ...] = ("weak", "strong")

    def __init__(self, weights: TasteWeights) -> None:
        if weights.general_coef:
            raise ValueError(
                "This taste reads General's view as an offset, which a search cannot give it "
                "yet: it is for calico eval taste --fit offset alone."
            )
        self.weights = weights
        self.version = f"taste@{weights.version}"

    def predict_proba(self, view: FeatureView) -> NDArray[np.float32]:
        w = self.weights
        if w.extractor:
            if w.extractor not in view.features:
                raise KeyError(f"The taste model reads {w.extractor!r}, which these photos lack.")
            x = view.features[w.extractor]
        else:
            x = view.embeddings
        strong = taste_probability(w, x)
        return np.stack([1.0 - strong, strong], axis=1).astype(np.float32)


def taste_probability(
    weights: TasteWeights,
    x: NDArray[np.floating],
    *,
    general: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """p(you like it) for each row of ``x`` (vectors of ``weights.feature``), with
    ``general`` (General's p("strong") per row) for a taste fitted with General as an
    offset."""
    logits = ((x - weights.mean) / weights.scale) @ weights.coef + weights.bias
    if weights.general_coef:
        if general is None:
            raise ValueError("This taste reads General's view as an offset: pass general.")
        g = (_logit(general) - weights.general_mean) / weights.general_scale
        logits = logits + weights.general_coef * g
    return expit(logits.astype(np.float64)).astype(np.float32)  # exp(-logit) overflows


TASTE_NEIGHBOURS = 10
"""How many of the nearest rated photos say how well the ratings know a photo: enough that
one look-alike does not make a photo familiar, few enough that a kind of photo rated at a
handful of places still counts."""

FAMILIAR_AT = 0.5
"""A photo whose nearest rated photos are as close as the median rated photo's are, at other
places, is wholly familiar: Yours alone scores it, as today. Below that, General's share
grows until, for a photo less familiar than every rated one, General alone scores it. So
leaning changes nothing where the ratings are typical, and only acts where they are thin."""

GENERAL_L2 = 1e-3
"""Next to nothing: calibrating General is one slope and one bias, fitted on hundreds of
ratings, with nothing to learn by heart."""


def familiarity(
    rated: NDArray[np.floating],
    places: Sequence[str],
    photos: NDArray[np.floating],
    *,
    k: int = TASTE_NEIGHBOURS,
) -> NDArray[np.float32]:
    """How well your ratings know each of ``photos``, from 0 (less than any rated photo) to
    1 (at least as well as the median rated photo, ``FAMILIAR_AT``).

    A photo's closeness is the mean cosine similarity of its ``k`` nearest ``rated`` photos.
    It is judged against each rated photo's closeness to the ratings at *other* places
    (``places``, one per row of ``rated``): a photo at a new place has no ratings from its
    own place beside it, and a rated photo always would. With ratings at one place only,
    each rated photo is compared with every other instead."""
    rated_u, photos_u = _unit(rated), _unit(photos)
    if len(rated_u) < 2:
        raise ValueError("Measuring how familiar a photo is needs two or more rated photos.")
    place = np.asarray(places)
    within = place[:, None] == place[None, :]
    if within.all():
        within = np.eye(len(place), dtype=bool)
    among = rated_u @ rated_u.T
    reference = np.sort(_closeness(np.where(within, -np.inf, among), k))
    rank = np.searchsorted(reference, _closeness(photos_u @ rated_u.T, k), side="right")
    return np.minimum(1.0, rank / len(reference) / FAMILIAR_AT).astype(np.float32)


def _closeness(similarity: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Mean of each row's ``k`` largest finite similarities (fewer where a row has fewer)."""
    found = np.isfinite(similarity).sum(axis=1)
    k = max(1, min(k, similarity.shape[1]))
    top = -np.sort(-similarity, axis=1)[:, :k]
    counted = np.minimum(found, k)
    total = np.where(np.isfinite(top), top, 0.0).sum(axis=1)
    return np.asarray(total / np.maximum(counted, 1))


def _unit(x: NDArray[np.floating]) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.asarray(x / np.where(norms > 0, norms, 1.0))


def fit_general(general: NDArray[np.floating], y: NDArray[np.integer]) -> TasteWeights:
    """General's p("strong") put on the scale of your likes: a slope and a bias on its
    log-odds, fitted to your ratings as Yours is. General's numbers come from a zero-shot
    softmax, so beside Yours's they would count for more or less than they should."""
    return fit_taste(_logit(general)[:, None], y, feature="general", l2=GENERAL_L2)


def general_probability(
    calibration: TasteWeights, general: NDArray[np.floating]
) -> NDArray[np.float32]:
    """p(you like it), by General's view (p("strong")) as ``fit_general`` calibrated it."""
    return taste_probability(calibration, _logit(general)[:, None])


def lean_on_general(
    yours: NDArray[np.floating],
    general: NDArray[np.floating],
    familiar: NDArray[np.floating],
) -> NDArray[np.float32]:
    """p(you like it), mixing Yours's and General's (calibrated, ``general_probability``)
    log-odds by how ``familiar`` (``familiarity``) each photo is: Yours's where your ratings
    know photos like it, General's where they do not (audit plan S6)."""
    w = np.asarray(familiar, dtype=np.float64)
    if ((w < 0) | (w > 1)).any() or not np.isfinite(w).all():
        raise ValueError("How familiar a photo is must be between 0 and 1.")
    mixed = w * _logit(yours) + (1 - w) * _logit(general)
    return expit(mixed).astype(np.float32)


SCALE_PRIOR = 20.0
"""How many ratings' worth a scale's share of likes is pulled toward every rating's: a
scale rated a handful of times says little about how much you like it."""


def scale_offsets(scales: Sequence[str], y: NDArray[np.integer]) -> dict[str, float]:
    """How much more (above 0) or less you like each scale than photos overall, in log-odds:
    each scale's share of likes, pulled toward the overall share by ``SCALE_PRIOR``. With
    likes and dislikes counting alike (as ``fit_taste`` counts them), the offset is the
    same whatever the overall share."""
    labels = np.asarray(y, dtype=np.float64)
    overall = float(labels.mean())
    offsets = {}
    for scale in sorted(set(scales)):
        at = np.asarray([s == scale for s in scales])
        share = (labels[at].sum() + SCALE_PRIOR * overall) / (at.sum() + SCALE_PRIOR)
        offsets[scale] = float(_logit(np.asarray([share]))[0] - _logit(np.asarray([overall]))[0])
    return offsets


@dataclass(frozen=True)
class ScaledTaste:
    """Yours learned apart from the scale of a shot: ``weights`` on each vector minus its
    scale's mean (``means``), ``offsets`` for how much each scale is liked, and
    ``calibration``, how much each of the two counts."""

    weights: TasteWeights
    means: dict[str, NDArray[np.float32]]
    overall: NDArray[np.float32]
    offsets: dict[str, float]
    calibration: TasteWeights


def fit_by_scale(
    x: NDArray[np.floating],
    y: NDArray[np.integer],
    scales: Sequence[str],
    *,
    feature: str,
    extractor: str = "",
    l2: float = TASTE_L2,
) -> ScaledTaste:
    """Yours fitted so the scale of a shot is learned once, as one number per scale, and the
    embedding learns what makes a photo strong within its scale. Fitted as today, close
    shots (liked least) teach a direction toward "close" that ranks every detail down,
    macros (liked often) with them (audit plan S5, 7 November 2026).

    The embedding's weights are held back by ``l2`` and the offsets are not, so added as
    they are the offsets would rank by scale alone. A slope for each and a bias, fitted on
    the same ratings (``GENERAL_L2``, as General is calibrated), say how much each counts."""
    x = np.asarray(x, dtype=np.float64)
    overall = x.mean(axis=0)
    means = {
        scale: x[np.asarray([s == scale for s in scales])].mean(axis=0)
        for scale in sorted(set(scales))
    }
    centred = x - np.stack([means[s] for s in scales])
    weights = fit_taste(centred, y, feature=feature, extractor=extractor, l2=l2)
    offsets = scale_offsets(scales, y)
    parts = np.column_stack(
        [_logit(taste_probability(weights, centred)), [offsets[s] for s in scales]]
    )
    return ScaledTaste(
        weights=weights,
        means={s: m.astype(np.float32) for s, m in means.items()},
        overall=overall.astype(np.float32),
        offsets=offsets,
        calibration=fit_taste(parts, y, feature=f"{feature} by scale", l2=GENERAL_L2),
    )


def scaled_probability(
    taste: ScaledTaste, x: NDArray[np.floating], scales: Sequence[str]
) -> NDArray[np.float32]:
    """p(you like it) under a taste ``fit_by_scale`` learned; a scale it never saw is
    centred on every rated photo's mean and given no offset."""
    centred = np.asarray(x, dtype=np.float64) - np.stack(
        [taste.means.get(s, taste.overall) for s in scales]
    )
    within = _logit(taste_probability(taste.weights, centred))
    offset = np.asarray([taste.offsets.get(s, 0.0) for s in scales])
    return taste_probability(taste.calibration, np.column_stack([within, offset]))
