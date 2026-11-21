"""Numeric building blocks shared by tasks and models.

Everything here is a pure function on arrays and returns values in [0, 1], so
signals from different sources can be weighted against each other directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from calico.domain import Evidence

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

FloatArray = NDArray[np.float32]


def percentile_rank(values: NDArray[np.floating]) -> FloatArray:
    """Each value's rank within ``values`` scaled to [0, 1]; ties share their mean rank.

    The lowest value maps to 0 and the highest to 1. A single value, or all-equal
    values, map to 0.5: no evidence either way. NaNs map to 0.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros(arr.shape, dtype=np.float32)
    finite = ~np.isnan(arr)
    n = int(finite.sum())
    if n == 0:
        return out
    if n == 1 or np.ptp(arr[finite]) == 0:
        out[finite] = 0.5
        return out
    sub = arr[finite]
    order = np.argsort(sub, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # average ranks over ties
    _, inverse = np.unique(sub, return_inverse=True)
    mean_rank = np.bincount(inverse, weights=ranks) / np.bincount(inverse)
    out[finite] = (mean_rank[inverse] / (n - 1)).astype(np.float32)
    return out


def pairwise_percentile(similarity: NDArray[np.floating]) -> FloatArray:
    """``percentile_rank`` over the off-diagonal entries of a symmetric matrix.

    Turns raw pairwise similarities (whose absolute scale depends on the
    embedder) into "how similar is this pair compared with all other pairs".
    The diagonal is set to 1.
    """
    sim = np.asarray(similarity, dtype=np.float64)
    n = sim.shape[0]
    out = np.ones((n, n), dtype=np.float32)
    if n < 2:
        return out
    iu = np.triu_indices(n, k=1)
    ranked = percentile_rank(sim[iu])
    out[iu] = ranked
    out[(iu[1], iu[0])] = ranked
    return out


def bhattacharyya(p: NDArray[np.floating], q: NDArray[np.floating]) -> FloatArray:
    """Overlap of probability distributions, row-wise or all-pairs.

    ``p`` (n, k) and ``q`` (m, k) -> (n, m) matrix of ``Σ sqrt(p·q)`` in [0, 1];
    1 means identical distributions.
    """
    sp = np.sqrt(np.clip(np.asarray(p, dtype=np.float64), 0, None))
    sq = np.sqrt(np.clip(np.asarray(q, dtype=np.float64), 0, None))
    out: FloatArray = np.clip(sp @ sq.T, 0.0, 1.0).astype(np.float32)
    return out


def histogram_intersection(a: NDArray[np.floating], b: NDArray[np.floating]) -> FloatArray:
    """All-pairs ``Σ min(a, b)`` for L1-normalised histograms: (n, k), (m, k) -> (n, m)."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    overlap = np.minimum(a64[:, None, :], b64[None, :, :]).sum(axis=2)
    out: FloatArray = np.clip(overlap, 0.0, 1.0).astype(np.float32)
    return out


@dataclass(frozen=True)
class Signal:
    """A named per-photo signal in [0, 1] and its intended weight."""

    name: str
    values: FloatArray
    weight: float
    describe: str = ""


def combine(signals: Sequence[Signal]) -> tuple[FloatArray, list[Signal]]:
    """Weighted mean of signals. Returns the scores and the signals with
    weights normalised to sum to 1 (zero-weight signals are dropped)."""
    active = [s for s in signals if s.weight > 0]
    if not active:
        raise ValueError("at least one signal must have positive weight")
    total = sum(s.weight for s in active)
    normalised = [
        Signal(s.name, np.asarray(s.values, np.float32), s.weight / total, s.describe)
        for s in active
    ]
    scores = np.zeros_like(normalised[0].values, dtype=np.float32)
    for s in normalised:
        scores += s.weight * s.values
    return scores, normalised


def evidence_for(
    index: int, signals: Sequence[Signal], details: Mapping[str, str] | None = None
) -> tuple[Evidence, ...]:
    """The evidence behind photo ``index``, given normalised signals from ``combine``."""
    details = details or {}
    return tuple(
        Evidence(
            signal=s.name,
            value=round(float(s.values[index]), 4),
            weight=round(s.weight, 4),
            detail=details.get(s.name, s.describe),
        )
        for s in signals
    )


def with_criteria(
    criteria_scores: NDArray[np.floating] | None,
    cand: Sequence[int],
    signals: list[Signal],
    describe_weight: float,
) -> tuple[list[Signal], FloatArray | None]:
    """``signals`` scaled to make room for the criteria text at ``describe_weight``, with the
    criteria (percentile within ``cand``) added; unchanged without usable text."""
    if criteria_scores is None or describe_weight <= 0:
        return signals, None
    pct = percentile_rank(criteria_scores[list(cand)])
    rest = 1 - describe_weight
    scaled = [Signal(s.name, s.values, s.weight * rest, s.describe) for s in signals]
    return [*scaled, Signal("criteria", pct, describe_weight, "matches criteria")], pct
