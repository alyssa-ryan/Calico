"""How far a measurement could move: intervals by place. Pure functions; no I/O.

Ratings cluster by place: photos of one place look alike and are rated in one sitting, so
treating them as independent makes every number look surer than it is (audit A10-3). An
interval here resamples the *places* (or searches, references, sessions: the ``unit``),
then the photos within each place drawn, and recomputes the metric on each resample. The
interval is the middle 95% of those values.

A comparison of two rankers on the same photos is measured as their *paired* difference:
both are computed on each resample, so what they share (a hard place, an easy one)
cancels. A difference whose interval holds zero has not been shown, whatever its sign.

Seeded (rule 7): the same ratings and seed give the same interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

RESAMPLES = 2000
"""Resamples per interval: enough that the 2.5th and 97.5th percentiles settle to about
0.01 on an AUC."""
SEED = 20261028
"""The default seed, so every ``calico eval`` run gives the same intervals."""
LEVEL = 0.95
MIN_UNITS = 5
"""Below this many places (or other units), redrawing them cannot show how far a value
could move: two runs drawn again give back the same two, and a range like [1.00, 1.00]
says nothing. Such a range is printed as too few to trust."""


@dataclass(frozen=True)
class Interval:
    """A value, the range it could take with other places drawn (``LEVEL``), and what it
    rests on: ``places`` (in ``unit``s) and the photos (or other answers) ``rated``."""

    value: float
    lo: float
    hi: float
    places: int
    rated: int
    unit: str = "place"

    @property
    def few(self) -> bool:
        """Too few units for the range to be trusted (``MIN_UNITS``)."""
        return self.places < MIN_UNITS

    @property
    def excludes_zero(self) -> bool:
        """Whether a difference is shown: its whole interval lies on one side of zero."""
        return self.lo > 0 or self.hi < 0


type Metric[T] = Callable[[Sequence[Sequence[T]]], float]
"""A measurement over places, each given as its photos (or other rated items)."""


def place_bootstrap[T](
    rows: Mapping[str, Sequence[T]],
    metric: Metric[T],
    *,
    n: int = RESAMPLES,
    seed: int = SEED,
    unit: str = "place",
) -> Interval:
    """``metric`` over every place in ``rows``, with its interval."""
    groups = _groups(rows)
    value = metric(groups) if groups else math.nan
    return _interval(value, [metric(drawn) for drawn in _resamples(groups, n, seed)], groups, unit)


def paired_difference[T](
    a: Metric[T],
    b: Metric[T],
    by_place: Mapping[str, Sequence[T]],
    *,
    n: int = RESAMPLES,
    seed: int = SEED,
    unit: str = "place",
) -> Interval:
    """``a - b``, both measured on the same places and photos, with its interval."""
    return compare(a, b, by_place, n=n, seed=seed, unit=unit).difference


@dataclass(frozen=True)
class Compared:
    """Two rankers on one measure: each with its interval, and ``a - b`` paired."""

    a: Interval
    b: Interval
    difference: Interval


def compare[T](
    a: Metric[T],
    b: Metric[T],
    by_place: Mapping[str, Sequence[T]],
    *,
    n: int = RESAMPLES,
    seed: int = SEED,
    unit: str = "place",
) -> Compared:
    """``a``, ``b`` and their paired difference, from one set of resamples."""
    groups = _groups(by_place)
    drawn = [(a(g), b(g)) for g in _resamples(groups, n, seed)]
    va, vb = (a(groups), b(groups)) if groups else (math.nan, math.nan)
    return Compared(
        a=_interval(va, [x for x, _ in drawn], groups, unit),
        b=_interval(vb, [y for _, y in drawn], groups, unit),
        difference=_interval(va - vb, [x - y for x, y in drawn], groups, unit),
    )


def macro_average[T](
    by_place: Mapping[str, Sequence[T]], metric: Callable[[Sequence[T]], float]
) -> float:
    """The mean of ``metric`` over places, each counting once (places where it is undefined
    are left out): beside a pooled value, it shows whether a few big places carry it."""
    values = [metric(rows) for rows in by_place.values() if rows]
    defined = [v for v in values if not math.isnan(v)]
    return sum(defined) / len(defined) if defined else math.nan


def _groups[T](rows: Mapping[str, Sequence[T]]) -> list[Sequence[T]]:
    """The places with something rated, in name order, so a seed always draws alike."""
    return [rows[name] for name in sorted(rows) if rows[name]]


def _resamples[T](groups: Sequence[Sequence[T]], n: int, seed: int) -> Iterator[list[list[T]]]:
    if not groups:
        return
    rng = np.random.default_rng(seed)
    for _ in range(n):
        drawn = []
        for g in rng.integers(0, len(groups), len(groups)):
            group = groups[int(g)]
            drawn.append([group[int(i)] for i in rng.integers(0, len(group), len(group))])
        yield drawn


def _interval(
    value: float, drawn: Sequence[float], groups: Sequence[Sequence[object]], unit: str
) -> Interval:
    defined = [v for v in drawn if not math.isnan(v)]
    tail = (1 - LEVEL) / 2 * 100
    lo, hi = (
        (float(np.percentile(defined, tail)), float(np.percentile(defined, 100 - tail)))
        if defined
        else (math.nan, math.nan)
    )
    return Interval(
        value=value,
        lo=lo,
        hi=hi,
        places=len(groups),
        rated=sum(len(g) for g in groups),
        unit=unit,
    )
