"""``calibration``: what a probability from Yours means.

Yours's p(you like it) comes from a regression that counts each class as much as the other
(``fit_taste``), so 0.9 is not "nine in ten you will like". What you said at places it never
learned (``measure_taste``'s forecasts) about the photos a taste showed first says how often you
liked what it gave each probability: one slope and one bias on its logit (Platt scaling). Two
numbers, because a few hundred answers would bend any freer curve to chance. Leaving one place
out, on 1,067 first picks at 46 places (10 November 2026), it scored a log loss of 0.469 against
0.534 for Yours's own probability and 0.537 for the liked share alone (decision 0010).

Pure: forecasts in, two numbers out. ``calico.trained`` keeps them with the taste they were read
for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

MIN_FORECASTS = 200
"""Judged photos a calibration needs: with fewer, two numbers are still fitted to chance."""

MIN_PLACES = 5
"""Places they come from, as ``evaluation.stats.MIN_UNITS``: one place's liking is not yours."""

SURE = 2.0
"""A slope must be this many standard errors above zero: a probability that does not rise
with the model's would only mislead, so then there is none."""

_EDGE = 1e-4
"""A model's 0 and 1 are not yours: its probability is held this far inside them."""


@dataclass(frozen=True)
class Calibration:
    """p(you like it) from Yours's own, as ``expit(slope * logit(p) + intercept)``, and what it
    was read from."""

    slope: float
    intercept: float
    photos: int
    liked: int
    places: int

    def probability(self, p: float | NDArray[np.floating]) -> NDArray[np.float64]:
        """How likely you are to like a photo Yours gives ``p`` (at a place it never learned)."""
        z = logit(np.clip(np.asarray(p, dtype=np.float64), _EDGE, 1 - _EDGE))
        return np.asarray(expit(self.slope * z + self.intercept), dtype=np.float64)

    def to_json(self) -> dict[str, float | int]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "photos": self.photos,
            "liked": self.liked,
            "places": self.places,
        }

    @staticmethod
    def from_json(data: Any) -> Calibration | None:
        """What ``to_json`` wrote, or None when there is none or it is damaged: a taste saved
        before probabilities, or half-written, shows none rather than a wrong one."""
        if not isinstance(data, dict):
            return None
        try:
            made = Calibration(
                slope=float(data["slope"]),
                intercept=float(data["intercept"]),
                photos=int(data["photos"]),
                liked=int(data["liked"]),
                places=int(data["places"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
        return made if np.isfinite([made.slope, made.intercept]).all() else None


def fit_calibration(forecasts: Sequence[tuple[float, bool]], *, places: int) -> Calibration | None:
    """Read how often you liked what Yours gave each probability, from ``forecasts`` (its
    p(you like it) for a photo at a place it never learned, and what you said) at ``places``
    places. None when there are too few, one kind of answer only, or the probability does not
    clearly rise with the model's (``SURE``)."""
    if len(forecasts) < MIN_FORECASTS or places < MIN_PLACES:
        return None
    p = np.clip(np.asarray([f[0] for f in forecasts], dtype=np.float64), _EDGE, 1 - _EDGE)
    y = np.asarray([f[1] for f in forecasts], dtype=np.float64)
    if y.min() == y.max() or not np.isfinite(p).all():
        return None
    x = np.stack([logit(p), np.ones_like(p)], axis=1)

    def loss(beta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        q = expit(x @ beta)
        nll = -(y * np.log(np.clip(q, 1e-12, None)) + (1 - y) * np.log(np.clip(1 - q, 1e-12, None)))
        return float(nll.sum()), x.T @ (q - y)

    fitted = minimize(loss, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    if not fitted.success:
        return None
    slope, intercept = (float(v) for v in fitted.x)
    q = expit(x @ fitted.x)
    information = x.T @ (x * (q * (1 - q))[:, None])
    error = float(np.sqrt(np.linalg.pinv(information)[0, 0]))
    if not slope > SURE * error:
        return None
    return Calibration(slope, intercept, len(y), int(y.sum()), places)
