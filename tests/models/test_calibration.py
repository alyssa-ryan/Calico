"""What a probability from Yours means: calibrated on what you said at places it never learned."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logit

from calico.evaluation import stats
from calico.models.calibration import (
    MIN_FORECASTS,
    MIN_PLACES,
    Calibration,
    fit_calibration,
)


def forecasts(n: int, slope: float, intercept: float, seed: int = 0) -> list[tuple[float, bool]]:
    """Yours's p(you like it) against what you said, when you like a photo with probability
    expit(slope * logit(p) + intercept): a model that is more sure than you are has slope < 1."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.99, size=n)
    liked = rng.random(n) < expit(slope * logit(p) + intercept)
    return [(float(a), bool(b)) for a, b in zip(p, liked, strict=True)]


def test_a_model_more_sure_than_you_are_is_pulled_back_toward_what_you_said() -> None:
    made = fit_calibration(forecasts(4000, slope=0.5, intercept=0.4), places=12)
    assert made is not None
    assert made.slope == pytest.approx(0.5, abs=0.1)
    assert made.intercept == pytest.approx(0.4, abs=0.1)
    # at p = 0.95 the model says 95%; the ratings say about expit(0.5 * logit(0.95) + 0.4) = 0.84
    assert float(made.probability(0.95)) == pytest.approx(0.84, abs=0.03)
    assert made.photos == 4000
    assert made.places == 12


def test_a_probability_rises_with_the_model_and_stays_between_zero_and_one() -> None:
    made = fit_calibration(forecasts(2000, slope=0.8, intercept=0.2), places=9)
    assert made is not None
    shown = made.probability(np.array([0.0, 0.1, 0.5, 0.9, 1.0]))
    assert np.all(np.diff(shown) > 0)
    assert shown[0] > 0
    assert shown[-1] < 1  # a model's 0 and 1 are not yours


def test_a_model_that_tells_nothing_apart_is_not_calibrated() -> None:
    # Whatever p says, you liked about the same share: a probability that does not rise with it
    # would only mislead (the fit says so rather than draw a flat line).
    rng = np.random.default_rng(1)
    flat = [
        (float(p), bool(b))
        for p, b in zip(rng.uniform(0.1, 0.9, 1000), rng.random(1000) < 0.8, strict=True)
    ]
    assert fit_calibration(flat, places=9) is None
    backwards = forecasts(2000, slope=-0.6, intercept=0.5)
    assert fit_calibration(backwards, places=9) is None


def test_too_few_photos_or_places_or_one_kind_of_answer_is_not_calibrated() -> None:
    enough = forecasts(MIN_FORECASTS, slope=0.7, intercept=0.3)
    assert fit_calibration(enough, places=9) is not None
    assert fit_calibration(enough[: MIN_FORECASTS - 1], places=9) is None
    assert fit_calibration(enough, places=4) is None  # below stats.MIN_UNITS
    assert fit_calibration([(p, True) for p, _ in enough], places=9) is None
    assert fit_calibration([(p, False) for p, _ in enough], places=9) is None
    assert fit_calibration([], places=0) is None


def test_a_calibration_is_kept_and_read_back_as_it_was() -> None:
    made = fit_calibration(forecasts(1000, slope=0.6, intercept=0.3), places=10)
    assert made is not None
    assert Calibration.from_json(made.to_json()) == made
    assert Calibration.from_json({"slope": "x"}) is None  # a damaged entry is no calibration
    assert Calibration.from_json(None) is None


def test_the_places_it_needs_are_the_places_a_range_needs() -> None:
    assert (
        MIN_PLACES == stats.MIN_UNITS
    )  # models cannot import evaluation, so the two are kept equal
