"""Intervals by place (plan, Workstream 7: A10-3, A05-3, A05-9)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from calico.evaluation.stats import (
    MIN_UNITS,
    Interval,
    compare,
    macro_average,
    paired_difference,
    place_bootstrap,
)

type Groups = Sequence[Sequence[float]]


def mean(groups: Groups) -> float:
    values = [v for g in groups for v in g]
    return sum(values) / len(values)


def places(*means: float, size: int = 20) -> dict[str, list[float]]:
    """Places whose photos each score their place's mean, give or take 0.1."""
    return {
        f"p{i}": [m + (0.1 if j % 2 else -0.1) for j in range(size)] for i, m in enumerate(means)
    }


def test_the_interval_holds_the_value_and_counts_what_it_rests_on() -> None:
    found = place_bootstrap(places(0.2, 0.5, 0.8), mean, n=200, seed=1)
    assert found.lo < found.value < found.hi
    assert math.isclose(found.value, 0.5)
    assert (found.places, found.rated, found.unit) == (3, 60, "place")


def test_the_same_seed_gives_the_same_interval() -> None:
    rows = places(0.2, 0.5, 0.8)
    assert place_bootstrap(rows, mean, n=200, seed=3) == place_bootstrap(rows, mean, n=200, seed=3)
    assert place_bootstrap(rows, mean, n=200, seed=3) != place_bootstrap(rows, mean, n=200, seed=4)


def test_places_are_resampled_so_places_that_differ_widen_it() -> None:
    alike = place_bootstrap(places(0.5, 0.5, 0.5, 0.5), mean, n=300, seed=1)
    apart = place_bootstrap(places(0.1, 0.4, 0.6, 0.9), mean, n=300, seed=1)
    # Photos alone vary by ±0.1 in both; only the places differ.
    assert apart.hi - apart.lo > 3 * (alike.hi - alike.lo)


def test_photos_are_resampled_within_each_place() -> None:
    found = place_bootstrap({"only": [0.0, 1.0] * 10}, mean, n=300, seed=1)
    assert found.lo < 0.5 < found.hi


def test_more_places_narrow_it() -> None:
    few = place_bootstrap(places(0.2, 0.8), mean, n=300, seed=1)
    many = place_bootstrap(places(*[0.2, 0.8] * 10), mean, n=300, seed=1)
    assert many.hi - many.lo < few.hi - few.lo


def test_a_paired_difference_cancels_what_both_share() -> None:
    rows = places(0.1, 0.4, 0.6, 0.9)

    def shifted(groups: Groups) -> float:
        return mean(groups) + 0.05

    found = paired_difference(shifted, mean, rows, n=300, seed=1)
    assert math.isclose(found.value, 0.05)
    assert math.isclose(found.lo, 0.05)
    assert math.isclose(found.hi, 0.05)
    assert found.excludes_zero


def test_a_difference_that_may_be_zero_does_not_exclude_it() -> None:
    rows = {"a": [1.0, 0.0] * 5, "b": [0.0, 1.0] * 5}

    def first_half(groups: Groups) -> float:
        return mean([g[: len(g) // 2] for g in groups])

    found = paired_difference(first_half, mean, rows, n=300, seed=1)
    assert found.lo < 0 < found.hi
    assert not found.excludes_zero


def test_resamples_where_the_metric_is_undefined_are_left_out() -> None:
    def only_with_both(groups: Groups) -> float:
        values = [v for g in groups for v in g]
        return math.nan if len(set(values)) < 2 else mean(groups)

    found = place_bootstrap({"a": [0.0], "b": [1.0]}, only_with_both, n=200, seed=1)
    assert found.lo == found.hi == 0.5  # only resamples holding both places count


def test_nothing_measured_has_no_interval() -> None:
    found = place_bootstrap({}, mean, n=50, seed=1)
    assert math.isnan(found.value)
    assert math.isnan(found.lo)
    assert not found.excludes_zero
    assert found.places == 0


def test_the_unit_names_what_was_resampled() -> None:
    assert place_bootstrap(places(0.5), mean, n=10, seed=1, unit="search").unit == "search"


def test_the_macro_average_counts_each_place_once() -> None:
    rows = {"big": [1.0] * 90, "small": [0.0] * 10}
    assert macro_average(rows, lambda photos: sum(photos) / len(photos)) == 0.5
    assert math.isnan(macro_average({"a": [1.0]}, lambda _: math.nan))


def test_an_interval_is_plain_data() -> None:
    found = Interval(value=0.7, lo=0.6, hi=0.8, places=4, rated=100)
    assert found.unit == "place"
    assert found.excludes_zero


def test_a_comparison_gives_each_side_and_their_difference_from_one_set_of_resamples() -> None:
    rows = places(0.1, 0.4, 0.6, 0.9)

    def higher(groups: Groups) -> float:
        return mean(groups) + 0.05

    found = compare(higher, mean, rows, n=300, seed=1)
    assert found.a.value - found.b.value == found.difference.value
    assert found.b == place_bootstrap(rows, mean, n=300, seed=1)
    assert found.difference == paired_difference(higher, mean, rows, n=300, seed=1)


def test_fewer_than_five_units_are_too_few_to_trust_a_range() -> None:
    assert place_bootstrap(places(0.2, 0.4, 0.6, 0.8), mean, n=10, seed=1).few
    assert not place_bootstrap(places(0.1, 0.2, 0.4, 0.6, 0.8), mean, n=10, seed=1).few
    assert MIN_UNITS == 5


# -- survivors of the mutation run, 10 November 2026 --------------------


def test_the_interval_is_the_middle_95_percent_of_the_resamples() -> None:
    draws = iter(range(1001))

    def counting(groups: Groups) -> float:
        return float(next(draws))  # the places as they are first, then each resample in turn

    found = place_bootstrap(places(0.1, 0.4, 0.6), counting, n=1000, seed=1)
    assert found.value == 0.0
    assert found.lo == pytest.approx(25.975)  # the 2.5th percentile of 1 to 1,000
    assert found.hi == pytest.approx(975.025)


def test_the_interval_does_not_depend_on_the_order_places_were_given_in() -> None:
    rows = places(0.2, 0.5, 0.8, 0.9)
    backwards = dict(reversed(list(rows.items())))
    assert place_bootstrap(rows, mean, n=200, seed=3) == place_bootstrap(
        backwards, mean, n=200, seed=3
    )


def test_an_interval_that_touches_zero_has_not_shown_a_difference() -> None:
    assert not Interval(value=0.1, lo=0.0, hi=0.2, places=5, rated=9).excludes_zero
    assert not Interval(value=-0.1, lo=-0.2, hi=0.0, places=5, rated=9).excludes_zero


def test_the_macro_average_is_the_mean_and_not_the_median() -> None:
    rows = {"a": [0.0], "b": [0.0], "c": [1.0]}
    assert macro_average(rows, lambda r: r[0]) == pytest.approx(1 / 3)
