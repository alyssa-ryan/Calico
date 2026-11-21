from __future__ import annotations

import numpy as np
import pytest

from calico.models.taste import (
    SCALE_PRIOR,
    TasteModel,
    familiarity,
    fit_by_scale,
    fit_general,
    fit_pairs,
    fit_taste,
    general_probability,
    lean_on_general,
    scale_offsets,
    scaled_probability,
    taste_probability,
)
from tests.factories import unit, view


def liked_along_first_axis(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Photos you like lean along the first axis; the rest is noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8)).astype(np.float32)
    y = (x[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(np.int8)
    return unit(x), y


def test_a_fitted_model_prefers_what_you_liked() -> None:
    x, y = liked_along_first_axis(200)
    weights = fit_taste(x, y, feature="clip@1")
    model = TasteModel(weights)
    test_x, test_y = liked_along_first_axis(200, seed=1)
    strong = model.predict_proba(view(test_x))[:, model.labels.index("strong")]
    # Strength ranks p("strong") by percentile, so the order is the contract.
    liked, disliked = strong[test_y == 1], strong[test_y == 0]
    assert (liked[:, None] > disliked[None, :]).mean() > 0.9
    assert np.allclose(model.predict_proba(view(test_x)).sum(axis=1), 1.0)


def test_a_photo_far_off_the_scale_scores_0_or_1_without_overflowing() -> None:
    """A taste fitted on a few ratings can set a tiny scale, so a photo unlike them gets a
    logit in the thousands; ``exp`` of it overflowed (a warning, an error in tests)."""
    x, y = liked_along_first_axis(40)
    weights = fit_taste(x, y, feature="clip@1")
    far = np.zeros((2, 8), dtype=np.float32)
    far[0, 0], far[1, 0] = 1e6, -1e6
    p = taste_probability(weights, far)
    assert np.isfinite(p).all()
    assert sorted(p.round(6).tolist()) == [0.0, 1.0]


def test_it_reads_the_embedding_it_was_trained_on() -> None:
    x, y = liked_along_first_axis(100)
    model = TasteModel(fit_taste(x, y, feature="siglip@1", extractor="siglip"))
    noise = unit(np.random.default_rng(5).normal(size=(4, 8)))
    main_is_noise = model.predict_proba(view(noise, {"siglip": x[:4]}))
    main_is_same = model.predict_proba(view(x[:4], {"siglip": x[:4]}))
    siglip_is_noise = model.predict_proba(view(x[:4], {"siglip": noise}))
    assert np.allclose(main_is_noise, main_is_same, atol=1e-5)
    assert not np.allclose(main_is_same, siglip_is_noise, atol=1e-3)


def test_a_photo_without_that_embedding_cannot_be_scored() -> None:
    x, y = liked_along_first_axis(50)
    model = TasteModel(fit_taste(x, y, feature="siglip@1", extractor="siglip"))
    with pytest.raises(KeyError, match="siglip"):
        model.predict_proba(view(np.ones((2, 3)), {"layout": np.ones((2, 4))}))


def test_the_version_names_what_it_learned_from() -> None:
    x, y = liked_along_first_axis(100)
    a = fit_taste(x, y, feature="clip@1")
    b = fit_taste(x, 1 - y, feature="clip@1")
    assert a.version != b.version
    assert a.version == fit_taste(x, y, feature="clip@1").version
    assert TasteModel(a).version.startswith("taste@")


def test_it_needs_likes_and_dislikes() -> None:
    x, _ = liked_along_first_axis(20)
    with pytest.raises(ValueError, match="both"):
        fit_taste(x, np.ones(20, dtype=np.int8), feature="clip@1")


def test_rare_dislikes_still_count_as_much_as_common_likes() -> None:
    # 90% likes: without balancing, "like everything" would be nearly as good.
    x, y = liked_along_first_axis(300, seed=3)
    keep = np.flatnonzero((y == 1) | (np.arange(300) % 9 == 0))
    model = TasteModel(fit_taste(x[keep], y[keep], feature="clip@1"))
    test_x, test_y = liked_along_first_axis(200, seed=4)
    strong = model.predict_proba(view(test_x))[:, 1]
    assert strong[test_y == 0].mean() < 0.5


def liked_by_general_and_first_axis(n: int, seed: int = 0) -> tuple[np.ndarray, ...]:
    """What you like is General's view plus a lean along the first axis General misses."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8)).astype(np.float32)
    general_logit = rng.normal(size=n)
    y = (x[:, 0] + general_logit > 0).astype(np.int8)
    return unit(x), y, (1.0 / (1.0 + np.exp(-general_logit))).astype(np.float32)


def pair_auc(score: np.ndarray, y: np.ndarray) -> float:
    return float((score[y == 1][:, None] > score[y == 0][None, :]).mean())


def test_with_general_as_an_offset_it_learns_only_what_general_misses() -> None:
    x, y, general = liked_by_general_and_first_axis(300)
    # Light: General is never held back, so a heavy l2 leaves the embedding too little say.
    alone = fit_taste(x, y, feature="clip@1", l2=10)
    offset = fit_taste(x, y, feature="clip@1", general=general, l2=10)
    assert alone.general_coef == 0.0
    assert offset.general_coef > 0
    test_x, test_y, test_general = liked_by_general_and_first_axis(300, seed=1)
    with_general = taste_probability(offset, test_x, general=test_general)
    assert pair_auc(with_general, test_y) > pair_auc(taste_probability(alone, test_x), test_y) + 0.1
    with pytest.raises(ValueError, match="General"):
        taste_probability(offset, test_x)  # it cannot score without General's view


def test_an_offset_taste_is_versioned_apart_and_not_yet_scored_in_a_search() -> None:
    x, y, general = liked_by_general_and_first_axis(100)
    alone, offset = (
        fit_taste(x, y, feature="clip@1"),
        fit_taste(x, y, feature="clip@1", general=general),
    )
    assert alone.version != offset.version
    with pytest.raises(ValueError, match="General"):
        TasteModel(offset)


def test_a_taste_is_not_learned_from_values_that_are_not_finite() -> None:
    """Audit A04-10: one NaN gave all-zero weights, a constant 0.5 saved as a taste."""
    x = np.random.default_rng(0).normal(size=(20, 4))
    y = (x[:, 0] > 0).astype(np.int8)
    x[3, 1] = np.nan
    with pytest.raises(ValueError, match="all finite numbers"):
        fit_taste(x, y, feature="f")


def test_a_taste_learns_only_likes_and_dislikes() -> None:
    x = np.random.default_rng(0).normal(size=(20, 4))
    y = (x[:, 0] > 0).astype(np.int8)
    y[0] = 2
    with pytest.raises(ValueError, match="1 for a like and 0"):
        fit_taste(x, y, feature="f")


def test_a_fit_that_does_not_converge_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    x = np.random.default_rng(0).normal(size=(20, 4))
    y = (x[:, 0] > 0).astype(np.int8)

    class Failed:
        success = False
        message = "ABNORMAL_TERMINATION_IN_LNSRCH"
        x = np.full(6, np.nan)

    monkeypatch.setattr("calico.models.taste.minimize", lambda *a, **k: Failed())
    with pytest.raises(ValueError, match="did not converge"):
        fit_taste(x, y, feature="f")


def test_a_photo_like_many_rated_photos_is_familiar_and_one_unlike_any_is_not() -> None:
    """Yours leans on General only where its ratings hold few photos like this one (S6)."""
    rng = np.random.default_rng(0)
    centre = np.zeros(8)
    centre[0] = 1.0
    rated = unit(centre + 0.2 * rng.normal(size=(60, 8)))
    places = [f"Mill {i % 3}" for i in range(60)]
    close = unit(centre + 0.02 * rng.normal(size=(5, 8)))  # nearer the ratings than most
    away = np.zeros((5, 8))
    away[:, 1] = -1.0  # a kind of photo nobody rated
    familiar = familiarity(rated, places, np.concatenate([close, unit(away)]))
    assert familiar.shape == (10,)
    assert ((familiar >= 0) & (familiar <= 1)).all()
    assert (familiar[:5] == 1).all()
    assert (familiar[5:] == 0).all()


def test_familiar_means_as_close_as_rated_photos_are_to_other_places() -> None:
    """A rated photo always has its own place's photos nearby; a photo at a new place does
    not. So the bar is how close rated photos are to ratings at *other* places, and a photo
    as close as the median rated photo is wholly familiar."""
    rng = np.random.default_rng(1)
    names = [f"P{i}" for i in range(20)]  # many places: leaving one out barely changes the rest
    rated = unit(rng.normal(size=(200, 8)))
    places = [names[i % 20] for i in range(200)]
    # Each rated photo, scored as a newcomer, against the ratings at the other places.
    familiar = np.concatenate(
        [
            familiarity(
                rated[[p != here for p in places]],
                [p for p in places if p != here],
                rated[[p == here for p in places]],
            )
            for here in names
        ]
    )
    assert 0.35 < float((familiar == 1).mean()) < 0.75  # about half are at or above the median
    assert familiar.min() < 0.5


def test_familiarity_needs_rated_photos_at_two_places_or_compares_each_with_the_rest() -> None:
    rng = np.random.default_rng(2)
    rated = unit(rng.normal(size=(20, 8)))
    familiar = familiarity(rated, ["Mill"] * 20, rated[:3])
    assert ((familiar >= 0) & (familiar <= 1)).all()


def test_general_is_put_on_the_scale_of_your_likes() -> None:
    """General's p("strong") is a zero-shot softmax, not a chance that you like it, so it is
    calibrated on your ratings before it is mixed with Yours."""
    rng = np.random.default_rng(3)
    logit = rng.normal(size=400)
    y = (3 * logit + rng.normal(size=400) > 0).astype(np.int8)
    squashed = 1 / (1 + np.exp(-0.2 * logit))  # right order, timid scale
    calibration = fit_general(squashed, y)
    p = general_probability(calibration, squashed)
    assert pair_auc(p, y) == pytest.approx(pair_auc(squashed, y))
    assert p.max() > 0.95
    assert p.min() < 0.05


def test_leaning_mixes_the_two_by_how_familiar_a_photo_is() -> None:
    yours = np.asarray([0.9, 0.9, 0.9], dtype=np.float32)
    general = np.asarray([0.1, 0.1, 0.1], dtype=np.float32)
    mixed = lean_on_general(yours, general, np.asarray([1.0, 0.5, 0.0]))
    assert mixed[0] == pytest.approx(0.9)
    assert mixed[1] == pytest.approx(0.5, abs=1e-6)  # halfway in log-odds
    assert mixed[2] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        lean_on_general(yours, general, np.asarray([1.5, 0.5, 0.0]))


def test_each_scale_gets_its_share_of_likes_pulled_toward_the_rest_where_rated_little() -> None:
    """You like close shots less than wide ones: one number per scale says so, instead of a
    direction in the embedding that also sweeps up macros (S5, 7 November 2026)."""
    scales = ["wide"] * 100 + ["close"] * 100 + ["macro"] * 4
    y = np.asarray([1] * 85 + [0] * 15 + [1] * 45 + [0] * 55 + [1] * 4, dtype=np.int8)
    offsets = scale_offsets(scales, y)
    assert offsets["wide"] > 0 > offsets["close"]
    assert 0 < offsets["macro"] < 1.0  # four likes are not certainty: pulled toward the rest
    rate = y.mean()
    unshrunk = np.log(0.999 / 0.001) - np.log(rate / (1 - rate))
    assert offsets["macro"] < unshrunk
    assert SCALE_PRIOR > 0


def scaled_likes(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Close-ups and macros lean one way in the embedding (axis 1); what you like is axis 0
    in every scale. You like most macros and few close-ups, and close-ups are common."""
    rng = np.random.default_rng(seed)
    scales = list(rng.choice(["wide", "close", "macro"], size=n, p=[0.5, 0.35, 0.15]))
    x = rng.normal(size=(n, 8))
    near = np.asarray([s != "wide" for s in scales])
    x[near, 1] += 2.0
    rate = np.asarray([{"wide": 0.8, "close": 0.3, "macro": 0.75}[s] for s in scales])
    cut = np.asarray([np.quantile(x[:, 0], 1 - r) for r in rate])
    y = (x[:, 0] > cut).astype(np.int8)
    return x.astype(np.float32), y, scales


def rank(p: np.ndarray) -> np.ndarray:
    """Each value's rank, 0 the lowest and 1 the highest."""
    return np.asarray(p.argsort().argsort() / (len(p) - 1))


def test_fitted_by_scale_a_liked_macro_is_not_ranked_down_with_the_close_ups() -> None:
    x, y, scales = scaled_likes(600)
    plain = fit_taste(x, y, feature="clip@1")
    scaled = fit_by_scale(x, y, scales, feature="clip@1")
    test_x, test_y, test_scales = scaled_likes(600, seed=1)
    today = taste_probability(plain, test_x)
    by_scale = scaled_probability(scaled, test_x, test_scales)
    macro = np.asarray([s == "macro" for s in test_scales])
    liked_macro = macro & (test_y == 1)
    assert rank(by_scale)[liked_macro].mean() > rank(today)[liked_macro].mean() + 0.05
    assert pair_auc(by_scale, test_y) > pair_auc(today, test_y)
    # How much the scale counts is learned beside the photo, not added at full strength to a
    # rule held back by l2: a liked close-up still ranks with the liked photos.
    liked_close = np.asarray([s == "close" for s in test_scales]) & (test_y == 1)
    assert rank(by_scale)[liked_close].mean() > 0.5
    unseen = scaled_probability(scaled, test_x[:3], ["aerial"] * 3)  # no aerial was rated
    assert np.isfinite(unseen).all()


def places_liked_unevenly(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within each place you like photos along the first axis; places sit apart along the
    second, and you like more at the places further along it."""
    rng = np.random.default_rng(seed)
    rows, liked, place = [], [], []
    for k in range(6):
        x = rng.normal(size=(40, 8))
        x[:, 1] += 2.0 * k
        share = 0.2 + 0.12 * k
        y = x[:, 0] > np.quantile(x[:, 0], 1 - share)
        rows.append(x)
        liked.append(y)
        place.append(np.full(40, k))
    return (
        np.concatenate(rows).astype(np.float32),
        np.concatenate(liked).astype(np.int8),
        np.concatenate(place),
    )


def same_place_pairs(y: np.ndarray, place: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (i, j)
            for i in np.flatnonzero(y == 1)
            for j in np.flatnonzero(y == 0)
            if place[i] == place[j]
        ],
        dtype=np.intp,
    )


def test_pairs_learn_what_tells_likes_apart_within_a_place_not_which_places_you_like() -> None:
    """A liked photo against a disliked one at the same place: the place's level cancels,
    so the fit learns only what decides within a place, which is what Strongest ranks."""
    x, y, place = places_liked_unevenly()
    pairs = same_place_pairs(y, place)
    paired = fit_pairs(x, pairs, feature="clip@1", l2=10.0)
    rated = fit_taste(x, y, feature="clip@1", l2=10.0)
    assert paired.coef[0] > 0
    assert abs(paired.coef[1]) < 0.2 * paired.coef[0]
    assert rated.coef[1] > 0.2 * rated.coef[0]  # a fit on ratings learns the places too
    assert paired.rated == len(np.unique(pairs))
    assert paired.feature == "clip@1"


def test_pairs_weigh_as_much_as_the_photos_they_come_from() -> None:
    """With many pairs per photo, l2 must hold the weights back as hard as it does for
    ratings: counting every pair twice changes nothing."""
    x, y, place = places_liked_unevenly()
    pairs = same_place_pairs(y, place)
    once = fit_pairs(x, pairs, feature="clip@1")
    twice = fit_pairs(x, np.concatenate([pairs, pairs]), feature="clip@1")
    np.testing.assert_allclose(once.coef, twice.coef, rtol=1e-3, atol=1e-6)


def test_pairs_need_one_pair_and_finite_features() -> None:
    x, y, place = places_liked_unevenly()
    with pytest.raises(ValueError, match="pair"):
        fit_pairs(x, np.zeros((0, 2), dtype=np.intp), feature="clip@1")
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_pairs(bad, same_place_pairs(y, place), feature="clip@1")
