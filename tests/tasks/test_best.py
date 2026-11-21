from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from calico.errors import CapabilityError, InvalidRequestError
from calico.models.calibration import Calibration
from calico.models.taste import TasteModel, TasteWeights
from calico.tasks.base import ReferenceUse
from calico.tasks.best import BestOptions, BestTask
from tests.factories import FakeModels, FeatureLabels, FeatureScore, context, photo, unit, view

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from calico.domain import Group, Photo
    from calico.models.base import Model
    from calico.selection.context import FeatureView, TaskContext

KINDS = (
    "establishing:exterior",
    "establishing:interior",
    "detail:object",
    "detail:texture",
    "portrait",
)


def make_models(*, aesthetic: bool = True, kind: bool = True) -> FakeModels:
    found: dict[str, Model] = {"quality": FeatureScore("quality", "q")}
    if aesthetic:
        found["aesthetic"] = FeatureLabels("aesthetic", "a", ("strong", "weak"))
    if kind:
        found["shot_kind"] = FeatureLabels("shot_kind", "k", KINDS)
    return FakeModels(found)


def pool(
    n: int,
    *,
    quality: Sequence[float] | None = None,
    strong: Sequence[float] | None = None,
    kinds: NDArray[np.float64] | None = None,
    emb: NDArray[np.float32] | None = None,
) -> tuple[list[Photo], FeatureView]:
    """n photos in separate folders (so no bursts), with controllable signals."""
    photos = [photo(f"{i}.jpg", location=f"L{i}") for i in range(n)]
    q = np.linspace(0, 1, n) if quality is None else np.asarray(quality, float)
    s = np.full(n, 0.5) if strong is None else np.asarray(strong, float)
    k = np.full((n, 5), 0.2) if kinds is None else kinds
    e = np.eye(n) + 0.01 if emb is None else emb
    return photos, view(e, {"q": q, "a": np.stack([s, 1 - s], axis=1), "k": k})


def run(
    photos: Sequence[Photo],
    v: FeatureView,
    *,
    options: dict[str, Any] | None = None,
    limit: int = 12,
    models: FakeModels | None = None,
    **kw: Any,
) -> tuple[TaskContext, list[Group]]:
    ctx = context(photos, v, models=models or make_models(), limit=limit, **kw)
    return ctx, BestTask().run(ctx, BestTask().parse_options(options or {}))


def paths(groups: list[Group]) -> list[str]:
    return [p.path.name for p in groups[0].picks]


def test_task_metadata() -> None:
    assert BestTask.name == "best"
    assert BestTask.reference_use is ReferenceUse.UNUSED
    assert BestTask.options_model is BestOptions
    assert BestTask.description


def test_options_reject_unknown_and_out_of_range() -> None:
    with pytest.raises(InvalidRequestError):
        BestTask().parse_options({"nope": 1})
    with pytest.raises(InvalidRequestError):
        BestTask().parse_options({"diversity": 2})


def test_strongest_by_quality_and_aesthetic_come_first() -> None:
    photos, v = pool(4, quality=[0.9, 0.1, 0.5, 0.3], strong=[0.9, 0.1, 0.5, 0.3])
    _, out = run(photos, v, options={"diversity": 0})
    assert paths(out) == ["0.jpg", "2.jpg", "3.jpg", "1.jpg"]


def test_limit_is_respected() -> None:
    photos, v = pool(6)
    _, out = run(photos, v, limit=3)
    assert len(out[0].picks) == 3


def test_kind_filter_keeps_only_that_kind() -> None:
    kinds = np.zeros((4, 5))
    kinds[[0, 2], 4] = 1.0  # portraits
    kinds[[1, 3], 0] = 1.0
    photos, v = pool(4, kinds=kinds)
    _, out = run(photos, v, options={"kind": "portrait"})
    assert sorted(paths(out)) == ["0.jpg", "2.jpg"]
    assert "portrait" in out[0].title


def test_kind_filter_understands_sub_kinds_and_parents() -> None:
    kinds = np.zeros((3, 5))
    kinds[0, 0] = kinds[1, 1] = kinds[2, 4] = 1.0
    photos, v = pool(3, kinds=kinds)
    _, out = run(photos, v, options={"kind": "establishing:interior"})
    assert paths(out) == ["1.jpg"]
    _, out = run(photos, v, options={"kind": "establishing"})
    assert sorted(paths(out)) == ["0.jpg", "1.jpg"]


def test_unknown_kind_lists_the_valid_ones() -> None:
    photos, v = pool(3)
    with pytest.raises(InvalidRequestError, match="portrait"):
        run(photos, v, options={"kind": "selfie"})


def test_kind_filter_without_a_kind_model_cannot_be_honoured() -> None:
    photos, v = pool(3)
    with pytest.raises(CapabilityError):
        run(photos, v, options={"kind": "portrait"}, models=make_models(kind=False))


def test_kind_filter_that_matches_nothing_says_so() -> None:
    kinds = np.zeros((3, 5))
    kinds[:, 0] = 1.0
    photos, v = pool(3, kinds=kinds)
    ctx, out = run(photos, v, options={"kind": "portrait"})
    assert out == []
    assert any("portrait" in n for n in ctx.trace.notes)


def test_criteria_reorders_results() -> None:
    emb = unit([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    photos, v = pool(3, quality=[0.9, 0.5, 0.1], strong=[0.9, 0.5, 0.1], emb=emb)
    _, plain = run(photos, v, options={"diversity": 0})
    _, steered = run(photos, v, options={"diversity": 0}, criteria=[0, 1], describe_weight=0.85)
    assert paths(plain)[0] == "0.jpg"
    assert paths(steered)[0] == "1.jpg"


def test_describe_weight_is_the_descriptions_share_of_the_score() -> None:
    photos, v = pool(3, emb=unit([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]))
    _, out = run(photos, v, criteria=[1, 0], describe_weight=0.3)
    weights = {e.signal: e.weight for e in out[0].picks[0].evidence}
    assert weights["criteria"] == pytest.approx(0.3, abs=1e-3)
    assert weights["quality"] / weights["aesthetic"] == pytest.approx(0.2 / 0.8, rel=1e-2)


def test_quality_is_a_fifth_of_strength_by_default() -> None:
    """Measured on your ratings (20 places, 30 October 2026): quality at 0.2 neither helps
    nor hurts, and at 0.35 it ranks your likes worse."""
    photos, v = pool(3)
    _, out = run(photos, v)
    weights = {e.signal: e.weight for e in out[0].picks[0].evidence}
    assert weights["quality"] == pytest.approx(0.2, abs=1e-3)
    assert weights["aesthetic"] == pytest.approx(0.8, abs=1e-3)


def test_the_description_counts_half_by_default() -> None:
    photos, v = pool(3, emb=unit([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]))
    _, out = run(photos, v, criteria=[1, 0])
    weights = {e.signal: e.weight for e in out[0].picks[0].evidence}
    assert BestTask.default_describe_weight == 0.5
    assert weights["criteria"] == pytest.approx(0.5, abs=1e-3)


def test_a_description_weighted_zero_does_not_count_or_title_the_run() -> None:
    photos, v = pool(3, emb=unit([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]))
    _, out = run(photos, v, criteria=[1, 0], criteria_text="fog", describe_weight=0.0)
    assert "criteria" not in {e.signal for e in out[0].picks[0].evidence}
    assert "fog" not in out[0].title


def test_criteria_weight_is_retired_with_a_pointer_to_describe_weight() -> None:
    with pytest.raises(InvalidRequestError, match="describe_weight"):
        BestTask().parse_options({"criteria_weight": 1.0})


def test_missing_aesthetic_degrades_with_a_note() -> None:
    photos, v = pool(4, quality=[0.9, 0.1, 0.5, 0.3])
    ctx, out = run(photos, v, models=make_models(aesthetic=False), options={"diversity": 0})
    assert paths(out)[0] == "0.jpg"
    assert any("aesthetic" in n for n in ctx.trace.notes)
    signals = {e.signal: e.weight for e in out[0].picks[0].evidence}
    assert signals == {"quality": pytest.approx(1.0)}


def test_no_usable_signal_is_an_error() -> None:
    photos, v = pool(3)
    with pytest.raises(CapabilityError):
        run(photos, v, models=FakeModels())


def test_one_pick_per_burst_with_alternates() -> None:
    t0 = datetime(2024, 1, 1, 10)
    photos = [photo(f"{i}.jpg", captured_at=t0 + timedelta(seconds=i)) for i in range(3)]
    photos.append(photo("far.jpg", captured_at=t0 + timedelta(hours=5)))
    v = view(
        [[1, 0], [1, 0], [1, 0], [0, 1]],
        {"q": [0.2, 0.9, 0.5, 0.4], "a": [[0.5, 0.5]] * 4},
    )
    _, out = run(photos, v)
    first = next(p for p in out[0].picks if p.path.name == "1.jpg")
    assert first.burst_size == 3
    assert [a.name for a in first.alternates] == ["2.jpg", "0.jpg"]
    assert len(out[0].picks) == 2


def test_why_a_photo_was_left_out_is_recorded_for_runs_explain() -> None:
    """A kind filter, a burst mate and the strength ranking each leave photos out; each
    leaves its reason in the trace."""
    t0 = datetime(2024, 1, 1, 10)
    photos = [photo(f"{i}.jpg", captured_at=t0 + timedelta(seconds=i)) for i in range(3)]
    photos.append(photo("far.jpg", captured_at=t0 + timedelta(hours=5)))
    kinds = np.zeros((4, 5))
    kinds[:3, 4] = 1.0  # the burst is portraits
    v = view(
        [[1, 0], [1, 0], [1, 0], [0, 1]],
        {"q": [0.2, 0.9, 0.5, 0.4], "a": [[0.5, 0.5]] * 4, "k": kinds},
    )
    ctx, _ = run(photos, v, options={"kind": "portrait"})
    columns = ctx.trace.columns
    assert columns["kind_probability"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert columns["burst_lead"][:3].tolist() == [0.0, 1.0, 0.0]  # 1.jpg leads its burst
    assert columns["strength_rank"][1] == 1.0


def test_reference_is_never_returned() -> None:
    photos, v = pool(4, quality=[0.99, 0.5, 0.4, 0.3])
    refs = view(unit([[1.0, 0.0, 0.0, 0.0]]), {})
    _, out = run(photos, v, references=refs, reference_keys=[photos[0].content_key])
    assert "0.jpg" not in paths(out)


def test_evidence_sums_to_the_pick_score() -> None:
    photos, v = pool(5, emb=unit(np.random.default_rng(0).normal(size=(5, 4))))
    _, out = run(photos, v, criteria=[1, 0, 0, 0])
    for pick in out[0].picks:
        assert sum(e.contribution for e in pick.evidence) == pytest.approx(pick.score, abs=2e-3)
        assert sum(e.weight for e in pick.evidence) == pytest.approx(1.0, abs=2e-3)
    assert out[0].score == pytest.approx(np.mean([p.score for p in out[0].picks]))


def test_diversity_avoids_near_duplicates() -> None:
    emb = unit([[1.0, 0.0], [0.999, 0.04], [0.0, 1.0]])
    photos, v = pool(3, quality=[0.9, 0.8, 0.5], strong=[0.9, 0.8, 0.5], emb=emb)
    _, plain = run(photos, v, limit=2, options={"diversity": 0})
    _, diverse = run(photos, v, limit=2, options={"diversity": 0.9})
    assert paths(plain) == ["0.jpg", "1.jpg"]
    assert paths(diverse) == ["0.jpg", "2.jpg"]


def test_picks_are_shown_strongest_first_though_variety_chose_them() -> None:
    emb = unit([[1.0, 0.0], [0.999, 0.04], [0.0, 1.0]])
    photos, v = pool(3, quality=[0.9, 0.8, 0.5], strong=[0.9, 0.8, 0.5], emb=emb)
    _, out = run(photos, v, limit=3, options={"diversity": 0.9})
    assert paths(out) == ["0.jpg", "1.jpg", "2.jpg"]  # variety chose 0, 2, 1
    scores = [p.score for g in out for p in g.picks]
    assert scores == sorted(scores, reverse=True)
    _, short = run(photos, v, limit=2, options={"diversity": 0.9})
    assert sorted(paths(short)) == ["0.jpg", "2.jpg"]  # which photos are chosen is still variety's


def test_equally_strong_picks_are_shown_in_path_order() -> None:
    emb = unit([[1.0, 0.0], [0.999, 0.04], [0.0, 1.0]])
    photos, v = pool(3, quality=[0.7] * 3, strong=[0.7] * 3, emb=emb)
    _, out = run(photos, v, limit=3, options={"diversity": 0.9})
    assert paths(out) == ["0.jpg", "1.jpg", "2.jpg"]  # variety chose 0, 2, 1


SAME_AS_YOURS = Calibration(1.0, 0.0, photos=500, liked=380, places=9)
"""The identity: a photo Yours gives 90% is one you liked 90% of the time."""


def yours(
    p: Sequence[float],
    v: FeatureView,
    *,
    calibration: Calibration | None = SAME_AS_YOURS,
    learned: Sequence[str] = (),
) -> FakeModels:
    """Yours as the aesthetic, giving photo i the probability ``p[i]`` (solved from the pool's
    embeddings, one axis per photo), calibrated as given (the identity: what it says is what
    you like)."""
    n = len(p)
    logits = np.log(np.asarray(p) / (1 - np.asarray(p)))
    taste = TasteWeights(
        feature="clip@1",
        extractor="",
        mean=np.zeros(n, dtype=np.float32),
        scale=np.ones(n, dtype=np.float32),
        coef=np.linalg.solve(np.asarray(v.embeddings, dtype=np.float64), logits).astype(np.float32),
        bias=0.0,
        rated=100,
        liked=70,
        places=tuple(learned),
        calibration=calibration,
    )
    models = make_models()
    return FakeModels(
        {**{k: models.get(k) for k in models.names()}, "aesthetic": TasteModel(taste)}
    )


LIKELY = [0.95, 0.9, 0.85, 0.6, 0.4, 0.2]


def line(groups: list[Group]) -> int | None:
    """Where the first group's list stops being good: the picks before this index are above it."""
    return None if groups[0].marker is None else groups[0].marker.before


def test_yours_says_how_likely_you_are_to_like_each_pick_and_where_the_list_stops_being_good() -> (
    None
):
    photos, v = pool(6, quality=LIKELY, strong=LIKELY)
    _, out = run(photos, v, limit=6, models=yours(LIKELY, v), options={"diversity": 0})
    picks = out[0].picks
    assert [pick.probability for pick in picks] == pytest.approx(LIKELY, abs=1e-3)
    assert line(out) == 3  # the first pick you are less than 80% likely to like
    assert out[0].marker is not None
    assert out[0].marker.probability == 0.8
    _, laxer = run(
        photos,
        v,
        limit=6,
        models=yours(LIKELY, v),
        options={"diversity": 0, "marker_probability": 0.5},
    )
    assert line(laxer) == 4
    _, all_above = run(
        photos,
        v,
        limit=6,
        models=yours(LIKELY, v),
        options={"diversity": 0, "marker_probability": 0.1},
    )
    assert line(all_above) == 6  # every pick is above the line


def test_general_says_nothing_of_how_likely_you_are_to_like_a_pick() -> None:
    photos, v = pool(6, quality=LIKELY, strong=LIKELY)
    _, out = run(photos, v, limit=6, options={"diversity": 0})
    assert all(pick.probability is None for pick in out[0].picks)
    assert out[0].marker is None


def test_yours_without_a_calibration_says_so_and_gives_no_probability() -> None:
    photos, v = pool(6, quality=LIKELY, strong=LIKELY)
    ctx, out = run(photos, v, limit=6, models=yours(LIKELY, v, calibration=None))
    assert all(pick.probability is None for pick in out[0].picks)
    assert out[0].marker is None
    assert any("calico train taste" in note for note in ctx.trace.notes)


def test_a_pick_at_a_place_yours_learned_has_no_probability_and_does_not_move_the_line() -> None:
    photos, v = pool(6, quality=LIKELY, strong=LIKELY)
    _, out = run(
        photos, v, limit=6, models=yours(LIKELY, v, learned=["L0", "L3"]), options={"diversity": 0}
    )
    by_name = {pick.path.name: pick.probability for pick in out[0].picks}
    assert by_name["0.jpg"] is None  # it scores its own training photos too high to be read
    assert by_name["3.jpg"] is None
    assert by_name["1.jpg"] == pytest.approx(0.9, abs=1e-3)
    assert line(out) == 4  # the first pick with a probability below the line is 4.jpg (0.4)


def test_a_description_reorders_the_list_so_no_line_is_drawn() -> None:
    photos, v = pool(6, quality=LIKELY, strong=LIKELY)
    ctx, out = run(
        photos,
        v,
        limit=6,
        models=yours(LIKELY, v),
        criteria=[0, 1, 0, 0, 0, 0],
        describe_weight=0.5,
    )
    assert any(pick.probability is not None for pick in out[0].picks)  # still how likely each is
    assert out[0].marker is None
    assert any("description" in note for note in ctx.trace.notes)


def test_the_probability_is_traced_for_every_photo_considered() -> None:
    photos, v = pool(6, quality=LIKELY, strong=LIKELY)
    ctx, _ = run(photos, v, limit=3, models=yours(LIKELY, v))
    traced = ctx.trace.columns["probability"]
    assert traced == pytest.approx(LIKELY, abs=1e-3)


def strong_look_alikes_and_one_weak_stranger(n: int = 10) -> tuple[list[Photo], FeatureView]:
    """``n`` strong photos of one subject, then one weak photo unlike any of them."""
    dim = n + 2
    emb = [np.eye(dim)[0] + 0.3 * np.eye(dim)[1 + i] for i in range(n)] + [np.eye(dim)[-1]]
    strength = [*np.linspace(0.9, 0.8, n), 0.1]
    return pool(n + 1, quality=strength, strong=strength, emb=unit(np.stack(emb)))


def test_diversity_never_reaches_down_to_weak_photos() -> None:
    photos, v = strong_look_alikes_and_one_weak_stranger()
    _, out = run(photos, v, limit=3, options={"diversity": 0.9})
    assert "10.jpg" not in paths(out)


def test_diversity_reach_says_how_far_down_the_ranking_diversity_may_choose() -> None:
    photos, v = strong_look_alikes_and_one_weak_stranger()
    _, far = run(photos, v, limit=3, options={"diversity": 0.9, "diversity_reach": 4})
    assert "10.jpg" in paths(far)  # within the 12 strongest, the stranger is the most different
    assert BestTask().parse_options({}).diversity_reach == 3


def test_signals_and_stages_are_traced_aligned_with_the_pool() -> None:
    photos, v = pool(4)
    kinds = np.zeros((4, 5))
    kinds[:3, 4] = 1.0
    kinds[3, 0] = 1.0
    photos, v = pool(4, kinds=kinds)
    ctx, _ = run(photos, v, options={"kind": "portrait"})
    for name in ("quality", "aesthetic", "score"):
        assert len(ctx.trace.columns[name]) == 4
    assert np.isnan(ctx.trace.columns["score"][3])
    assert not np.isnan(ctx.trace.columns["score"][0])
    assert ctx.trace.stages


def test_empty_pool_returns_nothing() -> None:
    ctx, out = run([], view(np.zeros((0, 2)), {"q": np.zeros(0), "a": np.zeros((0, 2))}))
    assert out == []
    assert ctx.trace.notes


def test_title_mentions_kind_and_criteria() -> None:
    kinds = np.zeros((3, 5))
    kinds[:, 4] = 1.0
    photos, v = pool(3, kinds=kinds)
    _, out = run(photos, v, options={"kind": "portrait"}, criteria=[1, 0, 0], criteria_text="snow")
    assert out[0].title == "Strongest portrait photographs matching “snow”"


def test_output_is_deterministic() -> None:
    photos, v = pool(6, quality=[0.5] * 6, strong=[0.5] * 6)
    assert run(photos, v)[1] == run(photos, v)[1]


def test_nan_quality_never_wins_and_is_noted() -> None:
    photos, v = pool(3, quality=[np.nan, 0.4, 0.6])
    ctx, out = run(photos, v, options={"diversity": 0, "aesthetic_weight": 0})
    assert paths(out) == ["2.jpg", "1.jpg", "0.jpg"]
    assert any("quality returned 1 invalid scores; treated as 0" in n for n in ctx.trace.notes)


def test_group_score_is_the_mean_pick_score() -> None:
    photos, v = pool(4)
    _, out = run(photos, v)
    assert out[0].score == pytest.approx(np.mean([p.score for p in out[0].picks]))


def test_ui_metadata() -> None:
    assert BestTask.default_limit == 24
    assert (BestTask.reference_label, BestTask.reference_help) == ("", "")


def test_fewer_distinct_moments_than_asked_for_are_noted() -> None:
    """Audit A03-15 (a): 24 asked for, bursts collapse to fewer, and nothing said why."""
    t0 = datetime(2023, 5, 1, 12, 0, 0)
    photos = [
        photo(f"b{i}.jpg", location="Mill", captured_at=t0 + timedelta(seconds=i)) for i in range(3)
    ]
    photos.append(photo("solo.jpg", location="Elsewhere"))
    e = np.vstack([np.tile([1.0, 0, 0], (3, 1)), [[0, 1.0, 0]]])
    v = view(e, {"q": [0.5, 0.9, 0.7, 0.4], "a": [[0.5, 0.5]] * 4})
    ctx, out = run(photos, v, limit=24)
    assert len(out[0].picks) == 2
    assert any("Only 2 distinct moments" in n for n in ctx.trace.notes)


def test_zero_quality_and_aesthetic_weights_with_a_description_are_noted() -> None:
    """Audit A03-15 (b): both zero meant "the description alone", silently."""
    photos, v = pool(4)
    ctx, out = run(
        photos, v, options={"quality_weight": 0, "aesthetic_weight": 0}, criteria=[1, 0, 0, 0]
    )
    assert out
    assert any("ranked by the description alone" in n for n in ctx.trace.notes)


def test_zero_quality_and_aesthetic_weights_alone_say_how_to_fix_it() -> None:
    photos, v = pool(4)
    with pytest.raises(InvalidRequestError, match="Set either above zero"):
        run(photos, v, options={"quality_weight": 0, "aesthetic_weight": 0})


# -- one pick per view ----------------------------------------------------------------

DAY = timedelta(days=1)
T0 = datetime(2023, 6, 10, 12, 0, 0)


def views_pool(
    shots: Sequence[tuple[str, str, float, int]],
) -> tuple[list[Photo], FeatureView]:
    """``(name, direction, strength, day)`` per photo: photos sharing a direction look alike
    as a whole (cosine about 0.98), and each is on its own day, so no two are one burst."""
    directions = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
    photos = [photo(name, location="Mill", captured_at=T0 + day * DAY) for name, _, _, day in shots]
    e = np.array(
        [
            np.add(directions[d], 0.1 * i * np.array([0, 0, 1.0]))
            for i, (_, d, _, _) in enumerate(shots)
        ]
    )
    s = np.array([strength for _, _, strength, _ in shots])
    return photos, view(unit(e), {"q": s, "a": np.stack([s, 1 - s], axis=1)})


def run_views(
    photos: Sequence[Photo],
    v: FeatureView,
    share: dict[frozenset[str], float],
    *,
    options: dict[str, Any] | None = None,
    limit: int = 12,
) -> tuple[TaskContext, list[Group], NDArray[np.intp] | None]:
    """Run ``best`` as the engine does: ask which pairs to compare, give each the share in
    ``share`` (by the two names; 0.05 when not given), then run."""
    task = BestTask()
    opts = task.parse_options({"diversity": 0, **(options or {})})
    ctx = context(photos, v, models=make_models(), limit=limit)
    pairs = task.patch_pairs(ctx, opts)
    if pairs is not None:
        shares = np.asarray(
            [
                share.get(frozenset((photos[a].path.name, photos[b].path.name)), 0.05)
                for a, b in pairs
            ],
            dtype=np.float32,
        )
        ctx = dataclasses.replace(ctx, compared_pairs=pairs, pair_shares=shares)
    return ctx, task.run(ctx, opts), pairs


def test_a_view_shot_on_two_occasions_is_one_pick_with_the_other_as_its_alternate() -> None:
    photos, v = views_pool(
        [("june.jpg", "x", 0.9, 0), ("april.jpg", "x", 0.8, 30), ("hall.jpg", "y", 0.5, 1)]
    )
    ctx, out, _ = run_views(photos, v, {frozenset(("june.jpg", "april.jpg")): 0.5}, limit=2)
    assert paths(out) == ["june.jpg", "hall.jpg"]
    first = out[0].picks[0]
    assert [p.name for p in first.alternates] == ["april.jpg"]
    assert first.burst_size == 2
    assert any("one pick per view" in n for n in ctx.trace.notes)


def test_look_alike_rooms_that_share_few_details_stay_two_picks() -> None:
    photos, v = views_pool(
        [("bay1.jpg", "x", 0.9, 0), ("bay2.jpg", "x", 0.8, 1), ("hall.jpg", "y", 0.5, 1)]
    )
    _, out, _ = run_views(photos, v, {frozenset(("bay1.jpg", "bay2.jpg")): 0.05}, limit=2)
    assert paths(out) == ["bay1.jpg", "bay2.jpg"]


def test_only_look_alikes_at_one_location_among_the_strongest_are_compared() -> None:
    photos, v = views_pool(
        [
            ("a.jpg", "x", 0.9, 0),
            ("b.jpg", "x", 0.8, 3),
            ("c.jpg", "y", 0.7, 5),
            ("weak.jpg", "x", 0.1, 9),
        ]
    )
    photos[1] = photo("b.jpg", location="Elsewhere", captured_at=photos[1].captured_at)
    _, _, pairs = run_views(photos, v, {}, options={"diversity_reach": 1}, limit=1)
    assert pairs is not None
    # b is at another location; c looks like nothing; weak is outside the strongest 2 x 1 x 1.
    assert pairs.tolist() == []
    _, _, pairs = run_views(photos, v, {}, options={"diversity_reach": 2}, limit=2)
    assert pairs is not None
    assert {frozenset((photos[a].path.name, photos[b].path.name)) for a, b in pairs} == {
        frozenset(("a.jpg", "weak.jpg"))
    }


def test_one_per_view_can_be_turned_off() -> None:
    photos, v = views_pool([("june.jpg", "x", 0.9, 0), ("april.jpg", "x", 0.8, 30)])
    _, out, pairs = run_views(
        photos, v, {frozenset(("june.jpg", "april.jpg")): 0.5}, options={"one_per_view": False}
    )
    assert pairs is None
    assert paths(out) == ["june.jpg", "april.jpg"]


def test_views_that_cannot_be_compared_part_by_part_are_noted() -> None:
    photos, v = views_pool([("june.jpg", "x", 0.9, 0), ("april.jpg", "x", 0.8, 30)])
    ctx, out = run(photos, v, options={"diversity": 0})
    assert paths(out) == ["june.jpg", "april.jpg"]
    assert any("may be shown more than once" in n for n in ctx.trace.notes)


def test_a_folded_frame_says_a_stronger_frame_of_its_view_was_shown() -> None:
    photos, v = views_pool([("june.jpg", "x", 0.9, 0), ("april.jpg", "x", 0.8, 30)])
    ctx, _, _ = run_views(photos, v, {frozenset(("june.jpg", "april.jpg")): 0.5})
    lead = ctx.trace.columns["view_lead"]
    assert lead.tolist() == [1.0, 0.0]


def test_the_next_view_moves_up_when_frames_fold() -> None:
    """Folding frees places on the first page for different views, not empty slots."""
    photos, v = views_pool(
        [
            ("x1.jpg", "x", 0.9, 0),
            ("x2.jpg", "x", 0.85, 10),
            ("x3.jpg", "x", 0.8, 20),
            ("z.jpg", "z", 0.3, 2),
        ]
    )
    together = {
        frozenset(p): 0.5
        for p in (("x1.jpg", "x2.jpg"), ("x1.jpg", "x3.jpg"), ("x2.jpg", "x3.jpg"))
    }
    _, out, _ = run_views(photos, v, together, limit=2)
    assert paths(out) == ["x1.jpg", "z.jpg"]
    assert [p.name for p in out[0].picks[0].alternates] == ["x2.jpg", "x3.jpg"]


def test_look_alikes_are_found_block_by_block_with_the_same_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wide reach (up to 2 x 100 x 500 photos) must not build one matrix of every pair."""
    shots = [(f"{i}.jpg", "xyz"[i % 3], 0.9 - 0.01 * i, i) for i in range(9)]
    photos, v = views_pool(shots)
    _, _, whole = run_views(photos, v, {})
    monkeypatch.setattr("calico.tasks.best._BLOCK", 2)
    _, _, blocked = run_views(photos, v, {})
    assert whole is not None
    assert blocked is not None
    assert len(whole) > 0
    assert sorted(map(tuple, whole.tolist())) == sorted(map(tuple, blocked.tolist()))


# -- survivors of the mutation run, 10 November 2026 --------------------


def test_a_view_needs_the_share_the_view_share_option_asks_for() -> None:
    photos, v = views_pool(
        [("june.jpg", "x", 0.9, 0), ("april.jpg", "x", 0.8, 30), ("hall.jpg", "y", 0.5, 1)]
    )
    share = {frozenset(("june.jpg", "april.jpg")): 0.5}
    _, one_view, _ = run_views(photos, v, share, limit=2)
    assert paths(one_view) == ["june.jpg", "hall.jpg"]  # 0.5 is above the default 0.17
    _, two_views, _ = run_views(photos, v, share, options={"view_share": 0.9}, limit=2)
    assert paths(two_views) == ["june.jpg", "april.jpg"]  # and below 0.9


def test_only_look_alikes_as_alike_as_view_similarity_asks_are_compared() -> None:
    photos, v = views_pool([("june.jpg", "x", 0.9, 0), ("april.jpg", "x", 0.8, 30)])
    _, _, pairs = run_views(photos, v, {})
    assert pairs is not None
    assert len(pairs) == 1
    _, _, strict = run_views(photos, v, {}, options={"view_similarity": 0.999})
    assert strict is not None
    assert strict.tolist() == []


def test_frames_are_one_burst_within_two_minutes_and_nine_tenths_alike_by_default() -> None:
    options = BestTask().parse_options({})
    assert (options.burst_window_seconds, options.burst_similarity) == (120, 0.9)


def test_equal_strengths_are_picked_in_path_order_whatever_the_order_they_were_given_in() -> None:
    _, v = pool(3, quality=[0.5] * 3, strong=[0.5] * 3)
    photos = [
        photo("c.jpg", location="L2"),
        photo("b.jpg", location="L1"),
        photo("a.jpg", location="L0"),
    ]
    _, out = run(photos, v, options={"diversity": 0}, limit=2)
    assert paths(out) == ["a.jpg", "b.jpg"]
