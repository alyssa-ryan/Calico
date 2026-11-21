from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from calico.errors import InvalidRequestError
from calico.selection.anchors import anchor_prior
from calico.tasks.base import ReferenceUse
from calico.tasks.triptych import NOT_CONNECTIONS, Connection, TriptychOptions, TriptychTask
from tests.factories import FakeModels, FeatureLabels, FeatureScore, context, photo, unit, view

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from calico.domain import Group, Photo
    from calico.models.base import Model
    from calico.selection.context import FeatureView, TaskContext

KINDS = ("establishing:exterior", "establishing:interior", "detail:object", "portrait")
MOODS = ("soft_light", "harsh_sun")


def make_models(*, mood: bool = False, kind: bool = False, quality: bool = True) -> FakeModels:
    found: dict[str, Model] = {}
    if quality:
        found["quality"] = FeatureScore("quality", "q")
    if mood:
        found["mood"] = FeatureLabels("mood", "m", MOODS)
    if kind:
        found["shot_kind"] = FeatureLabels("shot_kind", "k", KINDS)
    return FakeModels(found)


def world(
    n: int,
    *,
    locations: Sequence[str] | None = None,
    quality: Sequence[float] | None = None,
    emb: NDArray[np.float32] | None = None,
    extra: dict[str, NDArray[np.float64]] | None = None,
) -> tuple[list[Photo], FeatureView]:
    """n photos with mutually dissimilar embeddings unless ``emb`` says otherwise."""
    locs = locations or [f"L{i}" for i in range(n)]
    photos = [photo(f"{i:02d}.jpg", location=locs[i]) for i in range(n)]
    e = np.eye(n) + 0.001 if emb is None else emb
    q = np.linspace(0.2, 0.8, n) if quality is None else np.asarray(quality, float)
    return photos, view(e, {"q": q, **(extra or {})})


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
    return ctx, TriptychTask().run(ctx, TriptychTask().parse_options(options or {}))


def keys(group: Group) -> list[str]:
    return [p.content_key for p in group.picks]


def test_task_metadata() -> None:
    assert TriptychTask.name == "triptych"
    assert TriptychTask.reference_use is ReferenceUse.OPTIONAL
    assert TriptychTask.options_model is TriptychOptions
    assert TriptychTask.description
    defaults = TriptychTask().parse_options({})
    assert defaults.connections == (
        Connection.THEME,
        Connection.PLACE_TYPE,
        Connection.MOOD,
        Connection.COLOR,
    )
    assert (defaults.candidates, defaults.too_alike) == (1000, 0.7)


def test_connections_must_not_be_empty() -> None:
    with pytest.raises(InvalidRequestError):
        TriptychTask().parse_options({"connections": []})


def test_each_triptych_has_three_photos() -> None:
    photos, v = world(9)
    _, out = run(photos, v, options={"connections": ["theme"]})
    assert out
    assert all(len(g.picks) == 3 for g in out)


def test_location_connection_ignores_capitals_and_spaces() -> None:
    locs = ["Old Mill", "old  mill", "OLD MILL", "B", "C", "D"]
    photos, v = world(6, locations=locs)
    _, out = run(photos, v, limit=1, options={"connections": ["location"], "strength_weight": 0.0})
    assert {p.location.casefold().replace("  ", " ") for p in out[0].picks} == {"old mill"}
    ev = {e.signal: e for e in out[0].evidence}
    assert ev["location"].value == pytest.approx(1.0)


def test_location_connection_pairs_same_location_photos() -> None:
    locs = ["A", "A", "A", "B", "B", "B"]
    photos, v = world(6, locations=locs)
    _, out = run(photos, v, limit=1, options={"connections": ["location"], "strength_weight": 0.0})
    assert len({p.location for p in out[0].picks}) == 1
    ev = {e.signal: e for e in out[0].evidence}
    assert ev["location"].value == pytest.approx(1.0)
    assert "same location" in ev["location"].detail


def test_near_duplicates_never_share_a_triptych() -> None:
    emb = np.eye(6)
    emb[1] = emb[0] * 0.999 + 0.04 * np.eye(6)[1]  # photo 1 ~ photo 0
    photos, v = world(6, emb=unit(emb))
    _, out = run(photos, v, options={"connections": ["theme"], "burst_window_seconds": 0})
    sims = unit(emb) @ unit(emb).T
    assert sims[0, 1] > 0.95
    for g in out:
        idx = [int(Path(p.path).stem) for p in g.picks]
        assert all(sims[i, j] < 0.95 for i, j in combinations(idx, 2))


def look_alikes_and_relatives(n_alike: int = 3, others: int = 15) -> NDArray[np.float32]:
    """``n_alike`` photos 0.9 alike (one moment), two related to them at ~0.73, then photos
    unrelated to anything, so every percentile among the first few is high."""
    dim = 1 + n_alike + 2 + others
    rows = [
        np.sqrt(0.9) * np.eye(dim)[0] + np.sqrt(0.1) * np.eye(dim)[1 + i] for i in range(n_alike)
    ]
    rows += [
        np.sqrt(0.6) * np.eye(dim)[0] + np.sqrt(0.4) * np.eye(dim)[1 + n_alike + i]
        for i in range(2)
    ]
    rows += [np.eye(dim)[3 + n_alike + i] for i in range(others)]
    return unit(np.stack(rows))


THEME_ONLY = {"connections": ["theme"], "strength_weight": 0.0, "burst_window_seconds": 0}
DISTINCT = TriptychOptions().distinct_weight
"""What distinct takes of every story's score by default."""


def test_three_frames_of_one_moment_are_not_a_story() -> None:
    emb = look_alikes_and_relatives()
    photos, v = world(len(emb), emb=emb, quality=np.linspace(0.9, 0.1, len(emb)).tolist())
    _, out = run(photos, v, limit=1, options=THEME_ONLY)
    assert sum(int(p.path.stem) < 3 for p in out[0].picks) <= 2
    _, alike = run(photos, v, limit=1, options={**THEME_ONLY, "variety_weight": 0.0})
    assert sorted(int(p.path.stem) for p in alike[0].picks) == [0, 1, 2]  # as before variety


def test_an_anchor_with_many_look_alikes_still_finds_varied_companions() -> None:
    emb = look_alikes_and_relatives(n_alike=6)
    photos, v = world(len(emb), emb=emb)
    _, out = run(
        photos,
        v,
        limit=1,
        options={**THEME_ONLY, "companions": 3},
        references=view(v.embeddings[[0]], {}),
        reference_keys=[photos[0].content_key],
    )
    assert any(int(p.path.stem) in {6, 7} for p in out[0].picks)


def test_variety_is_evidence_that_says_how_alike_the_three_are() -> None:
    emb = look_alikes_and_relatives()
    photos, v = world(len(emb), emb=emb)
    _, out = run(photos, v, limit=1, options={**THEME_ONLY, "strength_weight": 0.4})
    ev = {e.signal: e for e in out[0].evidence}
    rest = 1 - DISTINCT  # what distinct leaves the rest
    assert ev["variety"].weight == pytest.approx(rest * 0.6 * 0.25)
    assert ev["theme"].weight == pytest.approx(rest * 0.6 * 0.75)
    assert 0.0 <= ev["variety"].value <= 1.0
    assert "alike" in ev["variety"].detail
    assert [e.signal for e in out[0].evidence][-2:] == ["variety", "distinct"]


def test_without_variety_weight_no_variety_evidence_is_shown() -> None:
    photos, v = world(6)
    _, out = run(photos, v, options={"connections": ["theme"], "variety_weight": 0.0})
    assert "variety" not in {e.signal for e in out[0].evidence}


def test_the_scorer_gives_the_tasks_own_stories_their_scores() -> None:
    emb = look_alikes_and_relatives()
    photos, v = world(len(emb), emb=emb, quality=np.linspace(0.9, 0.1, len(emb)).tolist())
    options = {"connections": ["theme"], "burst_window_seconds": 0}
    _, out = run(photos, v, limit=3, options=options)
    ctx = context(photos, v, models=make_models(), limit=3)
    scorer = TriptychTask().scorer(ctx, TriptychTask().parse_options(options), include=[])
    assert out
    for g in out:
        rows = [scorer.row(k) for k in keys(g)]
        assert scorer.score(*rows) == pytest.approx(g.score, abs=1e-6)
        assert scorer.allowed(*rows)


def test_the_scorer_scores_photos_outside_the_candidates_and_says_so() -> None:
    photos, v = world(10)  # quality rises with the index: 00.jpg is the weakest
    ctx = context(photos, v, models=make_models(), limit=3)
    options = TriptychTask().parse_options({"connections": ["theme"], "candidates": 3})
    scorer = TriptychTask().scorer(ctx, options, include=[photos[0].content_key])
    weakest, strongest = scorer.row(photos[0].content_key), scorer.row(photos[9].content_key)
    assert not scorer.shortlisted[weakest]
    assert scorer.shortlisted[strongest]
    assert len(scorer.keys) == 4


def test_the_scorer_never_allows_near_duplicates() -> None:
    emb = np.eye(5)
    emb[1] = emb[0] * 0.999 + 0.04 * np.eye(5)[1]
    photos, v = world(5, emb=unit(emb))
    ctx = context(photos, v, models=make_models(), limit=3)
    scorer = TriptychTask().scorer(ctx, TriptychTask().parse_options({}), include=[])
    rows = [scorer.row(p.content_key) for p in photos]
    assert not scorer.allowed(rows[0], rows[1], rows[2])
    assert scorer.allowed(rows[0], rows[2], rows[3])


def test_max_uses_one_means_no_photo_is_reused() -> None:
    photos, v = world(10)
    _, out = run(photos, v, options={"connections": ["theme"], "max_uses": 1})
    used = [k for g in out for k in keys(g)]
    assert len(used) == len(set(used))


def test_max_uses_allows_reuse_up_to_the_limit() -> None:
    photos, v = world(10)
    _, once = run(photos, v, options={"connections": ["theme"], "max_uses": 1})
    _, twice = run(photos, v, options={"connections": ["theme"], "max_uses": 2})
    counts: dict[str, int] = {}
    for g in twice:
        for k in keys(g):
            counts[k] = counts.get(k, 0) + 1
    assert max(counts.values()) <= 2
    assert len(twice) >= len(once)


def twins() -> tuple[list[Photo], FeatureView]:
    """Twelve photos at two places; by the view embedder they are six pairs of one view."""
    looks = np.repeat(np.eye(6), 2, axis=0) + 0.001
    return world(12, locations=["Mill"] * 6 + ["Harbor"] * 6, extra={"v": looks})


def views_shared(out: Sequence[Group]) -> int:
    """How many stories show a view an earlier story of the set showed."""
    seen: set[int] = set()
    shared = 0
    for g in out:
        views = {int(p.path.stem) // 2 for p in g.picks}
        shared += bool(views & seen)
        seen |= views
    return shared


def test_spend_alike_spends_the_other_frames_of_a_used_view() -> None:
    """Sets of stories repeated a subject because only the exact photo was spent
    (improvement loop, 14 November 2026): with ``spend_alike`` a used photo takes its
    look-alikes at the same place with it."""
    photos, v = twins()
    options = {"connections": ["theme"], "distinct_weight": 0.0}
    _, today = run(photos, v, options=options, view_feature="v")
    assert views_shared(today) > 0
    _, spent = run(photos, v, options={**options, "spend_alike": 0.9}, view_feature="v")
    assert views_shared(spent[:2]) == 0  # six views: two stories without a shared one
    assert views_shared(spent) < views_shared(today)


def test_spend_alike_still_makes_as_many_stories_as_without_it() -> None:
    """Its first form made 4 stories where 12 were asked for (attempt 001, 14 November
    2026): once nothing else can make a story, look-alikes come back into play, and the
    trace says how many stories that let in."""
    photos, v = twins()
    options = {"connections": ["theme"], "distinct_weight": 0.0}
    _, today = run(photos, v, options=options, view_feature="v")
    ctx, spent = run(photos, v, options={**options, "spend_alike": 0.9}, view_feature="v")
    assert len(spent) == len(today) > 2
    used = [k for g in spent for k in keys(g)]
    assert len(used) == len(set(used))  # max_uses still holds
    assert [g.title.split(":")[0] for g in spent] == [f"Story {n + 1}" for n in range(len(spent))]
    assert any(
        "spend_alike" in n and f"{len(spent) - 2} of {len(spent)}" in n for n in ctx.trace.notes
    )
    few_ctx, few = run(
        photos, v, options={**options, "spend_alike": 0.9}, view_feature="v", limit=2
    )
    assert len(few) == 2
    assert not [n for n in few_ctx.trace.notes if "spend_alike" in n]


def test_spend_alike_is_off_by_default_and_changes_nothing_then() -> None:
    photos, v = twins()
    options = {"connections": ["theme"], "distinct_weight": 0.0}
    assert TriptychTask().parse_options({}).spend_alike == 0
    _, today = run(photos, v, options=options, view_feature="v")
    _, off = run(photos, v, options={**options, "spend_alike": 0.0}, view_feature="v")
    assert [keys(g) for g in off] == [keys(g) for g in today]


def test_spend_alike_does_not_reach_another_place() -> None:
    looks = np.repeat(np.eye(6), 2, axis=0) + 0.001
    photos, v = world(12, locations=["Mill", "Harbor"] * 6, extra={"v": looks})
    options = {"connections": ["theme"], "distinct_weight": 0.0}
    _, today = run(photos, v, options=options, view_feature="v")
    _, spent = run(photos, v, options={**options, "spend_alike": 0.9}, view_feature="v")
    assert [keys(g) for g in spent] == [keys(g) for g in today]


def test_spend_alike_says_so_when_the_view_embedder_is_missing() -> None:
    photos, v = world(12, locations=["Mill"] * 12)
    options = {"connections": ["theme"], "distinct_weight": 0.0}
    _, today = run(photos, v, options=options, view_feature="v")
    ctx, out = run(photos, v, options={**options, "spend_alike": 0.9}, view_feature="v")
    assert [keys(g) for g in out] == [keys(g) for g in today]
    assert any("spend_alike" in n and "calico index" in n for n in ctx.trace.notes)


def test_limit_caps_the_number_of_triptychs() -> None:
    photos, v = world(12)
    _, out = run(photos, v, limit=2, options={"connections": ["theme"]})
    assert len(out) == 2


def test_in_pool_reference_becomes_the_middle_pick() -> None:
    photos, v = world(8)
    refs = view(v.embeddings[3:4], {})
    _, out = run(
        photos,
        v,
        options={"connections": ["theme"]},
        references=refs,
        reference_keys=[photos[3].content_key],
    )
    assert out
    for g in out:
        assert g.picks[1].content_key == photos[3].content_key
        assert photos[3].content_key not in {k for i, k in enumerate(keys(g)) if i != 1}


def test_external_reference_becomes_the_middle_pick() -> None:
    photos, v = world(8)
    refs = view(unit(np.eye(8)[0:1] + 0.5 * np.eye(8)[1:2]), {})
    ctx, out = run(photos, v, options={"connections": ["theme"]}, references=refs)
    assert out
    middle = out[0].picks[1]
    assert middle.path == ctx.reference_paths[0]
    assert middle.content_key == ctx.reference_keys[0]
    assert middle.score == pytest.approx(float(np.median(ctx.trace.columns["strength"])))
    assert len(out[0].picks) == 3


def test_every_reference_anchors_its_own_triptych() -> None:
    photos, v = world(10)
    refs = view(v.embeddings[[2, 6]], {})
    _, out = run(
        photos,
        v,
        options={"connections": ["theme"]},
        references=refs,
        reference_keys=[photos[2].content_key, photos[6].content_key],
    )
    assert {g.picks[1].content_key for g in out} == {photos[2].content_key, photos[6].content_key}


def test_kind_mix_prefers_different_kinds() -> None:
    kinds = np.zeros((6, 4))
    for i, k in enumerate([0, 0, 2, 2, 3, 3]):
        kinds[i, k] = 1.0
    photos, v = world(6, extra={"k": kinds})
    _, out = run(
        photos,
        v,
        limit=1,
        options={"connections": ["kind_mix"], "strength_weight": 0.0},
        models=make_models(kind=True),
    )
    picked = [int(p.path.stem) for p in out[0].picks]
    assert len({int(np.argmax(kinds[i])) for i in picked}) == 3


def test_kind_same_prefers_matching_kinds() -> None:
    kinds = np.zeros((6, 4))
    for i, k in enumerate([0, 3, 0, 3, 0, 3]):
        kinds[i, k] = 1.0
    photos, v = world(6, extra={"k": kinds})
    _, out = run(
        photos,
        v,
        limit=1,
        options={"connections": ["kind_same"], "strength_weight": 0.0},
        models=make_models(kind=True),
    )
    assert len({int(np.argmax(kinds[int(p.path.stem)])) for p in out[0].picks}) == 1


def test_mood_connection_uses_the_mood_model() -> None:
    mood = np.array([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    photos, v = world(6, extra={"m": mood})
    _, out = run(
        photos,
        v,
        limit=1,
        options={"connections": ["mood"], "strength_weight": 0.0},
        models=make_models(mood=True),
    )
    assert len({int(p.path.stem) // 3 for p in out[0].picks}) == 1
    assert out[0].evidence[1].detail.startswith("similar mood: ")


def test_color_connection_uses_the_palette() -> None:
    palette = np.zeros((6, 4))
    palette[:3, 0] = 1.0
    palette[3:, 3] = 1.0
    photos, v = world(6, extra={"palette": palette})
    _, out = run(photos, v, limit=1, options={"connections": ["color"], "strength_weight": 0.0})
    assert len({int(p.path.stem) // 3 for p in out[0].picks}) == 1


def test_place_type_connection_uses_the_place_model() -> None:
    place = np.array([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    photos, v = world(6, extra={"pt": place})
    models = FakeModels(
        {
            "quality": FeatureScore("quality", "q"),
            "place_type": FeatureLabels("place_type", "pt", ("hall", "street")),
        }
    )
    _, out = run(
        photos,
        v,
        limit=1,
        options={"connections": ["place_type"], "strength_weight": 0.0},
        models=models,
    )
    assert len({int(p.path.stem) // 3 for p in out[0].picks}) == 1


def test_surface_connection_joins_photos_of_one_surface() -> None:
    surface = np.array([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    photos, v = world(6, extra={"sf": surface})
    models = FakeModels(
        {
            "quality": FeatureScore("quality", "q"),
            "surface": FeatureLabels("surface", "sf", ("peeling_paint", "exposed_brick")),
        }
    )
    _, out = run(
        photos,
        v,
        limit=1,
        options={"connections": ["surface"], "strength_weight": 0.0},
        models=models,
    )
    assert len({int(p.path.stem) // 3 for p in out[0].picks}) == 1
    assert any(e.detail.startswith("similar surface: ") for e in out[0].evidence)


def test_missing_model_connection_is_dropped_with_a_note() -> None:
    photos, v = world(6)
    ctx, out = run(photos, v, options={"connections": ["theme", "mood"]})
    assert [e.signal for e in out[0].evidence if e.signal not in NOT_CONNECTIONS] == ["theme"]
    assert any("mood" in n for n in ctx.trace.notes)


def test_missing_palette_drops_color_with_a_note() -> None:
    photos, v = world(6)
    ctx, out = run(photos, v, options={"connections": ["theme", "color"]})
    assert "color" not in {e.signal for e in out[0].evidence}
    assert any("color" in n for n in ctx.trace.notes)


def test_no_available_connection_falls_back_to_theme() -> None:
    photos, v = world(6)
    ctx, out = run(photos, v, options={"connections": ["mood"]})
    assert [e.signal for e in out[0].evidence if e.signal not in NOT_CONNECTIONS] == ["theme"]
    assert any("theme" in n for n in ctx.trace.notes)
    assert out[0].title.startswith("Story 1: theme")


def test_group_evidence_sums_to_group_score() -> None:
    mood = np.random.default_rng(3).dirichlet([1, 1], size=8)
    palette = np.random.default_rng(4).dirichlet([1] * 4, size=8)
    photos, v = world(8, extra={"m": mood, "palette": palette})
    _, out = run(
        photos,
        v,
        options={"connections": ["theme", "mood", "color"], "strength_weight": 0.3},
        models=make_models(mood=True),
    )
    assert out
    for g in out:
        assert sum(e.weight for e in g.evidence) == pytest.approx(1.0)
        assert sum(e.contribution for e in g.evidence) == pytest.approx(g.score, abs=1e-6)
        assert g.evidence[0].signal == "strength"
        assert g.evidence[0].weight == pytest.approx((1 - DISTINCT) * 0.3)
        for p in g.picks:
            assert p.evidence[0].signal == "strength"
            assert p.evidence[0].value == pytest.approx(p.score)


def test_strength_weight_one_picks_the_strongest_photos() -> None:
    photos, v = world(8, quality=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.95, 0.9])
    _, out = run(photos, v, limit=1, options={"connections": ["theme"], "strength_weight": 1.0})
    assert {int(p.path.stem) for p in out[0].picks} >= {6, 7}


def test_titles_number_the_stories_and_name_the_connections() -> None:
    photos, v = world(8, extra={"palette": np.full((8, 3), 1 / 3)})
    _, out = run(photos, v, options={"connections": ["theme", "color"]})
    assert out[0].title == "Story 1: theme · color"
    assert out[1].title == "Story 2: theme · color"


def test_burst_mates_are_alternates() -> None:
    t = datetime(2024, 1, 1, 9)
    photos = [
        photo(f"{i:02d}.jpg", location="X", captured_at=t + timedelta(seconds=i)) for i in range(2)
    ]
    photos += [photo(f"{i:02d}.jpg", location=f"L{i}") for i in range(2, 8)]
    emb = np.eye(8) + 0.001
    emb[1] = emb[0]
    _, v = world(8, emb=unit(emb), quality=[0.9, 0.1, *[0.5] * 6])
    _, out = run(photos, v, options={"connections": ["theme"], "strength_weight": 1.0})
    first = next(p for g in out for p in g.picks if p.path.name == "00.jpg")
    assert [a.name for a in first.alternates] == ["01.jpg"]
    assert first.burst_size == 2


def test_too_few_photos_notes_and_returns_nothing() -> None:
    photos, v = world(2)
    ctx, out = run(photos, v)
    assert out == []
    assert ctx.trace.notes


def test_missing_quality_and_aesthetic_still_works_with_a_note() -> None:
    photos, v = world(6)
    ctx, out = run(photos, v, models=make_models(quality=False))
    assert out
    assert any("quality" in n for n in ctx.trace.notes)


def test_criteria_raise_strength_of_matching_photos() -> None:
    photos, v = world(6, quality=[0.5] * 6)
    _, out = run(
        photos,
        v,
        limit=1,
        criteria=np.eye(6)[4],
        options={"connections": ["theme"], "strength_weight": 1.0},
    )
    assert 4 in {int(p.path.stem) for p in out[0].picks}


def test_describe_weight_is_the_descriptions_share_of_strength() -> None:
    photos, v = world(6)
    ignored, _ = run(photos, v, criteria=np.eye(6)[0], describe_weight=0.0)
    alone, _ = run(photos, v, criteria=np.eye(6)[0], describe_weight=1.0)
    assert int(np.argmax(ignored.trace.columns["strength"])) == 5
    assert int(np.argmax(alone.trace.columns["strength"])) == 0


def test_the_description_counts_a_third_of_strength_by_default() -> None:
    assert TriptychTask.default_describe_weight == pytest.approx(1 / 3)


def test_strength_weighs_quality_as_strongest_does() -> None:
    """A fifth quality, the rest the aesthetic: one convention for every task."""
    found: dict[str, Model] = {
        "quality": FeatureScore("quality", "q"),
        "aesthetic": FeatureLabels("aesthetic", "a", ("strong", "weak")),
    }
    strong = np.array([1.0, 0.5, 0.0])
    photos, v = world(3, quality=[0.0, 0.5, 1.0], extra={"a": np.stack([strong, 1 - strong], 1)})
    ctx, _ = run(photos, v, models=FakeModels(found), options={"connections": ["theme"]})
    np.testing.assert_allclose(ctx.trace.columns["strength"], [0.8, 0.5, 0.2], atol=1e-5)


def test_strength_is_traced_for_the_pool() -> None:
    photos, v = world(6)
    ctx, _ = run(photos, v, options={"connections": ["theme"]})
    assert len(ctx.trace.columns["strength"]) == 6
    assert not np.isnan(ctx.trace.columns["strength"]).any()
    assert ctx.trace.stages


def test_output_is_deterministic() -> None:
    photos, v = world(10, quality=[0.5] * 10)
    assert run(photos, v)[1] == run(photos, v)[1]


def test_candidates_option_limits_the_working_set() -> None:
    photos, v = world(10, quality=list(np.linspace(0, 1, 10)))
    _, out = run(photos, v, options={"connections": ["theme"], "candidates": 3})
    assert len(out) == 1
    assert {int(p.path.stem) for p in out[0].picks} == {7, 8, 9}


def test_nan_quality_is_noted_and_treated_as_zero() -> None:
    photos, v = world(6, quality=[np.nan, 0.5, 0.5, 0.5, 0.5, 0.5])
    ctx, out = run(photos, v, options={"connections": ["theme"]})
    assert out
    assert any("quality returned 1 invalid scores; treated as 0" in n for n in ctx.trace.notes)
    assert all(np.isfinite(g.score) for g in out)


def test_external_reference_gets_median_strength_not_a_perfect_one() -> None:
    photos, v = world(8, quality=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    refs = view(unit(np.eye(8)[0:1] + 0.5 * np.eye(8)[1:2]), {})
    _, out = run(photos, v, options={"connections": ["theme"]}, references=refs)
    assert out[0].picks[1].score < 0.9


def test_a_dominant_anchor_can_anchor_two_stories_when_reuse_is_allowed() -> None:
    photos, v = world(10, quality=[0.99, *[0.3] * 9])
    options = {"connections": ["theme"], "strength_weight": 1.0}
    _, once = run(photos, v, options={**options, "max_uses": 1})
    _, twice = run(photos, v, options={**options, "max_uses": 2})
    anchors = [g.picks[1].path.name for g in twice]
    assert [g.picks[1].path.name for g in once].count("00.jpg") <= 1
    assert anchors.count("00.jpg") == 2
    assert all(
        sum(1 for g in twice for p in g.picks if p.path == path) <= 2
        for path in {p.path for g in twice for p in g.picks}
    )


def test_group_score_is_the_story_score() -> None:
    photos, v = world(6)
    _, out = run(photos, v, options={"connections": ["theme"]})
    assert out[0].score == pytest.approx(sum(e.contribution for e in out[0].evidence))


def test_ui_metadata() -> None:
    assert TriptychTask.default_limit == 6
    assert TriptychTask.reference_label == "Anchor photo"
    assert "middle" in TriptychTask.reference_help  # the UI badge says "optional"


def anchored(
    n: int, ref_rows: list[int], **kw: Any
) -> tuple[list[Group], list[Photo], TaskContext]:
    photos, v = world(n)
    refs = view(v.embeddings[ref_rows], {})
    ctx, out = run(
        photos,
        v,
        options={"connections": ["theme"], **kw.pop("options", {})},
        references=refs,
        reference_keys=[photos[r].content_key for r in ref_rows],
        **kw,
    )
    return out, photos, ctx


def test_one_anchor_yields_several_stories_with_distinct_companions() -> None:
    out, photos, _ = anchored(10, [0], limit=3)
    assert len(out) == 3
    assert {g.picks[1].content_key for g in out} == {photos[0].content_key}
    companions = [k for g in out for i, k in enumerate(keys(g)) if i != 1]
    assert len(companions) == len(set(companions)) == 6


def test_anchor_stories_stop_when_companions_run_out() -> None:
    out, _, _ = anchored(7, [0], limit=6)
    assert len(out) == 3  # six companions, two per story


def test_max_uses_lets_companions_repeat_across_anchor_stories() -> None:
    out, _, _ = anchored(5, [0], limit=6, options={"max_uses": 2})
    assert len(out) > 2  # one use per companion would stop at two stories
    counts = Counter(k for g in out for i, k in enumerate(keys(g)) if i != 1)
    assert max(counts.values()) <= 2


def test_several_anchors_share_the_limit_round_robin() -> None:
    out, photos, _ = anchored(12, [0, 5], limit=4)
    middles = [g.picks[1].content_key for g in out]
    assert middles == [photos[0].content_key, photos[5].content_key] * 2


def test_anchor_stories_are_titled_with_the_anchor_file() -> None:
    out, _, _ = anchored(8, [3], limit=2)
    assert [g.title for g in out] == ["Story 1 with 03.jpg", "Story 2 with 03.jpg"]


def test_external_anchor_is_in_every_story_and_titled_by_its_file() -> None:
    photos, v = world(9)
    refs = view(unit(np.eye(9)[0:1] + 0.5 * np.eye(9)[1:2]), {})
    ctx, out = run(photos, v, limit=3, options={"connections": ["theme"]}, references=refs)
    assert len(out) == 3
    assert all(g.picks[1].path == ctx.reference_paths[0] for g in out)
    assert out[0].title == "Story 1 with 0.jpg"


def test_the_anchors_own_burst_never_supplies_companions() -> None:
    t = datetime(2024, 1, 1, 9)
    photos = [
        photo(f"{i:02d}.jpg", location="X", captured_at=t + timedelta(seconds=i)) for i in range(3)
    ]
    photos += [photo(f"{i:02d}.jpg", location=f"L{i}") for i in range(3, 10)]
    emb = np.eye(10) + 0.001
    emb[1], emb[2] = emb[0], emb[0]
    _, v = world(10, emb=unit(emb), quality=[0.2, 0.9, 0.9, *[0.3] * 7])
    refs = view(v.embeddings[[0]], {})
    _, out = run(
        photos,
        v,
        limit=6,
        options={"connections": ["theme"], "max_similarity": 1.0},
        references=refs,
        reference_keys=[photos[0].content_key],
    )
    used = {p.path.name for g in out for p in g.picks}
    assert out
    assert not used & {"01.jpg", "02.jpg"}


def test_anchored_evidence_sums_and_output_is_deterministic() -> None:
    out, _, _ = anchored(9, [2], limit=3)
    for g in out:
        assert sum(e.contribution for e in g.evidence) == pytest.approx(g.score, abs=1e-6)
    assert out == anchored(9, [2], limit=3)[0]


# --- Arranged stories: the strongest connection, repetition, anchors by kind -------------

ARRANGED = ("establishing:exterior", "establishing:interior", "portrait")


def one_hot(rows: Sequence[int], width: int) -> NDArray[np.float64]:
    out = np.zeros((len(rows), width))
    out[np.arange(len(rows)), list(rows)] = 1.0
    return out


def alike(n: int, group: Sequence[float], dim_extra: int = 0) -> NDArray[np.float32]:
    """Photo i < len(group) shares a direction with the others in proportion ``group[i]``
    (so photos i and j are sqrt(group[i] * group[j]) alike); the rest are unrelated."""
    dim = 1 + n + dim_extra
    rows = []
    for i in range(n):
        share = group[i] if i < len(group) else 0.0
        rows.append(np.sqrt(share) * np.eye(dim)[0] + np.sqrt(1 - share) * np.eye(dim)[1 + i])
    return unit(np.stack(rows))


def one_room() -> tuple[list[Photo], FeatureView]:
    """Photos 0-3 are one room by the view embedder ("v", 0.85 alike); 0-2 are wide
    interiors and 3 a portrait. By theme 0-2 are 0.6 alike and 3 a little less, so a
    story by likeness alone is 0, 1 and 2. Photos 4-9 are other rooms, unrelated."""
    n = 10
    kinds = one_hot([1, 1, 1, 3, 1, 1, 1, 1, 1, 1], len(KINDS))
    return world(
        n,
        emb=alike(n, [0.6, 0.6, 0.6, 0.4]),
        quality=[0.5] * n,
        extra={"v": alike(n, [0.85] * 4).astype(np.float64), "k": kinds},
    )


REPETITION = {**THEME_ONLY, "variety_by": "repetition"}


def test_arranged_options_are_off_by_default() -> None:
    defaults = TriptychTask().parse_options({})
    assert (defaults.connect_by, defaults.variety_by, defaults.anchor_kinds) == (
        "average",
        "likeness",
        (),
    )
    assert defaults.same_space == 0.75


def test_repetition_holds_back_three_shots_of_one_room_at_one_kind() -> None:
    photos, v = one_room()
    models = make_models(kind=True)
    _, by_likeness = run(photos, v, limit=1, options=THEME_ONLY, models=models)
    assert sorted(int(p.path.stem) for p in by_likeness[0].picks) == [0, 1, 2]
    _, out = run(photos, v, limit=1, options=REPETITION, models=models, view_feature="v")
    picked = sorted(int(p.path.stem) for p in out[0].picks)
    assert picked != [0, 1, 2]
    ev = {e.signal: e for e in out[0].evidence}
    assert "one space" in ev["variety"].detail


def test_one_kind_of_shot_in_different_rooms_is_not_repetition() -> None:
    n = 6
    photos, v = world(
        n,
        emb=alike(n, [0.6] * n),
        extra={"v": np.eye(n), "k": one_hot([1] * n, len(KINDS))},
    )
    _, out = run(
        photos, v, limit=1, options=REPETITION, models=make_models(kind=True), view_feature="v"
    )
    ev = {e.signal: e for e in out[0].evidence}
    assert ev["variety"].value == pytest.approx(1.0)


def test_a_person_in_the_room_is_not_repetition() -> None:
    photos, v = one_room()
    ctx = context(photos, v, models=make_models(kind=True), view_feature="v")
    options = TriptychTask().parse_options(REPETITION)
    scorer = TriptychTask().scorer(ctx, options, include=[])
    rows = [scorer.row(photos[i].content_key) for i in range(4)]
    three_wide = scorer.variety_of(rows[0], rows[1], rows[2])
    with_portrait = scorer.variety_of(rows[0], rows[3], rows[1])
    assert three_wide == pytest.approx(0.0)
    assert with_portrait == pytest.approx(2 / 3)


def test_repetition_without_view_features_falls_back_to_likeness_with_a_note() -> None:
    photos, v = one_room()
    ctx, out = run(
        photos, v, limit=1, options=REPETITION, models=make_models(kind=True), view_feature="dino"
    )
    assert any("view embedder" in n for n in ctx.trace.notes)
    ev = {e.signal: e for e in out[0].evidence}
    assert "alike on average" in ev["variety"].detail


def test_repetition_without_kinds_of_shot_counts_one_space_alone_with_a_note() -> None:
    photos, v = one_room()
    ctx = context(photos, v, models=make_models(), view_feature="v")
    scorer = TriptychTask().scorer(ctx, TriptychTask().parse_options(REPETITION), include=[])
    assert any("shot_kind" in n for n in ctx.trace.notes)
    rows = [scorer.row(photos[i].content_key) for i in range(4)]
    assert scorer.variety_of(rows[0], rows[3], rows[1]) == pytest.approx(0.0)


def strong_and_middling() -> tuple[list[Photo], FeatureView]:
    """Anchor 0 with 1 and 2: one palette, nothing alike in subject. Anchor 0 with 3 and
    4: a little alike in subject and a little in palette. Photos 5-8 are more alike in
    subject among themselves, so 0, 3 and 4's subject is only middling; 9-11 unrelated.
    Connections are percentiles, so "middling" needs pairs that rank above it."""
    n = 12
    palette = np.zeros((n, n + 2))
    palette[[0, 1, 2], 0] = 1.0
    palette[[3, 4], 0] = 0.55
    palette[[3, 4], 1] = 0.45
    for i in range(5, n):
        palette[i, 2 + i] = 1.0
    dim = 2 + n
    shares = {0: (0, 0.3), 3: (0, 0.3), 4: (0, 0.3), 5: (1, 0.6), 6: (1, 0.6), 7: (1, 0.6)}
    shares[8] = (1, 0.6)
    rows = []
    for i in range(n):
        axis, share = shares.get(i, (0, 0.0))
        rows.append(np.sqrt(share) * np.eye(dim)[axis] + np.sqrt(1 - share) * np.eye(dim)[2 + i])
    return world(n, emb=unit(np.stack(rows)), quality=[0.5] * n, extra={"palette": palette})


def around_zero(photos: Sequence[Photo], v: FeatureView) -> dict[str, Any]:
    """Photo 0 as the anchor, so the stories compared are the ones told around it."""
    return {"references": view(v.embeddings[[0]], {}), "reference_keys": [photos[0].content_key]}


STRONGEST = {
    "connections": ["theme", "color"],
    "strength_weight": 0.0,
    "variety_weight": 0.0,
    "burst_window_seconds": 0,
}


def test_the_strongest_connection_can_carry_a_story() -> None:
    photos, v = strong_and_middling()
    anchor = around_zero(photos, v)
    _, average = run(photos, v, limit=1, options=STRONGEST, **anchor)
    assert sorted(int(p.path.stem) for p in average[0].picks) == [0, 3, 4]
    strongest = {**STRONGEST, "connect_by": "strongest"}
    _, out = run(photos, v, limit=1, options=strongest, **anchor)
    assert sorted(int(p.path.stem) for p in out[0].picks) == [0, 1, 2]


def test_the_strongest_connection_is_the_only_connection_evidence() -> None:
    photos, v = strong_and_middling()
    options = {**STRONGEST, "connect_by": "strongest", "strength_weight": 0.4}
    _, out = run(photos, v, limit=3, options=options)
    assert out
    for g in out:
        connections = [e for e in g.evidence if e.signal not in NOT_CONNECTIONS]
        assert len(connections) == 1
        assert connections[0].weight == pytest.approx((1 - DISTINCT) * 0.6)
        assert "strongest of theme, color" in connections[0].detail
        assert sum(e.weight for e in g.evidence) == pytest.approx(1.0)
        assert sum(e.contribution for e in g.evidence) == pytest.approx(g.score, abs=1e-6)


def kinds_world() -> tuple[list[Photo], FeatureView]:
    """Wide interiors 0-5 are the strongest; 6 a portrait, 7 an exterior, 8 a detail."""
    kinds = one_hot([1, 1, 1, 1, 1, 1, 3, 0, 2, 1, 1, 1], len(KINDS))
    quality = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2]
    return world(12, quality=quality, extra={"k": kinds})


def test_anchors_by_kind_take_turns_among_the_kinds_named() -> None:
    photos, v = kinds_world()
    models = make_models(kind=True)
    options = {"connections": ["theme"], "anchor_kinds": list(ARRANGED)}
    _, strongest = run(photos, v, limit=3, options={"connections": ["theme"]}, models=models)
    assert all(int(g.picks[1].path.stem) < 6 for g in strongest)
    _, out = run(photos, v, limit=3, options=options, models=models)
    anchors = [int(g.picks[1].path.stem) for g in out]
    assert anchors[0] == 7  # the first kind named, its strongest photo first
    assert sorted(KINDS[int(np.argmax(v.features["k"][a]))] for a in anchors) == sorted(ARRANGED)


def test_anchor_kinds_need_the_shot_kind_model_or_say_so() -> None:
    photos, v = kinds_world()
    options = {"connections": ["theme"], "anchor_kinds": list(ARRANGED)}
    ctx, out = run(photos, v, limit=3, options=options)
    assert out
    assert any("shot_kind" in n and "anchor" in n for n in ctx.trace.notes)


def test_an_unknown_anchor_kind_is_noted() -> None:
    photos, v = kinds_world()
    options = {"connections": ["theme"], "anchor_kinds": ["portrait", "aerial"]}
    ctx, out = run(photos, v, limit=2, options=options, models=make_models(kind=True))
    assert out
    assert any("aerial" in n for n in ctx.trace.notes)


@pytest.mark.parametrize(
    "extra",
    [
        {"connect_by": "strongest"},
        {"variety_by": "repetition"},
        {"connect_by": "strongest", "variety_by": "repetition", "anchor_kinds": list(ARRANGED)},
        {"tell_by": "roles"},
        {"distinct_weight": 0.3},
        {"tell_by": "roles", "distinct_weight": 0.3},
    ],
)
def test_the_scorer_gives_arranged_stories_their_scores(extra: dict[str, Any]) -> None:
    photos, v = one_room()
    models = make_models(kind=True)
    options = {"connections": ["theme", "kind_same"], "burst_window_seconds": 0, **extra}
    _, out = run(photos, v, limit=4, options=options, models=models, view_feature="v")
    ctx = context(photos, v, models=models, limit=4, view_feature="v")
    scorer = TriptychTask().scorer(ctx, TriptychTask().parse_options(options), include=[])
    assert out
    for g in out:
        rows = [scorer.row(k) for k in keys(g)]
        assert scorer.score(*rows) == pytest.approx(g.score, abs=1e-6)


@pytest.mark.parametrize("tell_by", ["pairs", "roles"])
def test_the_scorer_gives_every_signal_of_a_story_however_it_is_told(tell_by: str) -> None:
    photos, v = one_room()
    models = make_models(kind=True)
    options = {"connections": ["theme", "kind_same"], "burst_window_seconds": 0, "tell_by": tell_by}
    _, out = run(photos, v, limit=4, options=options, models=models, view_feature="v")
    ctx = context(photos, v, models=models, limit=4, view_feature="v")
    scorer = TriptychTask().scorer(ctx, TriptychTask().parse_options(options), include=[])
    # Without a scale or subject model, fresh and anchor cannot be measured: left out.
    roles = {"bond", "belong", "arrangement"} if tell_by == "roles" else set()
    assert out
    for g in out:
        signals = scorer.signals(*[scorer.row(k) for k in keys(g)])
        assert set(signals) == {"strength", "theme", "kind_same", "variety", "distinct", *roles}
        for e in g.evidence:  # what the story was told with is among them, as it was told
            assert signals[e.signal] == pytest.approx(e.value, abs=1e-6)


def test_anchor_kinds_with_no_photo_of_any_fall_back_to_the_strongest_with_a_note() -> None:
    photos, v = kinds_world()
    options = {"connections": ["theme"], "anchor_kinds": ["aerial"]}
    ctx, out = run(photos, v, limit=2, options=options, models=make_models(kind=True))
    assert len(out) == 2
    assert any("strongest photos anchor" in n for n in ctx.trace.notes)


SCALES = ("aerial", "wide", "medium", "close", "macro")
ROLES = {"connections": ["theme"], "burst_window_seconds": 0, "tell_by": "roles"}


def role_models(*, scale: bool = True, subject: bool = True) -> FakeModels:
    found: dict[str, Model] = {"quality": FeatureScore("quality", "q")}
    if scale:
        found["scale"] = FeatureLabels("scale", "s", SCALES)
    if subject:
        found["subject"] = FeatureLabels("subject", "life", ("living", "none"))
    return FakeModels(found)


def echo_and_carousel(
    n: int = 12, quality: Sequence[float] | None = None
) -> tuple[list[Photo], FeatureView]:
    """0 and 1: alike medium shots of two rooms. 2: a wide shot with a person, a little
    like them. 3-5: three wide shots of one room, alike. The rest: unrelated closer
    shots. By the mean of its pairs, 3-5 is the best connected story."""
    dim = n + 2
    emb = np.eye(dim)[2:].copy()
    for i, share in ((0, 0.7), (1, 0.7), (2, 0.15)):
        emb[i] = np.sqrt(share) * np.eye(dim)[0] + np.sqrt(1 - share) * emb[i]
    for i in (3, 4, 5):
        emb[i] = np.sqrt(0.6) * np.eye(dim)[1] + np.sqrt(0.4) * emb[i]
    rooms = np.eye(n)
    rooms[[3, 4, 5]] = alike(n, [0, 0, 0, 0.9, 0.9, 0.9])[[3, 4, 5], :n]
    scale = [2, 2, 1, 1, 1, 1, *[3 + i % 2 for i in range(n - 6)]]
    life = one_hot([0 if i == 2 else 1 for i in range(n)], 2)
    return world(
        n,
        emb=unit(emb),
        quality=[0.5] * n if quality is None else quality,
        extra={"s": one_hot(scale, len(SCALES)), "life": life, "v": unit(rooms).astype(float)},
    )


def role_scorer(photos: Sequence[Photo], v: FeatureView, options: dict[str, Any], **kw: Any) -> Any:
    models = kw.pop("models", role_models())
    ctx = context(photos, v, models=models, view_feature=kw.pop("view_feature", "v"), **kw)
    return ctx, TriptychTask().scorer(ctx, TriptychTask().parse_options(options), include=[])


def at(scorer: Any, photos: Sequence[Photo], *index: int) -> list[int]:
    return [scorer.row(photos[i].content_key) for i in index]


def test_stories_are_told_by_pairs_unless_roles_are_asked_for() -> None:
    assert TriptychTask().parse_options({}).tell_by == "pairs"


def test_roles_prefer_an_echo_around_a_person_to_three_shots_of_one_room() -> None:
    photos, v = echo_and_carousel()
    _, pairs = role_scorer(photos, v, {**ROLES, "tell_by": "pairs"})
    _, roles = role_scorer(photos, v, ROLES)
    for scorer, echo_wins in ((pairs, False), (roles, True)):
        echo = scorer.score(*at(scorer, photos, 0, 2, 1))
        carousel = scorer.score(*at(scorer, photos, 3, 4, 5))
        assert (echo > carousel) == echo_wins


def test_a_story_told_by_roles_explains_each_role_and_sums_to_its_score() -> None:
    photos, v = echo_and_carousel()
    _, out = run(photos, v, limit=1, options=ROLES, models=role_models(), view_feature="v")
    story = out[0]
    assert keys(story)[1] == photos[2].content_key
    ev = {e.signal: e for e in story.evidence}
    assert set(ev) == {"strength", "bond", "belong", "arrangement", "fresh", "anchor", "distinct"}
    assert "echo" in ev["arrangement"].detail
    assert "person or animal" in ev["anchor"].detail
    assert {"arrangement", "fresh", "anchor"} <= NOT_CONNECTIONS
    assert sum(e.weight for e in story.evidence) == pytest.approx(1.0)
    assert sum(e.contribution for e in story.evidence) == pytest.approx(story.score, abs=1e-6)


def test_a_change_of_scale_is_an_arrangement_and_one_room_at_one_scale_is_not_fresh() -> None:
    photos, v = echo_and_carousel()
    _, scorer = role_scorer(photos, v, ROLES)
    row = [scorer.row(p.content_key) for p in photos]
    wide_to_macro = scorer.roles_of(row[7], row[2], row[0])
    assert wide_to_macro["arrangement"] == pytest.approx(1.0)
    assert scorer.roles_of(row[3], row[4], row[5])["fresh"] == 0.0
    assert scorer.roles_of(row[3], row[2], row[4])["fresh"] == 1.0


def test_roles_without_scale_or_subject_or_view_features_say_what_they_lost() -> None:
    photos, v = echo_and_carousel()
    ctx, scorer = role_scorer(
        photos, v, ROLES, models=role_models(scale=False, subject=False), view_feature="dino"
    )
    notes = " ".join(ctx.trace.notes)
    assert "'scale'" in notes
    assert "'subject'" in notes
    assert "view embedder" in notes
    parts = scorer.roles_of(*at(scorer, photos, 3, 4, 5))
    assert set(parts) == {"bond", "belong", "arrangement"}  # fresh and anchor not measured
    assert "not measured" in notes


def test_a_role_that_cannot_be_measured_gives_its_weight_to_the_others() -> None:
    """Without the scale model, fresh (one space at one scale) cannot be told: a constant
    1.0 claiming "not three shots of one space" would be false."""
    photos, v = echo_and_carousel()
    _, groups = run(
        photos, v, limit=2, options=ROLES, models=role_models(scale=False), view_feature="v"
    )
    for g in groups:
        signals = {e.signal for e in g.evidence}
        assert "fresh" not in signals
        assert "anchor" in signals  # a person or animal can still be seen
        assert sum(e.weight for e in g.evidence) == pytest.approx(1.0)
        assert sum(e.weight * e.value for e in g.evidence) == pytest.approx(g.score, abs=1e-6)
        (anchor,) = [e for e in g.evidence if e.signal == "anchor"]
        assert "widest" not in anchor.detail


def test_options_that_telling_by_roles_does_not_use_are_named() -> None:
    photos, v = echo_and_carousel()
    roles = {**ROLES, "variety_weight": 0.5, "connect_by": "strongest"}
    ctx, chosen = run(photos, v, limit=1, options=roles, models=role_models(), view_feature="v")
    _, plain = run(photos, v, limit=1, options=ROLES, models=role_models(), view_feature="v")
    (note,) = [n for n in ctx.trace.notes if "by its roles" in n]
    assert "variety_weight" in note
    assert "connect_by" in note
    assert chosen[0].evidence == plain[0].evidence


def test_roles_try_anchors_by_the_shared_anchor_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    """``calico eval anchors`` measures this same prior (``selection.anchors``)."""
    calls: list[int] = []

    def spy(*args: Any, **kwargs: Any) -> NDArray[np.float32]:
        calls.append(1)
        return anchor_prior(*args, **kwargs)

    monkeypatch.setattr("calico.tasks.triptych.anchor_prior", spy)
    photos, v = echo_and_carousel()
    run(photos, v, limit=1, options=ROLES, models=role_models(), view_feature="v")
    assert calls


def test_how_many_shown_stories_each_photo_is_in_is_recorded() -> None:
    """So runs explain can say a strong photo was in no story chosen."""
    photos, v = echo_and_carousel()
    ctx, groups = run(photos, v, limit=1, options=ROLES, models=role_models(), view_feature="v")
    uses = ctx.trace.columns["uses"]
    shown = {p.content_key for p in groups[0].picks}
    for i, p in enumerate(photos):
        if ctx.trace.columns["in_working_set"][i] == 1:
            assert uses[i] == (1.0 if p.content_key in shown else 0.0)


def test_roles_let_a_weaker_photo_with_a_person_anchor() -> None:
    """Photo 2 is the weakest of 14: by strength it is not among the ten anchors tried.
    Strength counts little here, so the story is chosen by its roles."""
    n = 14
    quality = np.linspace(0.9, 0.3, n)
    quality[2] = 0.1
    photos, v = echo_and_carousel(n, quality=list(quality))
    roles = {**ROLES, "strength_weight": 0.2}
    pairs = {**roles, "tell_by": "pairs"}
    _, by_pairs = run(photos, v, limit=1, options=pairs, models=role_models())
    _, by_roles = run(photos, v, limit=1, options=roles, models=role_models(), view_feature="v")
    assert keys(by_pairs[0])[1] != photos[2].content_key
    assert keys(by_roles[0])[1] == photos[2].content_key


def two_of_one_view(
    apart: timedelta | None = timedelta(seconds=30),
) -> tuple[list[Photo], FeatureView]:
    """By theme 0-2 are 0.6 alike and 3 a little less, so a story by likeness alone is 0,
    1 and 2. By the view embedder ("v") 0 and 1 are one view (0.95 alike), shot ``apart``
    (None: 1 is undated); nothing else is alike. The rest are shot a day apart each."""
    n = 10
    photos, v = world(
        n,
        emb=alike(n, [0.6, 0.6, 0.6, 0.4]),
        quality=[0.5] * n,
        extra={"v": alike(n, [0.95, 0.95]).astype(np.float64)},
    )
    start = datetime(2024, 5, 1, 12)
    times: list[datetime | None] = [start + timedelta(days=i) for i in range(n)]
    times[1] = None if apart is None else start + apart
    dated = [
        photo(p.path.name, location=p.location, captured_at=t)
        for p, t in zip(photos, times, strict=True)
    ]
    return dated, v


def test_distinct_counts_by_default_and_falls_from_one_look_to_one_view() -> None:
    defaults = TriptychTask().parse_options({})
    assert defaults.distinct_weight == 0.15
    assert (defaults.distinct_from, defaults.distinct_until) == (0.80, 0.95)
    assert defaults.distinct_within == 600  # seconds: one moment
    assert "distinct" in NOT_CONNECTIONS


@pytest.mark.parametrize("tell_by", ["pairs", "roles"])
def test_distinct_holds_back_a_story_with_two_photos_of_one_view(tell_by: str) -> None:
    photos, v = two_of_one_view()
    options = {**THEME_ONLY, "tell_by": tell_by}
    alone = {**options, "distinct_weight": 0}
    _, by_likeness = run(photos, v, limit=1, options=alone, view_feature="v")
    assert {0, 1} <= {int(p.path.stem) for p in by_likeness[0].picks}
    _, out = run(photos, v, limit=1, options={**options, "distinct_weight": 0.5}, view_feature="v")
    assert not {0, 1} <= {int(p.path.stem) for p in out[0].picks}
    evidence = {e.signal: e for e in out[0].evidence}
    assert evidence["distinct"].weight == pytest.approx(0.5)  # strength counts nothing here
    assert evidence["distinct"].value == 1.0
    assert sum(e.weight for e in out[0].evidence) == pytest.approx(1.0)


@pytest.mark.parametrize("apart", [timedelta(days=7), None])
def test_two_photos_of_one_view_shot_at_other_times_are_an_echo_not_one_moment(
    apart: timedelta | None,
) -> None:
    photos, v = two_of_one_view(apart)
    options = {**THEME_ONLY, "distinct_weight": 0.5}
    _, out = run(photos, v, limit=1, options=options, view_feature="v")
    assert {0, 1} <= {int(p.path.stem) for p in out[0].picks}
    evidence = {e.signal: e for e in out[0].evidence}
    assert evidence["distinct"].value == 1.0
    assert "within 10 minutes" in evidence["distinct"].detail
    # Counting whenever they were shot, the same pair is held back.
    _, anytime = run(
        photos, v, limit=1, options={**options, "distinct_within": 0}, view_feature="v"
    )
    assert not {0, 1} <= {int(p.path.stem) for p in anytime[0].picks}


def test_a_story_of_two_photos_of_one_view_says_which_two() -> None:
    photos, v = two_of_one_view()
    ctx = context(photos, v, models=make_models(), limit=1, view_feature="v")
    options = TriptychTask().parse_options({**THEME_ONLY, "distinct_weight": 0.5})
    scorer = TriptychTask().scorer(ctx, options, include=[])
    rows = [scorer.row(photos[i].content_key) for i in (0, 2, 1)]
    assert scorer.signals(*rows)["distinct"] == pytest.approx(0.0, abs=1e-6)  # 0.95: one view
    slight = TriptychTask().parse_options({**THEME_ONLY, "distinct_weight": 0.02})
    (story,) = TriptychTask().run(ctx, slight)  # too slight to break up the pair
    detail = {e.signal: e for e in story.evidence}["distinct"].detail
    assert detail.startswith("closest pair shot within 10 minutes: ")
    assert {"00.jpg", "01.jpg"} <= set(detail.replace(",", " ").split())
    assert "0.95 alike" in detail
    rows = [scorer.row(photos[i].content_key) for i in (0, 2, 3)]
    assert scorer.signals(*rows)["distinct"] == 1.0


def test_without_the_view_embedder_distinct_is_not_measured_and_says_so() -> None:
    photos, v = two_of_one_view()
    options = {**THEME_ONLY, "distinct_weight": 0.5}
    ctx, out = run(photos, v, limit=1, options=options, view_feature="dinov2")  # not indexed
    assert out
    assert "distinct" not in {e.signal for e in out[0].evidence}
    assert sum(e.weight for e in out[0].evidence) == pytest.approx(1.0)
    assert any("Distinct" in n and "view embedder" in n for n in ctx.trace.notes)


def test_distinct_takes_its_share_of_the_whole_score_so_it_reorders_only_one_moment() -> None:
    photos, v = two_of_one_view()
    ctx = context(photos, v, models=make_models(), limit=1, view_feature="v")
    plain = {"connections": ["theme"], "burst_window_seconds": 0, "distinct_weight": 0}
    before = TriptychTask().scorer(ctx, TriptychTask().parse_options(plain), include=[])
    after = TriptychTask().scorer(
        ctx, TriptychTask().parse_options({**plain, "distinct_weight": 0.5}), include=[]
    )
    for story in ((0, 2, 3), (2, 3, 4), (4, 5, 6)):  # no two of one moment
        rows = [after.row(photos[i].content_key) for i in story]
        assert after.score(*rows) == pytest.approx(0.5 * before.score(*rows) + 0.5)


@pytest.mark.parametrize(
    "options",
    [
        {"too_alike": 0.95, "max_similarity": 0.95},
        {"too_alike": 0.99, "max_similarity": 0.5},
        {"distinct_from": 0.95, "distinct_until": 0.9},
    ],
)
def test_a_line_above_the_one_it_falls_to_is_refused(options: dict[str, float]) -> None:
    """Audit A03-3: variety (or distinct) was then 1 for every story, with no word."""
    with pytest.raises(InvalidRequestError, match="must be below"):
        TriptychTask().parse_options(options)


def test_anchor_kinds_are_noted_as_ignored_around_an_anchor_photo() -> None:
    """Audit A03-11."""
    photos, v = world(8)
    ctx, out = run(
        photos,
        v,
        options={"connections": ["theme"], "anchor_kinds": ["portrait"]},
        references=view(v.embeddings[3:4], {}),
        reference_keys=[photos[3].content_key],
    )
    assert out
    assert any("anchor_kinds is ignored" in n for n in ctx.trace.notes)


def test_photos_whose_embedding_or_palette_is_not_finite_are_noted() -> None:
    """Audit A03-16: such a photo could never be in a story, and nothing said why."""
    photos, v = world(8, extra={"palette": np.random.default_rng(0).random((8, 4))})
    v.embeddings[2] = np.nan
    v.features["palette"][5] = np.inf
    ctx, out = run(photos, v, options={"connections": ["theme", "color"]})
    assert out
    notes = " ".join(ctx.trace.notes)
    assert "1 photo has an embedding that is not finite" in notes
    assert "1 photo has a palette that is not finite" in notes


LIKELY = {"connections": ["theme"], "burst_window_seconds": 0, "anchor_by": "likely"}


def test_anchors_are_tried_by_strength_unless_the_likeliest_are_asked_for() -> None:
    assert TriptychTask().parse_options({}).anchor_by == "strength"


def test_likely_anchors_let_a_weaker_photo_with_a_person_anchor_todays_way() -> None:
    """Photo 2 is the weakest of 14 and the one wide shot with a person: by strength it is
    not among the ten anchors tried; by the anchor prior it is (calico eval anchors, 3
    November 2026: the prior found your anchors at 0.81, strength at 0.68)."""
    n = 14
    quality = np.linspace(0.9, 0.3, n)
    quality[2] = 0.1
    photos, v = echo_and_carousel(n, quality=list(quality))
    by_strength = {**LIKELY, "anchor_by": "strength"}
    _, strongest = role_scorer(photos, v, by_strength, limit=1)
    _, likeliest = role_scorer(photos, v, LIKELY, limit=1)
    (two,) = at(strongest, photos, 2)
    assert not strongest.anchored[two]
    assert likeliest.anchored[two]
    assert likeliest.anchored.sum() == strongest.anchored.sum() == 10


def test_likely_anchors_use_the_shared_anchor_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    """One prior for roles, today's way and calico eval anchors (``selection.anchors``)."""
    calls: list[int] = []

    def spy(*args: Any, **kwargs: Any) -> NDArray[np.float32]:
        calls.append(1)
        return anchor_prior(*args, **kwargs)

    monkeypatch.setattr("calico.tasks.triptych.anchor_prior", spy)
    photos, v = echo_and_carousel()
    ctx, out = run(photos, v, limit=1, options=LIKELY, models=role_models(), view_feature="v")
    assert out
    assert calls
    assert "anchor_prior" in ctx.trace.columns  # runs explain can say why it anchored


def test_likely_anchors_without_scale_or_subject_fall_back_to_the_strongest_saying_so() -> None:
    photos, v = echo_and_carousel()
    ctx, out = run(
        photos, v, limit=1, options=LIKELY, models=role_models(scale=False, subject=False)
    )
    assert out
    assert any("anchor_by" in n and "strongest" in n for n in ctx.trace.notes)


def test_likely_anchors_are_noted_as_ignored_around_an_anchor_photo() -> None:
    photos, v = world(8)
    ctx, out = run(
        photos,
        v,
        options={"connections": ["theme"], "anchor_by": "likely"},
        references=view(v.embeddings[3:4], {}),
        reference_keys=[photos[3].content_key],
    )
    assert out
    assert any("anchor_by is ignored" in n for n in ctx.trace.notes)


def test_the_scorer_marks_the_anchors_the_task_would_try() -> None:
    """The strongest ``max(3 * limit, 10)`` candidates, as ``run`` tries them; a photo a
    measurement asked to score is never one unless it is a candidate itself."""
    photos, v = world(14)
    ctx = context(photos, v, models=make_models(), limit=1)
    task = TriptychTask()
    weakest = photos[0].content_key
    scorer = task.scorer(ctx, task.parse_options({"candidates": 12}), include=[weakest])
    assert scorer.anchored.sum() == 10
    assert not scorer.anchored[scorer.row(weakest)]


def test_anchors_tried_are_three_per_story_unless_a_number_is_asked_for() -> None:
    assert TriptychTask().parse_options({}).anchors_tried is None
    with pytest.raises(InvalidRequestError):
        TriptychTask().parse_options({"anchors_tried": 0})


def test_asking_for_more_anchors_tries_more_of_the_candidates() -> None:
    """At a large place three per story is 18 of 1,000 candidates, which reached 4 of your
    28 finished stories' anchors (General, 5 November 2026)."""
    photos, v = world(14)
    ctx = context(photos, v, models=make_models(), limit=1)
    task = TriptychTask()
    scorer = task.scorer(ctx, task.parse_options({"anchors_tried": 12}), include=())
    assert scorer.anchored.sum() == 12
    every = task.scorer(ctx, task.parse_options({"anchors_tried": 100}), include=())
    assert every.anchored.sum() == every.shortlisted.sum()  # only candidates anchor


def test_more_anchors_tried_lets_a_weaker_photo_anchor() -> None:
    """Photo 2 is the weakest of 14: among the ten strongest it is not tried, among all
    fourteen it is."""
    n = 14
    quality = np.linspace(0.9, 0.3, n)
    quality[2] = 0.1
    photos, v = echo_and_carousel(n, quality=list(quality))
    plain = {"connections": ["theme"], "burst_window_seconds": 0}
    _, ten = role_scorer(photos, v, plain, limit=1)
    _, all_of_them = role_scorer(photos, v, {**plain, "anchors_tried": 14}, limit=1)
    (two,) = at(ten, photos, 2)
    assert not ten.anchored[two]
    assert all_of_them.anchored[two]


def test_anchors_tried_is_noted_as_ignored_around_an_anchor_photo() -> None:
    photos, v = world(8)
    ctx, out = run(
        photos,
        v,
        options={"connections": ["theme"], "anchors_tried": 20},
        references=view(v.embeddings[3:4], {}),
        reference_keys=[photos[3].content_key],
    )
    assert out
    assert any("anchors_tried is ignored" in n for n in ctx.trace.notes)
