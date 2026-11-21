from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from calico.errors import InvalidRequestError
from calico.tasks.base import ReferenceUse
from calico.tasks.same_view import SameViewOptions, SameViewTask
from tests.factories import context, photo, unit, view

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from calico.domain import Group, Photo
    from calico.selection.context import FeatureView, TaskContext

D0 = datetime(2023, 3, 1, 9, 0)


def pool(
    emb: ArrayLike,
    layout: ArrayLike | None = None,
    when: Sequence[datetime | None] | None = None,
    location: str | None = None,
) -> tuple[list[Photo], FeatureView]:
    e = unit(emb)
    n = len(e)
    lay = unit(layout) if layout is not None else unit(np.ones((n, 2)))
    stamps = when if when is not None else [D0 + timedelta(days=i) for i in range(n)]
    photos = [
        photo(f"{i}.jpg", location=location or f"L{i}", captured_at=stamps[i]) for i in range(n)
    ]
    return photos, view(e, {"layout": lay})


def run(
    photos: Sequence[Photo],
    v: FeatureView,
    *,
    ref_emb: ArrayLike | None = (1.0, 0.0),
    ref_layout: ArrayLike = (1.0, 0.0),
    options: dict[str, Any] | None = None,
    limit: int = 12,
    **kw: Any,
) -> tuple[TaskContext, list[Group]]:
    refs = None
    if ref_emb is not None:
        refs = view(np.asarray([ref_emb]), {"layout": np.asarray([ref_layout])})
    ctx = context(photos, v, references=refs, limit=limit, **kw)
    opts = {"min_similarity": 0.0, **(options or {})}
    return ctx, SameViewTask().run(ctx, SameViewTask().parse_options(opts))


def names(groups: list[Group]) -> list[str]:
    return [p.path.name for p in groups[0].picks]


def test_task_metadata() -> None:
    assert SameViewTask.name == "same_view"
    assert SameViewTask.reference_use is ReferenceUse.REQUIRED
    assert SameViewTask.options_model is SameViewOptions
    assert SameViewTask.description


def test_requires_a_reference_photo() -> None:
    photos, v = pool([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(InvalidRequestError, match="reference photo"):
        run(photos, v, ref_emb=None)


def test_output_is_chronological_whatever_the_score_order() -> None:
    emb = [[1.0, 0.0], [0.9, 0.4], [0.5, 0.8], [0.99, 0.1]]
    when = [D0 + timedelta(days=d) for d in (30, 10, 20, 0)]
    photos, v = pool(emb, when=when)
    _, out = run(photos, v)
    assert names(out) == ["3.jpg", "1.jpg", "2.jpg", "0.jpg"]


def test_undated_photos_come_last_in_path_order() -> None:
    photos, v = pool([[1.0, 0.0]] * 4, when=[None, D0, None, D0 + timedelta(days=1)])
    _, out = run(photos, v)
    assert names(out) == ["1.jpg", "3.jpg", "0.jpg", "2.jpg"]


def test_the_best_matches_are_kept_when_limited_then_ordered_by_time() -> None:
    emb = [[1.0, 0.0], [0.0, 1.0], [0.95, 0.3], [0.9, 0.4]]
    when = [D0 + timedelta(days=d) for d in (5, 1, 3, 4)]
    photos, v = pool(emb, when=when)
    _, out = run(photos, v, limit=3)
    assert names(out) == ["2.jpg", "3.jpg", "0.jpg"]


def test_every_close_match_is_returned_even_on_the_same_day() -> None:
    day = datetime(2023, 6, 1, 8)
    when = [day, day + timedelta(hours=6), day + timedelta(days=1)]
    photos, v = pool([[0.8, 0.6], [1.0, 0.0], [0.7, 0.7]], when=when)
    _, out = run(photos, v, options={"burst_window_seconds": 0})
    assert names(out) == ["0.jpg", "1.jpg", "2.jpg"]


def test_one_per_day_option_no_longer_exists() -> None:
    with pytest.raises(InvalidRequestError):
        SameViewTask().parse_options({"one_per_day": True})


def test_defaults_are_a_close_match_and_every_frame() -> None:
    options = SameViewTask().parse_options({})
    assert options.min_similarity == 0.75  # calibrated: see docs/roadmap.md
    assert options.collapse_bursts is False


def test_default_threshold_keeps_only_close_matches() -> None:
    photos, v = pool([[1.0, 0.0], [0.9, 0.44], [0.0, 1.0]], layout=[[1.0, 0.0]] * 3)
    refs = view(np.asarray([(1.0, 0.0)]), {"layout": np.asarray([(1.0, 0.0)])})
    ctx = context(photos, v, references=refs)
    out = SameViewTask().run(ctx, SameViewTask().parse_options({}))
    assert names(out) == ["0.jpg", "1.jpg"]


def test_a_match_just_below_the_old_default_is_kept() -> None:
    """Labelled same views often scored 0.75-0.80; the old 0.80 missed a third of them."""
    photos, v = pool([[1.0, 0.0], [0.77, 0.638]], layout=[[1.0, 0.0]] * 2)
    refs = view(np.asarray([(1.0, 0.0)]), {"layout": np.asarray([(1.0, 0.0)])})
    ctx = context(photos, v, references=refs)
    out = SameViewTask().run(ctx, SameViewTask().parse_options({"layout_weight": 0.0}))
    assert names(out) == ["0.jpg", "1.jpg"]


def test_more_matches_than_the_limit_keeps_the_best_in_time_order_and_says_so() -> None:
    emb = [[1.0, 0.0], [0.95, 0.31], [0.9, 0.44], [0.85, 0.53]]
    when = [D0 + timedelta(days=d) for d in (3, 2, 1, 0)]
    photos, v = pool(emb, when=when)
    ctx, out = run(photos, v, limit=2)
    assert names(out) == ["1.jpg", "0.jpg"]
    assert any("2 more photos matched; raise How many to see them." in n for n in ctx.trace.notes)


def test_nothing_matching_reports_the_closest_score() -> None:
    photos, v = pool([[0.7, 0.714], [0.0, 1.0]], layout=[[1.0, 0.0]] * 2)
    refs = view(np.asarray([(1.0, 0.0)]), {"layout": np.asarray([(1.0, 0.0)])})
    ctx = context(photos, v, references=refs)
    out = SameViewTask().run(ctx, SameViewTask().parse_options({"layout_weight": 0.0}))
    assert out == []
    assert any(
        "No photo matched at 0.75 or above; the closest scored 0.70." in n
        and "Lower min_similarity" in n
        for n in ctx.trace.notes
    )


def test_near_misses_are_counted_in_a_note() -> None:
    photos, v = pool([[1.0, 0.0], [0.7, 0.714], [0.0, 1.0]], layout=[[1.0, 0.0]] * 3)
    refs = view(np.asarray([(1.0, 0.0)]), {"layout": np.asarray([(1.0, 0.0)])})
    ctx = context(photos, v, references=refs)
    out = SameViewTask().run(ctx, SameViewTask().parse_options({"layout_weight": 0.0}))
    assert names(out) == ["0.jpg"]
    assert any(
        "1 more photo" in n and "within 0.10" in n and "Label for calibration" in n
        for n in ctx.trace.notes
    )


def test_layout_weight_changes_the_winner() -> None:
    # 0: same subject, other angle; 1: other subject, same angle
    photos, v = pool([[1.0, 0.0], [0.3, 0.95]], layout=[[0.0, 1.0], [1.0, 0.0]])
    _, subject = run(photos, v, limit=1, options={"layout_weight": 0.0})
    _, angle = run(photos, v, limit=1, options={"layout_weight": 1.0})
    assert names(subject) == ["0.jpg"]
    assert names(angle) == ["1.jpg"]


def test_min_similarity_drops_weak_matches() -> None:
    photos, v = pool([[1.0, 0.0], [0.0, 1.0]], layout=[[1.0, 0.0], [0.0, 1.0]])
    _, kept = run(photos, v)
    _, strict = run(photos, v, options={"min_similarity": 0.5})
    assert len(kept[0].picks) == 2
    assert names(strict) == ["0.jpg"]


def test_nothing_above_min_similarity_notes_and_returns_nothing() -> None:
    photos, v = pool([[0.0, 1.0]], layout=[[0.0, 1.0]])
    ctx, out = run(photos, v, options={"min_similarity": 0.9})
    assert out == []
    assert ctx.trace.notes


def test_reference_in_the_pool_is_never_returned() -> None:
    photos, v = pool([[1.0, 0.0], [0.9, 0.4]])
    _, out = run(photos, v, reference_keys=[photos[0].content_key])
    assert names(out) == ["1.jpg"]


def test_best_of_several_references_counts() -> None:
    photos, v = pool([[1.0, 0.0], [0.0, 1.0]], layout=[[1.0, 0.0], [0.0, 1.0]])
    refs = view([[0.0, 1.0], [1.0, 0.0]], {"layout": [[0.0, 1.0], [1.0, 0.0]]})
    ctx = context(photos, v, references=refs)
    out = SameViewTask().run(ctx, SameViewTask().parse_options({"min_similarity": 0.99}))
    assert sorted(names(out)) == ["0.jpg", "1.jpg"]
    assert out[0].title == "Same view as 2 references"


def test_bursts_collapse_to_the_best_frame() -> None:
    t = datetime(2023, 6, 1, 8)
    when = [t, t + timedelta(seconds=2), t + timedelta(days=3)]
    photos, v = pool([[0.95, 0.31], [1.0, 0.0], [0.9, 0.4]], when=when, location="Mill")
    _, every = run(photos, v)
    assert names(every) == ["0.jpg", "1.jpg", "2.jpg"]
    _, out = run(photos, v, options={"collapse_bursts": True})
    assert names(out) == ["1.jpg", "2.jpg"]
    assert [a.name for a in out[0].picks[0].alternates] == ["0.jpg"]
    assert out[0].picks[0].burst_size == 2


def test_evidence_sums_to_score_with_layout_weight_split() -> None:
    photos, v = pool(
        [[1.0, 0.0], [0.6, 0.8], [0.2, 0.9]], layout=[[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]
    )
    _, out = run(photos, v, options={"layout_weight": 0.25})
    for pick in out[0].picks:
        weights = {e.signal: e.weight for e in pick.evidence}
        assert weights == {"semantic": pytest.approx(0.75), "layout": pytest.approx(0.25)}
        assert sum(e.contribution for e in pick.evidence) == pytest.approx(pick.score, abs=1e-3)
        assert all("cosine" in e.detail and "z=" in e.detail for e in pick.evidence)


def test_criteria_blend_in_when_given() -> None:
    emb = [[1.0, 0.0], [0.9, 0.4], [0.8, 0.6]]
    photos, v = pool(emb)
    _, plain = run(photos, v, limit=1)
    _, steered = run(photos, v, limit=1, criteria=[0.0, 1.0], describe_weight=0.8)
    assert names(plain) == ["0.jpg"]
    assert names(steered) == ["2.jpg"]
    pick = steered[0].picks[0]
    assert {e.signal for e in pick.evidence} == {"semantic", "layout", "criteria"}
    assert sum(e.weight for e in pick.evidence) == pytest.approx(1.0)
    assert sum(e.contribution for e in pick.evidence) == pytest.approx(pick.score, abs=1e-3)


def test_the_description_counts_thirty_percent_by_default() -> None:
    photos, v = pool([[1.0, 0.0], [0.9, 0.4]])
    _, out = run(photos, v, criteria=[0.0, 1.0])
    weights = {e.signal: e.weight for e in out[0].picks[0].evidence}
    assert SameViewTask.default_describe_weight == 0.3
    assert weights["criteria"] == pytest.approx(0.3)


def test_title_names_the_reference() -> None:
    photos, v = pool([[1.0, 0.0]])
    _, out = run(photos, v)
    assert out[0].title == "Same view as 0.jpg"
    assert out[0].score == pytest.approx(np.mean([p.score for p in out[0].picks]))


def test_signals_are_traced_aligned_with_the_pool() -> None:
    photos, v = pool([[1.0, 0.0], [0.9, 0.4], [0.5, 0.5]])
    ctx, _ = run(photos, v, reference_keys=[photos[0].content_key])
    for name in ("semantic", "layout", "match", "score"):
        assert len(ctx.trace.columns[name]) == 3
    assert np.isnan(ctx.trace.columns["score"][0])
    assert ctx.trace.stages


def test_missing_layout_feature_falls_back_to_subject_with_a_note() -> None:
    photos = [photo("0.jpg", captured_at=D0), photo("1.jpg", location="B", captured_at=D0)]
    v = view([[1.0, 0.0], [0.0, 1.0]])
    refs = view([[1.0, 0.0]])
    ctx = context(photos, v, references=refs, limit=1)
    out = SameViewTask().run(ctx, SameViewTask().parse_options({}))
    assert names(out)[0] == "0.jpg"
    assert any("layout" in n for n in ctx.trace.notes)


def test_empty_pool() -> None:
    ctx, out = run([], view(np.zeros((0, 2)), {"layout": np.zeros((0, 2))}))
    assert out == []
    assert ctx.trace.notes


def test_output_is_deterministic() -> None:
    photos, v = pool([[1.0, 0.0]] * 5, when=[D0] * 5)
    assert run(photos, v)[1] == run(photos, v)[1]


def test_unknown_layout_is_traced_as_nan_not_zero() -> None:
    photos = [photo("0.jpg", captured_at=D0)]
    ctx = context(photos, view([[1.0, 0.0]]), references=view([[1.0, 0.0]]))
    SameViewTask().run(ctx, SameViewTask().parse_options({}))
    assert np.isnan(ctx.trace.columns["layout"]).all()


def test_nan_features_do_not_win() -> None:
    photos, v = pool([[1.0, 0.0], [0.5, 0.5], [0.8, 0.6]])
    v.embeddings[0] = np.nan
    ctx, out = run(photos, v, limit=1)
    assert "0.jpg" not in names(out)
    assert any("invalid" in n for n in ctx.trace.notes)


def test_ui_metadata() -> None:
    assert SameViewTask.default_limit == 200
    assert SameViewTask.reference_label == "Reference photo"
    assert "oldest first" in SameViewTask.reference_help


# -- matching part by part: a shortlist, then two tiers -----------------------------------


def patch_ctx(
    shares: Sequence[float],
    *,
    when: Sequence[datetime | None] | None = None,
    criteria: ArrayLike | None = None,
    view_vectors: ArrayLike | None = None,
) -> TaskContext:
    n = len(shares)
    photos, v = pool(np.ones((n, 2)), when=when)
    if view_vectors is not None:
        v = view(v.embeddings, {"layout": v.features["layout"], "dinov2": unit(view_vectors)})
    refs = view(
        np.asarray([[1.0, 0.0]]), {"layout": np.asarray([[1.0, 0.0]]), "dinov2": unit([[1.0, 0.0]])}
    )
    ctx = context(photos, v, references=refs, criteria=criteria)
    return dataclasses.replace(
        ctx, patch_scores=np.asarray(shares, dtype=np.float32), view_feature="dinov2"
    )


def tiers(ctx: TaskContext, **options: Any) -> list[Group]:
    task = SameViewTask()
    return task.run(ctx, task.parse_options(options))


def test_the_shortlist_is_photos_alike_as_a_whole_best_first() -> None:
    ctx = patch_ctx([0.0] * 4, view_vectors=[[1.0, 0.1], [0.0, 1.0], [1.0, 0.5], [1.0, 0.3]])
    task = SameViewTask()
    shortlist = task.patch_shortlist(ctx, task.parse_options({"shortlist_similarity": 0.75}))
    assert shortlist is not None
    assert [ctx.photos[i].path.name for i in shortlist] == ["0.jpg", "3.jpg", "2.jpg"]


def test_a_long_shortlist_is_cut_to_its_size_and_says_so() -> None:
    ctx = patch_ctx([0.0] * 4, view_vectors=[[1.0, 0.1], [1.0, 0.2], [1.0, 0.3], [1.0, 0.4]])
    task = SameViewTask()
    options = task.parse_options({"shortlist_similarity": 0.5, "shortlist_size": 2})
    shortlist = task.patch_shortlist(ctx, options)
    assert shortlist is not None
    assert len(shortlist) == 2
    assert any("The 2 photos most alike" in n and "2 less alike" in n for n in ctx.trace.notes)


def test_when_no_photo_is_alike_enough_to_compare_it_says_which_option_to_lower() -> None:
    ctx = patch_ctx([0.0] * 3, view_vectors=[[0.0, 1.0], [0.1, 1.0], [0.2, 1.0]])
    task = SameViewTask()
    shortlist = task.patch_shortlist(ctx, task.parse_options({}))
    assert shortlist is not None
    assert len(shortlist) == 0
    (note,) = [n for n in ctx.trace.notes if "alike enough as a whole" in n]
    assert "shortlist_similarity" in note


def test_photos_never_compared_detail_by_detail_are_not_said_to_share_nothing() -> None:
    """Lowering possible_share cannot help when nothing was compared; and "0% shared"
    would claim a comparison that was never made."""
    ctx = patch_ctx([float("nan")] * 3)
    tiers(ctx)
    assert not any("possible_share" in n for n in ctx.trace.notes)
    assert np.isnan(ctx.trace.columns["patch_share"]).all()


def test_the_reference_is_never_shortlisted() -> None:
    photos, v = pool([[1.0, 0.0], [1.0, 0.1]])
    refs = view(np.asarray([[1.0, 0.0]]), {"layout": np.asarray([[1.0, 0.0]])})
    ctx = context(photos, v, references=refs, reference_keys=[photos[0].content_key])
    task = SameViewTask()
    shortlist = task.patch_shortlist(ctx, task.parse_options({}))
    assert shortlist is not None
    assert [ctx.photos[i].path.name for i in shortlist] == ["1.jpg"]  # the main embedding


def test_sure_matches_are_the_timeline_and_possible_ones_follow_by_share() -> None:
    when = [D0 + timedelta(days=d) for d in (3, 1, 2, 0, 4)]
    ctx = patch_ctx([0.30, 0.40, 0.20, 0.10, 0.22], when=when)
    sure, possible = tiers(ctx)
    assert sure.title.startswith("Same view as")
    assert [p.path.name for p in sure.picks] == ["1.jpg", "0.jpg"]  # oldest first
    assert possible.title == "Possibly the same view"
    assert [p.path.name for p in possible.picks] == ["4.jpg", "2.jpg"]  # closest first


def test_with_nothing_possible_there_is_one_group() -> None:
    (sure,) = tiers(patch_ctx([0.30, 0.05]))
    assert [p.path.name for p in sure.picks] == ["0.jpg"]


def test_with_nothing_sure_the_timeline_is_empty_and_the_possible_are_shown() -> None:
    sure, possible = tiers(patch_ctx([0.20, 0.05]))
    assert sure.picks == ()
    assert [p.path.name for p in possible.picks] == ["0.jpg"]


def test_with_no_match_at_all_a_note_says_how_close_the_closest_came() -> None:
    ctx = patch_ctx([0.12, 0.05])
    (sure,) = tiers(ctx)
    assert sure.picks == ()
    assert any("12%" in n and "possible_share" in n for n in ctx.trace.notes)


def test_photos_left_off_the_shortlist_are_no_match() -> None:
    sure, possible = tiers(patch_ctx([0.30, float("nan"), 0.2]))
    assert [p.path.name for p in sure.picks] == ["0.jpg"]
    assert [p.path.name for p in possible.picks] == ["2.jpg"]


def test_the_possible_tier_is_kept_short_and_says_how_many_more() -> None:
    ctx = patch_ctx([0.19, 0.20, 0.21, 0.22])
    _, possible = tiers(ctx, possible_limit=2)
    assert [p.path.name for p in possible.picks] == ["3.jpg", "2.jpg"]
    assert any("2 more" in n and "possibly" in n for n in ctx.trace.notes)


def test_possible_may_not_be_stricter_than_sure() -> None:
    with pytest.raises(InvalidRequestError, match="possible_share"):
        SameViewTask().parse_options({"sure_share": 0.2, "possible_share": 0.3})


def test_patch_shares_are_traced_and_are_the_evidence() -> None:
    ctx = patch_ctx([0.30, 0.20], criteria=(1.0, 0.0))
    sure, _ = tiers(ctx)
    assert np.allclose(ctx.trace.columns["patch_share"], [0.30, 0.20])
    (pick,) = sure.picks
    signals = {e.signal: e for e in pick.evidence}
    assert set(signals) == {"patches", "criteria"}
    assert signals["patches"].value == pytest.approx(0.30)
    assert "30%" in signals["patches"].detail
    assert sum(e.weight for e in pick.evidence) == pytest.approx(1.0)
