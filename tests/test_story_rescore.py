"""A rated story is re-scored as the task scored it (audit 10 November 2026, B-2, SY-1).

``calico eval stories`` used to re-add a story's score from its strength and the mean of its
connections, which ignores ``distinct`` (31 October) and turns the roles' weights into
connection weight. The stories here come from the task itself, so what is checked is that the
re-score agrees with the task, not with a copy of its formula."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from calico.evaluation import RatedStory, _story_score
from calico.tasks.triptych import TriptychTask
from tests.factories import context
from tests.tasks.test_triptych import (
    THEME_ONLY,
    echo_and_carousel,
    keys,
    look_alikes_and_relatives,
    make_models,
    role_models,
    run,
    world,
)

if TYPE_CHECKING:
    from calico.domain import Group, Photo
    from calico.selection.context import FeatureView, TaskContext

NEAR_DUPLICATE = 0.95
OPTIONS = {**THEME_ONLY, "strength_weight": 0.4}
"""The task's defaults for variety (0.25 of the side, falling from 0.7) and for distinct."""


def rated(group: Group, photos: list[Photo], v: FeatureView) -> RatedStory:
    """The story as the owner's ratings keep it: its evidence and its photos' embeddings."""
    rows = {p.content_key: i for i, p in enumerate(photos)}
    vectors = np.stack([v.embeddings[rows[k]] for k in keys(group)])
    return RatedStory(
        run_id="r", vectors=vectors, marks=(0, 0, 0), evidence=group.evidence, stated="liked"
    )


def told(options: dict[str, Any], **kw: Any) -> tuple[list[Photo], FeatureView, Group, Any]:
    emb = look_alikes_and_relatives()
    photos, v = world(len(emb), emb=emb, quality=np.linspace(0.9, 0.1, len(emb)).tolist())
    ctx, out = run(photos, v, limit=1, options=options, **kw)
    return photos, v, out[0], ctx


def scorer(ctx: TaskContext, options: dict[str, Any]) -> Any:
    task = TriptychTask()
    return task.scorer(ctx, task.parse_options(options), include=[])


def test_counting_variety_as_the_task_does_gives_back_the_recorded_score() -> None:
    photos, v, group, _ = told(OPTIONS)
    assert {"variety", "distinct"} <= {e.signal for e in group.evidence}
    story = rated(group, photos, v)
    task = TriptychTask().parse_options(OPTIONS)
    again = _story_score(story, task.too_alike, task.variety_weight, NEAR_DUPLICATE)
    assert again == pytest.approx(group.score, abs=1e-6)


def test_a_story_told_by_roles_keeps_its_roles_and_takes_no_variety() -> None:
    photos, v = echo_and_carousel()
    options = {"connections": ["theme"], "burst_window_seconds": 0, "tell_by": "roles"}
    _, out = run(photos, v, limit=1, options=options, models=role_models(), view_feature="v")
    story = rated(out[0], photos, v)
    assert {"bond", "belong", "arrangement", "fresh", "anchor"} <= {
        e.signal for e in out[0].evidence
    }
    # roles take no variety, so counting none of it leaves the story as it was told
    assert _story_score(story, None, 0.0, NEAR_DUPLICATE) == pytest.approx(out[0].score, abs=1e-6)
    assert _story_score(story, 0.7, 0.0, NEAR_DUPLICATE) == pytest.approx(out[0].score, abs=1e-6)


@pytest.mark.parametrize(("too_alike", "weight"), [(0.8, 0.5), (0.6, 0.25), (0.9, 0.0)])
def test_counting_variety_another_way_is_what_the_task_scores_with_those_options(
    too_alike: float, weight: float
) -> None:
    photos, v, group, _ = told(OPTIONS)
    ctx = context(photos, v, models=make_models(), limit=1)
    other = scorer(ctx, {**OPTIONS, "too_alike": too_alike, "variety_weight": weight})
    rows = [other.row(k) for k in keys(group)]
    story = rated(group, photos, v)
    assert _story_score(story, too_alike, weight, NEAR_DUPLICATE) == pytest.approx(
        other.score(*rows), abs=1e-6
    )


def test_no_variety_is_the_task_told_without_variety_and_keeps_distinct() -> None:
    photos, v, group, _ = told(OPTIONS)
    ctx = context(photos, v, models=make_models(), limit=1)
    plain = scorer(ctx, {**OPTIONS, "variety_weight": 0.0})
    rows = [plain.row(k) for k in keys(group)]
    story = rated(group, photos, v)
    without = _story_score(story, None, 0.0, NEAR_DUPLICATE)
    assert without == pytest.approx(plain.score(*rows), abs=1e-6)
    assert without != pytest.approx(group.score, abs=1e-3)  # it was told with variety
    assert any(e.signal == "distinct" for e in group.evidence)  # which the old re-score lost
