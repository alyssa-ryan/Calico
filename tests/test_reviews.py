"""Reviews: which question each answer answers, and the rules that read ratings (2b)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from calico.domain import Group, Pick, SelectionRequest, SelectionResult
from calico.errors import InvalidRequestError
from calico.labels import LabelItem, LabelSet, LabelStore, ShownBy, StoryArm, StoryItem, StoryPhoto
from calico.ratings_store import Answer, RatingsJournal
from calico.reviews import (
    Ratings,
    answered,
    labels_of,
    reviewed,
    story_label,
    taste_labels,
    unfit,
)
from calico.runs import RunStore
from tests import factories as f

if TYPE_CHECKING:
    from collections.abc import Callable

    from calico.ratings_store import Review

    type Fold = Callable[..., dict[Review, Answer]]

T0 = datetime(2026, 10, 1, 9, tzinfo=UTC)


def answer(use: str, value: int | None, *, key: str = "k", days: int = 0,
           scope: object = None) -> Answer:  # fmt: skip
    return Answer.model_validate(
        {"at": T0 + timedelta(days=days), "use": use, "scope": scope, "subject": [key],
         "value": value}
    )  # fmt: skip


@pytest.fixture
def current(tmp_path: Path) -> Callable[..., dict[Review, Answer]]:
    """Each review's rating after ``answers``, folded by a real journal."""

    def fold(*answers: Answer) -> dict[Review, Answer]:
        journal = RatingsJournal(tmp_path / "journal")
        journal.append(*answers)
        return journal.ratings()

    return fold


BEST_GENERAL = {"task": "best", "template": "general"}


# -- the taste rule (owner decision Q1) ------------------------------------------


def test_a_later_taste_sessions_dont_like_replaces_an_earlier_like(current: Fold) -> None:  # A05-1
    labels = taste_labels(current(answer("taste", 2), answer("taste", 0, days=3)))
    assert labels.liked == {"k": False}


def test_a_newer_not_sure_changes_only_its_own_review(current: Fold) -> None:
    labels = taste_labels(
        current(answer("compare", 2), answer("taste", 0, days=1), answer("taste", 1, days=2))
    )
    assert labels.liked == {"k": True}  # the taste review is now unsure; Compare's like stands
    assert labels.disagree == 0


def test_a_result_thumb_and_a_compare_rating_stay_two_reviews_and_the_newest_decides(
    current: Fold,
) -> None:
    thumbs_down = answer("result", -1, scope=BEST_GENERAL, days=5)
    labels = taste_labels(current(answer("compare", 2), thumbs_down))
    assert labels.liked == {"k": False}
    assert labels.from_session == {"k": False}
    assert labels.disagree == 1


def test_each_label_says_which_session_its_deciding_answer_came_from(current: Fold) -> None:
    """So evaluation can read how that session drew the photo (``LabelItem.stratum``)."""
    earlier = answer("taste", 2).model_copy(update={"session": "first"})
    later = answer("taste", 0, days=3).model_copy(update={"session": "second"})
    thumb = answer("result", 1, key="j", scope=BEST_GENERAL)
    labels = taste_labels(current(earlier, later, thumb))
    assert labels.session == {"k": "second", "j": None}


def test_not_sure_alone_is_left_out_and_counted(current: Fold) -> None:
    labels = taste_labels(current(answer("taste", 1), answer("taste", 2, key="j")))
    assert labels.liked == {"j": True}
    assert labels.unsure == 1


def test_thumbs_on_stories_and_other_reviews_never_reach_taste(current: Fold) -> None:
    on_a_story = answer("result", -1, scope={"task": "triptych", "template": "general"})
    labels = taste_labels(current(on_a_story, answer("anchor", 2), answer("describe", 0)))
    assert labels.liked == {}


def test_what_a_new_session_leaves_out_is_every_standing_answer_for_its_use(
    current: Fold,
) -> None:
    standing = current(
        answer("taste", 1, key="unsure"),
        answer("compare", 2, key="compared"),
        answer("result", 1, scope=BEST_GENERAL, key="thumbed"),
        answer("result", 1, scope={"task": "triptych", "template": "general"}, key="story"),
        answer("taste", 0, key="withdrawn"),
        answer("taste", None, key="withdrawn", days=1),
        answer("anchor", 2, key="anchor"),
    )
    assert reviewed(standing, "taste") == {"unsure", "compared", "thumbed"}
    assert reviewed(standing, "anchor") == {"anchor"}


def test_a_control_answer_is_a_review_of_its_own_and_never_a_taste_label(current: Fold) -> None:
    again = {"control": "yes"}
    asked = current(answer("compare", 2), answer("compare", 0, days=3, scope=again))
    assert len(asked) == 2  # the re-ask does not replace the answer it repeats
    assert taste_labels(asked).liked == {"k": True}  # and a flip does not relabel the photo
    assert reviewed(asked, "taste") == {"k"}


# -- writing answers -------------------------------------------------------------


@pytest.fixture
def ratings(tmp_path: Path) -> Ratings:
    return Ratings(
        LabelStore(tmp_path / "labels"),
        RatingsJournal(tmp_path / "ratings"),
        RunStore(tmp_path / "runs"),
        clock=lambda: T0,
    )


def saved(ratings: Ratings, kind: str = "taste", **extra: object) -> LabelSet:
    made = LabelSet.model_validate(
        {
            "set_id": f"20261001-090000-{kind[:4]}",
            "created_at": T0,
            "text": "fog",
            "folders": ["/a"],
            "scorer": "t",
            "kind": kind,
            "items": [LabelItem(content_key=k, path=Path("/a") / k) for k in ("a", "b")],
            **extra,
        }
    )
    ratings.labels.save(made)
    return made


def test_a_photo_rating_is_appended_and_the_set_file_is_never_rewritten(
    ratings: Ratings, tmp_path: Path
) -> None:
    made = saved(ratings, "describe")
    before = (tmp_path / "labels" / f"{made.set_id}.json").read_bytes()
    ratings.rate_photo(made.set_id, "a", 2)
    ratings.rate_photo(made.set_id, "b", 0)
    ratings.rate_photo(made.set_id, "b", None)  # withdrawn
    assert labels_of(made, ratings.current()) == {"a": 2}
    assert (tmp_path / "labels" / f"{made.set_id}.json").read_bytes() == before
    (first, *_) = ratings.journal.answers()
    assert (first.use, first.scope, first.session) == ("describe", "fog", made.set_id)


def test_a_compare_rating_records_what_showed_the_photo(ratings: Ratings) -> None:
    made = saved(
        ratings,
        "compare",
        items=[
            LabelItem(
                content_key="a",
                path=Path("/a/a"),
                shown_by=(ShownBy(template="yours", run_id="r1", rank=2),),
            )
        ],
    )
    ratings.rate_photo(made.set_id, "a", 2)
    (only,) = ratings.journal.answers()
    assert [(s.template, s.run, s.rank) for s in only.shown] == [("yours", "r1", 2)]


def test_a_photo_not_in_the_set_is_refused(ratings: Ratings) -> None:
    made = saved(ratings)
    with pytest.raises(InvalidRequestError, match="not in this labelling set"):
        ratings.rate_photo(made.set_id, "zzz", 2)


def test_two_taste_sessions_share_one_rating_of_a_photo(ratings: Ratings) -> None:
    first = saved(ratings)
    second = saved(ratings, set_id="20261003-090000-tast")
    ratings.rate_photo(first.set_id, "a", 2)
    ratings.rate_photo(second.set_id, "a", 0)
    assert labels_of(first, ratings.current()) == {"a": 0}  # the newer answer, on both pages


def story_set(ratings: Ratings) -> LabelSet:
    return saved(
        ratings,
        "stories",
        items=[],
        stories=[
            StoryItem(
                run_id="r",
                photos=tuple(StoryPhoto(content_key=k, path=Path("/a") / k) for k in "xyz"),
                evidence=(),
                arms=(StoryArm(template="general", variety=True),),
            )
        ],
    )


def test_a_story_is_rated_whole_and_its_fit_marks_are_kept_apart(ratings: Ratings) -> None:
    made = story_set(ratings)
    ratings.rate_story(made.set_id, 0, 0, doesnt_fit=["z"])
    (story,) = made.stories
    now = ratings.current()
    assert story_label(story, now) == 0
    assert unfit(story, now) == {"z"}
    assert {a.use for a in ratings.journal.answers()} == {"story", "story_fit"}
    ratings.rate_story(made.set_id, 0, 2, doesnt_fit=[])
    now = ratings.current()
    assert (story_label(story, now), unfit(story, now)) == (2, set())


def test_a_story_or_photo_not_in_the_set_is_refused(ratings: Ratings) -> None:
    made = story_set(ratings)
    with pytest.raises(InvalidRequestError, match="No story 3"):
        ratings.rate_story(made.set_id, 3, 2, doesnt_fit=[])
    with pytest.raises(InvalidRequestError, match="not in story 0"):
        ratings.rate_story(made.set_id, 0, 2, doesnt_fit=["q"])


def run_of(ratings: Ratings, template: str) -> SelectionResult:
    picks = tuple(
        Pick(path=Path("/a") / k, content_key=k, location="Mill", captured_at=None, score=1)
        for k in ("a", "b")
    )
    result = SelectionResult(
        run_id=f"20261001-090000-{template}",
        created_at=T0,
        request=SelectionRequest(task="best", folders=(Path("/a"),), template=template),
        groups=(Group(title="g", score=1, picks=picks),),
    )
    ratings.runs.save(result, [f.photo("a.jpg"), f.photo("b.jpg")], {})
    return result


def test_a_thumb_is_a_review_of_its_task_and_template_and_can_be_cleared(
    ratings: Ratings,
) -> None:
    general, yours = run_of(ratings, "general"), run_of(ratings, "yours")
    ratings.rate_result(general.run_id, "b", 1)
    ratings.rate_result(yours.run_id, "b", -1, note="soft")
    assert ratings.on_result(general) == {"b": 1}
    assert ratings.on_result(yours) == {"b": -1}
    (_, last) = ratings.journal.answers()
    assert (last.note, last.shown[0].rank, last.shown[0].run) == ("soft", 1, yours.run_id)
    ratings.rate_result(general.run_id, "b", None)
    assert ratings.on_result(general) == {}


def test_a_thumb_on_a_photo_the_run_did_not_pick_is_refused(ratings: Ratings) -> None:
    general = run_of(ratings, "general")
    with pytest.raises(InvalidRequestError, match="not part of this run"):
        ratings.rate_result(general.run_id, "zzz", 1)


def test_a_control_photo_has_its_own_review_and_nothing_says_who_showed_it(
    ratings: Ratings,
) -> None:
    made = saved(
        ratings,
        "compare",
        items=[
            LabelItem(
                content_key="a",
                path=Path("/a/a"),
                shown_by=(ShownBy(template="yours", run_id="r1", rank=0),),
            ),
            LabelItem(content_key="c", path=Path("/a/c"), control=True),
        ],
    )
    ratings.rate_photo(made.set_id, "a", 2)
    ratings.rate_photo(made.set_id, "c", 0)
    assert labels_of(made, ratings.current()) == {"a": 2, "c": 0}  # the page shows both answers
    assert labels_of(made, ratings.current(), controls=False) == {"a": 2}  # evaluation does not
    assert answered(made, ratings.current()) == 2
    by_photo = {x.subject[0]: x for x in ratings.journal.answers()}
    assert by_photo["c"].scope == {"control": "yes"}
    assert by_photo["c"].shown == ()
    assert taste_labels(ratings.current()).liked == {"a": True}


# -- survivors of the mutation run, 10 November 2026 --------------------


def test_the_newest_clear_answer_decides_whichever_use_sorts_last(current: Fold) -> None:
    thumbs_up = answer("result", 1, scope=BEST_GENERAL, days=0)
    labels = taste_labels(current(thumbs_up, answer("compare", 0, days=5)))
    assert labels.liked == {"k": False}  # Compare's newer "don't like", though "result" sorts later
    assert labels.disagree == 1
