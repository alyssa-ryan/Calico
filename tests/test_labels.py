from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from calico.domain import Evidence
from calico.errors import (
    InvalidRequestError,
    LabelSetNotFoundError,
    LabelSetUnreadableError,
)
from calico.labels import (
    LabelItem,
    LabelSet,
    LabelStore,
    ShownBy,
    StoryArm,
    StoryItem,
    StoryPhoto,
    TasteSeen,
)
from calico.ratings_store import RatingsJournal
from calico.reviews import Ratings, answered, labels_of, story_label, unfit
from calico.runs import RunStore

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


def rating(store: LabelStore) -> Ratings:
    """Answers to ``store``'s sets, in a journal beside it."""
    home = store.labels_dir
    return Ratings(store, RatingsJournal(home / "ratings"), RunStore(home / "runs"))


def label_set(
    store: LabelStore, n: int = 3, text: str = "fog over water", now: datetime = NOW
) -> LabelSet:
    items = tuple(
        LabelItem(content_key=f"k{i}", path=Path(f"/archive/Mill/{i}.jpg")) for i in range(n)
    )
    made = LabelSet(
        set_id=store.new_id(now),
        created_at=now,
        text=text,
        folders=(Path("/archive"),),
        scorer="baseline@1",
        items=items,
    )
    store.save(made)
    return made


def test_a_saved_set_loads_back_unchanged(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    made = label_set(store)
    assert store.load(made.set_id) == made
    assert answered(made, rating(store).current()) == 0


def test_labels_are_saved_one_at_a_time_and_can_be_cleared(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    made = label_set(store)
    ratings = rating(store)
    ratings.rate_photo(made.set_id, "k0", 2)
    ratings.rate_photo(made.set_id, "k2", 0)
    assert labels_of(made, ratings.current()) == {"k0": 2, "k2": 0}
    assert store.load(made.set_id) == made  # the set itself is written once
    ratings.rate_photo(made.set_id, "k0", None)
    assert answered(made, ratings.current()) == 1


def test_labelling_a_photo_outside_the_set_is_refused(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    made = label_set(store)
    with pytest.raises(InvalidRequestError, match="not in this labelling set"):
        rating(store).rate_photo(made.set_id, "other", 1)


@pytest.mark.parametrize("set_id", ["nope", "../runs", "a/b", ""])
def test_unknown_or_malformed_ids_are_not_found(tmp_path: Path, set_id: str) -> None:
    with pytest.raises(LabelSetNotFoundError):
        LabelStore(tmp_path).load(set_id)


def test_all_sets_newest_first(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    first = label_set(store, text="a")
    label_set(store, text="b", now=datetime(2026, 10, 14, tzinfo=UTC))
    assert [s.text for s in store.all()] == ["b", "a"]
    assert store.all()[1] == first


def test_a_set_saved_by_a_newer_calico_is_left_out_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An older release reading labels a newer one wrote (a new kind of session) must
    still show every set it can read."""
    store = LabelStore(tmp_path)
    readable = label_set(store, text="a")
    newer = readable.model_dump(mode="json") | {
        "set_id": "20991231-000000-0000",
        "kind": "a kind of session from a newer Calico",
    }
    (tmp_path / "20991231-000000-0000.json").write_text(json.dumps(newer))
    assert store.all() == [readable]
    assert "20991231-000000-0000" in caplog.text


def test_deleting_a_set_removes_it(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    made = label_set(store)
    store.delete(made.set_id)
    assert store.all() == []
    with pytest.raises(LabelSetNotFoundError):
        store.delete(made.set_id)


def test_a_deleted_set_goes_to_the_trash_not_away(tmp_path: Path) -> None:
    """A set cannot be recomputed, so a delete must be undoable by hand."""
    store = LabelStore(tmp_path)
    made = label_set(store)
    store.delete(made.set_id)
    (trashed,) = (tmp_path / ".trash").glob(f"{made.set_id}.*.json")
    assert LabelSet.model_validate_json(trashed.read_text()) == made


def test_a_set_that_cannot_be_read_is_left_out_named_and_left_untouched(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    good = label_set(store)
    bad = tmp_path / "20990101-000000-bad0.json"
    bad.write_text('{"set_id": "x"')  # truncated by a crash, or written by another branch
    assert store.all() == [good]
    ((path, reason),) = store.unreadable()
    assert path == bad
    assert reason
    with pytest.raises(LabelSetUnreadableError, match="left untouched"):
        store.load(bad.stem)
    assert bad.read_text() == '{"set_id": "x"'


def test_a_set_that_cannot_be_read_can_still_be_deleted(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    bad = tmp_path / "20990101-000000-bad0.json"
    bad.write_text("not json")
    store.delete(bad.stem)
    assert not bad.exists()
    assert list((tmp_path / ".trash").glob("20990101-000000-bad0.*.json"))


def test_a_set_from_a_newer_calico_is_read_rated_and_never_rewritten(tmp_path: Path) -> None:
    """An older Calico would drop the fields it does not know if it saved the set; answers
    go to the journal, so the set never needs saving."""
    store = LabelStore(tmp_path)
    made = label_set(store)
    path = tmp_path / f"{made.set_id}.json"
    newer = path.read_text().replace('"set_id"', '"schema_version": 99, "future": 1, "set_id"')
    path.write_text(newer)
    assert store.load(made.set_id).set_id == made.set_id
    rating(store).rate_photo(made.set_id, "k0", 2)
    assert path.read_text() == newer


def test_saving_a_new_set_never_replaces_one_with_the_same_id(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    made = label_set(store)
    with pytest.raises(InvalidRequestError, match="already exists"):
        store.save(made.model_copy(update={"text": "something else"}))
    assert store.load(made.set_id) == made


def test_labels_from_two_processes_are_all_kept(tmp_path: Path) -> None:
    """The web app and a second Calico (another serve, the CLI) each have their own store;
    labelling the same set at once must lose nothing."""
    first, second = rating(LabelStore(tmp_path)), rating(LabelStore(tmp_path))
    made = label_set(first.labels, n=40)
    threads = [
        threading.Thread(target=ratings.rate_photo, args=(made.set_id, f"k{i}", i % 3))
        for i, ratings in ((i, first if i % 2 else second) for i in range(40))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert answered(made, first.current()) == 40


def test_concurrent_labels_are_all_kept(tmp_path: Path) -> None:
    """The web app labels from many request threads at once; no click may be lost."""
    ratings = rating(LabelStore(tmp_path))
    made = label_set(ratings.labels, n=40)
    threads = [
        threading.Thread(target=ratings.rate_photo, args=(made.set_id, f"k{i}", i % 3))
        for i in range(40)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert answered(made, ratings.current()) == 40


def test_a_missing_directory_means_no_sets(tmp_path: Path) -> None:
    assert LabelStore(tmp_path / "absent").all() == []


def test_sets_saved_before_kinds_existed_are_descriptions(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    made = label_set(store)
    raw = (tmp_path / f"{made.set_id}.json").read_text()
    assert '"kind"' in raw
    old = raw.replace('"kind":"describe",', "").replace('"kind": "describe",', "")
    (tmp_path / f"{made.set_id}.json").write_text(old)
    loaded = store.load(made.set_id)
    assert (loaded.kind, loaded.reference) == ("describe", None)


def test_a_same_view_set_keeps_its_reference(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    made = label_set(store).model_copy(
        update={
            "set_id": store.new_id(NOW),
            "kind": "same_view",
            "reference": Path("/archive/Mill/ref.jpg"),
        }
    )
    store.save(made)
    assert store.load(made.set_id).reference == Path("/archive/Mill/ref.jpg")


def test_a_compare_set_remembers_which_templates_showed_each_photo(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    both = (
        ShownBy(template="general", run_id="g", rank=3),
        ShownBy(template="yours", run_id="y", rank=0),
    )
    made = label_set(store).model_copy(update={"set_id": store.new_id(NOW), "kind": "compare"})
    made = made.model_copy(
        update={"items": (made.items[0].model_copy(update={"shown_by": both}), *made.items[1:])}
    )
    store.save(made)
    assert store.load(made.set_id).items[0].shown_by == both
    assert made.items[1].shown_by == ()  # every other kind of set shows by no template


def story_set(store: LabelStore, n: int = 2) -> LabelSet:
    stories = tuple(
        StoryItem(
            run_id=f"run{i}",
            photos=tuple(
                StoryPhoto(content_key=f"s{i}k{j}", path=Path(f"/archive/Mill/{i}-{j}.jpg"))
                for j in range(3)
            ),
            evidence=(Evidence(signal="strength", value=0.8, weight=0.4),),
            arms=(StoryArm(template="general", variety=i == 0),),
        )
        for i in range(n)
    )
    made = LabelSet(
        set_id=store.new_id(NOW),
        created_at=NOW,
        text="Stories: Mill",
        folders=(Path("/archive/Mill"),),
        scorer="triptych",
        items=(),
        kind="stories",
        stories=stories,
    )
    store.save(made)
    return made


def test_a_story_is_rated_whole_with_the_photos_that_do_not_fit_it(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    made = story_set(store)
    ratings = rating(store)
    assert store.load(made.set_id) == made
    assert (answered(made, ratings.current()), made.total) == (0, 2)
    ratings.rate_story(made.set_id, 1, 0, doesnt_fit=["s1k2"])
    now = ratings.current()
    assert story_label(made.stories[1], now) == 0
    assert unfit(made.stories[1], now) == {"s1k2"}
    assert story_label(made.stories[0], now) is None
    assert (answered(made, now), made.total) == (1, 2)
    ratings.rate_story(made.set_id, 1, None, doesnt_fit=[])
    now = ratings.current()
    assert (story_label(made.stories[1], now), unfit(made.stories[1], now)) == (None, set())


def test_rating_a_story_or_photo_outside_the_set_is_refused(tmp_path: Path) -> None:
    store = LabelStore(tmp_path)
    made = story_set(store)
    with pytest.raises(InvalidRequestError, match="No story 2"):
        rating(store).rate_story(made.set_id, 2, 1, doesnt_fit=[])
    with pytest.raises(InvalidRequestError, match="not in story 0"):
        rating(store).rate_story(made.set_id, 0, 1, doesnt_fit=["s1k0"])


def test_stories_rated_before_arranged_stories_were_told_today_s_way() -> None:
    arm = StoryArm.model_validate({"template": "general", "variety": True})
    assert arm.way == "today"


def test_stories_saved_as_arranged_read_as_the_arranged_way() -> None:
    saved = {"template": "general", "variety": True, "arranged": True}
    assert StoryArm.model_validate(saved).way == "arranged"
    assert StoryArm.model_validate({**saved, "arranged": False}).way == "today"
    assert "arranged" not in StoryArm(template="yours", variety=True, way="roles").model_dump()


def test_stories_told_by_roles_still_read_after_the_slot_went_to_another_way() -> None:
    """Sessions of 28 October to 2 November 2026 told by roles; later ones by the variant
    under test. Both read, so ``calico eval stories`` reports each."""
    for way in ("roles", "deeper", "likely_anchors"):
        saved = {"template": "general", "variety": True, "way": way}
        assert StoryArm.model_validate(saved).way == way


def test_a_taste_stamp_from_before_cross_fitting_loads_as_not_cross_fitted() -> None:
    old = TasteSeen.model_validate({"version": "abc123", "seen": False})
    assert old.cross_fit is False
    fitted = TasteSeen(version="abc123", seen=False, cross_fit=True)
    assert TasteSeen.model_validate_json(fitted.model_dump_json()) == fitted
