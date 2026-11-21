"""Moving every rating you gave into the journal, once (plan, Workstream 2b: migration)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from calico.catalog.store import Catalog
from calico.domain import Group, Pick, SelectionRequest, SelectionResult
from calico.labels import (
    LabelItem,
    LabelSet,
    LabelStore,
    ShownBy,
    StoryArm,
    StoryItem,
    StoryPhoto,
)
from calico.ratings_migration import migrate, open_journal
from calico.ratings_store import RatingsJournal
from calico.runs import RunStore
from calico.settings import Settings
from tests import factories as f

if TYPE_CHECKING:
    from calico.labels import Label

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
NOW = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)


class Home:
    """A calico home holding every kind of rating the old files keep."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.labels = LabelStore(root / "labels")
        self.runs = RunStore(root / "runs")
        self.catalog = Catalog(root / "catalog.sqlite")
        self.journal = RatingsJournal(root / "ratings", clock=lambda: NOW)

    def migrate(self) -> Counter[str]:
        self.report = migrate(self.labels, self.catalog, self.runs, self.journal)
        return Counter(a.use for a in self.journal.answers())

    def lines(self) -> list[dict[str, Any]]:
        return [json.loads(a.line()) for a in self.journal.answers()]

    def label_set(
        self,
        kind: str,
        labels: dict[str, Label | None],
        *,
        at: datetime = T0,
        **extra: object,
    ) -> LabelSet:
        made = LabelSet.model_validate(
            {
                "set_id": f"{at:%Y%m%d-%H%M%S}-{kind[:4]}",
                "created_at": at,
                "text": extra.pop("text", f"a {kind} session"),
                "folders": [self.root / "archive"],
                "scorer": "test",
                "kind": kind,
                "items": [
                    LabelItem(content_key=k, path=self.root / "archive" / k, label=v)
                    for k, v in labels.items()
                ],
                **extra,
            }
        )
        self.labels.save(made)
        return made

    def run(self, run_id: str, task: str, template: str, keys: list[str]) -> None:
        picks = tuple(
            Pick(path=Path("/a") / k, content_key=k, location="Mill", captured_at=None, score=1)
            for k in keys
        )
        self.runs.save(
            SelectionResult(
                run_id=run_id,
                created_at=T0,
                request=SelectionRequest(task=task, folders=(Path("/a"),), template=template),
                groups=(Group(title="g", score=1, picks=picks),),
            ),
            [f.photo(f"{k}.jpg") for k in keys],
            {},
        )


@pytest.fixture
def home(tmp_path: Path) -> Home:
    return Home(tmp_path / "home")


def story(
    keys: tuple[str, str, str], label: Label | None, unfit: tuple[str, ...] = ()
) -> StoryItem:
    return StoryItem(
        run_id="20261001-090000-tell",
        photos=tuple(
            StoryPhoto(content_key=k, path=Path("/a") / k, fits=k not in unfit) for k in keys
        ),
        evidence=(),
        arms=(
            StoryArm(template="general", variety=True, way="today"),
            StoryArm(template="yours", variety=True, way="roles"),
        ),
        label=label,
    )


def test_each_session_kind_becomes_answers_of_its_use(home: Home) -> None:
    home.label_set("describe", {"a": 2, "b": 0, "c": None}, text="fog")
    home.label_set("same_view", {"a": 1}, reference=Path("/a/ref.jpg"))
    home.label_set("taste", {"a": 2, "b": 1})
    home.label_set("anchor", {"d": 0})
    home.label_set(
        "compare",
        {},
        items=[
            LabelItem(
                content_key="e",
                path=Path("/a/e"),
                label=2,
                shown_by=(ShownBy(template="yours", run_id="r9", rank=4),),
            )
        ],
    )
    assert home.migrate() == {"describe": 2, "same_view": 1, "taste": 2, "anchor": 1, "compare": 1}
    by_use = {(line["use"], line["subject"][0]): line for line in home.lines()}
    fog = by_use["describe", "a"]
    assert fog == {
        "schema": 1,
        "at": "2026-10-01T09:00:00Z",
        "use": "describe",
        "scope": "fog",
        "subject": ["a"],
        "value": 2,
        "session": "20261001-090000-desc",
        "shown": [],
        "taste": None,
        "approximate_time": True,
        "source": "labels/20261001-090000-desc.json",
    }
    assert by_use["same_view", "a"]["scope"] == "/a/ref.jpg"
    assert by_use["taste", "b"]["value"] == 1  # not sure is an answer, kept as one
    assert by_use["compare", "e"]["shown"] == [
        {"template": "yours", "run": "r9", "rank": 4, "way": None}
    ]


def test_stories_become_story_verdicts_and_fit_marks_never_thumbs(home: Home) -> None:
    home.label_set(
        "stories",
        {},
        stories=[
            story(("a", "b", "c"), 2),
            story(("d", "e", "f"), 0, unfit=("f",)),
            story(("g", "h", "i"), None, unfit=("g",)),  # a fit mark without a verdict
            story(("j", "k", "l"), None),  # not rated at all
        ],
    )
    assert home.migrate() == {"story": 2, "story_fit": 2}
    lines = home.lines()
    verdicts = {tuple(x["subject"]): x["value"] for x in lines if x["use"] == "story"}
    assert verdicts == {("a", "b", "c"): 2, ("d", "e", "f"): 0}
    fits = {tuple(x["subject"]): x["value"] for x in lines if x["use"] == "story_fit"}
    assert fits == {("d", "e", "f", "f"): False, ("g", "h", "i", "g"): False}
    assert lines[0]["shown"] == [
        {"template": "general", "run": None, "rank": None, "way": "today"},
        {"template": "yours", "run": None, "rank": None, "way": "roles"},
    ]


def test_thumbs_become_result_answers_scoped_by_their_runs_task_and_template(
    home: Home,
) -> None:
    home.run("20261002-100000-best", "best", "yours", ["x", "y"])
    home.run("20261002-110000-trip", "triptych", "general", ["z"])
    home.catalog.add_feedback("20261002-100000-best", Path("/a/y"), "y", 1)
    home.catalog.add_feedback("20261002-110000-trip", Path("/a/z"), "z", -1, "doesn't fit")
    home.catalog.add_feedback("20261002-120000-gone", Path("/a/w"), "w", 1)
    assert home.migrate() == {"result": 3}
    best, trip, gone = home.lines()
    assert best["scope"] == {"task": "best", "template": "yours"}
    assert best["shown"] == [
        {"template": "yours", "run": "20261002-100000-best", "rank": 1, "way": None}
    ]
    assert best["value"] == 1
    assert "approximate_time" not in best  # 👍/👎 kept their own time
    assert best["source"] == "catalog feedback"
    assert trip["scope"] == {"task": "triptych", "template": "general"}
    assert trip["note"] == "doesn't fit"
    assert gone["scope"] == {"task": None}
    assert home.report.lost_runs == 1


def test_counts_are_checked_against_the_sources(home: Home) -> None:
    home.label_set("taste", {"a": 2, "b": 0, "c": None})
    home.label_set("stories", {}, stories=[story(("a", "b", "c"), 1, unfit=("a",))])
    home.catalog.add_feedback("20261002-120000-gone", Path("/a/w"), "w", 1)
    home.migrate()
    report = home.report
    assert report.complete
    assert {c.use: (c.found, c.written, c.in_journal) for c in report.counts if c.found} == {
        "taste": (2, 2, 2),
        "story": (1, 1, 1),
        "story_fit": (1, 1, 1),
        "result": (1, 1, 1),
    }
    assert report.sets == {"taste": 1, "stories": 1}
    assert report.feedback_rows == 1


def test_migrating_twice_adds_nothing(home: Home) -> None:
    home.label_set("taste", {"a": 2, "b": 0})
    home.catalog.add_feedback("20261002-120000-gone", Path("/a/w"), "w", 1)
    home.migrate()
    first = (home.root / "ratings" / "2026-10.jsonl").read_text()
    home.migrate()
    assert (home.root / "ratings" / "2026-10.jsonl").read_text() == first
    assert home.report.complete
    assert sum(c.written for c in home.report.counts) == 0


def test_a_migration_cut_short_resumes_where_it_stopped(home: Home) -> None:
    home.label_set("taste", {"a": 2, "b": 0, "c": 2, "d": 0})
    home.migrate()
    month = home.root / "ratings" / "2026-10.jsonl"
    whole = month.read_text().splitlines()
    month.write_text(whole[0] + "\n" + whole[1][:20])  # a crash in the second line
    home.migrate()
    assert home.report.complete
    assert sum(c.written for c in home.report.counts) == 3
    assert sorted(a.subject[0] for a in home.journal.answers()) == ["a", "b", "c", "d"]


def test_the_newer_session_is_the_rating_whatever_order_the_files_are_read(home: Home) -> None:
    home.label_set("taste", {"a": 2}, at=T0)
    home.label_set("taste", {"a": 0}, at=T0 + timedelta(days=3))
    home.migrate()
    (rating,) = home.journal.ratings().values()
    assert (rating.value, rating.session) == (0, "20261004-090000-tast")


def test_thumbs_on_a_photo_keep_their_order(home: Home) -> None:
    home.run("20261002-100000-best", "best", "general", ["x"])
    home.catalog.add_feedback("20261002-100000-best", Path("/a/x"), "x", 1)
    home.catalog.add_feedback("20261002-100000-best", Path("/a/x"), "x", -1)
    home.migrate()
    (rating,) = home.journal.ratings().values()
    assert rating.value == -1


def test_the_old_files_are_read_and_never_changed(home: Home) -> None:
    made = home.label_set("taste", {"a": 2})
    home.catalog.add_feedback("20261002-120000-gone", Path("/a/w"), "w", 1)
    label_file = home.root / "labels" / f"{made.set_id}.json"
    before = label_file.read_bytes()
    home.migrate()
    assert label_file.read_bytes() == before
    assert len(home.catalog.feedback()) == 1


def test_a_set_that_cannot_be_read_is_named_and_left_alone(home: Home) -> None:
    home.label_set("taste", {"a": 2})
    damaged = home.root / "labels" / "20261005-000000-bad.json"
    damaged.write_text("{not json", encoding="utf-8")
    home.migrate()
    ((path, _why),) = home.report.unreadable
    assert path == damaged
    assert damaged.read_text(encoding="utf-8") == "{not json"
    assert home.report.complete  # complete for what could be read


def test_an_empty_home_migrates_to_nothing(home: Home) -> None:
    assert home.migrate() == {}
    assert home.report.complete


# -- opening the journal ---------------------------------------------------------


def settings_for(home: Home) -> Settings:
    return Settings(home=home.root)


def test_the_journal_is_migrated_the_first_time_it_is_opened(home: Home) -> None:
    home.label_set("taste", {"a": 2})
    journal = open_journal(settings_for(home))
    assert [a.subject for a in journal.answers()] == [("a",)]
    assert (home.root / "ratings" / ".migrated").is_file()


def test_once_migrated_opening_never_migrates_again(home: Home) -> None:
    made = home.label_set("taste", {"a": 2})
    open_journal(settings_for(home))
    later = made.model_copy(update={"set_id": "20261009-000000-late"})
    home.labels.save(later)  # an older Calico, rolled back to, saved a session
    assert len(list(open_journal(settings_for(home)).answers())) == 1


def test_a_first_open_cut_short_migrates_again_on_the_next(home: Home) -> None:
    home.label_set("taste", {"a": 2, "b": 0})
    home.migrate()  # wrote its lines, then stopped before noting it was done
    month = home.root / "ratings" / "2026-10.jsonl"
    month.write_text(month.read_text().splitlines()[0] + "\n")
    journal = open_journal(settings_for(home))
    assert sorted(a.subject[0] for a in journal.answers()) == ["a", "b"]
