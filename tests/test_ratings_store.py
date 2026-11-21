"""The ratings journal: every answer you gave, one line each (plan, Workstream 2b)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from calico.ratings_store import Answer, RatingsJournal, Shown, review

if TYPE_CHECKING:
    from pathlib import Path

T0 = datetime(2026, 10, 28, 14, 3, 11, tzinfo=timezone(timedelta(hours=1)))


def answer(
    value: int | bool | None,
    *,
    use: str = "taste",
    subject: tuple[str, ...] = ("k1",),
    scope: str | dict[str, str | None] | None = None,
    at: datetime = T0,
) -> Answer:
    return Answer.model_validate(
        {"at": at, "use": use, "scope": scope, "subject": subject, "value": value}
    )


def journal(tmp_path: Path, now: datetime = T0) -> RatingsJournal:
    return RatingsJournal(tmp_path / "ratings", clock=lambda: now)


def test_an_answer_is_one_line_in_the_month_it_was_written(tmp_path: Path) -> None:
    store = journal(tmp_path)
    store.append(
        answer(2, use="story", subject=("a", "b", "c")).model_copy(
            update={
                "session": "20261028-140001-ab12",
                "shown": (Shown(template="yours", run="r1", rank=3, way="roles"),),
                "taste": "taste@9f3c",
            }
        )
    )
    (line,) = (tmp_path / "ratings" / "2026-10.jsonl").read_text().splitlines()
    assert json.loads(line) == {
        "schema": 1,
        "at": "2026-10-28T14:03:11+01:00",
        "use": "story",
        "scope": None,
        "subject": ["a", "b", "c"],
        "value": 2,
        "session": "20261028-140001-ab12",
        "shown": [{"template": "yours", "run": "r1", "rank": 3, "way": "roles"}],
        "taste": "taste@9f3c",
    }


def test_a_rating_is_the_newest_answer_to_its_review(tmp_path: Path) -> None:
    store = journal(tmp_path)
    store.append(answer(2))
    store.append(answer(0, at=T0 + timedelta(days=1)))
    (rating,) = store.ratings().values()
    assert rating.value == 0
    assert len(list(store.answers())) == 2  # the older answer is kept as history


def test_the_order_written_decides_not_the_time_recorded(tmp_path: Path) -> None:
    store = journal(tmp_path)
    store.append(answer(2, at=T0))
    store.append(answer(0, at=T0 - timedelta(days=300)))  # migrated: an old, approximate time
    assert next(iter(store.ratings().values())).value == 0


def test_an_answer_replaces_only_the_same_review(tmp_path: Path) -> None:
    store = journal(tmp_path)
    store.append(answer(2, use="taste"))
    store.append(answer(1, use="compare"))  # not sure, in another use case
    store.append(answer(1, use="result", scope={"task": "best", "template": "general"}))
    store.append(answer(-1, use="result", scope={"task": "best", "template": "yours"}))
    store.append(answer(0, use="describe", scope="fog"))
    store.append(answer(2, use="describe", scope="mist"))
    assert len(store.ratings()) == 6


def test_scope_order_does_not_make_two_reviews(tmp_path: Path) -> None:
    store = journal(tmp_path)
    store.append(answer(1, use="result", scope={"task": "best", "template": "general"}))
    store.append(answer(-1, use="result", scope={"template": "general", "task": "best"}))
    (rating,) = store.ratings().values()
    assert rating.value == -1


def test_clearing_an_answer_withdraws_the_rating(tmp_path: Path) -> None:
    store = journal(tmp_path)
    store.append(answer(2))
    store.append(answer(None))
    assert store.ratings() == {}


def test_review_identity_is_use_scope_and_subject() -> None:
    a = answer(1, use="result", subject=("k",), scope={"task": "best", "template": "yours"})
    assert review(a) == ("result", '{"task":"best","template":"yours"}', ("k",))


def test_a_torn_last_line_is_skipped_and_the_next_answer_starts_a_new_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = journal(tmp_path)
    store.append(answer(2, subject=("k1",)))
    month = tmp_path / "ratings" / "2026-10.jsonl"
    with month.open("a", encoding="utf-8") as fh:
        fh.write('{"schema": 1, "at": "2026-10-28T1')  # a crash mid-write
    store.append(answer(0, subject=("k2",)))
    assert {r.subject: r.value for r in store.ratings().values()} == {("k1",): 2, ("k2",): 0}
    assert store.skipped == 1
    assert "1 line" in caplog.text


def test_a_line_from_a_newer_calico_is_left_out_and_counted(tmp_path: Path) -> None:
    store = journal(tmp_path)
    store.append(answer(2))
    month = tmp_path / "ratings" / "2026-10.jsonl"
    newer = {**json.loads(month.read_text()), "schema": 2, "subject": ["k9"]}
    with month.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(newer) + "\n")
    assert [r.subject for r in store.ratings().values()] == [("k1",)]
    assert store.skipped == 1


def test_months_are_read_oldest_first(tmp_path: Path) -> None:
    journal(tmp_path, now=T0).append(answer(2))
    journal(tmp_path, now=T0 + timedelta(days=10)).append(answer(0))  # November
    assert sorted(p.name for p in (tmp_path / "ratings").iterdir() if p.suffix == ".jsonl") == [
        "2026-10.jsonl",
        "2026-11.jsonl",
    ]
    assert next(iter(journal(tmp_path).ratings().values())).value == 0


def test_an_empty_or_missing_journal_has_no_ratings(tmp_path: Path) -> None:
    store = journal(tmp_path)
    assert store.ratings() == {}
    assert not store.exists()
    store.append(answer(1))
    assert store.exists()


def test_an_answer_needs_a_time_zone_and_a_known_use() -> None:
    with pytest.raises(ValidationError):
        answer(1, at=datetime(2026, 10, 28, 14, 3))  # naive: which clock?
    with pytest.raises(ValidationError):
        answer(1, use="vibes")
    assert answer(1, at=datetime(2026, 10, 28, tzinfo=UTC)).at.tzinfo is not None


@pytest.mark.parametrize(
    ("use", "good", "bad"),
    [("story", 2, 3), ("story_fit", False, 0), ("result", -1, 0), ("taste", 1, True)],
)
def test_each_use_answers_on_its_own_scale(use: str, good: int | bool, bad: int | bool) -> None:
    subject = ("a", "b", "c", "b") if use == "story_fit" else ("k",)
    assert answer(good, use=use, subject=subject).value == good
    with pytest.raises(ValidationError, match="answer is"):
        answer(bad, use=use, subject=subject)
