"""A file of one number per photo (``calico eval taste --scores``): what it may hold, and what
it says when it holds something else (rule 5: never fail silently)."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from calico.errors import InvalidRequestError
from calico.evaluation.scores import read_scores

if TYPE_CHECKING:
    from pathlib import Path


def write(path: Path, *rows: object) -> Path:
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return path


def test_a_photo_and_its_number_are_a_key_and_a_score_on_one_line(tmp_path: Path) -> None:
    path = write(
        tmp_path / "judge.jsonl",
        {"key": "a", "score": 1.5},
        "",
        {"key": "b", "score": -2},
        {"key": "c", "score": 0, "p_yes": 0.5, "text": "No"},
    )
    scores = read_scores(path)
    assert scores.values == {"a": 1.5, "b": -2.0, "c": 0.0}


def test_the_number_may_be_named_by_the_field_the_file_calls_it(tmp_path: Path) -> None:
    path = write(
        tmp_path / "judge.jsonl",
        {"key": "a", "p_yes": 0.9, "logit_yes_no": 14.75},
        {"key": "b", "p_yes": 0.1, "logit_yes_no": -3.0},
    )
    assert read_scores(path, "logit_yes_no").values == {"a": 14.75, "b": -3.0}
    with pytest.raises(InvalidRequestError, match=r"line 1.*no number called 'score'"):
        read_scores(path)


def test_the_file_is_named_by_its_stem_and_a_digest_so_other_numbers_count_as_another_try(
    tmp_path: Path,
) -> None:
    first = read_scores(write(tmp_path / "judge.jsonl", {"key": "a", "score": 1}))
    again = read_scores(write(tmp_path / "judge.jsonl", {"key": "a", "score": 1}))
    other = read_scores(write(tmp_path / "judge.jsonl", {"key": "a", "score": 2}))
    assert first.name.startswith("judge@")
    assert again.name == first.name
    assert other.name != first.name


def test_a_photo_scored_twice_with_the_same_number_is_one_photo(tmp_path: Path) -> None:
    path = write(
        tmp_path / "s.jsonl",
        {"key": "a", "score": 1},
        {"key": "a", "score": 1.0},
        {"key": "b", "score": 2},
    )  # a run resumed writes a photo again
    assert read_scores(path).values == {"a": 1.0, "b": 2.0}


def test_a_photo_scored_two_ways_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "s.jsonl", {"key": "a", "score": 1}, {"key": "a", "score": 2})
    with pytest.raises(InvalidRequestError, match=r"'a'.*two different"):
        read_scores(path)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("not json", r"line 1.*not JSON"),
        ('["a", 1]', r"line 1.*one JSON object"),
        ({"score": 1}, r"line 1.*no 'key'"),
        ({"key": "a", "score": "high"}, r"line 1.*not a number"),
        ({"key": "a", "score": True}, r"line 1.*not a number"),
        ('{"key": "a", "score": NaN}', r"line 1.*not a finite number"),
        ('{"key": "a", "score": Infinity}', r"line 1.*not a finite number"),
    ],
)
def test_a_line_that_is_not_a_photo_and_a_number_is_named(
    tmp_path: Path, row: object, message: str
) -> None:
    with pytest.raises(InvalidRequestError, match=message):
        read_scores(write(tmp_path / "s.jsonl", row))


def test_a_file_with_no_scores_is_refused(tmp_path: Path) -> None:
    (tmp_path / "s.jsonl").write_text("\n\n")
    with pytest.raises(InvalidRequestError, match="no scores"):
        read_scores(tmp_path / "s.jsonl")


def test_every_score_is_finite(tmp_path: Path) -> None:
    scores = read_scores(write(tmp_path / "s.jsonl", {"key": "a", "score": 3}))
    assert all(math.isfinite(v) for v in scores.values.values())
