from __future__ import annotations

from pathlib import Path

import pytest

from calico.errors import LegacyRatingsError
from calico.ratings import (
    Rating,
    by_photo,
    from_legacy_rows,
    read_legacy,
    verdict,
)


def row(path: str, stars: str = "", vote: str = "") -> dict[str, str]:
    return {"image_path": path, "stars": stars, "vote": vote, "note": ""}


def test_stars_and_thumbs_become_one_scale() -> None:
    ratings = from_legacy_rows(
        [row("/a/1.jpg", stars="5"), row("/a/2.jpg", stars="1"), row("/a/3.jpg", stars="3"),
         row("/a/4.jpg", vote="up"), row("/a/5.jpg", vote="down")]
    )  # fmt: skip
    assert [(r.path.name, r.score, r.source) for r in ratings] == [
        ("1.jpg", 1.0, "stars"),
        ("2.jpg", 0.0, "stars"),
        ("3.jpg", 0.5, "stars"),
        ("4.jpg", 1.0, "thumbs"),
        ("5.jpg", 0.0, "thumbs"),
    ]


def test_rows_without_a_rating_are_left_out() -> None:
    assert from_legacy_rows([row("/a/1.jpg"), row("/a/2.jpg", stars="x")]) == []


def test_a_legacy_file_is_read_into_ratings(tmp_path: Path) -> None:
    legacy = tmp_path / "feedback.csv"
    legacy.write_text("image_path,stars,vote\n/a/1.jpg,5,\n/a/2.jpg,,down\n", encoding="utf-8")
    assert [(r.path.name, r.score) for r in read_legacy(legacy)] == [("1.jpg", 1.0), ("2.jpg", 0.0)]


def test_a_legacy_file_that_is_not_text_says_so(tmp_path: Path) -> None:
    legacy = tmp_path / "feedback.csv"
    legacy.write_bytes(b"image_path,stars\n\xff\xfe\x00/a/1.jpg,5\n")
    with pytest.raises(LegacyRatingsError, match=r"feedback\.csv is not UTF-8 text"):
        read_legacy(legacy)


@pytest.mark.parametrize("header", ["a,b", "image_path,note", "stars,vote"])
def test_a_legacy_file_without_the_rating_columns_names_them(tmp_path: Path, header: str) -> None:
    legacy = tmp_path / "feedback.csv"
    legacy.write_text(f"{header}\n1,2\n", encoding="utf-8")
    with pytest.raises(LegacyRatingsError, match=r"expected image_path and stars or vote"):
        read_legacy(legacy)


def test_an_empty_legacy_file_says_so(tmp_path: Path) -> None:
    legacy = tmp_path / "feedback.csv"
    legacy.write_text("", encoding="utf-8")
    with pytest.raises(LegacyRatingsError, match="has no header"):
        read_legacy(legacy)


def test_paths_are_compared_as_the_catalog_stores_them() -> None:
    (rating,) = from_legacy_rows([row("/a/Café/1.jpg", stars="4")])
    assert rating.path == Path("/a/Café/1.jpg")


def test_several_ratings_of_one_photo_are_averaged() -> None:
    a, b = Path("/a/1.jpg"), Path("/a/2.jpg")
    scores = by_photo([Rating(a, 1.0, "stars"), Rating(a, 0.5, "thumbs"), Rating(b, 0.0, "stars")])
    assert scores == {a: pytest.approx(0.75), b: 0.0}


@pytest.mark.parametrize(("score", "label"), [(1.0, 2), (0.75, 2), (0.5, 1), (0.25, 0), (0.0, 0)])
def test_a_verdict_is_liked_disliked_or_in_between(score: float, label: int) -> None:
    assert verdict(score) == label
