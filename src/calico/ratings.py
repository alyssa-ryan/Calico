"""Your judgements of photos as one scale: how much you liked each, in [0, 1].

Ratings come from two places: the 👍/👎 on Calico's Strongest results (the ratings
journal, ``Sessions.thumbs``), and the previous system's ``feedback.csv`` (stars on some
screens, thumbs on others). Stars map linearly (1 → 0, 5 → 1), thumbs to 0 or 1; a photo
rated several times gets the mean. They are noisy (several scales, rated over
months, mostly photos a gallery had already chosen), so they are used to
*measure* how well a score agrees with you, within each location, before anything
is trained on them. Pure functions, apart from ``read_legacy``, which reads the file and
says what is wrong with one it cannot use (audit A07-5), rather than finding no ratings.
"""

from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from calico.errors import LegacyRatingsError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


type Source = Literal["stars", "thumbs", "calico"]

LIKED = 0.75
"""A photo whose mean rating is at least this is one you liked (4 or 5 stars, 👍)."""
DISLIKED = 0.25
"""At most this, one you did not (1 or 2 stars, 👎). Between the two is left out."""

STARS = 5
LEGACY_EXPECTED = "expected image_path and stars or vote, as the previous system's feedback.csv has"


@dataclass(frozen=True)
class Rating:
    path: Path
    score: float
    source: Source


def from_legacy_rows(rows: Iterable[Mapping[str, str]]) -> list[Rating]:
    """Ratings from the previous system's ``feedback.csv`` rows (``image_path``,
    ``stars`` 1-5, ``vote`` up/down). Rows with neither are skipped."""
    ratings = []
    for row in rows:
        path = Path(unicodedata.normalize("NFC", row.get("image_path", "")))
        stars, vote = row.get("stars", "").strip(), row.get("vote", "").strip()
        if stars.isdigit() and 1 <= int(stars) <= STARS:
            ratings.append(Rating(path, (int(stars) - 1) / (STARS - 1), "stars"))
        elif vote in {"up", "down"}:
            ratings.append(Rating(path, 1.0 if vote == "up" else 0.0, "thumbs"))
    return ratings


def read_legacy(path: Path) -> list[Rating]:
    """The ratings in a ``feedback.csv`` from the previous system. A file that is not UTF-8
    text, or whose header lacks the columns ``from_legacy_rows`` reads, is refused."""
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            columns = set(reader.fieldnames or ())
            if not columns:
                raise LegacyRatingsError(f"{path} has no header; {LEGACY_EXPECTED}.")
            if "image_path" not in columns or not columns & {"stars", "vote"}:
                raise LegacyRatingsError(
                    f"{path} has no rating column (it has {', '.join(sorted(columns))}); "
                    f"{LEGACY_EXPECTED}."
                )
            return from_legacy_rows(reader)
    except UnicodeDecodeError as exc:
        raise LegacyRatingsError(
            f"{path} is not UTF-8 text ({exc.reason} at byte {exc.start}). Save it as UTF-8 "
            "CSV and pass it again."
        ) from exc
    except (OSError, csv.Error) as exc:
        raise LegacyRatingsError(f"Cannot read {path}: {exc}") from exc


def by_photo(ratings: Iterable[Rating]) -> dict[Path, float]:
    """The mean rating of each photo."""
    totals: dict[Path, list[float]] = {}
    for rating in ratings:
        totals.setdefault(rating.path, []).append(rating.score)
    return {path: sum(scores) / len(scores) for path, scores in totals.items()}


def verdict(score: float) -> int:
    """2 liked, 1 in between, 0 not: the label scale ``calico.evaluation`` measures with."""
    if score >= LIKED:
        return 2
    if score <= DISLIKED:
        return 0
    return 1
