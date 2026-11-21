"""A file of one number per photo, to measure beside General and Yours
(``calico eval taste --scores FILE``).

Any scorer that can give a photo one number, higher meaning better, can be judged this way
without Calico running it: a vision-language model asked a question, another aesthetic
predictor, a score kept by hand. The file is JSON Lines, one object per photo: its content
key (``key``) and the number (``score``, or the field ``--score-field`` names, such as the
``logit_yes_no`` the judge in the 11 November 2026 audit writes). A run that was resumed may
write a photo twice with the same number; two different numbers for one photo are refused.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from calico.errors import InvalidRequestError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

DIGEST_BYTES = 4


@dataclass(frozen=True)
class Scores:
    """What a file says: a ``name`` for the report and the tries, and each photo's number."""

    name: str
    values: Mapping[str, float]


def read_scores(path: Path, field: str = "score") -> Scores:
    """The numbers in ``path``, by photo. Named by the file's stem and a digest of the
    numbers, so other numbers count as another try (``calico eval``'s record of tries), as a
    reworded prompt file does, and a file written again the same way does not."""
    values: dict[str, float] = {}
    seen: dict[str, int] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        key, value = _row(path, n, line, field)
        if key in values and values[key] != value:
            raise InvalidRequestError(
                f"{path.name}: the photo {key!r} has two different scores (lines {seen[key]} "
                f"and {n}). Score each photo once, or write the same number again."
            )
        values.setdefault(key, value)
        seen.setdefault(key, n)
    if not values:
        raise InvalidRequestError(
            f"{path.name} has no scores: one JSON object per line, with the photo's content "
            f"key as 'key' and its number as {field!r} (--score-field names another field)."
        )
    digest = hashlib.blake2b(
        "\n".join(f"{k} {v!r}" for k, v in sorted(values.items())).encode(),
        digest_size=DIGEST_BYTES,
    ).hexdigest()
    return Scores(f"{path.stem}@{digest}", values)


def _row(path: Path, n: int, line: str, field: str) -> tuple[str, float]:
    where = f"{path.name} line {n}"
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        raise InvalidRequestError(f"{where} is not JSON ({error.msg}).") from error
    if not isinstance(row, dict):
        raise InvalidRequestError(f"{where} is not one JSON object, a photo and its number.")
    key = row.get("key")
    if not isinstance(key, str) or not key:
        raise InvalidRequestError(f"{where} has no 'key': the photo's content key, as text.")
    if field not in row:
        raise InvalidRequestError(
            f"{where} has no number called {field!r}. Name the field the file calls it with "
            "--score-field."
        )
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidRequestError(f"{where}: {field!r} is not a number.")
    if not math.isfinite(value):
        raise InvalidRequestError(f"{where}: {field!r} is not a finite number.")
    return key, float(value)
