"""``<home>/experiments.jsonl``: every setting a ``calico eval`` measured (audit A10-3).

Trying twelve settings on the same ratings and keeping the best makes the best look
better than it is. So each eval that can vary a setting adds one line naming what it
measured, and its report says how many settings have been tried so far: read its best as
a best of that many. Looks at the hold-out alone are counted apart: each one acted on uses
some of it up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from calico.storage import append_lines

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True)
class Tried:
    """Distinct settings measured with one eval on the ratings left in, and how often the
    current hold-out alone was reported on."""

    settings: int
    holdout_looks: int


def record(
    path: Path,
    command: str,
    tried: Sequence[str],
    *,
    holdout: bool,
    at: datetime,
    since: datetime | None = None,
) -> Tried:
    """Note that ``command`` measured each of ``tried``, and say how much has been tried
    (looks at the hold-out counted from ``since``)."""
    line = {"at": at.isoformat(), "command": command, "tried": list(tried), "holdout": holdout}
    path.parent.mkdir(parents=True, exist_ok=True)
    append_lines(path, [json.dumps(line, ensure_ascii=False)])
    return tried_so_far(path, command, since=since)


def tried_so_far(path: Path, command: str, since: datetime | None = None) -> Tried:
    """Looks at the hold-out count from ``since``, the day the current one was drawn: a hold-out
    retired and drawn again starts with none, its old looks having been at other places. Every
    setting tried counts whatever the hold-out, the ratings having been looked at all the same.
    A line cut short (a crash mid-write) or written by hand wrongly is skipped."""
    settings: set[str] = set()
    looks = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    for raw in lines:
        try:
            entry = json.loads(raw)
            if entry["command"] != command:
                continue
            if not entry["holdout"]:
                settings.update(str(t) for t in entry["tried"])
            elif since is None or _aware(datetime.fromisoformat(entry["at"])) >= _aware(since):
                looks += 1
        except (ValueError, KeyError, TypeError):
            continue
    return Tried(settings=len(settings), holdout_looks=looks)


def _aware(when: datetime) -> datetime:
    """A time written without its zone is read as UTC, which is how Calico writes them."""
    return when if when.tzinfo else when.replace(tzinfo=UTC)
