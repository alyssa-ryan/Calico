"""``<home>/holdout.toml``: the hold-out (``evaluation.holdout``), written once.

A hold-out redrawn after its numbers were seen would be one more setting tuned on them,
so the file is created exclusively and never replaced by Calico. Delete it by hand only
to start measuring afresh, knowing every number before is no longer comparable.
"""

from __future__ import annotations

import difflib
import json
import tomllib
from datetime import datetime
from typing import TYPE_CHECKING

from calico.errors import HoldoutError
from calico.evaluation.holdout import HoldOut
from calico.storage import create_exclusive

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

HEADER = """\
# Drawn once by calico holdout draw. Never tuned on: every calico eval leaves these out,
# and --holdout reports on them alone, for a final candidate. Do not edit.
"""


def read_holdout(path: Path) -> HoldOut | None:
    """The hold-out at ``path``, or None if none was drawn."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise HoldoutError(_unreadable(path, exc)) from exc
    try:
        drawn_at = data["drawn_at"]
        if not isinstance(drawn_at, datetime):
            raise TypeError("drawn_at is not a date and time")
        return HoldOut(
            places=_names(data, "places"),
            references=_names(data, "references"),
            story_sessions=_names(data, "story_sessions"),
            seed=int(data["seed"]),
            drawn_at=drawn_at,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HoldoutError(_unreadable(path, exc)) from exc


def write_holdout(path: Path, holdout: HoldOut) -> None:
    """Write ``holdout`` to ``path``, which must not exist yet."""

    def names(values: tuple[str, ...]) -> str:
        return "[" + ", ".join(json.dumps(v, ensure_ascii=False) for v in values) + "]"

    text = (
        HEADER
        + f"drawn_at = {holdout.drawn_at.isoformat()}\n"
        + f"seed = {holdout.seed}\n"
        + f"places = {names(holdout.places)}\n"
        + f"references = {names(holdout.references)}\n"
        + f"story_sessions = {names(holdout.story_sessions)}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        create_exclusive(path, text)
    except FileExistsError as exc:
        raise HoldoutError(
            f"The hold-out was already drawn ({path}). It is drawn once: drawing again after "
            "seeing its numbers would tune on it. See it with: calico holdout show"
        ) from exc


def no_such_locations(names: Sequence[str], judged: Iterable[str]) -> HoldoutError:
    """What ``calico holdout draw --leave-out`` says when the catalog has no location of a name
    (with the judged places it is closest to): a misspelt name would leave the place it meant
    in the draw, and the draw is once."""
    pool = sorted(set(judged))
    said = []
    for name in names:
        close = difflib.get_close_matches(name, pool, n=3, cutoff=0.6)
        hint = f" (did you mean {' or '.join(map(repr, close))}?)" if close else ""
        said.append(repr(name) + hint)
    return HoldoutError(
        f"The catalog has no location named {', '.join(said)}. A misspelt name would leave that "
        "place in the draw, and the hold-out is drawn once. Write each name as calico locations "
        "lists it."
    )


def _names(data: dict[str, object], key: str) -> tuple[str, ...]:
    values = data[key]
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise TypeError(f"{key} is not a list of names")
    return tuple(values)


def _unreadable(path: Path, exc: Exception) -> str:
    return (
        f"{path} cannot be read ({exc}). Restore it from a backup (calico backup --verify "
        "FILE lists what one holds); deleting it would mean drawing a new hold-out, and "
        "every number measured before would no longer be comparable."
    )
