"""``<home>/controls.toml``: the pool of control photos (``evaluation.controls``), written once.

A pool drawn again would be a different set of photos, and the asks so far would no longer
add up. The file is created exclusively and never replaced by Calico. Delete it by hand only
to start the test-retest afresh, knowing what was asked before no longer counts.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from calico.errors import ControlsError
from calico.evaluation.controls import Control
from calico.storage import create_exclusive

if TYPE_CHECKING:
    from pathlib import Path

HEADER = """\
# Drawn once by calico controls draw. Photos you rated before, asked again inside Compare
# Strongest sessions without saying so, to measure how consistent you are. Do not edit.
"""


@dataclass(frozen=True)
class Controls:
    photos: tuple[Control, ...]
    """Each photo and how it was rated when the pool was drawn."""
    seed: int
    drawn_at: datetime


def read_controls(path: Path) -> Controls | None:
    """The pool at ``path``, or None if none was drawn."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ControlsError(_unreadable(path, exc)) from exc
    try:
        drawn_at = data["drawn_at"]
        if not isinstance(drawn_at, datetime):
            raise TypeError("drawn_at is not a date and time")
        return Controls(
            photos=tuple(Control(str(p["key"]), bool(p["liked"])) for p in data["photos"]),
            seed=int(data["seed"]),
            drawn_at=drawn_at,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ControlsError(_unreadable(path, exc)) from exc


def write_controls(path: Path, controls: Controls) -> None:
    """Write ``controls`` to ``path``, which must not exist yet."""
    photos = "".join(
        f"\n[[photos]]\nkey = {json.dumps(p.key)}\nliked = {'true' if p.liked else 'false'}\n"
        for p in controls.photos
    )
    text = (
        HEADER
        + f"drawn_at = {controls.drawn_at.isoformat()}\n"
        + f"seed = {controls.seed}\n"
        + photos
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        create_exclusive(path, text)
    except FileExistsError as exc:
        raise ControlsError(
            f"The control photos were already drawn ({path}). They are drawn once: a second "
            "pool would be other photos, and what was asked so far would not add up. See "
            "them with: calico controls show"
        ) from exc


def _unreadable(path: Path, exc: Exception) -> str:
    return (
        f"{path} cannot be read ({exc}). Restore it from a backup (calico backup --verify "
        "FILE lists what one holds); deleting it would start the test-retest afresh, and "
        "what was asked before would no longer count."
    )
