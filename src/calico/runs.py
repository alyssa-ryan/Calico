"""Saved runs: every selection is written to disk so it can be revisited and explained.

A run directory holds two files:

* ``result.json`` -- the ``SelectionResult`` exactly as returned to the caller.
* ``signals.csv`` -- one row per photo in the pool with every signal the task
  recorded (``Trace.record``) and where it ended up. This is what answers
  "why was this photo *not* picked?" without re-running anything.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from calico.domain import SelectionResult
from calico.errors import CalicoError, RunNotFoundError, RunUnreadableError
from calico.storage import write_atomic

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from calico.domain import Photo

RESULT_FILE = "result.json"
SIGNALS_FILE = "signals.csv"
BASE_COLUMNS = ("path", "content_key", "location", "captured_at", "picked")


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    created_at: datetime
    task: str
    criteria: str
    groups: int
    picks: int


class RunStore:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir

    @staticmethod
    def new_run_id(now: datetime) -> str:
        """Sortable by time, and unique: saving never replaces a run with the same id."""
        return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(4)}"

    def path(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def save(
        self,
        result: SelectionResult,
        photos: Sequence[Photo],
        columns: Mapping[str, NDArray[np.float32]],
    ) -> Path:
        """Save a new run: its signals first, then ``result.json``, each atomically. A run is
        listed once its result exists, so it is never listed without its signals, and an
        existing run with the same id is never replaced."""
        run_dir = self.path(result.run_id)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_dir.mkdir()
        except FileExistsError:
            raise CalicoError(
                f"A run {result.run_id!r} already exists in {self.runs_dir}, so nothing was "
                "saved over it. Run the selection again."
            ) from None
        placement = {
            pick.content_key: f"{g}.{r}"
            for g, group in enumerate(result.groups, start=1)
            for r, pick in enumerate(group.picks, start=1)
        }
        names = sorted(columns)
        table = io.StringIO(newline="")
        writer = csv.writer(table)
        writer.writerow([*BASE_COLUMNS, *names])
        for i, photo in enumerate(photos):
            writer.writerow(
                [
                    photo.path,
                    photo.content_key,
                    photo.location,
                    photo.captured_at.isoformat() if photo.captured_at else "",
                    placement.get(photo.content_key, ""),
                    *(_cell(columns[name][i]) for name in names),
                ]
            )
        write_atomic(run_dir / SIGNALS_FILE, table.getvalue())
        write_atomic(run_dir / RESULT_FILE, result.model_dump_json(indent=2))
        return run_dir

    def load(self, run_id: str) -> SelectionResult:
        path = self.path(run_id) / RESULT_FILE
        if not path.exists():
            raise RunNotFoundError(f"No run {run_id!r} in {self.runs_dir}. See: calico runs list")
        try:
            return SelectionResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError) as exc:
            raise RunUnreadableError(
                f"Run {run_id!r} cannot be read by this version of Calico; it was probably "
                f"saved by another version (another branch). Open it with that version. "
                f"Details: {_first_error(exc)}"
            ) from exc

    def recent(self, limit: int = 20) -> list[RunSummary]:
        """Newest first, leaving out (and logging) runs this version cannot read, so one
        such run never hides the others; ``unreadable`` names them."""
        summaries = []
        for run_id in self._run_ids()[:limit]:
            try:
                result = self.load(run_id)
            except RunUnreadableError as exc:
                logger.warning("%s", exc)
                continue
            summaries.append(
                RunSummary(
                    run_id=result.run_id,
                    created_at=result.created_at,
                    task=result.request.task,
                    criteria=result.request.criteria,
                    groups=len(result.groups),
                    picks=sum(len(g.picks) for g in result.groups),
                )
            )
        return summaries

    def unreadable(self, limit: int = 20) -> list[tuple[str, str]]:
        """Among the newest ``limit`` runs, those this version cannot read, and why."""
        found = []
        for run_id in self._run_ids()[:limit]:
            try:
                self.load(run_id)
            except RunUnreadableError as exc:
                found.append((run_id, str(exc)))
        return found

    def _run_ids(self) -> list[str]:
        if not self.runs_dir.exists():
            return []
        return sorted(
            (d.name for d in self.runs_dir.iterdir() if (d / RESULT_FILE).exists()), reverse=True
        )

    def signals(self, run_id: str) -> list[dict[str, str]]:
        path = self.path(run_id) / SIGNALS_FILE
        if not path.exists():
            self.load(run_id)  # says so if the run itself does not exist
            raise RunNotFoundError(f"Run {run_id!r} has no {SIGNALS_FILE}.")
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def explain(self, run_id: str, needle: str) -> list[dict[str, str]]:
        """Signal rows for photos whose path contains ``needle`` (e.g. a file name)."""
        return [row for row in self.signals(run_id) if needle in row["path"]]


def left_out_because(row: Mapping[str, str]) -> str | None:
    """Why a photo that was not picked was left out, from the signals its run recorded
    (a row of ``signals.csv``); None when they do not say, as when it simply scored lower.
    The first reason that applies is given, in the order the run left photos out."""

    def number(name: str) -> float | None:
        text = row.get(name, "")
        return float(text) if text else None

    dropped, kind, rank = number("dropped"), number("kind_probability"), number("strength_rank")
    checks = [
        (dropped == 1, "it has no capture time, so it could not be placed on an occasion"),
        (dropped == 2, "it is a copy of a frame saved twice, and was counted once"),
        (
            kind is not None and number("score") is None,
            f"it is not the kind asked for (probability {kind or 0:.2f})",
        ),
        (number("view_lead") == 0, "a stronger frame of its view was shown instead"),
        (number("burst_lead") == 0, "a stronger frame of its burst was shown instead"),
        (
            rank is not None,
            (
                f"it is the {_ordinal(int(rank or 0))} strongest moment, below the picks or "
                "too like one"
            ),
        ),
        (number("uses") == 0 and number("in_working_set") == 1, "it was in no story chosen"),
    ]
    return next((text for applies, text in checks if applies), None)


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _cell(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.4f}"


def _first_error(exc: ValidationError | ValueError) -> str:
    if isinstance(exc, ValidationError) and exc.errors():
        error = exc.errors()[0]
        return f"{'.'.join(str(p) for p in error['loc'])}: {error['msg']}"
    return str(exc).splitlines()[0]
