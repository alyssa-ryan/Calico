"""Every answer you gave, kept in an append-only journal: ``<home>/ratings/YYYY-MM.jsonl``.

A rating belongs to one review: the question asked (``use``), what else makes two questions
the same (``scope``: the text for "describe", the reference for "same_view", the run's task
and template for 👍/👎 on a result), and what was judged (``subject``). The newest answer to
a review is its rating; older answers to the same review are kept as history, and answers
to other reviews of the same photo are never replaced (owner decision 1 in
``docs/audits/2026-10-28-plan.md``).

Why a journal and not the label files: an append cannot lose another process's answer,
every answer keeps its own time, and the history is one folder to back up. A line is
filed in the month it was *written*, so the order of the files and lines is the order the
answers were given, whatever time a migrated answer records. A line this Calico cannot
read (cut short by a crash, or written by a newer Calico) is left out and counted.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from calico.storage import append_lines, locked

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA: Literal[1] = 1

type Use = Literal[
    "describe", "same_view", "taste", "compare", "anchor", "story", "story_fit", "result"
]
type Scope = str | dict[str, str | None] | None
type Review = tuple[str, str, tuple[str, ...]]
"""``(use, scope as canonical JSON, subject)``: two answers with one review answer the same
question about the same thing."""

_SCALES: dict[str, frozenset[int]] = {
    "story": frozenset({0, 1, 2}),
    "result": frozenset({-1, 1}),
}
"""Each use's answers; ``story_fit`` is true or false, and the rest 0, 1 or 2."""
_THREE_WAY = frozenset({0, 1, 2})


class Shown(BaseModel):
    """What showed the subject when it was judged: kept beside the answer, never shown to you
    in a blind session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    template: str | None = None
    run: str | None = None
    rank: int | None = None
    way: str | None = None


class Answer(BaseModel):
    """One line of the journal. ``subject`` is a photo's content key; a story's three keys in
    order, the anchor in the middle; or, for ``story_fit``, the story's three keys and then
    the photo judged. ``value`` is ``None`` when an answer is withdrawn."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_version: Literal[1] = Field(default=SCHEMA, alias="schema")
    at: AwareDatetime
    use: Use
    scope: Scope = None
    subject: tuple[str, ...] = Field(min_length=1)
    value: bool | int | None
    session: str | None = None
    shown: tuple[Shown, ...] = ()
    taste: str | None = None
    approximate_time: bool = False
    """True for an answer migrated from a file that recorded only when its set was made."""
    source: str | None = None
    """Where a migrated answer came from (``labels/<set>.json``, ``catalog feedback``)."""
    note: str | None = None
    """What you wrote beside a 👍/👎, if anything."""

    @model_validator(mode="after")
    def _on_its_scale(self) -> Self:
        if self.value is None:
            return self
        if self.use == "story_fit":
            if not isinstance(self.value, bool):
                raise ValueError("a story_fit answer is true or false")
        elif isinstance(self.value, bool) or self.value not in _SCALES.get(self.use, _THREE_WAY):
            scale = sorted(_SCALES.get(self.use, _THREE_WAY))
            raise ValueError(f"a {self.use} answer is one of {scale}, not {self.value!r}")
        return self

    def line(self) -> str:
        data = self.model_dump(mode="json", by_alias=True)
        for optional in ("approximate_time", "source", "note"):  # written only when set
            if not data[optional]:
                del data[optional]
        return json.dumps(data, ensure_ascii=False, separators=(", ", ": "))


def review(answer: Answer) -> Review:
    return review_of(answer.use, answer.scope, answer.subject)


def review_of(use: Use, scope: Scope, subject: tuple[str, ...]) -> Review:
    """The review a question about ``subject`` belongs to; scope order never matters."""
    canonical = json.dumps(scope, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return use, canonical, subject


def _now() -> datetime:
    return datetime.now().astimezone()


class RatingsJournal:
    """The ``ratings/`` folder. ``skipped`` counts the lines the last read left out."""

    def __init__(self, folder: Path, clock: Callable[[], datetime] = _now) -> None:
        self.folder = folder
        self._clock = clock
        self.skipped = 0

    def exists(self) -> bool:
        return self.folder.is_dir()

    def append(self, *answers: Answer) -> None:
        """Add ``answers`` in one write, in order."""
        with locked(self.folder / ".lock"):
            append_lines(self.folder / f"{self._clock():%Y-%m}.jsonl", [a.line() for a in answers])

    def answers(self) -> Iterator[Answer]:
        """Every answer, in the order given."""
        self.skipped = 0
        unread: list[str] = []
        for month in sorted(self.folder.glob("*.jsonl")) if self.exists() else ():
            for number, text in enumerate(month.read_text(encoding="utf-8").splitlines(), 1):
                if not text.strip():
                    continue
                try:
                    yield Answer.model_validate_json(text)
                except ValidationError:
                    self.skipped += 1
                    unread.append(f"{month.name}:{number}")
        if unread:
            logger.warning(
                "%d line%s of the ratings journal could not be read and %s left out (cut "
                "short by a crash, or written by a newer Calico): %s",
                len(unread),
                "" if len(unread) == 1 else "s",
                "was" if len(unread) == 1 else "were",
                ", ".join(unread),
            )

    def ratings(self) -> dict[Review, Answer]:
        """Each review's newest answer; a withdrawn one leaves no rating."""
        rated: dict[Review, Answer] = {}
        for answer in self.answers():
            key = review(answer)
            rated.pop(key, None)  # re-insert, so the dict is in order of the newest answers
            if answer.value is not None:
                rated[key] = answer
        return rated
