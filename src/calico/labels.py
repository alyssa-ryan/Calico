"""Labelling sets: the sessions you rate photos and stories in, as Calico sampled them.

A *labelling set* is one session: a search (text + folders) and a fixed sample of photos
from those folders, a reference and photos to judge against it, a taste, anchor or Compare
Strongest session, or a session of stories. Each is one JSON file in
``<calico home>/labels/``, written once and never rewritten: what was sampled, in what
order, and what showed it. Your answers are kept in the ratings journal
(``calico.ratings_store``, read through ``calico.reviews``); sets written before it hold
their answers too, which the migration copied into the journal. Photos are identified by
content key, so renaming folders never loses a label.
"""

from __future__ import annotations

import logging
import re
import secrets
import shutil
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    model_validator,
)

from calico.domain import Evidence
from calico.errors import (
    InvalidRequestError,
    LabelSetNotFoundError,
    LabelSetUnreadableError,
)
from calico.storage import create_exclusive, locked

if TYPE_CHECKING:
    from collections.abc import Iterator

type Label = Literal[0, 1, 2]
"""0 = doesn't match, 1 = partly, 2 = matches."""

logger = logging.getLogger(__name__)

_ID = re.compile(r"^[\w-]+$")
REFERENCES = "references"
"""Where same-view sets keep a copy of a reference that is not in a collection."""
TRASH = ".trash"
"""Where a deleted set goes, with the time it was deleted: labels cannot be recomputed, so
a delete can be undone by moving the file back."""
SCHEMA = 1
"""The ``schema_version`` this Calico writes. A set with a higher one was written by a
newer Calico: it is read, but never saved, since saving would drop what this one does not
know."""


class ShownBy(BaseModel):
    """A template that showed a photo in a Compare Strongest session: its run, and where
    the photo was among that run's picks (0 = first)."""

    model_config = ConfigDict(frozen=True)

    template: Literal["general", "yours"]
    run_id: str
    rank: int


class TasteSeen(BaseModel):
    """The trained taste that chose Yours's picks in a Compare Strongest session, and whether
    it had already learned from that place (``None``: it did not record its places). A taste
    judged on photos from places it learned from is judged in-sample. ``cross_fit``: Yours was
    not the saved taste but one fitted without that place and the places next to it, for this
    session alone (``Sessions.cross_fitted_taste``), so ``seen`` is False however often the
    saved taste learned it; such a session is read apart from the fresh ones."""

    model_config = ConfigDict(frozen=True)

    version: str
    seen: bool | None
    cross_fit: bool = False


class LabelItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    content_key: str
    path: Path
    label: Label | None = None
    """Legacy: where the answer was kept until the ratings journal (``calico.reviews``);
    read only by the migration."""
    shown_by: tuple[ShownBy, ...] = ()
    """Compare Strongest only: every template that showed the photo, kept from you while
    you rate."""
    stratum: str | None = None
    """Taste and anchor sessions: what the photo was drawn from (a quarter of today's
    strength, a kind of photo), kept from you while you rate."""
    inclusion: float | None = None
    """... and the chance an unrated photo there had of being drawn, so evaluation can
    weight the sample (``evaluation.Drawn``)."""
    control: bool = False
    """Compare Strongest only: a photo you rated before, asked again among the picks to
    measure how consistent you are (``evaluation.controls``). Shown by neither taste, kept
    from you while you rate, and answered as a review of its own."""


class StoryPhoto(BaseModel):
    model_config = ConfigDict(frozen=True)

    content_key: str
    path: Path
    fits: bool = True
    """Legacy, read only by the migration: False when you marked this photo alone as not
    fitting the story. The journal keeps the marks now (``story_fit``)."""


type StoryWay = Literal["today", "arranged", "roles", "deeper", "likely_anchors"]
"""How a session told a story: today's way, or the way under test then. ``arranged``:
anchors taking turns by kind of shot (27 October 2026); ``roles``: told by roles (28
October to 2 November); ``deeper``: from 1,000 candidates, not 500 (3 November);
``likely_anchors``: anchors tried by the anchor prior (from 4 November,
``evaluation.STORY_VARIANT``)."""

CANDIDATE_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*"
"""What a candidate told beside today's way may be called (``sessions.StoryVariant``)."""

type CandidateWay = Annotated[str, StringConstraints(pattern=rf"^candidate:{CANDIDATE_NAME}$")]
"""A story told with a named candidate's options, ``candidate:NAME``
(``calico eval new-stories --variant NAME``)."""


class StoryArm(BaseModel):
    """How a story was told: under which template, with variety counting or not, and which
    way (``StoryWay``). Sessions before arranged stories were told only today's way, and
    those of 27 October saved ``arranged`` as a flag."""

    model_config = ConfigDict(frozen=True)

    template: Literal["general", "yours"]
    variety: bool
    way: StoryWay | CandidateWay = "today"

    @model_validator(mode="before")
    @classmethod
    def _arranged_flag(cls, data: Any) -> Any:
        if isinstance(data, dict) and "arranged" in data:
            flag = data["arranged"]
            data = {k: v for k, v in data.items() if k != "arranged"}
            data.setdefault("way", "arranged" if flag else "today")
        return data


class StoryItem(BaseModel):
    """One story to rate 2 (works), 1 (not sure) or 0 (doesn't). ``evidence`` is what its
    run said of it, so it can be re-scored without the run; ``arms`` are every way of
    telling it that told it, kept from you while you rate."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    photos: tuple[StoryPhoto, ...]
    """Left, middle (the anchor), right."""
    evidence: tuple[Evidence, ...]
    arms: tuple[StoryArm, ...]
    label: Label | None = None
    """Legacy, read only by the migration; the journal keeps the verdicts now."""


class LabelSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA
    """Absent from sets written before it existed, which are schema 1."""
    set_id: str
    created_at: datetime
    text: str
    folders: tuple[Path, ...]
    scorer: str
    """The scorer (name@version) whose ranking chose the sample; see ``calico.criteria``."""
    items: tuple[LabelItem, ...]
    kind: Literal["describe", "same_view", "taste", "stories", "compare", "anchor"] = "describe"
    """What the labels judge: matching ``text`` (2 matches, 1 partly, 0 no), being the
    same view as ``reference`` (2 same view, 1 same place re-aimed, 0 not), your taste
    (2 like, 1 not sure, 0 don't like), ``stories`` (2 works, 1 not sure, 0 doesn't),
    ``compare``: your taste again, for the first picks of each template at one place, or
    ``anchor``: whether the photo could anchor a story (2 could, 1 not sure, 0 could not)."""
    reference: Path | None = None
    taste: TasteSeen | None = None
    """Compare Strongest only: the taste that chose Yours's picks."""
    stories: tuple[StoryItem, ...] = ()
    """A story session's stories; its ``items`` are empty."""

    @property
    def total(self) -> int:
        return len(self.items) + len(self.stories)


class LabelStore:
    """The labelling sets in one folder. Every change is a read-modify-write under a lock
    that other Calico processes respect too, and is written atomically."""

    def __init__(self, labels_dir: Path) -> None:
        self.labels_dir = labels_dir
        self._lock = threading.Lock()

    @staticmethod
    def new_id(now: datetime) -> str:
        """Sortable by time, and unique: saving never replaces a set with the same id."""
        return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(4)}"

    def save(self, label_set: LabelSet) -> Path:
        """Save a new set. One with the same id is never replaced."""
        path = self.labels_dir / f"{label_set.set_id}.json"
        try:
            create_exclusive(path, label_set.model_dump_json(indent=1))
        except FileExistsError:
            raise InvalidRequestError(
                f"A labelling set {label_set.set_id!r} already exists, so nothing was saved "
                "over it. Start the session again."
            ) from None
        return path

    def load(self, set_id: str) -> LabelSet:
        path = self._existing(set_id)
        try:
            return LabelSet.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError, UnicodeDecodeError) as exc:
            raise LabelSetUnreadableError(
                f"{path} cannot be read by this version of Calico ({_first_error(exc)}). It "
                "was left untouched: it may have been written by another version (another "
                "branch), so open it with that one, or move it out of "
                f"{self.labels_dir}."
            ) from exc

    def scan(self) -> tuple[list[LabelSet], list[tuple[Path, str]]]:
        """Every set, newest first, and every file that cannot be read with why: one such
        file never hides the others."""
        if not self.labels_dir.is_dir():
            return [], []
        found, unreadable = [], []
        for path in sorted(self.labels_dir.glob("*.json"), key=lambda p: p.stem, reverse=True):
            try:
                found.append(self.load(path.stem))
            except LabelSetUnreadableError as exc:
                logger.warning("%s", exc)
                cause = exc.__cause__
                unreadable.append((path, _first_error(cause) if cause else str(exc)))
        return found, unreadable

    def all(self) -> list[LabelSet]:
        """Newest first, leaving out (and logging) sets that cannot be read."""
        return self.scan()[0]

    def unreadable(self) -> list[tuple[Path, str]]:
        """The files in the folder that cannot be read as sets, and why."""
        return self.scan()[1]

    def keep_reference(self, set_id: str, source: Path) -> Path:
        """A copy of a same-view set's reference beside the labels: an uploaded reference is
        pruned after a week, but the labels stay meaningful only while it exists."""
        if not _ID.match(set_id):
            raise LabelSetNotFoundError(f"No labelling set {set_id!r}.")
        folder = self.labels_dir / REFERENCES
        folder.mkdir(parents=True, exist_ok=True)
        kept = folder / f"{set_id}{source.suffix.lower()}"
        shutil.copyfile(source, kept)
        return kept

    def delete(self, set_id: str) -> None:
        """Move the set, and its kept reference, to the trash. It need not be readable:
        a file this Calico cannot read can still be cleared away."""
        with self._changing(set_id):
            path = self._existing(set_id)
            trash = self.labels_dir / TRASH
            trash.mkdir(parents=True, exist_ok=True)
            when = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            path.replace(trash / f"{set_id}.{when}.json")
            for kept in (self.labels_dir / REFERENCES).glob(f"{set_id}.*"):
                kept.replace(trash / f"{kept.stem}.{when}{kept.suffix}")

    def _existing(self, set_id: str) -> Path:
        path = self.labels_dir / f"{set_id}.json"
        if not _ID.match(set_id) or not path.is_file():
            raise LabelSetNotFoundError(
                f"No labelling set {set_id!r} in {self.labels_dir}. See: calico eval list"
            )
        return path

    @contextmanager
    def _changing(self, set_id: str) -> Iterator[None]:
        """Change one set: this process's threads one at a time, and other processes too."""
        with self._lock, locked(self.labels_dir / ".lock"):
            yield


def _first_error(exc: BaseException) -> str:
    if isinstance(exc, ValidationError) and exc.errors():
        error = exc.errors()[0]
        return f"{'.'.join(str(p) for p in error['loc'])}: {error['msg']}"
    return str(exc).splitlines()[0] if str(exc) else type(exc).__name__
