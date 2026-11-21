"""Every rating the old files hold, written once into the ratings journal.

The label sets (``labels/*.json``) and the catalog's ``feedback`` table are read, never
changed: nothing is deleted, and an older Calico still reads them as before. Each answer is
written as its review's use (owner decision 1 in ``docs/audits/2026-10-28-plan.md``):

* a labelled photo in a set, as the set's use, at the set's ``created_at`` with
  ``approximate_time`` (sets never recorded when each photo was rated);
* a rated story, as a ``story`` verdict; each photo marked as not fitting, as a
  ``story_fit`` of false, never as a thumb;
* each 👍/👎, as a ``result`` answer scoped by its run's task and template, at its own time.
  A run that cannot be read leaves the task unknown, and is counted.

Answers are written oldest first, so the newest answer to a review is its rating: a later
session's answer replaces an earlier session's (audit A05-1 had the oldest win). Each line
names its ``source``. Before writing, the lines already in the journal are compared, so
migrating again adds nothing, and a migration cut short resumes where it stopped.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, get_args

from calico.catalog.store import Catalog
from calico.errors import MigrationError, RunNotFoundError, RunUnreadableError
from calico.labels import LabelStore
from calico.ratings_store import Answer, RatingsJournal, Shown, Use
from calico.reviews import scope_of, shown_by, story_keys, story_shown, use_of
from calico.runs import RunStore
from calico.storage import locked, write_atomic

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from calico.catalog.store import FeedbackRow
    from calico.labels import LabelSet, StoryItem
    from calico.settings import Settings

logger = logging.getLogger(__name__)

USES: tuple[Use, ...] = get_args(Use.__value__)
FEEDBACK = "catalog feedback"
MIGRATED = ".migrated"
"""Written in the journal folder once a migration has finished: until then, every opening
migrates (adding only what is missing), so one cut short is completed."""


def open_journal(settings: Settings) -> RatingsJournal:
    """The home's ratings journal, migrated from the old files the first time. Every reader
    and writer of ratings opens it here, so no answer is written before the old ones are in.
    A second process opening it meanwhile waits for the migration, then finds it done."""
    journal = RatingsJournal(settings.ratings_dir)
    note = journal.folder / MIGRATED
    if note.is_file():
        return journal
    with locked(journal.folder / ".migrate.lock"):
        if note.is_file():
            return journal
        catalog = Catalog(settings.catalog_path) if settings.catalog_path.exists() else None
        try:
            report = migrate(
                LabelStore(settings.labels_dir), catalog, RunStore(settings.runs_dir), journal
            )
        finally:
            if catalog is not None:
                catalog.close()
        if not report.complete:
            missing = sum(c.found - c.in_journal for c in report.counts)
            raise MigrationError(
                f"{missing} ratings could not be written to {journal.folder}; nothing was "
                "deleted. Run calico ratings migrate to see which."
            )
        found = {c.use: c.found for c in report.counts if c.found}
        write_atomic(note, json.dumps({"at": datetime.now(UTC).isoformat(), "found": found}))
        logger.info("ratings migrated into %s: %s", journal.folder, found)
    return journal


@dataclass(frozen=True)
class UseCount:
    """For one use: answers in the old files, written by this run, and found in the journal
    afterwards (``in_journal`` equals ``found`` when nothing is missing)."""

    use: Use
    found: int
    written: int
    in_journal: int


@dataclass(frozen=True)
class MigrationReport:
    journal: Path
    counts: tuple[UseCount, ...]
    sets: dict[str, int]
    """Label sets read, by kind."""
    feedback_rows: int
    lost_runs: int
    """👍/👎 on runs that are gone or cannot be read: migrated with the task unknown."""
    unreadable: tuple[tuple[Path, str], ...]
    """Label sets this Calico cannot read, with why: left as they are, and not migrated."""

    @property
    def complete(self) -> bool:
        return all(c.in_journal == c.found for c in self.counts)


def migrate(
    labels: LabelStore, catalog: Catalog | None, runs: RunStore, journal: RatingsJournal
) -> MigrationReport:
    """Write every old answer that is not in ``journal`` yet, then count them all again."""
    sets, unreadable = labels.scan()
    feedback = catalog.feedback() if catalog is not None else []
    found, lost = _from_feedback(feedback, runs)
    for label_set in sets:
        found += _from_set(label_set)
    found.sort(key=lambda a: a.at)  # stable: answers of one time keep their files' order

    present = _migrated_lines(journal)
    todo = []
    for answer in found:
        if present[answer.line()] > 0:
            present[answer.line()] -= 1
        else:
            todo.append(answer)
    journal.append(*todo)

    after = _migrated_lines(journal)
    in_journal: Counter[str] = Counter()
    for answer in found:
        if after[answer.line()] > 0:
            after[answer.line()] -= 1
            in_journal[answer.use] += 1
    by_use = Counter(a.use for a in found)
    written = Counter(a.use for a in todo)
    return MigrationReport(
        journal=journal.folder,
        counts=tuple(UseCount(u, by_use[u], written[u], in_journal[u]) for u in USES),
        sets=dict(Counter(s.kind for s in sets)),
        feedback_rows=len(feedback),
        lost_runs=lost,
        unreadable=tuple(unreadable),
    )


def _migrated_lines(journal: RatingsJournal) -> Counter[str]:
    return Counter(a.line() for a in journal.answers() if a.source is not None)


def _from_set(label_set: LabelSet) -> list[Answer]:
    common: dict[str, Any] = {
        "at": _aware(label_set.created_at),
        "session": label_set.set_id,
        "approximate_time": True,
        "source": f"labels/{label_set.set_id}.json",
    }
    if label_set.kind == "stories":
        return [a for item in label_set.stories for a in _from_story(item, common)]
    return [
        Answer(
            use=use_of(label_set),
            scope=scope_of(label_set),
            subject=(item.content_key,),
            value=item.label,
            shown=shown_by(item),
            **common,
        )
        for item in label_set.items
        if item.label is not None
    ]


def _from_story(item: StoryItem, common: dict[str, Any]) -> Iterable[Answer]:
    keys, shown = story_keys(item), story_shown(item)
    if item.label is not None:
        yield Answer(use="story", subject=keys, value=item.label, shown=shown, **common)
    for photo in item.photos:
        if not photo.fits:
            yield Answer(
                use="story_fit",
                subject=(*keys, photo.content_key),
                value=False,
                shown=shown,
                **common,
            )


def _from_feedback(rows: Iterable[FeedbackRow], runs: RunStore) -> tuple[list[Answer], int]:
    found, lost = [], 0
    known: dict[str, tuple[str, str | None, list[str]] | None] = {}
    for row in rows:
        if row.run_id not in known:
            known[row.run_id] = _run(runs, row.run_id)
        run = known[row.run_id]
        if run is None:
            lost += 1
            scope: dict[str, str | None] = {"task": None}
            shown = Shown(run=row.run_id)
        else:
            task, template, picks = run
            scope = {"task": task, "template": template}
            rank = picks.index(row.content_key) if row.content_key in picks else None
            shown = Shown(template=template, run=row.run_id, rank=rank)
        found.append(
            Answer(
                at=_aware(row.created_at),
                use="result",
                scope=scope,
                subject=(row.content_key,),
                value=row.rating,
                shown=(shown,),
                source=FEEDBACK,
                note=row.note or None,
            )
        )
    return found, lost


def _run(runs: RunStore, run_id: str) -> tuple[str, str | None, list[str]] | None:
    """A run's task, template and picks in the order shown; None if it cannot be read."""
    try:
        result = runs.load(run_id)
    except (RunNotFoundError, RunUnreadableError):
        return None
    picks = [p.content_key for g in result.groups for p in g.picks]
    return result.request.task, result.request.template, picks


def _aware(when: datetime) -> datetime:
    """Calico has always saved UTC; a time saved without its zone is read as UTC."""
    return when if when.tzinfo is not None else when.replace(tzinfo=UTC)
