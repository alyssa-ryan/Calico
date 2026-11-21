"""Which review each answer belongs to, and every rule that reads ratings, in one place.

A rating belongs to one review: ``(use, scope, subject)`` (owner decision 1 in
``docs/audits/2026-10-28-plan.md``). A label set's photos are reviewed as the set's use
(``describe`` scoped by its text, ``same_view`` by its reference, the rest unscoped); a story
as ``story``, a photo within it as ``story_fit``; a 👍/👎 on a result as ``result``, scoped by
the run's task and template. The newest answer to a review is its rating, so a photo rated
in two taste sessions has one taste rating, the later one, shown on both sessions' pages.

``Ratings`` writes answers to the journal and looks ratings up. Label set files are now
write-once session definitions: what was sampled and what showed it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from calico.errors import InvalidRequestError
from calico.ratings_store import Answer, Shown, review_of

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    from calico.domain import SelectionResult
    from calico.labels import Label, LabelItem, LabelSet, LabelStore, StoryItem
    from calico.ratings_store import RatingsJournal, Review, Scope, Use
    from calico.runs import RunStore

type Current = Mapping[Review, Answer]
"""Each review's newest answer: ``RatingsJournal.ratings()``."""

_USE_OF_KIND: dict[str, Use] = {
    "describe": "describe",
    "same_view": "same_view",
    "taste": "taste",
    "compare": "compare",
    "anchor": "anchor",
    "stories": "story",
}

_LABELS: dict[object, Label] = {0: 0, 1: 1, 2: 2}
"""An answer's value as a label (0, 1 or 2); only photo and story reviews are looked up."""
_THUMBS: dict[object, Literal[-1, 1]] = {-1: -1, 1: 1}

TASTE_RULE = (
    "each photo's newest clear answer (like or don't like) across taste sessions, Compare "
    "Strongest and 👍/👎 on Strongest results; not sure is left out"
)
"""How Yours gets one label per photo from several use cases (owner decision Q1)."""


def use_of(label_set: LabelSet) -> Use:
    return _USE_OF_KIND[label_set.kind]


def scope_of(label_set: LabelSet) -> Scope:
    """What else makes two of the set's questions the same: the text it searched for, or
    the reference photo as the set records it."""
    if label_set.kind == "describe":
        return label_set.text
    if label_set.kind == "same_view" and label_set.reference is not None:
        return str(label_set.reference)
    return None


def photo_review(label_set: LabelSet, content_key: str) -> Review:
    return review_of(use_of(label_set), scope_of(label_set), (content_key,))


CONTROL_SCOPE: dict[str, str | None] = {"control": "yes"}
"""What marks an answer to a control photo (``evaluation.controls``): asked again among a Compare
Strongest session's picks, it is a review of its own, so it neither replaces the answer it
repeats nor counts in any taste, share or training."""


def is_control(answer: Answer) -> bool:
    return isinstance(answer.scope, dict) and answer.scope.get("control") == "yes"


def item_review(label_set: LabelSet, item: LabelItem) -> Review:
    scope = CONTROL_SCOPE if item.control else scope_of(label_set)
    return review_of(use_of(label_set), scope, (item.content_key,))


def story_keys(story: StoryItem) -> tuple[str, ...]:
    """A story's subject: its three content keys in order, the anchor in the middle."""
    return tuple(p.content_key for p in story.photos)


def story_review(story: StoryItem) -> Review:
    return review_of("story", None, story_keys(story))


def fit_review(story: StoryItem, content_key: str) -> Review:
    return review_of("story_fit", None, (*story_keys(story), content_key))


def result_scope(result: SelectionResult) -> dict[str, str | None]:
    """A 👍 on a General Strongest result and a 👎 on Yours's are two reviews."""
    return {"task": result.request.task, "template": result.request.template}


def result_review(result: SelectionResult, content_key: str) -> Review:
    return review_of("result", result_scope(result), (content_key,))


def shown_by(item: LabelItem) -> tuple[Shown, ...]:
    return tuple(Shown(template=s.template, run=s.run_id, rank=s.rank) for s in item.shown_by)


def story_shown(story: StoryItem) -> tuple[Shown, ...]:
    return tuple(Shown(template=a.template, way=a.way) for a in story.arms)


def labels_of(label_set: LabelSet, current: Current, *, controls: bool = True) -> dict[str, Label]:
    """The rating of each of the set's photos that has one, by content key. ``controls=False``
    leaves out the control photos, for evaluation; the page shows them like any other."""
    found: dict[str, Label] = {}
    for item in label_set.items:
        if item.control and not controls:
            continue
        answer = current.get(item_review(label_set, item))
        if answer is not None and (label := _LABELS.get(answer.value)) is not None:
            found[item.content_key] = label
    return found


def story_label(story: StoryItem, current: Current) -> Label | None:
    answer = current.get(story_review(story))
    return None if answer is None else _LABELS.get(answer.value)


def unfit(story: StoryItem, current: Current) -> set[str]:
    """The story's photos marked as not fitting it."""
    return {
        p.content_key
        for p in story.photos
        if (a := current.get(fit_review(story, p.content_key))) is not None and a.value is False
    }


def answered(label_set: LabelSet, current: Current) -> int:
    """How many of the set's questions have an answer standing."""
    if label_set.kind == "stories":
        return sum(story_label(story, current) is not None for story in label_set.stories)
    return len(labels_of(label_set, current))


@dataclass(frozen=True)
class TasteLabels:
    """One label per photo for Yours (``TASTE_RULE``), and what the rule left out."""

    liked: dict[str, bool]
    from_session: dict[str, bool]
    """Whether the deciding answer came from a session (taste or Compare), which samples a
    place's whole range, rather than a 👍/👎 on what Calico already ranked high."""
    unsure: int
    """Photos whose every answer was not sure."""
    disagree: int
    """Photos liked in one use case and not in another; the newest answer decided."""
    session: dict[str, str | None] = field(default_factory=dict)
    """The label set the deciding answer was given in (None for a 👍/👎 on a result), so
    evaluation can read how that session drew the photo (``LabelItem.stratum``)."""


def reviewed(current: Current, use: Use) -> set[str]:
    """The photos with a standing answer for ``use``: what a new session of it leaves out,
    so a second session at a place asks about other photos (audit A05-5). For taste, every
    answer the taste rule reads (``TASTE_RULE``); not sure is an answer too."""
    reads = _for_taste if use == "taste" else lambda a: a.use == use
    return {a.subject[0] for a in current.values() if a.value is not None and reads(a)}


def _for_taste(answer: Answer) -> bool:
    if is_control(answer):
        return False  # asked again to measure you, not to say what you like
    return answer.use in ("taste", "compare") or (
        answer.use == "result"
        and isinstance(answer.scope, dict)
        and answer.scope.get("task") == "best"
    )


def taste_labels(current: Current) -> TasteLabels:
    answers: dict[str, list[Answer]] = defaultdict(list)
    for answer in current.values():
        if _for_taste(answer):
            answers[answer.subject[0]].append(answer)
    liked, from_session, session, unsure, disagree = {}, {}, {}, 0, 0
    for key in sorted(answers):
        clear = [a for a in answers[key] if verdict(a) is not None]
        if not clear:
            unsure += 1
            continue
        if len({verdict(a) for a in clear}) > 1:
            disagree += 1
        newest = max(clear, key=lambda a: (a.at, a.use))
        liked[key] = bool(verdict(newest))
        from_session[key] = newest.use != "result"
        session[key] = newest.session
    return TasteLabels(
        liked=liked,
        from_session=from_session,
        unsure=unsure,
        disagree=disagree,
        session=session,
    )


def verdict(answer: Answer) -> bool | None:
    """Like (True), don't like (False), or not sure (None) on any of taste's scales."""
    if answer.use == "result":
        return answer.value == 1
    return {2: True, 0: False}.get(answer.value)  # type: ignore[arg-type]


def _now() -> datetime:
    return datetime.now().astimezone()


class Ratings:
    """Answers to label sets and results: checked against what was asked, appended to the
    journal, and looked up by review."""

    def __init__(
        self,
        labels: LabelStore,
        journal: RatingsJournal,
        runs: RunStore,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.labels = labels
        self.journal = journal
        self.runs = runs
        self._clock = clock

    def current(self) -> dict[Review, Answer]:
        return self.journal.ratings()

    def rate_photo(self, set_id: str, content_key: str, label: Label | None) -> None:
        """Answer (or, with None, withdraw) the set's question about one of its photos."""
        label_set = self.labels.load(set_id)
        item = next((i for i in label_set.items if i.content_key == content_key), None)
        if item is None:
            raise InvalidRequestError(f"{content_key!r} is not in this labelling set.")
        self.journal.append(
            self._answer(
                label_set,
                use=use_of(label_set),
                scope=CONTROL_SCOPE if item.control else scope_of(label_set),
                subject=(content_key,),
                value=label,
                shown=shown_by(item),
            )
        )

    def rate_story(
        self, set_id: str, index: int, label: Label | None, *, doesnt_fit: Collection[str]
    ) -> None:
        """Rate story ``index`` as a whole, and mark which of its photos do not fit it; a
        photo marked before and not now has its mark withdrawn."""
        label_set = self.labels.load(set_id)
        if not 0 <= index < len(label_set.stories):
            raise InvalidRequestError(f"No story {index} in this labelling set.")
        story = label_set.stories[index]
        keys = story_keys(story)
        strangers = set(doesnt_fit) - set(keys)
        if strangers:
            raise InvalidRequestError(
                f"{', '.join(sorted(strangers))} is not in story {index} of this set."
            )
        current = self.current()
        was_unfit = unfit(story, current)
        shown = story_shown(story)
        answers = [
            self._answer(label_set, use="story", scope=None, subject=keys, value=label, shown=shown)
        ]
        for key in keys:
            now_unfit = key in doesnt_fit
            if now_unfit != (key in was_unfit):
                answers.append(
                    self._answer(
                        label_set,
                        use="story_fit",
                        scope=None,
                        subject=(*keys, key),
                        value=False if now_unfit else None,
                        shown=shown,
                    )
                )
        self.journal.append(*answers)

    def rate_result(
        self, run_id: str, content_key: str, rating: int | None, note: str = ""
    ) -> None:
        """👍 (1), 👎 (-1) or neither (None) on one of a run's picks."""
        result = self.runs.load(run_id)
        picks = [p.content_key for g in result.groups for p in g.picks]
        if content_key not in picks:
            raise InvalidRequestError("That photo is not part of this run.")
        self.journal.append(
            Answer(
                at=self._clock(),
                use="result",
                scope=result_scope(result),
                subject=(content_key,),
                value=rating,
                shown=(
                    Shown(
                        template=result.request.template,
                        run=run_id,
                        rank=picks.index(content_key),
                    ),
                ),
                note=note or None,
            )
        )

    def on_result(
        self, result: SelectionResult, current: Current | None = None
    ) -> dict[str, Literal[-1, 1]]:
        """The 👍/👎 standing on each of a run's picks."""
        current = self.current() if current is None else current
        found: dict[str, Literal[-1, 1]] = {}
        for key in sorted({p.content_key for g in result.groups for p in g.picks}):
            answer = current.get(result_review(result, key))
            if answer is not None and (thumb := _THUMBS.get(answer.value)) is not None:
                found[key] = thumb
        return found

    def _answer(self, label_set: LabelSet, **fields: Any) -> Answer:
        taste = label_set.taste.version if label_set.taste is not None else None
        return Answer(at=self._clock(), session=label_set.set_id, taste=taste, **fields)
