"""Clear matches to "Describe it": narrowing the pool to them, and marking picks.

A photo clearly matches when the describe model is confident it does (see
``calico.criteria.CLEAR_MATCH``). How confident a model can be depends on how
concrete the description is: "a staircase" often clears the line, "golden hour"
rarely does, so "no clear match" says the description is beyond the model as
often as that the thing is absent. The notes say so in both cases, and nothing
is filtered on a guess: with no clear match, the closest photos are shown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from calico.criteria import CriteriaScores
    from calico.domain import Group, Photo, SelectionRequest
    from calico.selection.context import FeatureView, Trace


def narrow_to_clear_matches(
    request: SelectionRequest,
    photos: list[Photo],
    view: FeatureView,
    criteria: CriteriaScores | None,
    trace: Trace,
) -> tuple[list[Photo], FeatureView, CriteriaScores | None]:
    """The pool a task should see: only clear matches when asked for and there are some."""
    if not photos:
        return photos, view, criteria
    text = request.criteria.strip()
    clear = None if criteria is None else criteria.clear_matches
    if clear is not None and not clear.any():
        trace.note(
            f"Calico isn't confident any photo here clearly matches “{text}”, so these are "
            "the closest. It is rarely confident about light or mood, even when they are there."
            + (" Your example photos still shaped the order." if request.examples else "")
        )
    if request.only_clear_matches:
        if not text:
            trace.note("“Only clear matches” needs a description in “Describe it”; ignored.")
        elif clear is None:
            trace.note(
                f"Calico cannot tell which photos clearly match “{text}”, "
                "so “Only clear matches” was ignored."
            )
        elif clear.any():
            keep = np.flatnonzero(clear)
            trace.note(
                f"Only the {len(keep)} of {len(photos)} photos that clearly match "
                f"“{text}” were considered."
            )
            photos, view = [photos[i] for i in keep], view.take(keep)
            criteria = None if criteria is None else criteria.take(keep)
    if criteria is not None and criteria.match is not None:
        trace.record("criteria_match", criteria.match)
    return photos, view, criteria


def mark_clear_matches(
    groups: Sequence[Group], photos: Sequence[Photo], criteria: CriteriaScores | None
) -> tuple[Group, ...]:
    """Each pick marked with whether it clearly matches; photos outside the pool (a
    reference) and models that cannot tell leave it unknown."""
    clear = None if criteria is None else criteria.clear_matches
    if clear is None:
        return tuple(groups)
    by_key = {p.content_key: bool(c) for p, c in zip(photos, clear, strict=True)}
    return tuple(
        g.model_copy(
            update={
                "picks": tuple(
                    p.model_copy(update={"clear_match": by_key.get(p.content_key)}) for p in g.picks
                )
            }
        )
        for g in groups
    )
