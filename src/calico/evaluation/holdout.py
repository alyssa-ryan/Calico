"""Which ratings no tuning may see: the hold-out, drawn once (audit A10-3). Pure; no I/O.

Every setting chosen on your ratings and then reported on the same ratings looks better
than it is. So some places, same-view references and story sessions are set aside by a
seeded draw, left out of every ``calico eval``, and kept for one last check of a final
candidate (``--holdout``). ``calico.holdout`` keeps the draw in ``<home>/holdout.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

HOLDOUT_SIZES = {"places": 4, "references": 5, "story_sessions": 2}
"""How many of each to hold out, as the owner chose (29 October 2026): 4 of 15 judged
taste places, 5 of 19 same-view references, 2 of 16 story sessions."""

HOLDOUT_SHARE = 1 / 3
"""Never more than this share of what there is, so a small home keeps most of its ratings
for measuring."""


NEAR_SHARE = 0.10
"""A location is next to another when at least this share of the smaller one's files (content
keys) are filed under both: a trip filed under each of its sites and again under an umbrella
folder, which no folder name shows. On the owner's catalog (10 November 2026), 86 of 434
locations are next to another, and one hop from a drawn location is enough: counted as chains,
a catch-all folder joins unrelated sites into one 17-location trip."""

NEAR_FILES = 3
"""... and at least this many files, so one stray copy is not a trip."""


def near_places(
    size: int,
    shared: Mapping[str, int],
    sizes: Mapping[str, int],
    *,
    share: float | None = None,
    at_least: int | None = None,
) -> frozenset[str]:
    """The locations next to one that holds ``size`` files: those in ``shared`` (files in
    common, by location) holding ``at_least`` of them and ``share`` of the smaller of the two
    (``sizes``: files at each location). One hop: what is next to a neighbour is not."""
    share = NEAR_SHARE if share is None else share
    at_least = NEAR_FILES if at_least is None else at_least
    return frozenset(
        other
        for other, files in shared.items()
        if files >= at_least
        and (smaller := min(size, sizes.get(other, size))) > 0
        and files / smaller >= share
    )


@dataclass(frozen=True)
class HoldOut:
    """The places (catalogued locations), same-view references (as their label sets name
    them) and story sessions (label set ids) no tuning may see."""

    places: tuple[str, ...]
    references: tuple[str, ...]
    story_sessions: tuple[str, ...]
    seed: int
    drawn_at: datetime

    def holds_place(self, location: str) -> bool:
        return location in self.places


def draw_holdout(
    places: Sequence[str],
    references: Sequence[str],
    story_sessions: Sequence[str],
    *,
    seed: int,
    at: datetime,
    sizes: dict[str, int] = HOLDOUT_SIZES,
) -> HoldOut:
    """Draw each kind at random, the same for the same names and seed in any order. Each
    kind has its own stream, so a reference added later does not change which places are
    drawn."""

    def draw(kind: int, names: Sequence[str], size: int) -> tuple[str, ...]:
        pool = sorted(set(names))
        k = min(size, int(len(pool) * HOLDOUT_SHARE))
        chosen = np.random.default_rng([seed, kind]).choice(len(pool), k, replace=False)
        return tuple(pool[int(i)] for i in sorted(chosen))

    return HoldOut(
        places=draw(0, places, sizes["places"]),
        references=draw(1, references, sizes["references"]),
        story_sessions=draw(2, story_sessions, sizes["story_sessions"]),
        seed=seed,
        drawn_at=at,
    )
