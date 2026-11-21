"""Helpers that turn selected indices into domain objects and trace columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from calico.domain import Pick

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from calico.domain import Evidence, Photo


def make_pick(
    photos: Sequence[Photo],
    index: int,
    score: float,
    evidence: tuple[Evidence, ...],
    alternates: Sequence[int] = (),
    *,
    probability: float | None = None,
) -> Pick:
    """A ``Pick`` for ``photos[index]``; ``alternates`` are its other frames (of its burst,
    or of its view), best first; ``probability`` how likely you are to like it, if known."""
    p = photos[index]
    return Pick(
        path=p.path,
        content_key=p.content_key,
        location=p.location,
        captured_at=p.captured_at,
        score=float(score),
        evidence=evidence,
        burst_size=1 + len(alternates),
        alternates=tuple(photos[a].path for a in alternates),
        probability=None if probability is None else float(probability),
    )


def scatter(n: int, indices: Sequence[int], values: NDArray[np.floating]) -> NDArray[np.float32]:
    """Place ``values`` at ``indices`` in an all-NaN array of length ``n``.

    Trace columns must align with the full pool; photos a task never considered
    stay NaN so ``runs explain`` can tell "not considered" from "scored zero".
    """
    out = np.full(n, np.nan, dtype=np.float32)
    out[np.asarray(indices, dtype=np.intp)] = values
    return out


def ordinal(fraction: float) -> str:
    """0.74 -> '74th' (a percentile for human-readable evidence details)."""
    n = round(float(fraction) * 100)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
