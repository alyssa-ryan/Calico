"""One definition of "how strong is this photo", shared by ``best`` and ``triptych``.

Strength mixes technical quality, aesthetic strength (percentile of p("strong"))
and match to the criteria text (percentile of the criteria score). ``criteria_share`` is
the criteria's share of the result (the request's ``describe_weight``): quality and
aesthetic split the rest by their own weights. Without criteria the rest is everything.
Unavailable models are dropped with a note and the remaining weights renormalise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from calico.errors import CapabilityError
from calico.models.base import LabelModel, ScoreModel
from calico.selection.scoring import Signal, combine, percentile_rank

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from calico.selection.context import TaskContext, Trace

QUALITY_WEIGHT = 0.2
"""Quality's share of strength beside the aesthetic (``AESTHETIC_WEIGHT``), for every task.
Measured with ``calico eval taste`` on 975 ratings at 20 places (30 October 2026),
strength minus the aesthetic alone, for General: -0.01 [-0.03, 0.01] at 0.2, -0.04
[-0.07, 0.00] at 0.35 (the old default), -0.10 [-0.15, -0.03] at 0.5; for Yours at 0.35,
-0.04 [-0.07, -0.01]. On the four hold-out places at 0.2: +0.01 [-0.02, 0.04]. Quality
does not tell your likes apart beyond the aesthetic, but it keeps its share so a blurred
or black frame, which taste sessions hold few of, still falls."""
AESTHETIC_WEIGHT = 0.8
"""The aesthetic's share of strength beside quality (``QUALITY_WEIGHT``)."""


def finite_or_zero(trace: Trace, name: str, values: NDArray[np.floating]) -> NDArray[np.float32]:
    """Replace non-finite values with 0, saying so: a third-party model may emit NaN,
    and NaN would otherwise win every argmax."""
    arr = np.asarray(values, dtype=np.float32)
    bad = ~np.isfinite(arr)
    if bad.any():
        trace.note(f"{name} returned {int(bad.sum())} invalid scores; treated as 0")
        arr = np.where(bad, np.float32(0.0), arr).astype(np.float32)
    return arr


def photo_strength(
    ctx: TaskContext,
    indices: Sequence[int],
    *,
    quality_weight: float,
    aesthetic_weight: float,
    criteria_share: float,
) -> tuple[NDArray[np.float32], list[Signal]]:
    """Strength of ``ctx.photos[i]`` for each ``i`` in ``indices`` (same order).

    Returns the scores and the normalised signals behind them. With no usable
    signal, every photo scores a neutral 0.5 and the signal list is empty.
    """
    sub = ctx.view.take(list(indices))
    base: list[Signal] = []
    for name, weight in (("quality", quality_weight), ("aesthetic", aesthetic_weight)):
        try:
            model = ctx.models.get(name)
        except CapabilityError as exc:
            ctx.trace.note(f"The {name!r} model is unavailable, so it does not count: {exc}")
            continue
        if isinstance(model, ScoreModel):
            values = np.clip(finite_or_zero(ctx.trace, name, model.score(sub)), 0, 1)
            base.append(Signal(name, values, weight, "technical quality"))
        elif isinstance(model, LabelModel) and "strong" in model.labels:
            strong = model.predict_proba(sub)[:, model.labels.index("strong")]
            ranked = percentile_rank(finite_or_zero(ctx.trace, name, strong))
            base.append(Signal(name, ranked, weight, "aesthetic"))
        else:
            ctx.trace.note(f"The {name!r} model has an unexpected shape, so it does not count.")

    base = [s for s in base if s.weight > 0]
    total = sum(s.weight for s in base)
    with_criteria = ctx.criteria_scores is not None and criteria_share > 0
    rest = 1 - criteria_share if with_criteria else 1.0
    signals = [Signal(s.name, s.values, rest * s.weight / total, s.describe) for s in base]
    signals = [s for s in signals if s.weight > 0]
    if ctx.criteria_scores is not None and with_criteria:
        sims = finite_or_zero(ctx.trace, "criteria", ctx.criteria_scores[list(indices)])
        signals.append(
            Signal("criteria", percentile_rank(sims), criteria_share, "matches criteria")
        )
    if not signals:
        ctx.trace.note("No strength signal is available; every photo counts as equally strong.")
        return np.full(len(sub), 0.5, dtype=np.float32), []
    return combine(signals)
