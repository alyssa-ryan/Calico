"""Maximal marginal relevance: a ranked list that is not twelve views of one doorway."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def mmr(
    relevance: NDArray[np.floating],
    embeddings: NDArray[np.floating],
    k: int,
    *,
    diversity: float,
    max_similarity: float | None = None,
) -> list[int]:
    """Greedy MMR with lambda = 1 - ``diversity``.

    The first pick is the most relevant. Each next pick maximises
    ``lambda * relevance - (1 - lambda) * (closest cosine to anything picked)``.
    ``diversity=0`` is therefore plain ranking. With ``max_similarity`` set,
    candidates at least that close to a pick are ineligible outright, so the
    result may be shorter than ``k``. Ties go to the lower index.
    """
    n = len(relevance)
    k = min(k, n)
    if k <= 0:
        return []
    lam = 1.0 - diversity
    # NaN would win np.argmax; an unscorable photo must never be selected.
    rel = np.nan_to_num(np.asarray(relevance, dtype=np.float64), nan=-np.inf)
    emb = np.asarray(embeddings, dtype=np.float64)

    available = np.isfinite(rel)
    if not available.any():
        return []
    closest = np.full(n, -np.inf)
    selected: list[int] = []
    nxt = int(np.argmax(np.where(available, rel, -np.inf)))
    while True:
        selected.append(nxt)
        available[nxt] = False
        closest = np.maximum(closest, np.nan_to_num(emb @ emb[nxt], nan=0.0))
        if len(selected) >= k:
            break
        eligible = available.copy()
        if max_similarity is not None:
            eligible &= closest < max_similarity
        if not eligible.any():
            break
        with np.errstate(invalid="ignore"):
            objective = np.where(eligible, lam * rel - (1.0 - lam) * closest, -np.inf)
        nxt = int(np.argmax(objective))
    return selected
