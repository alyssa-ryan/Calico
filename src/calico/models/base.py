"""Models: features -> judgements about photos.

There are two shapes of model, and every model is one of them:

* ``ScoreModel`` -- one number per photo in [0, 1] (e.g. technical quality).
* ``LabelModel`` -- a probability distribution over named labels per photo
  (e.g. shot kind: establishing / detail / portrait).

**Sub-models** are hierarchical labels: ``"establishing:wide"`` and
``"establishing:close"`` are sub-labels of ``"establishing"``. ``rollup``
sums sub-label probabilities into their parent, so a task can ask the coarse
question or the fine one from the same prediction.

A model can be zero-shot (text prompts against the embedding; see
``zero_shot.py``) or trained (e.g. a linear probe fitted to ratings). Tasks
cannot tell the difference, which is the point: a better model drops in under
the same name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from calico.selection.context import FeatureView

LABEL_SEPARATOR = ":"


@runtime_checkable
class ScoreModel(Protocol):
    name: str
    version: str
    description: str

    def score(self, view: FeatureView) -> NDArray[np.float32]:
        """Return shape ``(len(view),)``, each value in [0, 1]."""
        ...


@runtime_checkable
class LabelModel(Protocol):
    name: str
    version: str
    description: str
    labels: tuple[str, ...]

    def predict_proba(self, view: FeatureView) -> NDArray[np.float32]:
        """Return shape ``(len(view), len(labels))``; each row sums to 1."""
        ...


type Model = ScoreModel | LabelModel


class ModelProvider(Protocol):
    """Looks up models by name. Raises ``CapabilityError`` if one cannot run here
    (e.g. a zero-shot model when the embedder has no text encoder)."""

    def names(self) -> list[str]: ...

    def get(self, name: str) -> Model: ...

    def versions(self) -> dict[str, str]:
        """``{name: version}`` for every model fetched so far (recorded with each run)."""
        ...


def rollup(
    labels: tuple[str, ...], probs: NDArray[np.float32], depth: int = 1
) -> tuple[tuple[str, ...], NDArray[np.float32]]:
    """Merge labels to their first ``depth`` levels, summing probabilities.

    >>> labels = ("establishing:wide", "establishing:close", "detail")
    >>> rollup(labels, np.array([[0.5, 0.2, 0.3]], dtype=np.float32))[0]
    ('establishing', 'detail')
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")
    parents: list[str] = []
    for label in labels:
        parent = LABEL_SEPARATOR.join(label.split(LABEL_SEPARATOR)[:depth])
        if parent not in parents:
            parents.append(parent)
    out = np.zeros((probs.shape[0], len(parents)), dtype=np.float32)
    for i, label in enumerate(labels):
        parent = LABEL_SEPARATOR.join(label.split(LABEL_SEPARATOR)[:depth])
        out[:, parents.index(parent)] += probs[:, i]
    return tuple(parents), out
