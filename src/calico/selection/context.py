"""What a task is given to work with, and how it reports what it did.

A task never touches the filesystem, the catalog or a model checkpoint. The
engine resolves the pool, computes every feature, and hands the task a
``TaskContext`` of plain arrays. That is what makes tasks unit-testable with
synthetic vectors, and why a new task is usually a single file.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from calico.domain import StageStat

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from numpy.typing import NDArray

    from calico.domain import Photo, SelectionRequest
    from calico.models.base import ModelProvider


@dataclass(frozen=True)
class FeatureView:
    """Feature matrices with one row per photo, in the same order as the photos.

    ``embeddings`` is the configured semantic embedding (unit-norm rows).
    ``features`` holds every other extractor's output by extractor name, and
    ``columns`` names the columns of extractors that have meaningful columns.
    """

    embeddings: NDArray[np.float32]
    features: Mapping[str, NDArray[np.float32]]
    columns: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.embeddings)
        for name, matrix in self.features.items():
            if len(matrix) != n:
                raise ValueError(f"feature {name!r} has {len(matrix)} rows; embeddings have {n}")

    def __len__(self) -> int:
        return len(self.embeddings)

    def column(self, extractor: str, column: str) -> NDArray[np.float32]:
        """One named column, e.g. ``view.column("technical", "sharpness")``."""
        names = self.columns.get(extractor)
        if names is None or column not in names:
            raise KeyError(f"{extractor!r} has no column {column!r} (columns: {names})")
        return self.features[extractor][:, names.index(column)]

    def take(self, indices: NDArray[np.intp] | list[int]) -> FeatureView:
        idx = np.asarray(indices, dtype=np.intp)
        return FeatureView(
            embeddings=self.embeddings[idx],
            features={k: v[idx] for k, v in self.features.items()},
            columns=self.columns,
        )


@dataclass
class Trace:
    """The task's lab notebook. Saved with every run; read by ``calico runs explain``.

    * ``note`` -- something the user should see (a fallback, an ignored option).
    * ``stage`` -- a named step and how it narrowed the candidates.
    * ``record`` -- a per-photo signal, aligned with ``TaskContext.photos``, so a
      run can answer "why was *this* photo not picked?" after the fact.
    """

    notes: list[str] = field(default_factory=list)
    stages: list[StageStat] = field(default_factory=list)
    columns: dict[str, NDArray[np.float32]] = field(default_factory=dict)

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    @contextmanager
    def stage(self, name: str, count_in: int) -> Iterator[StageCounter]:
        counter = StageCounter(count_in)
        start = time.perf_counter()
        yield counter
        self.stages.append(
            StageStat(
                name=name,
                count_in=count_in,
                count_out=counter.count_out,
                seconds=round(time.perf_counter() - start, 4),
            )
        )

    def record(self, name: str, values: NDArray[np.floating] | list[float]) -> None:
        self.columns[name] = np.asarray(values, dtype=np.float32)


@dataclass
class StageCounter:
    count_in: int
    count_out: int = -1

    def __post_init__(self) -> None:
        self.count_out = self.count_in


@dataclass
class TaskContext:
    """Everything a task may use. Built by the engine; built by hand in tests.

    ``references`` are the request's reference images (which need not be in the
    catalog), with the same features as the pool. ``reference_paths`` and
    ``reference_keys`` are aligned with its rows; a task uses the keys to drop
    (or reuse) a reference that is also in the pool.
    ``criteria_scores`` says how well each photo matches the request's criteria
    text (higher is better; aligned with ``photos``; see ``calico.criteria``),
    or is ``None`` if there was no usable text (the engine has noted why).
    """

    request: SelectionRequest
    photos: tuple[Photo, ...]
    view: FeatureView
    models: ModelProvider
    references: FeatureView | None = None
    reference_paths: tuple[Path, ...] = ()
    reference_keys: tuple[str, ...] = ()
    criteria_scores: NDArray[np.float32] | None = None
    patch_scores: NDArray[np.float32] | None = None
    """How much of each photo is a reference, part by part (``selection.patch_match``), for
    the photos the task's ``patch_shortlist`` asked for; NaN for the rest. ``None`` when the
    task asked for none, or the view embedder cannot compare patches (noted)."""
    compared_pairs: NDArray[np.intp] | None = None
    """Pairs of photos (rows of two indices into ``photos``) the task's ``patch_pairs`` asked
    to compare with each other part by part."""
    pair_shares: NDArray[np.float32] | None = None
    """How much each of ``compared_pairs`` shares, part by part (``selection.patch_match``);
    NaN where a photo could not be read. ``None`` when the task asked for none, or the view
    embedder cannot compare patches (noted)."""
    view_feature: str = ""
    """The ``view.features`` entry that tells the same view from look-alikes (the settings'
    ``view_embedder``); "" when it is the main embedding."""
    trace: Trace = field(default_factory=Trace)
    limit: int = 0
    """How many results to return. The engine always sets it; 0 means "use ``request.limit``"
    (or 12), which keeps hand-built test contexts short."""

    def view_vectors(self) -> NDArray[np.float32]:
        """The view embedder's vectors, or the main embedding's when it is the view embedder
        (or was not computed)."""
        name = self.view_feature
        found = self.view.features.get(name) if name else None
        return self.view.embeddings if found is None else found

    def __post_init__(self) -> None:
        if len(self.photos) != len(self.view):
            raise ValueError(f"{len(self.photos)} photos but {len(self.view)} feature rows")
        if self.criteria_scores is not None and len(self.criteria_scores) != len(self.photos):
            raise ValueError(
                f"{len(self.photos)} photos but {len(self.criteria_scores)} criteria scores"
            )
        if self.patch_scores is not None and len(self.patch_scores) != len(self.photos):
            raise ValueError(f"{len(self.photos)} photos but {len(self.patch_scores)} patch scores")
        if self.limit <= 0:
            self.limit = self.request.limit or 12
        n_refs = 0 if self.references is None else len(self.references)
        if not len(self.reference_paths) == len(self.reference_keys) == n_refs:
            raise ValueError("reference_paths, reference_keys and references must align")
