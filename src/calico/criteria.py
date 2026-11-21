"""How well each photo matches the "Describe it" text.

A ``CriteriaScorer`` turns the text into one score per photo (higher is a
better match) and, when its model can say, how likely each photo is to match.
The engine runs it once per request and hands tasks the scores
(``TaskContext.criteria_scores``), so tasks never see text and every task gets
the same reading of it. ``calico eval criteria`` compares scorers against your
labels, which is how a better reading earns its place: register it in
``SCORERS`` under a new name, measure, then make it the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from calico.domain import place_key
from calico.errors import UnknownPluginError
from calico.looks import discover_looks
from calico.reading import Hemisphere, Reading, read_description
from calico.selection.scoring import percentile_rank

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from calico.domain import Photo
    from calico.features.base import Embedder
    from calico.looks import Look
    from calico.selection.context import FeatureView, Trace
    from calico.settings import Settings


CLEAR_MATCH = 0.5
"""A photo at least this likely to match is a clear match: the line SigLIP was trained to
draw. On 30 labelled searches, 94% of the photos above it were labelled Matches and 1% No;
whole folders of photos without snow or bicycles had none above it."""


@dataclass(frozen=True)
class CriteriaScores:
    """A scorer's answer, aligned with the photos.

    ``scores`` rank (higher matches better; only the order means anything).
    ``match`` is how likely each photo is to match, in [0, 1], or None when the
    model only ranks (CLIP: its similarities shift with every description, so no
    one value means "matches").
    """

    scores: NDArray[np.float32]
    match: NDArray[np.float32] | None = None

    @property
    def clear_matches(self) -> NDArray[np.bool_] | None:
        return None if self.match is None else self.match >= CLEAR_MATCH

    def take(self, indices: NDArray[np.intp]) -> CriteriaScores:
        return CriteriaScores(
            self.scores[indices], None if self.match is None else self.match[indices]
        )


class CriteriaScorer(Protocol):
    name: str
    version: str
    """Bump when the scores change, so evaluations and runs say which reading they used."""
    embedder: Embedder

    def score(
        self, text: str, photos: Sequence[Photo], view: FeatureView, trace: Trace
    ) -> CriteriaScores | None:
        """Scores for each photo, or None when there is no usable text (noting why, if any)."""
        ...


def _vectors(embedder: Embedder, view: FeatureView) -> NDArray[np.float32]:
    """The photos in ``embedder``'s space: its own feature when it is not the main embedding
    (SigLIP for "Describe it" beside CLIP for everything else), else the embedding itself."""
    return view.features.get(embedder.name, view.embeddings)


def _readable(embedder: Embedder, text: str, trace: Trace) -> bool:
    """Whether ``embedder`` can read ``text``; notes what it cannot (rule 5: never silently)."""
    if not embedder.supports_text:
        trace.note(
            f"“{text}” was ignored: the {embedder.name!r} embedder cannot read text. "
            'Install the ml extra and set describe_embedder = "siglip".'
        )
        return False
    kept = embedder.readable_prefix(text)
    if kept != text:
        ignored = text[len(kept) :].strip() if text.startswith(kept) else text
        trace.note(
            f"The criteria are longer than the {embedder.name!r} text encoder reads: "
            f"it used “{kept}” and ignored “{ignored}”. Shorten them to the key phrases."
        )
    return True


class BaselineScorer:
    """The whole text as one CLIP embedding; each photo scores its cosine with it."""

    name = "baseline"
    version = "1"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def score(
        self, text: str, photos: Sequence[Photo], view: FeatureView, trace: Trace
    ) -> CriteriaScores | None:
        text = text.strip()
        if not text or not _readable(self.embedder, text, trace):
            return None
        vector = self.embedder.encode_texts([text])[0]
        sims = (_vectors(self.embedder, view) @ vector).astype(np.float32)
        return CriteriaScores(sims, self.embedder.match_probability(sims))


AVOID_WEIGHT = 1.0
"""A thing to avoid costs its similarity in full: "a sky without clouds" sends a cloudy
sky (similar to both) to zero and a clear one stays high."""
METADATA_WEIGHT = 2.0
"""Larger than any spread in how a photo looks (cosines span about 0.1-0.35; blended
looks lie in [0, 1]), so photos that fit when and where rank first, then by look.
It ranks; it does not filter."""
MEASURED_SHARE = 0.5
"""With both a subject ("stairs") and a look ("golden hour"), each counts half."""


class PartsScorer:
    """The description read into parts (``calico.reading``) and each part scored by what
    answers it best: the image model for what to look for and avoid, capture dates for
    when, location names for where.

    Given a looks vocabulary (the ``parts-looks`` scorer), light, colour and mood words
    are answered by measurements instead. Not the default: it lifted the three labelled
    searches that suggested it, but on six fresh searches it was worse than the image
    model (AUC 0.60 vs 0.74), because whole-photo measurements track light within one
    shoot but not across sites. Kept so better measurements can be compared later."""

    def __init__(
        self, embedder: Embedder, hemisphere: Hemisphere = "north", looks: Sequence[Look] = ()
    ) -> None:
        self.embedder = embedder
        self.hemisphere = hemisphere
        self.looks = tuple(looks)
        self.name = "parts-looks" if self.looks else "parts"
        self.version = "1"

    def score(
        self, text: str, photos: Sequence[Photo], view: FeatureView, trace: Trace
    ) -> CriteriaScores | None:
        text = text.strip()
        if not text:
            return None
        reading = read_description(
            text,
            sorted({p.location for p in photos}),
            hemisphere=self.hemisphere,
            looks=self.looks,
        )
        if reading.avoids or reading.uses_metadata or reading.looks:
            parts = " · ".join(f"{kind} {value}" for kind, value in reading.summary())
            trace.note(f"Read “{text}” as: {parts}.")
        words = [w for w in (reading.looks_for, *reading.avoids) if w]
        read = bool(words) and _readable(self.embedder, ", ".join(words), trace)
        seen, likely = self._looks(reading, view) if read else (None, None)
        measured = _measured(reading, view, trace)
        if measured is not None:
            seen = (
                measured
                if seen is None
                else (1 - MEASURED_SHARE) * percentile_rank(seen) + MEASURED_SHARE * measured
            )
        fits = _fits(reading, photos, trace)
        if seen is None and fits is None:
            if not words:  # (words the model could not read have said so already)
                trace.note(f"“{text}” has nothing Calico can look for, so it was ignored.")
            return None
        total = np.zeros(len(photos), dtype=np.float32)
        if seen is not None:
            total += seen
        if fits is not None:
            total += METADATA_WEIGHT * fits.mean(axis=0)
        return CriteriaScores(total, _match(reading, read, likely, fits, len(photos)))

    def _looks(
        self, reading: Reading, view: FeatureView
    ) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]:
        """Similarity to what to look for, less similarity to what to avoid; and, if the
        model can say, how likely each photo shows the one and none of the others."""
        words = [w for w in (reading.looks_for, *reading.avoids) if w]
        sims = (_vectors(self.embedder, view) @ self.embedder.encode_texts(words).T).astype(
            np.float32
        )
        first_avoid = 1 if reading.looks_for else 0
        looks = sims[:, 0] if reading.looks_for else np.zeros(len(view), dtype=np.float32)
        if reading.avoids:
            looks = looks - AVOID_WEIGHT * sims[:, first_avoid:].max(axis=1)
        p = self.embedder.match_probability(sims)
        if p is None:
            return looks.astype(np.float32), None
        shows = p[:, 0] if reading.looks_for else np.ones(len(view), dtype=np.float32)
        likely: NDArray[np.float32] = (shows * np.prod(1 - p[:, first_avoid:], axis=1)).astype(
            np.float32
        )
        return looks.astype(np.float32), likely


def _measured(reading: Reading, view: FeatureView, trace: Trace) -> NDArray[np.float32] | None:
    """How strongly each photo shows the look words, as a mean percentile within the pool."""
    if not reading.looks:
        return None
    wanted: dict[str, int] = {}
    for look in reading.looks:
        for measurement, sign in look.signals:
            wanted[measurement] = wanted.get(measurement, 0) + sign
    columns = view.columns.get("technical", ())
    usable = {m: s for m, s in wanted.items() if s and m in columns}
    if not usable:
        names = ", ".join(f"“{look.name}”" for look in reading.looks)
        trace.note(
            f"{names} cannot be measured here: these photos have no technical "
            "measurements, so it was left out. Re-index them with: calico index FOLDER"
        )
        return None
    ranks = [
        percentile_rank(np.sign(s) * view.features["technical"][:, columns.index(m)])
        for m, s in usable.items()
    ]
    return np.mean(ranks, axis=0).astype(np.float32)


def _match(
    reading: Reading,
    read: bool,
    likely: NDArray[np.float32] | None,
    fits: NDArray[np.float32] | None,
    n: int,
) -> NDArray[np.float32] | None:
    """How likely each photo is to match every part: what it shows, when and where.

    Unknown when a part cannot be judged so: text the model could not read, a model
    that only ranks, or a look that is measured (a percentile, not a likelihood). A
    photo with no capture date counts as an even chance of fitting "when".
    """
    if reading.looks or (read and likely is None):
        return None
    if (reading.looks_for or reading.avoids) and not read:
        return None
    match = np.ones(n, dtype=np.float32) if likely is None else likely
    if fits is not None:
        match = match * fits.prod(axis=0)
    return match.astype(np.float32)


def _fits(reading: Reading, photos: Sequence[Photo], trace: Trace) -> NDArray[np.float32] | None:
    """How well each photo fits each of when and where, one row per part: 1 fits, 0 does
    not, 0.5 unknown (no date)."""
    if not reading.uses_metadata:
        return None
    columns: list[list[float]] = []
    if reading.months or reading.years:
        undated = sum(p.captured_at is None for p in photos)
        if undated:
            trace.note(
                f"{undated} photo{' has' if undated == 1 else 's have'} no capture date, so "
                "“when” cannot rule them in or out; they rank between the photos that fit "
                "and those that do not."
            )
    if reading.months:
        columns.append(
            [
                0.5 if p.captured_at is None else float(p.captured_at.month in reading.months)
                for p in photos
            ]
        )
    if reading.years:
        first, last = reading.years
        columns.append(
            [
                0.5 if p.captured_at is None else float(first <= p.captured_at.year <= last)
                for p in photos
            ]
        )
    if reading.places:
        wanted = {place_key(name) for name in reading.places}
        columns.append([float(place_key(p.location) in wanted) for p in photos])
    return np.asarray(columns, dtype=np.float32)


EXAMPLE_WEIGHT = 0.25
"""How much example photos count beside the text. On the labels, three matches from the
same search at 0.25 lifted the rest of it (AUC 0.86 -> 0.89, precision@10 0.81 -> 0.84);
higher weights were no better and lower ones gained less."""


def example_similarity(
    embedder: Embedder, view: FeatureView, examples: FeatureView
) -> NDArray[np.float32]:
    """How like the example photos each photo looks: its mean cosine to them, in the
    reader's own space (so text and examples are compared on the same vectors)."""
    sims = _vectors(embedder, view) @ _vectors(embedder, examples).T
    return sims.mean(axis=1).astype(np.float32)


def blend_examples(
    text: CriteriaScores | None, similarity: NDArray[np.float32], weight: float
) -> CriteriaScores:
    """The text's scores plus ``weight`` times the similarity to the examples.

    A fixed weight, not a percentile blend, so each photo is scored on its own (the
    evaluation scores only a labelled sample). Whether a photo clearly matches stays
    the text's call: examples have no line of their own, so alone they only rank.
    """
    if text is None:
        return CriteriaScores(similarity.astype(np.float32))
    return CriteriaScores((text.scores + weight * similarity).astype(np.float32), text.match)


SCORERS: dict[str, Callable[[Embedder, Settings], CriteriaScorer]] = {
    "baseline": lambda embedder, _settings: BaselineScorer(embedder),
    "parts": lambda embedder, settings: PartsScorer(embedder, settings.hemisphere),
    "parts-looks": lambda embedder, settings: PartsScorer(
        embedder, settings.hemisphere, discover_looks(settings.looks_files)
    ),
}


def criteria_scorer(name: str, embedder: Embedder, settings: Settings) -> CriteriaScorer:
    if name not in SCORERS:
        raise UnknownPluginError(
            f"No criteria scorer {name!r}. Available: {', '.join(sorted(SCORERS))}"
        )
    return SCORERS[name](embedder, settings)


def scorer_key(scorer: CriteriaScorer) -> str:
    """Which reading, and which model read it, e.g. ``parts@1 (siglip)``."""
    return f"{scorer.name}@{scorer.version} ({scorer.embedder.name})"
