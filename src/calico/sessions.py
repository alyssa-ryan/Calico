"""Rating sessions and measurement: everything built on the engine that is not a search.

Each ``label_*`` samples photos (or tells stories) for you to judge and saves a labelling
set; each ``evaluate_*`` measures today's scores against what you judged; ``train_taste``
learns Yours from it. They change every few days, as a new kind of session is added, while
the selection pipeline rarely does, so they live apart from ``Engine`` and reach it only
through its public steps (``select``, ``feature_view``, ``template``, ``patch_scores``...).
Nothing in ``engine.py`` imports this module.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import itertools
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from calico.catalog.indexer import Indexer
from calico.catalog.place_names import similar_places
from calico.controls import Controls, read_controls, write_controls
from calico.criteria import (
    SCORERS,
    criteria_scorer,
    scorer_key,
)
from calico.domain import SelectionRequest
from calico.errors import (
    CapabilityError,
    HoldoutError,
    ImageDecodeError,
    InvalidRequestError,
    RunNotFoundError,
    RunUnreadableError,
)
from calico.evaluation import (
    CANDIDATE_SESSION_STORIES,
    COMPARE_PICKS,
    DISAGREEMENT_COUNT,
    EXAMPLE_WEIGHTS,
    GIVEN,
    NEAR_SHORTLIST,
    OUTSIDE_SHORTLIST,
    PATCH_SAMPLE_BANDS,
    ROLE_STORIES,
    STORY_L2,
    STORY_SESSION_STORIES,
    STORY_SWAPS_PER_SIDE,
    STORY_TOO_ALIKE,
    STORY_VARIANT,
    STORY_VARIETY_WEIGHTS,
    VALUE_DEPTH,
    VALUE_STRATA,
    AnchorRating,
    AnchorReport,
    ComparedPhoto,
    CompareReport,
    Evaluated,
    ExampleEvaluated,
    FinishedReport,
    FinishedStory,
    LabelledSearch,
    LabelledView,
    LearnedStories,
    LocationRatings,
    Pool,
    RatedStory,
    RatingsReport,
    Scored,
    SignalledStory,
    StoryArmKey,
    StoryReport,
    StoryReports,
    StoryVerdict,
    TastePhoto,
    TasteReport,
    ViewGeometry,
    ViewReport,
    auc,
    clear_calls,
    count_copies_once,
    finished_intervals,
    learn_stories,
    measure_anchors,
    measure_compare,
    measure_examples,
    measure_signals,
    measure_stories,
    measure_taste,
    measure_view_geometry,
    measure_views,
    photos_to_compare,
    sample_by_match,
    sample_by_value,
    sample_for_anchors,
    sample_for_labelling,
    score_search,
    stories_to_rate,
    taste_apart,
    taste_disagreements,
)
from calico.evaluation.controls import (
    CONTROL_POOL,
    CONTROLS_PER_SESSION,
    Ask,
    Candidate,
    Retest,
    draw_controls,
    measure_retest,
    pick_controls,
)
from calico.evaluation.holdout import HOLDOUT_SIZES, HoldOut, draw_holdout, near_places
from calico.evaluation.scores import Scores, read_scores
from calico.evaluation.stats import SEED
from calico.features import build_extractor
from calico.features.base import Embedder, PatchExtractor, feature_key
from calico.features.technical import TechnicalExtractor
from calico.holdout import no_such_locations, read_holdout, write_holdout
from calico.labels import (
    CANDIDATE_NAME,
    LabelItem,
    LabelSet,
    ShownBy,
    StoryArm,
    StoryItem,
    StoryPhoto,
    TasteSeen,
)
from calico.models.base import LabelModel, ScoreModel
from calico.models.calibration import fit_calibration
from calico.models.registry import WithTaste
from calico.models.taste import TASTE_L2, TasteModel, fit_taste
from calico.models.taxonomy import discover_taxonomies, load_taxonomy
from calico.models.zero_shot import ZeroShotLabelModel
from calico.paths import normalise
from calico.ratings import Rating, by_photo, verdict
from calico.reviews import (
    TASTE_RULE,
    TasteLabels,
    is_control,
    labels_of,
    photo_review,
    reviewed,
    story_keys,
    story_label,
    taste_labels,
    unfit,
)
from calico.reviews import verdict as clear_answer
from calico.selection.anchors import SCALES, anchor_prior, expected_scale, wide_view
from calico.selection.bursts import group_bursts, representatives
from calico.selection.clock import on_one_clock
from calico.selection.context import FeatureView, TaskContext, Trace
from calico.selection.copies import copy_of
from calico.selection.patch_match import PatchMeasure, consistent_shares, patch_shares
from calico.selection.strength import photo_strength
from calico.selection.view_match import best_cosine, view_match
from calico.tasks.triptych import TriptychOptions, TriptychTask

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator, Mapping, Sequence

    from numpy.typing import NDArray

    from calico.catalog.indexer import ProgressFn
    from calico.domain import Evidence, Photo, SelectionResult
    from calico.engine import Engine
    from calico.evaluation import Disagreements, Drawn, FirstPicksFn, TasteFit
    from calico.features.base import ImageExtractor
    from calico.models.base import ModelProvider
    from calico.models.calibration import Calibration
    from calico.models.taste import TasteWeights
    from calico.ratings_store import Review, Use
    from calico.reviews import Current
    from calico.tasks.triptych import StoryScorer


logger = logging.getLogger(__name__)

type _Rated = tuple[
    str,
    tuple[str, ...],
    tuple[Evidence, ...],
    tuple[int, ...],
    StoryVerdict | None,
    tuple[StoryArmKey, ...],
]
"""A rated story before its embeddings are read: what it is compared within (run or
session), its photos, its evidence, each photo's mark, its stated verdict and arms."""

type PlaceProgressFn = Callable[[str, int, int], None]
"""Called as ``progress(location, done, total)`` while Strongest's first picks at a place
compare its views part by part (one pick per view). Only pairs no search has compared are
compared, so a place is slow once and the next measurement there is not."""


@dataclass(frozen=True)
class StoryVariant:
    """A named candidate's story options, told beside today's way under one template in a
    rating session (``Sessions.label_stories``)."""

    name: str
    template: Literal["general", "yours"]
    options: Mapping[str, object]

    def __post_init__(self) -> None:
        if not re.fullmatch(CANDIDATE_NAME, self.name):
            raise InvalidRequestError(
                f"A candidate's name is letters, digits, '.', '_' and '-', not {self.name!r}: "
                "use the name in its config, such as sy-vw55."
            )
        if not self.options:
            raise InvalidRequestError(
                f"Candidate {self.name!r} has no options, so it would tell stories today's "
                "way: give each with -o KEY=VALUE."
            )

    @property
    def way(self) -> str:
        return f"candidate:{self.name}"


class Sessions:
    """Labelling sessions and evaluations on one engine. Holds no state of its own, so
    ``Sessions(engine)`` is as cheap to make as it is to use once."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def label_set(
        self, text: str, folders: Sequence[Path], progress: ProgressFn | None = None
    ) -> LabelSet:
        """Sample photos from ``folders`` for you to label against ``text``, and save the set.

        The current scorer's ranking picks the sample (see
        ``calico.evaluation.sample_for_labelling``); the same search always
        samples the same photos, so a set can be rebuilt and compared.
        """
        text = text.strip()
        if not text:
            raise InvalidRequestError("Say what to look for: labelling needs a description.")
        resolved = self.engine.validated_folders(folders)
        self.engine.indexer.index(resolved, progress=progress)
        trace = Trace()
        photos, view = self.engine.feature_view(self.engine.gather(resolved, trace), trace)
        criteria = self.engine.criteria.score(text, photos, view, trace)
        if criteria is None:
            raise CapabilityError(" ".join(trace.notes) or "The text could not be read.")
        scores = criteria.scores
        if not photos:
            raise InvalidRequestError("No readable photos were found in the chosen folders.")
        order = sorted(range(len(photos)), key=lambda i: (-float(scores[i]), str(photos[i].path)))
        by_key = {p.content_key: p for p in photos}
        seed_text = "\n".join([text, *sorted(str(f) for f in resolved)])
        seed = int.from_bytes(hashlib.blake2b(seed_text.encode(), digest_size=8).digest())
        sample = sample_for_labelling([photos[i].content_key for i in order], seed)
        now = self.engine.clock()
        made = LabelSet(
            set_id=self.engine.labels.new_id(now),
            created_at=now,
            text=text,
            folders=tuple(resolved),
            scorer=scorer_key(self.engine.criteria),
            items=tuple(LabelItem(content_key=k, path=by_key[k].path) for k in sample),
        )
        self.engine.labels.save(made)
        return made

    def evaluate_criteria(
        self, scorers: Sequence[str] | None = None, embedders: Sequence[str] | None = None
    ) -> list[Evaluated]:
        """Score each scorer (default: all registered), read by each embedder (default: the
        describe embedder, then every other one that reads text), on every labelled search.

        Uses the catalog as it is; it does not re-index. Labelled photos that
        are no longer catalogued are counted in ``missing`` and left out.
        """
        chosen = [
            criteria_scorer(n, reader, self.engine.settings)
            for reader in self._readers(embedders)
            for n in (scorers or sorted(SCORERS))
        ]
        rows = []
        for label_set, labels in self._labelled_sets():
            trace = Trace()
            photos, view = self._labelled_photos(label_set, labels, trace)
            present = {p.content_key: labels[p.content_key] for p in photos}
            for scorer in chosen:
                values = scorer.score(label_set.text, photos, view, trace)
                if values is None or not present:
                    continue
                keys = [p.content_key for p in photos]
                scores = dict(zip(keys, values.scores.tolist(), strict=True))
                clear = values.clear_matches
                rows.append(
                    Evaluated(
                        set_id=label_set.set_id,
                        text=label_set.text,
                        scorer=scorer_key(scorer),
                        score=score_search(scores, present),
                        missing=len(labels) - len(present),
                        clear=None
                        if clear is None
                        else clear_calls(dict(zip(keys, clear.tolist(), strict=True)), present),
                        scores=scores,
                        labels=present,
                    )
                )
        return rows

    def evaluate_examples(
        self, k: int = 3, draws: int = 5, weights: Sequence[float] = EXAMPLE_WEIGHTS
    ) -> list[ExampleEvaluated]:
        """How describing by example would rank your labelled searches, read by the
        configured scorer and describe embedder (see ``evaluation.measure_examples``)."""
        searches = []
        for label_set, labels in self._labelled_sets():
            trace = Trace()
            photos, view = self._labelled_photos(label_set, labels, trace)
            if not photos:
                continue
            searches.append(
                LabelledSearch(
                    set_id=label_set.set_id,
                    text=label_set.text,
                    folders=label_set.folders,
                    keys=[p.content_key for p in photos],
                    taken=on_one_clock(photos),  # cameras set to other zones compare
                    labels={p.content_key: labels[p.content_key] for p in photos},
                    view=view,
                    text_scores=self.engine.criteria.score(label_set.text, photos, view, trace),
                    frames=self.engine.copy_vectors(view),
                )
            )
        return measure_examples(
            searches, self.engine.criteria.embedder, k=k, draws=draws, weights=weights
        )

    def label_taste(self, folders: Sequence[Path], progress: ProgressFn | None = None) -> LabelSet:
        """Sample photos from ``folders`` for you to rate: would you want Calico to pick it?
        One frame per burst, drawn by value (``evaluation.sample_by_value``): mostly what
        General and Yours would show first there, above all where they disagree, and some
        at random from the whole place. Without a trained Yours, General's first picks
        alone, and the set's scorer says so."""
        resolved = self.engine.validated_folders(folders)
        photos, view, ranked, rated, frame_of = self._one_frame_each(resolved, progress, "taste")
        trace = Trace()
        template, yours_models = self.engine.template("yours", trace)
        general_first = self._shown_first(photos, view, self.engine.models, frame_of)
        yours_first = (
            self._shown_first(photos, view, yours_models, frame_of) if template == "yours" else []
        )
        sample = sample_by_value(
            [photos[i].content_key for i in ranked],
            general_first,
            yours_first,
            _seed("taste", resolved),
            exclude=rated,
        )
        drawn_from = (
            "General's and Yours's first picks"
            if template == "yours"
            else "General's first picks: Yours is not trained"
        )
        now = self.engine.clock()
        made = LabelSet(
            set_id=self.engine.labels.new_id(now),
            created_at=now,
            text="Your taste: " + ", ".join(f.name for f in resolved),
            folders=tuple(resolved),
            scorer=f"taste, drawn by value from {drawn_from}",
            items=_drawn_items(sample, photos, resolved, "for your taste"),
            kind="taste",
        )
        self.engine.labels.save(made)
        return made

    def label_anchors(
        self, folders: Sequence[Path], progress: ProgressFn | None = None
    ) -> LabelSet:
        """Sample photos from ``folders`` for you to rate: could this photo anchor a story?
        One frame per burst, each kind of photo (its scale, and whether a person or animal
        is in it) taking its turn, spread over today's general strength
        (``evaluation.sample_for_anchors``). Without the scale or subject vocabulary (an
        embedder that reads no text) the sample is spread over strength alone, and the
        set's scorer says so."""
        resolved = self.engine.validated_folders(folders)
        photos, view, ranked, rated, _ = self._one_frame_each(resolved, progress, "anchor")
        scale, living, lacking = self._scale_and_subject(view, Trace())
        kinds = [
            (
                ("" if scale is None else SCALES[round(float(scale[i]))])
                + ("" if living is None or living[i] < 0.5 else ", a person or animal")
            )
            for i in range(len(photos))
        ]
        sample = sample_for_anchors(
            [(photos[i].content_key, kinds[i]) for i in ranked],
            _seed("anchor", resolved),
            exclude=rated,
        )
        now = self.engine.clock()
        made = LabelSet(
            set_id=self.engine.labels.new_id(now),
            created_at=now,
            text="Could it anchor a story: " + ", ".join(f.name for f in resolved),
            folders=tuple(resolved),
            scorer="anchor (each kind in turn, over general strength"
            + (f"; without {' or '.join(lacking)})" if lacking else ")"),
            items=_drawn_items(sample, photos, resolved, "as a possible anchor"),
            kind="anchor",
        )
        self.engine.labels.save(made)
        return made

    def evaluate_anchors(self, *, only_holdout: bool = False) -> AnchorReport:
        """How well each signal Calico has finds the photos you said could anchor a story,
        within each session's place (``evaluation.measure_anchors``): general strength,
        Yours's (once trained), a person or animal, a wide view, and the anchors stories
        told by roles try first (half strength, half person-or-wide)."""
        held = self._held(only_holdout)
        reach = self._reach(held)
        rows = []
        skipped = []
        for label_set, labels in self._labelled_sets("anchor"):
            trace = Trace()
            photos, view = self.engine.feature_view(
                self.engine.gather(list(label_set.folders), trace), trace
            )
            if not photos:  # its folders emptied or unmounted: one dead session, not all
                skipped.append(label_set.text)
                continue
            signals = {"general strength": self._place_strength(photos, view, self.engine.models)}
            template, yours = self.engine.template("yours", trace)
            if template == "yours":
                signals["yours strength"] = self._place_strength(photos, view, yours)
            scale, living, _ = self._scale_and_subject(view, trace)
            if living is not None:
                signals["person or animal"] = living
            if scale is not None:
                signals["wide view"] = wide_view(scale)
            if living is not None or scale is not None:
                signals["tried first by roles"] = anchor_prior(
                    signals["general strength"], scale=scale, living=living
                )
            place = " · ".join(f.name for f in label_set.folders)
            rows += [
                AnchorRating(
                    place=place,
                    could=labels[p.content_key] == 2,
                    signals={k: float(v[i]) for k, v in signals.items()},
                )
                for i, p in enumerate(photos)
                if labels.get(p.content_key) in (0, 2) and (p.location in reach) == only_holdout
            ]
        return dataclasses.replace(measure_anchors(rows), skipped=skipped)

    def _one_frame_each(
        self, resolved: Sequence[Path], progress: ProgressFn | None, use: Use
    ) -> tuple[list[Photo], FeatureView, list[int], set[str], dict[str, str]]:
        """The photos of ``resolved`` with their features, one frame per burst (the
        strongest) best first by today's general strength: what rating sessions sample;
        the keys of those frames whose burst already has a rating for ``use``; and, for
        every photo's key, the key of its burst's frame. A burst is reviewed whole, so a
        rated photo's near-twin is not drawn in its place."""
        self.engine.indexer.index(resolved, progress=progress)
        trace = Trace()
        photos, view = self.engine.feature_view(self.engine.gather(resolved, trace), trace)
        if not photos:
            raise InvalidRequestError("No readable photos were found in the chosen folders.")
        options = self.engine.task("best").parse_options({})
        strength = self._place_strength(photos, view, self.engine.models)
        bursts = group_bursts(
            photos,
            view.embeddings,
            window_seconds=options.burst_window_seconds,
            min_similarity=options.burst_similarity,
        )
        chosen = representatives(bursts, strength)
        ranked = sorted(
            (rep for rep, _ in chosen), key=lambda i: (-float(strength[i]), str(photos[i].path))
        )
        done = reviewed(self.engine.ratings.current(), use)
        rated = {
            photos[rep].content_key
            for rep, rest in chosen
            if any(photos[i].content_key in done for i in (rep, *rest))
        }
        frame_of = {
            photos[i].content_key: photos[rep].content_key
            for rep, rest in chosen
            for i in (rep, *rest)
        }
        return photos, view, ranked, rated, frame_of

    def _shown_first(
        self,
        photos: Sequence[Photo],
        view: FeatureView,
        models: ModelProvider,
        frame_of: Mapping[str, str],
    ) -> list[str]:
        """What Strongest would show first among ``photos`` with ``models`` (the first
        ``VALUE_DEPTH``), each as its burst's frame in a session: Yours may show another
        frame of a burst than General."""
        task = self.engine.task("best")
        options = task.parse_options({})
        ctx = TaskContext(
            request=SelectionRequest(task="best", folders=(photos[0].path.parent,)),
            photos=tuple(photos),
            view=view,
            models=models,
            trace=Trace(),
            limit=VALUE_DEPTH,
            view_feature=self.engine.view_feature(),
        )
        ctx = self.engine.compare_parts(task, options, ctx)
        shown = [p.content_key for g in task.run(ctx, options) for p in g.picks]
        return list(dict.fromkeys(frame_of[k] for k in shown))

    def _place_strength(
        self, photos: Sequence[Photo], view: FeatureView, models: ModelProvider
    ) -> NDArray[np.float32]:
        """Each photo's strength among ``photos``, as Strongest ranks without a description."""
        options = self.engine.task("best").parse_options({})
        ctx = TaskContext(
            request=SelectionRequest(task="best", folders=(photos[0].path.parent,)),
            photos=tuple(photos),
            view=view,
            models=models,
            trace=Trace(),
        )
        strength, _ = photo_strength(
            ctx,
            range(len(photos)),
            quality_weight=options.quality_weight,
            aesthetic_weight=options.aesthetic_weight,
            criteria_share=0.0,
        )
        return strength

    def _scale_and_subject(
        self, view: FeatureView, trace: Trace
    ) -> tuple[NDArray[np.float32] | None, NDArray[np.float32] | None, list[str]]:
        """Each photo's scale (0 aerial ... 4 macro) and how surely a person or animal is
        in it, None where the vocabulary cannot be read, with the names of those missing."""
        scale = living = None
        lacking = []
        for name in ("scale", "subject"):
            try:
                model = self.engine.models.get(name)
            except CapabilityError as exc:
                trace.note(f"The {name!r} vocabulary is unavailable: {exc}")
                lacking.append(name)
                continue
            if not isinstance(model, LabelModel):
                lacking.append(name)
                continue
            probs = model.predict_proba(view)
            if name == "scale" and (found := expected_scale(model.labels, probs)) is not None:
                scale = found
            elif name == "subject" and "living" in model.labels:
                living = probs[:, model.labels.index("living")].astype(np.float32)
            else:
                lacking.append(name)
        return scale, living, lacking

    def label_stories(
        self,
        folders: Sequence[Path],
        progress: ProgressFn | None = None,
        *,
        per_way: int | None = None,
        variant: StoryVariant | None = None,
    ) -> LabelSet:
        """Tell stories from ``folders`` for you to rate as a whole: under each template
        (Yours once it is trained), today's way and the variant under test
        (``STORY_VARIANT``), so the ratings say which way tells stories that work for you.
        With ``variant``, today's way and that candidate's options under its template
        alone, ``CANDIDATE_SESSION_STORIES`` a way.
        (Sessions until 26 October 2026 told them with variety and without, and confirmed
        variety; until 2 November, by roles beside today's.) Each story is
        rated once however many ways told it, shuffled, so the order says nothing of how
        it was told. Every telling is an ordinary run, so ``calico runs explain`` works on
        it."""
        resolved = self.engine.validated_folders(folders)
        yours, _ = self.engine.template("yours", Trace())
        tested: str
        if variant is None:
            tested, tested_options = STORY_VARIANT
            templates: tuple[Literal["general", "yours"], ...] = tuple(
                dict.fromkeys(("general", yours))
            )
            told_each = STORY_SESSION_STORIES if per_way is None else per_way
        else:
            if variant.template == "yours" and yours != "yours":
                raise InvalidRequestError(
                    f"Candidate {variant.name!r} is for Yours, which has not been trained yet. "
                    "Rate photos with Rate for your taste (the Label page), then run: "
                    "calico train taste"
                )
            tested, tested_options = variant.way, variant.options
            templates = (variant.template,)
            told_each = CANDIDATE_SESSION_STORIES if per_way is None else per_way
        ways: list[tuple[Literal["general", "yours"], str]] = [
            (template, way) for template in templates for way in ("today", tested)
        ]
        results = [
            self.engine.select(
                SelectionRequest(
                    task="triptych",
                    folders=tuple(resolved),
                    template=template,
                    limit=told_each,
                    options=dict(tested_options) if way == tested else {},
                ),
                progress,
            )
            for template, way in ways
        ]
        told = [
            ((template, True, way), [tuple(p.content_key for p in g.picks) for g in r.groups])
            for (template, way), r in zip(ways, results, strict=True)
        ]
        seed_text = "\n".join(["stories", *sorted(str(f) for f in resolved)])
        seed = int.from_bytes(hashlib.blake2b(seed_text.encode(), digest_size=8).digest())
        stories = []
        for chosen in stories_to_rate(told, seed):
            way, index = chosen.told
            group = results[way].groups[index]
            stories.append(
                StoryItem(
                    run_id=results[way].run_id,
                    photos=tuple(
                        StoryPhoto(content_key=p.content_key, path=p.path) for p in group.picks
                    ),
                    evidence=group.evidence,
                    arms=tuple(StoryArm(template=t, variety=v, way=w) for t, v, w in chosen.arms),
                )
            )
        if not stories:
            raise InvalidRequestError(
                "No stories could be told from those folders: a story needs three photos "
                "that are not frames of one burst. Choose a folder with more photos."
            )
        now = self.engine.clock()
        made = LabelSet(
            set_id=self.engine.labels.new_id(now),
            created_at=now,
            text="Stories: " + ", ".join(f.name for f in resolved),
            folders=tuple(resolved),
            scorer="triptych: "
            + ", ".join(t if w == "today" else f"{t} {w}" for t, w in ways)
            + ("" if variant is None else " " + json.dumps(dict(sorted(variant.options.items())))),
            items=(),
            kind="stories",
            stories=tuple(stories),
        )
        self.engine.labels.save(made)
        return made

    def label_list(self, paths: Sequence[Path], name: str) -> LabelSet:
        """The photos at ``paths``, for you to rate for your taste (like, not sure, don't
        like): a session of photos someone chose, not ones Calico drew. Made to judge the
        photos two versions of a task disagree on, so it is blind by construction: each
        photo is shown once, in a shuffle that depends on the photos alone, and the session
        keeps nothing about where a photo came from. The ratings are taste ratings: Yours
        learns from them, and the whole-range measures leave them out (``evaluation.GIVEN``),
        as they do Compare Strongest's."""
        wanted = sorted({normalise(Path(p)) for p in paths})
        if not wanted:
            raise InvalidRequestError(
                "The list holds no photos: give the path of at least one photo in the catalog."
            )
        known: dict[Path, Photo] = {}
        for folder in sorted({p.parent for p in wanted}):
            known.update(
                (path, record.photo)
                for path, record in self.engine.catalog.records_under(folder).items()
            )
        missing = [p for p in wanted if p not in known]
        if missing:
            shown = ", ".join(str(p) for p in missing[:5])
            more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            raise InvalidRequestError(
                f"{len(missing)} of the {len(wanted)} photos listed are not in the catalog: "
                f"{shown}{more}. Check the paths, or index their folders first: "
                "calico index FOLDER"
            )
        by_key: dict[str, Photo] = {}
        for path in wanted:
            by_key.setdefault(known[path].content_key, known[path])
        keys = sorted(by_key)
        seed = int.from_bytes(
            hashlib.blake2b("\n".join(["list", *keys]).encode(), digest_size=8).digest()
        )
        order = np.random.default_rng(seed).permutation(len(keys))
        now = self.engine.clock()
        made = LabelSet(
            set_id=self.engine.labels.new_id(now),
            created_at=now,
            text=f"Your taste: {name}",
            folders=tuple(sorted({p.path.parent for p in by_key.values()})),
            scorer="taste, photos given in a list",
            items=tuple(
                LabelItem(content_key=keys[int(i)], path=by_key[keys[int(i)]].path, stratum=GIVEN)
                for i in order
            ),
            kind="taste",
        )
        self.engine.labels.save(made)
        return made

    def label_compare(
        self,
        folders: Sequence[Path],
        progress: ProgressFn | None = None,
        *,
        per_template: int = COMPARE_PICKS,
        cross_fit: bool = False,
    ) -> LabelSet:
        """The first picks of General and of Yours at ``folders``, for you to rate (like,
        not sure, don't like) without knowing whose they are: they say directly whose first
        picks you prefer (``calico eval compare``), and they are good-against-great
        ratings for Yours to learn from, which an even taste sample rarely holds. Each
        template's picks are an ordinary run; a photo both chose is shown once, credited
        to both, and the order is shuffled. With ``cross_fit``, Yours is not the saved taste
        but one fitted without these places (``cross_fitted_taste``), so a place the saved
        taste has learned is fresh to it: it needs ratings, not a saved taste."""
        resolved = self.engine.validated_folders(folders)
        fitted: TasteWeights | None = None
        if cross_fit:
            fitted = self.cross_fitted_taste(resolved, progress)
        elif self.engine.template("yours", Trace())[0] != "yours":
            raise InvalidRequestError(
                "Compare Strongest needs a trained Yours template that reads these photos: "
                "rate photos with Rate for your taste (the Label page), then run: "
                "calico train taste"
            )
        templates: tuple[Literal["general", "yours"], ...] = ("general", "yours")
        results = [
            self.engine.select(
                SelectionRequest(
                    task="best", folders=tuple(resolved), template=template, limit=per_template
                ),
                progress,
                taste=fitted if template == "yours" else None,
            )
            for template in templates
        ]
        shown = [[p.content_key for g in r.groups for p in g.picks] for r in results]
        paths = {p.content_key: p.path for r in results for g in r.groups for p in g.picks}
        seed_text = "\n".join(["compare", *sorted(str(f) for f in resolved)])
        seed = int.from_bytes(hashlib.blake2b(seed_text.encode(), digest_size=8).digest())
        items = tuple(
            LabelItem(
                content_key=chosen.content_key,
                path=paths[chosen.content_key],
                shown_by=tuple(
                    ShownBy(template=templates[way], run_id=results[way].run_id, rank=rank)
                    for way, rank in chosen.shown_by
                ),
            )
            for chosen in photos_to_compare(shown, seed)
        )
        if not items:
            raise InvalidRequestError("No readable photos were found in the chosen folders.")
        now = self.engine.clock()
        made = LabelSet(
            set_id=self.engine.labels.new_id(now),
            created_at=now,
            text="Compare Strongest: " + ", ".join(f.name for f in resolved),
            folders=tuple(resolved),
            scorer="best: " + ", ".join(templates),
            items=self._with_controls(items, seed),
            kind="compare",
            taste=(
                TasteSeen(version=fitted.version, seen=False, cross_fit=True)
                if fitted is not None
                else self._taste_seen(
                    {p.location for r in results for g in r.groups for p in g.picks}
                )
            ),
        )
        self.engine.labels.save(made)
        return made

    def _with_controls(self, items: tuple[LabelItem, ...], seed: int) -> tuple[LabelItem, ...]:
        """``items`` with this session's control photos mixed in, when a pool was drawn
        (``evaluation.controls``): the ones asked least so far, never one the session shows
        already nor one from a location it shows, in a shuffle with the picks."""
        pool = read_controls(self.engine.settings.controls_path)
        if pool is None:
            return items
        asked = Counter(
            i.content_key
            for s in self.engine.labels.all()
            if s.kind == "compare"
            for i in s.items
            if i.control
        )
        found = {
            p.content_key: p
            for p in self.engine.catalog.photos_with_keys([c.key for c in pool.photos])
        }
        shown = {i.content_key for i in items}
        here = {p.location for p in self.engine.catalog.photos_with_keys(list(shown))}
        avoid = shown | {k for k, p in found.items() if p.location in here}
        picked = pick_controls(
            [c.key for c in pool.photos if c.key in found],
            asked,
            avoid,
            k=CONTROLS_PER_SESSION,
            seed=seed,
        )
        mixed = [
            *items,
            *(LabelItem(content_key=k, path=found[k].path, control=True) for k in picked),
        ]
        return tuple(
            mixed[int(i)] for i in np.random.default_rng([seed, 1]).permutation(len(mixed))
        )

    def _taste_seen(self, places: set[str]) -> TasteSeen | None:
        weights = self.engine.trained.load_taste()
        if weights is None:
            return None
        # Learned from the location itself, or from one next to it (files in common): a trip
        # filed under each of its sites is one place to a taste, whatever the folders are called.
        reach = places | {o for others in self._near(places).values() for o in others}
        seen = bool(reach & set(weights.places)) if weights.places else None
        return TasteSeen(version=weights.version, seen=seen)

    def evaluate_compare(
        self, *, only_holdout: bool = False, min_bursts: int | None = None
    ) -> CompareReport:
        """Whose first picks you liked more, in each Compare Strongest session, oldest
        first, each named by the place most of its photos are from. A photo rated in two
        sessions counts once, in the newer. Sessions whose taste had already learned from
        their place, or cannot say, are reported apart (``in_sample``): there Yours is
        judged on what it learned from. Those whose Yours was fitted without their place
        (``label_compare(cross_fit=True)``) are reported apart too (``cross_fitted``), never
        mixed with the saved taste's. With ``min_bursts``, those of the first and that kind
        whose pool held fewer bursts are reported apart (``small``): no size is built in, and
        the roadmap asks 100 of a place read for the bar."""
        held = self._held(only_holdout)
        reach = self._reach(held)
        rated = {s.set_id: labels for s, labels in self._labelled_sets("compare")}
        fresh: list[tuple[str, list[ComparedPhoto]]] = []
        seen: list[tuple[str, list[ComparedPhoto]]] = []
        crossed: list[tuple[str, list[ComparedPhoto]]] = []
        for label_set in reversed(self.engine.labels.all()):
            if label_set.kind != "compare":
                continue
            labels = rated.get(label_set.set_id, {})
            found = Counter(
                p.location
                for p in self.engine.catalog.photos_with_keys(
                    [i.content_key for i in label_set.items if not i.control]
                )
            )
            place = found.most_common(1)[0][0] if found else label_set.text
            if (place in reach) != only_holdout:
                continue
            yours = self._yours_run(label_set)
            marker = _marker_drawn(yours)
            bursts = _bursts_held(yours)
            stamp = label_set.taste
            (
                crossed
                if stamp is not None and stamp.cross_fit
                else fresh
                if stamp is not None and stamp.seen is False
                else seen
            ).append(
                (
                    place,
                    [
                        ComparedPhoto(
                            liked=labels[i.content_key] == 2,
                            shown_by=frozenset(s.template for s in i.shown_by),
                            ranks=tuple((s.template, s.rank) for s in i.shown_by),
                            marker=marker,
                            bursts=bursts,
                        )
                        for i in label_set.items
                        if labels.get(i.content_key) in {0, 2}
                    ],
                )
            )
        report = measure_compare(fresh, min_bursts=min_bursts)
        return dataclasses.replace(
            report,
            in_sample=measure_compare(seen) if seen else None,
            cross_fitted=measure_compare(crossed, min_bursts=min_bursts) if crossed else None,
        )

    def _yours_run(self, label_set: LabelSet) -> SelectionResult | None:
        """The saved run of Yours's picks in a Compare session; None where there is none or
        this version cannot read it."""
        run_id = next(
            (s.run_id for i in label_set.items for s in i.shown_by if s.template == "yours"), None
        )
        if run_id is None:
            return None
        try:
            return self.engine.runs.load(run_id)
        except (RunNotFoundError, RunUnreadableError):
            return None

    def label_same_view(
        self, reference: Path, folders: Sequence[Path], progress: ProgressFn | None = None
    ) -> LabelSet:
        """Sample photos from ``folders`` for you to judge against ``reference``: are they
        the same view? Drawn band by band of what same view uses: patch share where the view
        embedder can compare patches (``evaluation.PATCH_SAMPLE_BANDS``), else whole-photo
        match (``evaluation.sample_by_match``), so the labels show where same views stop,
        not only that the closest are right."""
        resolved = self.engine.validated_folders(folders)
        reference = normalise(Path(reference))
        self.engine.indexer.index(resolved, progress=progress)
        trace = Trace()
        photos, view = self.engine.feature_view(self.engine.gather(resolved, trace), trace)
        refs = self.engine.references([reference])
        keep = [i for i, p in enumerate(photos) if p.content_key not in refs["reference_keys"]]
        if not keep:
            raise InvalidRequestError("No photos to compare with the reference in those folders.")
        seed_text = "\n".join([str(reference), *sorted(str(f) for f in resolved)])
        seed = int.from_bytes(hashlib.blake2b(seed_text.encode(), digest_size=8).digest())
        keys = [photos[i].content_key for i in keep]
        ctx = TaskContext(
            request=SelectionRequest(task="same_view", folders=tuple(resolved)),
            photos=tuple(photos),
            view=view,
            models=self.engine.models,
            trace=trace,
            view_feature=self.engine.view_feature(),
            **refs,
        )
        shares = self._patch_sample_values(ctx, progress)
        if shares is not None:
            sample = sample_by_match(keys, [shares[i] for i in keep], seed, PATCH_SAMPLE_BANDS)
            scorer = f"same_view patches ({self.engine.settings.view_embedder})"
        else:
            layout_weight, _ = self._same_view_defaults()
            match = view_match(view.take(keep), refs["references"], layout_weight, trace).match
            sample = sample_by_match(keys, match.tolist(), seed)
            scorer = f"same_view match (layout {layout_weight:g})"
        by_key = {p.content_key: p for p in photos}
        now = self.engine.clock()
        set_id = self.engine.labels.new_id(now)
        name = reference.name
        if not any(c.contains(reference) for c in self.engine.settings.collections):
            reference = self.engine.labels.keep_reference(set_id, reference)
        made = LabelSet(
            set_id=set_id,
            created_at=now,
            text=f"Same view as {name}",
            folders=tuple(resolved),
            scorer=scorer,
            items=tuple(LabelItem(content_key=k, path=by_key[k].path) for k in sample),
            kind="same_view",
            reference=reference,
        )
        self.engine.labels.save(made)
        return made

    def _patch_sample_values(
        self, ctx: TaskContext, progress: ProgressFn | None = None
    ) -> list[float] | None:
        """Each photo's patch share, for drawing a labelling sample: shortlisted photos at
        their share, photos just outside the shortlist at ``OUTSIDE_SHORTLIST``, the rest
        below every band. ``None`` when the view embedder cannot compare patches."""
        if not isinstance(
            self.engine.extractors.get(self.engine.settings.view_embedder), PatchExtractor
        ):
            return None
        task = self.engine.task("same_view")
        shortlist = task.patch_shortlist(ctx, task.parse_options({}))
        empty = np.zeros(0, dtype=np.intp)
        scores = self.engine.patch_scores(
            ctx, shortlist if shortlist is not None else empty, progress
        )
        if scores is None or ctx.references is None:
            return None
        name = ctx.view_feature
        mine, theirs = (
            (ctx.view.features[name], ctx.references.features[name])
            if name
            else (ctx.view.embeddings, ctx.references.embeddings)
        )
        near = best_cosine(mine, theirs) >= NEAR_SHORTLIST
        outside = np.where(near, OUTSIDE_SHORTLIST, -1.0)
        values: list[float] = np.where(np.isfinite(scores), scores, outside).tolist()
        return values

    def evaluate_same_view(
        self,
        embedder: str | None = None,
        layout_weight: float | None = None,
        *,
        patches: bool = False,
        only_holdout: bool = False,
    ) -> ViewReport:
        """How well match separates the views you labelled, and the ``min_similarity``
        that would serve them best.

        ``embedder`` gives "same subject" in place of the main embedding, and
        ``layout_weight`` replaces the task's mix: how candidates are compared before one
        is adopted. An embedder Calico does not compute for every photo is built and run
        on the labelled photos only, and cached; the catalog is otherwise used as it is.
        ``patches`` scores each photo by its patch share with the reference instead
        (``selection.patch_match``, with the view embedder), judged from the task's ``sure_share``.
        """
        held = self._held(only_holdout)
        if patches:
            return self._evaluate_patches(held, only_holdout)
        default_weight, default = self._same_view_defaults()
        weight = default_weight if layout_weight is None else layout_weight
        semantic, indexer = None, self.engine.indexer
        if embedder is not None and embedder != self.engine.settings.embedder:
            semantic = embedder
            extra = self.engine.extractors.get(embedder) or build_extractor(
                embedder, self.engine.settings
            )
            indexer = Indexer(
                self.engine.settings,
                self.engine.catalog,
                {**self.engine.extractors, embedder: extra},
            )
        searches, skipped, copies = [], [], 0
        for label_set, labels in self._labelled_sets("same_view"):
            reference = label_set.reference
            if (str(reference) in held.references) != only_holdout:
                continue
            if reference is None or not reference.is_file():
                skipped.append(label_set.set_id)
                continue
            trace = Trace()
            photos, view = self._labelled_photos(label_set, labels, trace)
            frames = self.engine.copy_vectors(view)
            if semantic is not None:
                _, matrices = indexer.features_for_files([p.path for p in photos])
                view = self.engine.as_view(matrices)
            _, matrices = indexer.features_for_files([reference])
            ref_view = self.engine.as_view(matrices)
            match = view_match(view, ref_view, weight, trace, semantic=semantic).match
            search, folded = count_copies_once(
                LabelledView(
                    set_id=label_set.set_id,
                    text=label_set.text,
                    match={p.content_key: float(m) for p, m in zip(photos, match, strict=True)},
                    labels={p.content_key: labels[p.content_key] for p in photos},
                ),
                [p.content_key for p in photos],
                frames,
            )
            searches.append(search)
            copies += folded
        return dataclasses.replace(
            measure_views(searches, default),
            skipped=skipped,
            copies=copies,
            embedder=embedder or self.engine.settings.embedder,
            layout_weight=weight,
        )

    def _evaluate_patches(self, held: HoldOut, only_holdout: bool) -> ViewReport:
        (searches,), skipped, copies = self._patch_searches(held, only_holdout, (patch_shares,))
        return dataclasses.replace(
            measure_views(searches, self._sure_share()),
            skipped=skipped,
            copies=copies,
            embedder=f"{self.engine.settings.view_embedder} (patches)",
            layout_weight=0.0,
        )

    def evaluate_view_geometry(self, *, only_holdout: bool = False) -> ViewGeometry:
        """Today's patch share against the share of its matches that sit in place
        (``selection.patch_match.consistent_shares``, audit A10-7), on your labelled same
        views: over every labelled photo, and where the share is least sure
        (``evaluation.GEOMETRY_BAND``). Each photo's patches are read once for both."""
        held = self._held(only_holdout)
        (share, consistent), skipped, copies = self._patch_searches(
            held, only_holdout, (patch_shares, consistent_shares)
        )
        measured = measure_view_geometry(share, consistent, sure=self._sure_share())
        name = f"{self.engine.settings.view_embedder} (patches"

        def named(reports: list[ViewReport]) -> list[ViewReport]:
            today, in_place = reports
            return [
                dataclasses.replace(
                    r, skipped=skipped, copies=copies, embedder=e, layout_weight=0.0
                )
                for r, e in ((today, f"{name})"), (in_place, f"{name} in place)"))
            ]

        return dataclasses.replace(
            measured, whole=named(measured.whole), within_band=named(measured.within_band)
        )

    def _patch_searches(
        self, held: HoldOut, only_holdout: bool, measures: Sequence[PatchMeasure]
    ) -> tuple[list[list[LabelledView]], list[str], int]:
        """Each labelled same-view search, scored by each of ``measures`` against its
        reference, a frame saved twice counted once; the sets whose reference is gone; and
        how many copies were folded in."""
        trace = Trace()
        extractor = self.engine.patch_extractor(trace)
        if extractor is None:
            raise CapabilityError(trace.notes[-1])
        searches: list[list[LabelledView]] = [[] for _ in measures]
        skipped, copies = [], 0
        for label_set, labels in self._labelled_sets("same_view"):
            reference = label_set.reference
            if (str(reference) in held.references) != only_holdout:
                continue
            if reference is None or not reference.is_file():
                skipped.append(label_set.set_id)
                continue
            photos, view = self._labelled_photos(label_set, labels, trace)
            (key,), _ = self.engine.indexer.features_for_files([reference])
            measured = self.engine.patch_measures(
                extractor, photos, self.engine.reference_images([reference], [key]), measures
            )
            keys = [p.content_key for p in photos]
            frames = self.engine.copy_vectors(view)
            for found, values in zip(searches, measured, strict=True):
                search, folded = count_copies_once(
                    LabelledView(
                        set_id=label_set.set_id,
                        text=label_set.text,
                        match={k: float(m) for k, m in zip(keys, values, strict=True)},
                        labels={k: labels[k] for k in keys},
                    ),
                    keys,
                    frames,
                )
                found.append(search)
            copies += folded
        return searches, skipped, copies

    def _sure_share(self) -> float:
        return float(self.engine.task("same_view").parse_options({}).sure_share)

    def evaluate_ratings(
        self,
        ratings: Sequence[Rating],
        quality_weights: Sequence[float] = (),
        *,
        only_holdout: bool = False,
    ) -> RatingsReport:
        """How well today's strength, and each signal in it, agrees with your ratings.

        Measured within each location, against that location's whole pool, because
        that is how a request ranks (quality and aesthetic are percentiles within the
        pool). ``quality_weights`` adds strength with quality at each weight (aesthetic
        the rest), to see whether the default mix suits you. Uses the catalog as it is.
        """
        wanted = by_photo(ratings)
        roots = tuple(c.root for c in self.engine.settings.collections)
        photos = (
            self.engine.unique_by_content(self.engine.catalog.photos_under(roots)) if roots else []
        )
        held = self._held(only_holdout)
        reach = self._reach(held)
        found = [p for p in photos if p.path in wanted and (p.location in reach) == only_holdout]
        options = self.engine.task("best").parse_options({})
        rows, one_sided = [], 0
        scored: dict[str, list[Scored]] = {}
        for location in sorted({p.location for p in found}):
            trace = Trace()
            pool, view = self.engine.feature_view(
                [p for p in photos if p.location == location], trace
            )
            labels = {i: verdict(wanted[p.path]) for i, p in enumerate(pool) if p.path in wanted}
            if not {0, 2} <= set(labels.values()):
                one_sided += 1
                continue
            ctx = TaskContext(
                request=SelectionRequest(task="best", folders=roots),
                photos=tuple(pool),
                view=view,
                models=self.engine.models,
                trace=trace,
            )
            strength, signals = photo_strength(
                ctx,
                range(len(pool)),
                quality_weight=options.quality_weight,
                aesthetic_weight=options.aesthetic_weight,
                criteria_share=0.0,
            )
            rated = sorted(labels)
            y = [labels[i] for i in rated]
            scores = {s.name: s.values[rated].tolist() for s in signals}
            scores["strength"] = strength[rated].tolist()
            for weight in quality_weights:
                mixed, _ = photo_strength(
                    ctx,
                    range(len(pool)),
                    quality_weight=weight,
                    aesthetic_weight=1 - weight,
                    criteria_share=0.0,
                )
                scores[f"strength (quality {weight:g})"] = mixed[rated].tolist()
            rows.append(
                LocationRatings(
                    location=location,
                    rated=len(rated),
                    liked=y.count(2),
                    disliked=y.count(0),
                    auc={name: auc(values, y) for name, values in scores.items()},
                )
            )
            scored[location] = [
                Scored(liked=label == 2, scores={name: v[j] for name, v in scores.items()})
                for j, label in enumerate(y)
                if label in (0, 2)
            ]
        return RatingsReport(
            rows=rows,
            rated=len(found),
            missing=len(wanted) - len(found),
            one_sided=one_sided,
            measured=measure_signals(scored, baseline="strength"),
        )

    def thumbs(self) -> list[Rating]:
        """Your 👍/👎 standing on Strongest results, as ratings of the photos they are on.
        Thumbs on stories judge how a photo fits the story, so they are left out. A photo
        the catalog no longer has keeps a rating under its content key, so the report
        counts it as missing rather than dropping it."""
        answers = [
            a
            for a in self.engine.ratings.current().values()
            if a.use == "result"
            and isinstance(a.scope, dict)
            and a.scope.get("task") == "best"
            and a.value in (-1, 1)
        ]
        keys = [a.subject[0] for a in answers]
        paths = {p.content_key: p.path for p in self.engine.catalog.photos_with_keys(keys)}
        return [
            Rating(
                paths.get(a.subject[0], Path(f"content key {a.subject[0]}")),
                1.0 if a.value == 1 else 0.0,
                "calico",
            )
            for a in answers
        ]

    def _same_view_defaults(self) -> tuple[float, float]:
        """The registered same-view task's default layout weight and ``min_similarity``."""
        options = self.engine.task("same_view").parse_options({})
        return float(options.layout_weight), float(options.min_similarity)

    def _labelled_sets(
        self, *kinds: Literal["describe", "same_view", "taste", "compare", "anchor"]
    ) -> Iterator[tuple[LabelSet, dict[str, int]]]:
        """Each label set of ``kinds`` (default: describe) with a rating standing, and those
        ratings by content key, newest set first. A review counts once, in the newest set
        that asked it, so a photo rated again in a later session is not counted twice (audit
        A05-5); a Compare Strongest photo counts once for each state of the taste that chose
        Yours's picks (it had learned the place, or not), since those are reported apart: a
        later session at a place the taste had learned must not take its photos from an
        earlier one it had not (10 November 2026 audit, B-6)."""
        current = self.engine.ratings.current()
        seen: set[tuple[Review, bool | None]] = set()
        for label_set in self.engine.labels.all():
            if label_set.kind not in (kinds or ("describe",)):
                continue
            state = label_set.taste.seen if label_set.taste is not None else None
            labels: dict[str, int] = {}
            for key, label in labels_of(label_set, current, controls=False).items():
                if ((review := photo_review(label_set, key)), state) not in seen:
                    seen.add((review, state))
                    labels[key] = label
            if labels:
                yield label_set, labels

    def _labelled_photos(
        self, label_set: LabelSet, labels: Mapping[str, int], trace: Trace
    ) -> tuple[list[Photo], FeatureView]:
        """The labelled photos still catalogued, with their features. A feature added since
        they were indexed (a new extractor) is computed for them now, not left to drop them."""
        under = self.engine.catalog.photos_under(list(label_set.folders))
        found = self.engine.unique_by_content([p for p in under if p.content_key in labels])
        keys = [p.content_key for p in found]
        complete = set(keys)
        for extractor in self.engine.extractors.values():
            complete &= self.engine.catalog.has_features(feature_key(extractor), keys)
        for photo in (p for p in found if p.content_key not in complete):
            try:
                self.engine.indexer.features_for_files([photo.path])
            except ImageDecodeError:
                trace.note(f"{photo.path.name} could not be read and was left out.")
        return self.engine.feature_view(found, trace)

    def _readers(self, names: Sequence[str] | None) -> list[Embedder]:
        if names is None:
            others = [
                ex
                for name, ex in self.engine.extractors.items()
                if name != self.engine.settings.describe_embedder
                and isinstance(ex, Embedder)
                and ex.supports_text
            ]
            return [self.engine.describer, *others]
        readers = []
        for name in names:
            reader = self.engine.extractors.get(name)
            if not isinstance(reader, Embedder):
                readable = [
                    n for n, ex in self.engine.extractors.items() if isinstance(ex, Embedder)
                ]
                raise InvalidRequestError(
                    f"No embedder {name!r} here. Available: {', '.join(readable)}"
                )
            readers.append(reader)
        return readers

    def _embedding_alone(self, name: str) -> ImageExtractor | None:
        """``name`` built to be computed for the rated photos alone, or None when Calico
        computes it for every photo. Refused unless it is an embedder: a taste is learned
        on an embedding, not on a score or a layout."""
        found = self.engine.extractors.get(name)
        extractor = found or build_extractor(name, self.engine.settings)
        if not isinstance(extractor, Embedder):
            raise InvalidRequestError(
                f"{name!r} is not an embedding a taste can be learned on. Name an embedder, "
                "such as clip or clip_large."
            )
        return None if found is not None else extractor

    def _yours_candidate(self, name: str) -> ImageExtractor:
        """The embedding a candidate Yours is learned on, against today's Yours."""
        if name == self.engine.settings.embedder:
            raise InvalidRequestError(
                f"Yours is already learned on {name!r}, so there is nothing to compare it "
                "with. Name another embedder, such as clip_large."
            )
        rival = self.engine.extractors.get(name) or build_extractor(name, self.engine.settings)
        if not isinstance(rival, Embedder):
            raise InvalidRequestError(
                f"{name!r} is not an embedding a taste can be learned on. Name an embedder, "
                "such as clip_large."
            )
        return rival

    def _general_reader(self, name: str) -> Embedder:
        """The embedder that reads today's wording for a candidate General: one that reads
        text, other than the one General reads with today."""
        if name == self.engine.settings.embedder:
            raise InvalidRequestError(
                f"General already reads its wording with {name!r}, so there is nothing to "
                "compare it with. Name another embedder, such as clip_large."
            )
        reader = self.engine.extractors.get(name) or build_extractor(name, self.engine.settings)
        if not isinstance(reader, Embedder) or not reader.supports_text:
            raise InvalidRequestError(
                f"{name!r} cannot read General's wording: it has no text encoder. Name one "
                "that reads text, such as clip_large."
            )
        return reader

    def _read_by(self, reader: Embedder) -> tuple[LabelModel, str]:
        """Today's wording for General's aesthetic, read zero-shot by ``reader``, and its
        name for the report and the tries (its version names the reader)."""
        taxonomy = discover_taxonomies(self.engine.settings.taxonomy_dirs)["aesthetic"]
        model = ZeroShotLabelModel(taxonomy, reader)
        return model, model.version

    def _taste_candidates(self) -> list[str]:
        """Embeddings a taste can be learned on: those computed for every photo."""
        names = [
            self.engine.settings.embedder,
            self.engine.settings.describe_embedder,
            self.engine.settings.view_embedder,
        ]
        return [
            n for i, n in enumerate(names) if n in self.engine.extractors and n not in names[:i]
        ]

    def holdout(self) -> HoldOut | None:
        """The hold-out, if one was drawn (``calico.holdout``)."""
        return read_holdout(self.engine.settings.holdout_path)

    def draw_holdout(
        self,
        seed: int = SEED,
        leave_out: Collection[str] = (),
        places: int | None = None,
        *,
        write: bool = True,
    ) -> HoldOut:
        """Set aside, once, some of what you rated for a last check no tuning has seen:
        places judged for your taste (with likes and dislikes), same-view references, and
        story sessions (``evaluation.draw_holdout``). ``leave_out`` names places never drawn,
        with the places next to them: ones you designed against, which no check can then
        call unseen. ``places`` holds out that many places, not ``HOLDOUT_SIZES``'s. With
        ``write`` false it only says what it would draw: the draw is the same, and nothing is
        written."""
        labels = taste_labels(self.engine.ratings.current())
        photos = self._taste_photos(self.engine.settings.embedder, labels)
        judged = [
            place
            for place in {p.location for p in photos if p.from_session}
            if len({p.liked for p in photos if p.from_session and p.location == place}) == 2
        ]
        if leave_out:
            known = self.engine.catalog.keys_at(sorted(set(leave_out)))
            if unknown := [name for name in dict.fromkeys(leave_out) if name not in known]:
                raise no_such_locations(unknown, judged)
            gone = set(leave_out) | {o for near in self._near(leave_out).values() for o in near}
            judged = [place for place in judged if place not in gone]
        references = [
            str(label_set.reference)
            for label_set, _ in self._labelled_sets("same_view")
            if label_set.reference is not None
        ]
        current = self.engine.ratings.current()
        stories = sorted({s[0] for s in self._stories_rated_in_sessions(current)})
        sizes = HOLDOUT_SIZES if places is None else {**HOLDOUT_SIZES, "places": places}
        drawn = draw_holdout(
            judged, references, stories, seed=seed, at=self.engine.clock(), sizes=sizes
        )
        if write:
            write_holdout(self.engine.settings.holdout_path, drawn)
        return drawn

    def draw_controls(self, seed: int = SEED, size: int = CONTROL_POOL) -> Controls:
        """Set aside, once, the pool of control photos (``evaluation.controls``): photos rated
        in Compare Strongest sessions, where first picks come from, outside what the hold-out
        reaches (its places, the locations next to them, and the frames also filed under
        them)."""
        labels = taste_labels(self.engine.ratings.current())
        compared = {s.set_id for s in self.engine.labels.all() if s.kind == "compare"}
        keys = [k for k, s in labels.session.items() if s in compared]
        if not keys:
            raise InvalidRequestError(
                "No photo has been rated in a Compare Strongest session yet, so there is "
                "nothing to ask again. Rate one first (the Label page, or calico eval "
                "new-compare)."
            )
        held = self._held(only_holdout=False)
        reach = self._reach(held)
        photos = self.engine.catalog.photos_with_keys(keys)
        at_held, _ = self._held_filing(photos)
        drawn = draw_controls(
            [
                Candidate(p.content_key, p.location, labels.liked[p.content_key])
                for p in photos
                if p.location not in reach and p.content_key not in at_held
            ],
            seed=seed,
            size=size,
        )
        if not drawn:
            raise InvalidRequestError(
                "No photo rated in a Compare Strongest session lies outside the hold-out and "
                "the locations next to it, so none can be asked again. The pool is drawn "
                "once: rate a session at another location first."
            )
        controls = Controls(photos=drawn, seed=seed, drawn_at=self.engine.clock())
        write_controls(self.engine.settings.controls_path, controls)
        return controls

    def retest(self) -> Retest:
        """What the control photos, asked again, say about how consistent you are
        (``evaluation.controls``). Of a photo answered twice in one session, the newer counts:
        the first was a correction."""
        pool = read_controls(self.engine.settings.controls_path)
        if pool is None:
            raise InvalidRequestError(
                "No control photos have been drawn yet. Draw them once with: calico controls draw"
            )
        newest: dict[tuple[str, str], Ask] = {}
        for a in sorted(self.engine.ratings.journal.answers(), key=lambda a: a.at):
            if is_control(a) and (liked := clear_answer(a)) is not None:
                key, session = a.subject[0], a.session or ""
                newest[key, session] = Ask(key=key, at=a.at, liked=liked, session=session)
        return measure_retest({c.key: c.liked for c in pool.photos}, list(newest.values()))

    def _near(self, places: Collection[str]) -> dict[str, frozenset[str]]:
        """For each of ``places``, the other locations next to it: holding a tenth or more of
        the smaller one's files (``evaluation.holdout.near_places``), as a trip filed under
        each of its sites and again under an umbrella folder does."""
        own = self.engine.catalog.keys_at(sorted(set(places)))
        filed = self.engine.catalog.locations_of(sorted({k for keys in own.values() for k in keys}))
        shared: dict[str, Counter[str]] = {place: Counter() for place in own}
        for place, keys in own.items():
            for key in keys:
                shared[place].update(o for o in filed.get(key, ()) if o != place)
        others = {o for counts in shared.values() for o in counts}
        sizes = {o: len(k) for o, k in self.engine.catalog.keys_at(sorted(others)).items()}
        return {place: near_places(len(own[place]), n, sizes) for place, n in shared.items()}

    def _reach(self, held: HoldOut) -> frozenset[str]:
        """The held-out places and the locations next to them: what no tuning may see."""
        return frozenset(held.places) | {o for n in self._near(held.places).values() for o in n}

    def _held(self, only_holdout: bool) -> HoldOut:
        """What is held out (nothing, before a draw). Asking for the hold-out alone before
        one was drawn is refused: there is nothing to report on."""
        held = self.holdout()
        if held is None:
            if only_holdout:
                raise HoldoutError(
                    "No hold-out has been drawn yet, so there is nothing to report on alone. "
                    "Draw it once with: calico holdout draw"
                )
            return HoldOut((), (), (), seed=0, drawn_at=datetime.min.replace(tzinfo=UTC))
        return held

    def evaluate_taste(
        self,
        extractors: Sequence[str] | None = None,
        l2s: Sequence[float] = (TASTE_L2,),
        *,
        only_holdout: bool = False,
        quality_weights: Sequence[float] | None = None,
        technical: Literal["2", "3"] | None = None,
        aesthetic: str | None = None,
        fit: TasteFit | None = None,
        progress: PlaceProgressFn | None = None,
        prompts: Path | None = None,
        blend: bool = False,
        general_embedder: str | None = None,
        yours_embedder: str | None = None,
        scores: Path | None = None,
        score_field: str = "score",
    ) -> list[TasteReport]:
        """Would Yours rank your ratings better than General, at places it never learned
        from? One report per embedding, regularisation and quality weight (quality's share
        of strength; default: Strongest's), over the whole range and among the photos
        Strongest would show first (``evaluation.measure_taste``).

        With ``technical``, each report also measures General's strength with that version
        of the technical measures (``features.technical``) against today's, paired: it is
        computed for the rated photos alone, so a candidate is judged without indexing the
        archive again. With ``aesthetic`` (an extractor that gives one number per photo,
        higher looking better, such as ``laion_aesthetic``), each also measures General with
        that number as its aesthetic against today's, paired, computed the same way. With
        ``fit``, each also measures Yours fitted that way (``evaluation.TASTE_FITS``)
        against today's, paired. With ``prompts`` (a file worded as General's aesthetic
        vocabulary is), each also measures General with that wording against today's,
        paired, read from the stored embeddings; no search uses it. With
        ``general_embedder``, each also measures today's wording read by that embedder,
        computed for the rated photos alone. With ``blend``, that candidate is averaged
        with today's General by rank, half each. With ``yours_embedder``, each also
        measures Yours learned on that embedding (computed the same way) against today's
        Yours, paired (``TasteFit`` "embedding"). With either, only today's embedding is
        measured unless others are named. With ``scores`` (a file of one number per photo,
        ``evaluation.scores``), each also measures those numbers beside General and Yours, on
        the photos the file scores: Strongest's first picks are not read for them, and a
        photo thumbed on a result is judged too, among the photos a taste chose.
        ``progress`` hears where Strongest's first picks compare a place's views part by part
        (``PlaceProgressFn``)."""
        given = _scores_given(
            scores,
            score_field,
            alongside=technical is not None
            or aesthetic is not None
            or prompts is not None
            or general_embedder is not None
            or yours_embedder is not None
            or fit is not None
            or blend,
        )
        options = self.engine.task("best").parse_options({})
        if quality_weights is None:
            quality_weights = [
                options.quality_weight / (options.quality_weight + options.aesthetic_weight)
            ]
        candidate = None if technical is None else self._technical_candidate(technical)
        named = [c for c in (aesthetic, prompts, general_embedder) if c is not None]
        if blend and not named:
            raise InvalidRequestError(
                "--blend averages today's General with a candidate: name one with "
                "--aesthetic, --prompts or --general-embedder."
            )
        if len(named) > 1:
            raise InvalidRequestError(
                "Measure one candidate for General at a time: --aesthetic, --prompts or "
                "--general-embedder."
            )
        general = None if aesthetic is None else self._aesthetic_candidate(aesthetic)
        reader = None if general_embedder is None else self._general_reader(general_embedder)
        worded = (
            self._prompts_candidate(prompts)
            if prompts is not None
            else None
            if reader is None
            else self._read_by(reader)
        )
        if yours_embedder is not None and fit is not None:
            raise InvalidRequestError("Measure Yours one way at a time: --fit or --yours-embedder.")
        rival = None if yours_embedder is None else self._yours_candidate(yours_embedder)
        if rival is not None:
            fit = "embedding"
        # A candidate is measured once, not once per embedding: each report counts as a try.
        alone_today = rival is not None or reader is not None
        names = list(
            extractors
            or ([self.engine.settings.embedder] if alone_today else self._taste_candidates())
        )
        alone = {name: self._embedding_alone(name) for name in names}
        if fit == "centred" and (rated_only := [n for n in names if alone[n] is not None]):
            raise InvalidRequestError(
                f"--fit centred measures each photo against its place's mean over every photo "
                f"there, and {', '.join(map(repr, rated_only))} is computed for the rated "
                "photos alone. Measure it without --fit centred."
            )
        first_picks = self._first_picks(progress) if given is None else None
        notes = self._general_notes()
        labels = taste_labels(self.engine.ratings.current())
        held = self._held(only_holdout)
        _require_rated(given, labels.liked, scores)
        plain = (
            given is None
            and candidate is None
            and general is None
            and worded is None
            and reader is None
            and rival is None
            and fit is None
            and not blend
        )
        reports = []
        for name in names:
            embedding = alone[name]
            photos = self._taste_photos(
                name,
                labels,
                candidate,
                aesthetic=general,
                worded=None if worded is None else worded[0],
                centred=fit == "centred",
                scaled=fit == "by_scale",
                embedding=embedding,
                reader=reader,
                rival=rival,
            )
            read = [
                p
                for p in photos
                if (candidate is None or p.candidate_quality is not None)
                and ((general is None and reader is None) or p.candidate_general is not None)
                and (rival is None or p.candidate_vector is not None)
                and np.isfinite(p.vector).all()
            ]
            photos, unread = read, len(photos) - len(read)
            if not only_holdout:  # the hold-out is neither judged nor learned from
                photos = taste_apart(photos, held.places).photos
            photos = photos if given is None else _with_scores(photos, given)
            if fit == "top_pairs" and not any(not p.whole_range for p in photos):
                raise InvalidRequestError(
                    "Fitting Yours on pairs among the first picks learns from photos a taste "
                    "chose to show, and none is rated yet: rate a Compare Strongest session "
                    "(the Label page, or: calico eval new-compare FOLDER)."
                )
            extractor = "" if name == self.engine.settings.embedder else name
            feature = (
                self.engine.taste_feature(extractor)
                if embedding is None
                else feature_key(embedding)
            )
            pools = self._taste_pools(photos, name) if plain and embedding is None else None
            noted = notes
            noted += () if given is None else (_scores_note(given, labels.liked, photos),)
            if embedding is not None:
                noted += (
                    (
                        f"First picks not measured on {name!r}: it is computed for the rated "
                        "photos alone, and what Strongest would show first needs every photo "
                        "at a place."
                    ),
                )
            reports += [
                dataclasses.replace(
                    measured := measure_taste(
                        photos,
                        feature=feature,
                        extractor=extractor,
                        quality_weight=share,
                        l2=l2,
                        first_picks=first_picks if embedding is None else None,
                        judged=held.places if only_holdout else None,
                        untrained=held.places,
                        fit=fit,
                        general_blend=blend,
                        pools=pools,
                    ),
                    counted=TASTE_RULE,
                    notes=(*noted, _strength_note(measured, plain)),
                    unsure=labels.unsure,
                    disagree=labels.disagree,
                    technical="" if candidate is None else feature_key(candidate),
                    aesthetic_candidate=feature_key(general)
                    if general is not None
                    else ""
                    if worded is None
                    else worded[1],
                    unread=unread,
                    yours_candidate="" if rival is None else feature_key(rival),
                    scores=None
                    if measured.scores is None or given is None
                    else dataclasses.replace(measured.scores, name=given.name),
                )
                for l2 in l2s
                for share in quality_weights
            ]
        return reports

    def taste_disagreements(
        self, extractor: str | None = None, *, count: int = DISAGREEMENT_COUNT
    ) -> tuple[Disagreements, dict[str, Path]]:
        """The rated photos General and Yours rank furthest apart, by whom you sided with
        (``evaluation.taste_disagreements``), and where each one is. Yours is learned on
        ``extractor`` (default: the main embedding, which Yours uses); the hold-out is
        neither judged nor learned from."""
        options = self.engine.task("best").parse_options({})
        share = options.quality_weight / (options.quality_weight + options.aesthetic_weight)
        name = extractor or self.engine.settings.embedder
        held = self._held(only_holdout=False)
        photos = taste_apart(
            self._taste_photos(name, taste_labels(self.engine.ratings.current())), held.places
        ).photos
        chosen = "" if name == self.engine.settings.embedder else name
        found = taste_disagreements(
            photos,
            feature=self.engine.taste_feature(chosen),
            extractor=chosen,
            quality_weight=share,
            untrained=held.places,
            count=count,
            pools=self._taste_pools(photos, name),
        )
        keys = [
            d.content_key
            for group in (
                found.general_missed,
                found.general_wrong,
                found.yours_missed,
                found.yours_wrong,
            )
            for d in group
        ]
        paths = {p.content_key: p.path for p in self.engine.catalog.photos_with_keys(keys)}
        return found, paths

    def _scale_model(self) -> LabelModel:
        """The built-in ``scale`` vocabulary (aerial to macro), read zero-shot from the main
        embedding, for Yours fitted by scale."""
        embedder = self.engine.embedder
        if not embedder.supports_text:
            raise CapabilityError(
                f"Fitting Yours by scale reads each photo's scale with a text encoder, and the "
                f"embedder {embedder.name!r} has none: install the ml extra and set embedder = "
                '"clip".'
            )
        taxonomy = discover_taxonomies(self.engine.settings.taxonomy_dirs)["scale"]
        return ZeroShotLabelModel(taxonomy, embedder)

    def _prompts_candidate(self, path: Path) -> tuple[LabelModel, str]:
        """General's aesthetic worded as ``path`` says, read zero-shot from the main
        embedding as today's is, and its name for the report and the tries: the file's
        name and a digest of its wording, so each wording counts as a try of its own."""
        taxonomy = load_taxonomy(path)
        if "strong" not in taxonomy.labels:
            raise InvalidRequestError(
                f"{path} has no [labels.strong]: General's aesthetic ranks photos by how "
                "likely each is to be strong. Start from models/taxonomies/aesthetic.toml."
            )
        embedder = self.engine.embedder
        if not embedder.supports_text:
            raise CapabilityError(
                f"Wording for General is read by a text encoder, and the embedder "
                f"{embedder.name!r} has none: install the ml extra and set embedder = "
                '"clip".'
            )
        digest = hashlib.blake2b(
            "\n".join(
                f"{label}: {p}"
                for label, group in zip(taxonomy.labels, taxonomy.prompts, strict=True)
                for p in group
            ).encode(),
            digest_size=4,
        ).hexdigest()
        return ZeroShotLabelModel(taxonomy, embedder), f"{path.stem}@{digest}"

    def _technical_candidate(self, version: Literal["2", "3"]) -> TechnicalExtractor:
        today = self.engine.extractors["technical"]
        if today.version == version:
            raise InvalidRequestError(
                f"technical@{version} is already what Calico uses, so there is nothing to "
                "compare it with. Name the other version."
            )
        return TechnicalExtractor(noise_aware=version == "3")

    def _aesthetic_candidate(self, name: str) -> ImageExtractor:
        """The extractor ``name`` as General's aesthetic: it must give one number a photo."""
        extractor = self.engine.extractors.get(name) or build_extractor(name, self.engine.settings)
        if extractor.dim != 1:
            raise InvalidRequestError(
                f"{name!r} gives {extractor.dim} numbers for each photo, but an aesthetic "
                "for General is one number per photo. Name an extractor such as "
                "laion_aesthetic."
            )
        return extractor

    def _features_of(
        self, photos: Sequence[Photo], extractor: ImageExtractor
    ) -> dict[str, NDArray[np.float32]]:
        """``extractor``'s features for ``photos``, computed for those not yet cached (under
        its own key, beside today's) and read back. A photo that cannot be read now (moved,
        or on a drive not mounted) is missing from the result."""
        key = feature_key(extractor)
        keys = [p.content_key for p in photos]
        have = self.engine.catalog.has_features(key, keys)
        missing = [p.path for p in photos if p.content_key not in have]
        indexer = Indexer(self.engine.settings, self.engine.catalog, {extractor.name: extractor})
        for start in range(0, len(missing), _DECODE_BATCH):
            batch = missing[start : start + _DECODE_BATCH]
            try:
                indexer.features_for_files(batch)
            except ImageDecodeError:  # one by one, so one unreadable file costs only itself
                for path in batch:
                    with contextlib.suppress(ImageDecodeError):
                        indexer.features_for_files([path])
        return self.engine.catalog.get_features(key, keys)

    def _first_picks(self, progress: PlaceProgressFn | None = None) -> FirstPicksFn:
        """What Strongest would show first at a place: run as a search of the whole place
        would be (bursts collapsed, one pick per view, varied), but in memory, so measuring
        saves no runs. Pairs it compares part by part are remembered, as a search's are,
        and ``progress`` hears of those it compares now. Each place's photos, and General's
        picks, are read once."""
        task = self.engine.task("best")
        options = task.parse_options({})
        roots = tuple(c.root for c in self.engine.settings.collections)
        everything: list[Photo] = []
        pools: dict[str, tuple[list[Photo], FeatureView]] = {}
        general: dict[tuple[str, int], list[str]] = {}

        def first_picks(location: str, taste: TasteWeights | None, k: int) -> list[str]:
            if taste is None and (location, k) in general:
                return general[location, k]
            trace = Trace()
            if not everything:
                everything.extend(self.engine.gather(roots, trace))
            if location not in pools:
                pools[location] = self.engine.feature_view(
                    [p for p in everything if p.location == location], trace
                )
            photos, view = pools[location]
            ctx = TaskContext(
                request=SelectionRequest(task="best", folders=roots, limit=k),
                photos=tuple(photos),
                view=view,
                models=self.engine.models
                if taste is None
                else WithTaste(self.engine.models, TasteModel(taste)),
                trace=trace,
                limit=k,
                view_feature=self.engine.view_feature(),
            )
            if photos:  # one pick per view: compared as a search would compare them
                ctx = self.engine.compare_parts(
                    task,
                    options,
                    ctx,
                    None
                    if progress is None
                    else lambda done, total, _step: progress(location, done, total),
                )
            shown = [p.content_key for g in task.run(ctx, options) for p in g.picks]
            if taste is None:
                general[location, k] = shown
            return shown

        return first_picks

    def train_taste(
        self,
        extractor: str | None = None,
        l2: float = TASTE_L2,
        *,
        force: bool = False,
        accept_missing: bool = False,
    ) -> TasteTraining:
        """Measure, then learn your taste from every rating and save it, but only if it
        beats General at places it never learned from (``force`` saves it regardless).
        Refused while a labelling set cannot be read, unless ``accept_missing``: the taste
        would be learned without its ratings."""
        unreadable = self.engine.labels.unreadable()
        if unreadable and not accept_missing:
            names = ", ".join(path.name for path, _ in unreadable)
            raise InvalidRequestError(
                f"{_count(len(unreadable), 'labelling set')} cannot be read ({names}), so "
                "your taste would be learned without their ratings. calico doctor says why; "
                "fix or move them, or train anyway with --accept-missing."
            )
        name = extractor or self.engine.settings.embedder
        if name not in self.engine.extractors:
            raise InvalidRequestError(
                f"Yours learned on {name!r} could not run in a search: Calico does not compute "
                f"it for every photo. Measure it with: calico eval taste --embedder {name}"
            )
        (report,) = self.evaluate_taste([name], [l2])
        if not (force or report.yours_wins):
            return TasteTraining(report=report, saved=False)
        weights = self._learned_taste(name, l2, report)
        self.engine.trained.save_taste(weights)
        return TasteTraining(report=report, saved=True, calibration=weights.calibration)

    def cross_fitted_taste(
        self, folders: Sequence[Path], progress: ProgressFn | None = None
    ) -> TasteWeights:
        """Yours as it would be had it never seen the places of ``folders``, nor the places
        next to them (files in common), nor copies of their photos: learned from every other
        rating but the hold-out's, for one search alone and never saved. At a place the saved
        taste has learned, this is the taste to judge: it is the saved taste's own recipe on
        the ratings that tell nothing about the place, so the place is fresh to it (decision
        0017). It is not the saved taste without a place: it learns from every current
        rating, so it also holds what was rated since the saved one was trained."""
        resolved = self.engine.validated_folders(folders)
        self.engine.indexer.index(resolved, progress=progress)
        places = set(self.engine.catalog.locations_under(resolved))
        if not places:
            raise InvalidRequestError("No readable photos were found in the chosen folders.")
        reach = places | {o for others in self._near(places).values() for o in others}
        name = self.engine.settings.embedder
        (report,) = self.evaluate_taste([name], [TASTE_L2])
        return self._learned_taste(name, TASTE_L2, report, reach)

    def _learned_taste(
        self,
        name: str,
        l2: float,
        report: TasteReport,
        leave_out: Collection[str] = (),
    ) -> TasteWeights:
        """Yours learned from every rating but the hold-out's and those at ``leave_out`` (kept
        for a fresh last check, with their copies and other names): what ``train_taste`` saves
        and what ``cross_fitted_taste`` shows a Compare session. What a probability means is
        read from what you said about the photos a taste showed first, at places it had not
        learned (``report``): those are the photos a pick is; read from the whole range it
        overstates them (decision 0010). Where places are left out, theirs are not read."""
        held = self._held(only_holdout=False)
        every = self._taste_photos(name, taste_labels(self.engine.ratings.current()))
        photos = taste_apart(every, {*held.places, *leave_out}).photos
        if len({p.liked for p in photos}) < 2:
            if leave_out:
                raise InvalidRequestError(
                    f"Yours fitted without {', '.join(sorted(leave_out))} would have nothing to "
                    "learn from: no other place holds photos you liked and photos you did not. "
                    "Rate more places with Rate for your taste (the Label page)."
                )
            raise InvalidRequestError(
                "Learning your taste needs photos you like and photos you don't: rate some "
                "with Rate for your taste (the Label page)."
            )
        weights = fit_taste(
            np.stack([p.vector for p in photos]),
            np.asarray([p.liked for p in photos], dtype=np.int8),
            feature=report.feature,
            extractor="" if name == self.engine.settings.embedder else name,
            l2=l2,
        )
        groups = {p.place for p in every if p.location in leave_out} | set(leave_out)
        gone = set(leave_out) | {p.location for p in every if p.place in groups}
        first = [f for f in report.forecasts if f.first_pick and f.place not in gone]
        calibration = fit_calibration(
            [(f.yours, f.liked) for f in first], places=len({f.place for f in first})
        )
        return dataclasses.replace(
            weights,
            places=tuple(sorted({p.location for p in photos})),
            calibration=calibration,
        )

    def _taste_photos(
        self,
        extractor: str,
        labels: TasteLabels,
        candidate: ImageExtractor | None = None,
        *,
        aesthetic: ImageExtractor | None = None,
        worded: LabelModel | None = None,
        centred: bool = False,
        scaled: bool = False,
        embedding: ImageExtractor | None = None,
        reader: Embedder | None = None,
        rival: ImageExtractor | None = None,
    ) -> list[TastePhoto]:
        """Every photo with a label for your taste (``reviews.TASTE_RULE``), with its
        ``extractor`` vector and today's general scores, its quality with a ``candidate``
        technical extractor and its candidate ``aesthetic`` for General where they can be read,
        or General's p("strong") with a candidate wording (``worded``), (``centred``) its
        place's mean vector, and (``scaled``) its scale. With ``embedding`` (an embedder
        not computed for every photo), each vector is that one's, computed for the rated
        photos alone; one that cannot be read now is NaN. With ``reader``, the ``worded``
        General reads that embedder's vectors, computed the same way; with ``rival``, each
        photo also carries its vector in that embedding as its ``candidate_vector``. Every
        label is one of your stored answers: nothing else makes one."""
        rated = {k: (liked, labels.from_session[k]) for k, liked in labels.liked.items()}
        for_value = {  # (session, photo) drawn for its value: left out of the whole-range AUC
            (s.set_id, item.content_key)
            for s in self.engine.labels.all()
            if s.kind == "taste"
            for item in s.items
            if item.stratum is not None and (item.stratum in VALUE_STRATA or item.stratum == GIVEN)
        } | {  # shown first by today's General or Yours (Compare Strongest): chosen by them too
            (s.set_id, item.content_key)
            for s in self.engine.labels.all()
            if s.kind == "compare"
            for item in s.items
        }
        trace = Trace()
        photos, view = self.engine.feature_view(
            self.engine.catalog.photos_with_keys(list(rated)), trace
        )
        if not photos:
            return []
        if embedding is not None:
            found = self._features_of(photos, embedding)
            unread = np.full(embedding.dim, np.nan, dtype=np.float32)
            vectors = np.stack([found.get(p.content_key, unread) for p in photos])
        elif extractor == self.engine.settings.embedder:
            vectors = view.embeddings
        else:
            vectors = view.features[extractor]
        quality_model = self.engine.models.get("quality")
        quality = (
            quality_model.score(view)
            if isinstance(quality_model, ScoreModel)
            else np.full(len(view), 0.5, dtype=np.float32)
        )
        general = self._general_aesthetic(view)
        candidate_quality: list[float | None] = [None] * len(photos)
        if candidate is not None and isinstance(quality_model, ScoreModel):
            found = self._features_of(photos, candidate)
            today = view.features["technical"]
            rows = np.stack([found.get(p.content_key, today[i]) for i, p in enumerate(photos)])
            scored = quality_model.score(
                dataclasses.replace(view, features={**view.features, "technical": rows})
            )
            candidate_quality = [
                float(scored[i]) if p.content_key in found else None for i, p in enumerate(photos)
            ]
        candidate_general: list[float | None] = [None] * len(photos)
        if aesthetic is not None:
            scores = self._features_of(photos, aesthetic)
            candidate_general = [
                float(scores[p.content_key][0])
                if p.content_key in scores and np.isfinite(scores[p.content_key][0])
                else None
                for p in photos
            ]
        if worded is not None and reader is None:
            strong = worded.predict_proba(view)[:, worded.labels.index("strong")]
            candidate_general = [float(v) for v in strong]
        elif worded is not None and reader is not None:
            seen = self._features_of(photos, reader)
            unseen = np.zeros(reader.dim, dtype=np.float32)
            rows = np.stack([seen.get(p.content_key, unseen) for p in photos])
            strong = worded.predict_proba(dataclasses.replace(view, embeddings=rows))[
                :, worded.labels.index("strong")
            ]
            candidate_general = [
                float(strong[i]) if p.content_key in seen else None for i, p in enumerate(photos)
            ]
        rivals: dict[str, NDArray[np.float32]] = (
            {} if rival is None else self._features_of(photos, rival)
        )
        scales = [""] * len(photos)
        if scaled:
            model = self._scale_model()
            scales = [model.labels[i] for i in model.predict_proba(view).argmax(axis=1)]
        frames = _same_frames([p.content_key for p in photos], self.engine.copy_vectors(view))
        places = {
            name: group.names[0]
            for group in similar_places(Counter(p.location for p in photos))
            for name in group.names
        }
        means = self._place_means(extractor, {p.location for p in photos}) if centred else {}
        at_held, beside = self._held_filing(photos)
        return [
            TastePhoto(
                location=at_held.get(p.content_key, p.location),
                liked=rated[p.content_key][0],
                from_session=rated[p.content_key][1],
                vector=vectors[i],
                general=float(general[i]),
                quality=float(quality[i]),
                content_key=p.content_key,
                same_frame=frames.get(p.content_key, ""),
                same_place=beside.get(p.location) or places.get(p.location, ""),
                candidate_quality=candidate_quality[i],
                candidate_general=candidate_general[i],
                place_mean=means.get(p.location),
                collection=p.collection,
                scale=scales[i],
                candidate_vector=rivals.get(p.content_key),
                whole_range=(labels.session.get(p.content_key) or "", p.content_key)
                not in for_value,
            )
            for i, p in enumerate(photos)
        ]

    def _held_filing(self, photos: Sequence[Photo]) -> tuple[dict[str, str], dict[str, str]]:
        """What the hold-out reaches beyond a photo's first path (10 November 2026 audit, B-1):
        the held-out place each frame is also filed under (that is its place, for the hold-out),
        and, for each location next to a held-out place, that place (it is grouped with it, as
        a second name for it is)."""
        held = self.holdout()
        if held is None or not held.places:
            return {}, {}
        filed = self.engine.catalog.locations_of([p.content_key for p in photos])
        at_held = {}
        for p in photos:
            here = sorted(set(held.places) & filed.get(p.content_key, frozenset()))
            if here:
                at_held[p.content_key] = here[0]
        beside: dict[str, str] = {}
        for place, others in self._near(held.places).items():
            for other in sorted(others):
                beside.setdefault(other, place)
        return at_held, beside

    def _place_means(
        self, extractor: str, locations: Collection[str]
    ) -> dict[str, NDArray[np.float32]]:
        """Each named place's mean ``extractor`` vector over every frame catalogued there,
        rated or not: the rated photos are drawn by strength, so their own mean would lean
        to whatever you rated most."""
        key = feature_key(self.engine.extractors[extractor])
        out = {}
        for location, keys in self.engine.catalog.keys_at(sorted(set(locations) - {""})).items():
            found = self.engine.catalog.get_features(key, keys).values()
            finite = [v for v in found if np.isfinite(v).all()]
            if finite:
                out[location] = np.mean(finite, axis=0).astype(np.float32)
        return out

    def _taste_pools(self, photos: Sequence[TastePhoto], extractor: str) -> dict[str, Pool]:
        """Every location with a rated photo you could be judged on, with all it holds: what
        Strongest ranks when you search it (``evaluation.Pool``), so strength is scored over
        it and not among the rated photos alone. None at all where the quality or the
        aesthetic model cannot run, so strength falls back to the rated photos, as before."""
        quality_model = self.engine.models.get("quality")
        if not isinstance(quality_model, ScoreModel) or self._aesthetic_model() is None:
            return {}
        both = {
            place
            for place in {p.location for p in photos if p.from_session}
            if len({p.liked for p in photos if p.from_session and p.location == place}) == 2
        }
        pools = {}
        for location, keys in self.engine.catalog.keys_at(sorted(both)).items():
            pooled, view = self.engine.feature_view(
                self.engine.catalog.photos_with_keys(keys), Trace()
            )
            if not pooled:
                continue
            vectors = (
                view.embeddings
                if extractor == self.engine.settings.embedder
                else view.features[extractor]
            )
            pools[location] = Pool(
                keys=tuple(p.content_key for p in pooled),
                vectors=vectors,
                general=self._general_aesthetic(view),
                quality=np.asarray(quality_model.score(view), dtype=np.float32),
            )
        return pools

    def _general_aesthetic(self, view: FeatureView) -> NDArray[np.float32]:
        """Today's general aesthetic, p("strong"); neutral where it cannot run (no text),
        which ``_general_notes`` says."""
        model = self._aesthetic_model()
        if model is None:
            return np.full(len(view), 0.5, dtype=np.float32)
        return model.predict_proba(view)[:, model.labels.index("strong")].astype(np.float32)

    def _aesthetic_model(self) -> LabelModel | None:
        try:
            model = self.engine.models.get("aesthetic")
        except CapabilityError:
            return None
        if not isinstance(model, LabelModel) or "strong" not in model.labels:
            return None
        return model

    def _general_notes(self) -> tuple[str, ...]:
        """What General falls back on when measured against Yours: a constant compared with
        a learned taste says nothing about either (audit A01-7)."""
        notes = []
        if self._aesthetic_model() is None:
            notes.append(
                f"General's aesthetic could not run with the embedder "
                f"{self.engine.settings.embedder!r} (it needs one that reads text, such as "
                "clip), so General is a constant 0.5 here and Yours is measured against "
                "that, not against General."
            )
        if not isinstance(self.engine.models.get("quality"), ScoreModel):
            notes.append("The quality model is missing, so quality counts as 0.5 for every photo.")
        return tuple(notes)

    def evaluate_stories(
        self,
        variants: Sequence[tuple[float, float]] | None = None,
        *,
        only_holdout: bool = False,
    ) -> StoryReports:
        """Would variety rank the stories you liked above the ones you rejected? Stories
        rated whole in a session, and stories with a 👍/👎 on a photo of a story result, are
        each re-scored from the evidence saved with them and their photos' embeddings
        (``evaluation.measure_stories``), and reported apart: a verdict on a story and a
        thumb on one of its photos are different answers (audit A05-10)."""
        if variants is None:
            variants = list(itertools.product(STORY_TOO_ALIKE, STORY_VARIETY_WEIGHTS))
        current = self.engine.ratings.current()
        held = self._held(only_holdout)
        sessions = [
            s
            for s in self._stories_rated_in_sessions(current)
            if (s[0] in held.story_sessions) == only_holdout
        ]
        # Story results are not part of the hold-out: they are reported on everything else.
        results = [] if only_holdout else list(self._stories_rated_on_results(current))
        # By key, not through the catalog's photos: an uploaded anchor is pruned after a
        # week, but its embedding stays.
        vectors = self.engine.catalog.get_features(
            feature_key(self.engine.extractors[self.engine.settings.embedder]),
            sorted({k for _, keys, *_ in [*sessions, *results] for k in keys}),
        )
        near_duplicate = self.engine.task("triptych").parse_options({}).max_similarity
        return StoryReports(
            sessions=self._measure_stories(sessions, vectors, variants, near_duplicate),
            results=self._measure_stories(results, vectors, variants, near_duplicate),
        )

    @staticmethod
    def _measure_stories(
        rated: Sequence[_Rated],
        vectors: Mapping[str, NDArray[np.float32]],
        variants: Sequence[tuple[float, float]],
        near_duplicate: float,
    ) -> StoryReport:
        stories, missing = [], 0
        for batch, keys, evidence, marks, stated, arms in rated:
            if not all(k in vectors for k in keys) or not evidence:
                missing += 1  # a photo since removed, or a story saved without its evidence
                continue
            stories.append(
                RatedStory(
                    run_id=batch,
                    vectors=np.stack([vectors[k] for k in keys]),
                    marks=marks,
                    evidence=tuple(evidence),
                    stated=stated,
                    arms=arms,
                )
            )
        report = measure_stories(stories, variants, near_duplicate=near_duplicate)
        return dataclasses.replace(report, missing=missing)

    def _stories_rated_on_results(self, current: Current) -> Iterator[_Rated]:
        """Stories with a 👍/👎 standing on a photo of a story result, judged within the run
        it was given on."""
        marks: dict[str, dict[str, int]] = {}
        for answer in current.values():
            scope = answer.scope
            if answer.use == "result" and isinstance(scope, dict) and scope["task"] == "triptych":
                run = answer.shown[0].run if answer.shown else None
                if run is not None and isinstance(answer.value, int):
                    marks.setdefault(run, {})[answer.subject[0]] = answer.value
        for run_id in sorted(marks):
            try:
                groups = self.engine.runs.load(run_id).groups
            except (RunNotFoundError, RunUnreadableError):
                continue
            for group in groups:
                keys = tuple(p.content_key for p in group.picks)
                if any(k in marks[run_id] for k in keys):
                    rating = tuple(marks[run_id].get(k, 0) for k in keys)
                    yield run_id, keys, group.evidence, rating, None, ()

    def _stories_rated_in_sessions(self, current: Current) -> Iterator[_Rated]:
        """Stories rated whole in a session, judged within their session. A photo marked as
        not fitting is -1, every other 0: the marks are ``story_fit`` answers, never
        thumbs."""
        verdicts: dict[int, StoryVerdict] = {2: "liked", 1: "unsure", 0: "rejected"}
        for label_set in self.engine.labels.all():
            for story in label_set.stories:
                label = story_label(story, current)
                if label is None:
                    continue
                strangers = unfit(story, current)
                yield (
                    label_set.set_id,
                    story_keys(story),
                    story.evidence,
                    tuple(-1 if p.content_key in strangers else 0 for p in story.photos),
                    verdicts[label],
                    tuple((a.template, a.variety, a.way) for a in story.arms),
                )

    def evaluate_finished_stories(
        self,
        path: Path,
        templates: Sequence[str] = ("general", "yours"),
        variants: Sequence[Mapping[str, Any]] = ({},),
    ) -> list[FinishedReport]:
        """Would the story task tell the stories you finished yourself? ``path`` is a JSON
        file of ``{"stories": {name: {"paths": [left, middle, right]}}}``. Each story is
        scored as the task scores one, over the photos at its places, against every story
        made by swapping one of its photos for one of the task's candidates; and its photos
        are checked against the candidates (``TriptychTask.scorer``). One report per
        template and per set of story options (``variants``)."""
        task = self._story_task()
        roots = [c.root for c in self.engine.settings.collections]
        found, pools, missing = self._finished_pools(path)
        reports = []
        for wanted in templates:
            trace = Trace()
            template, models = self.engine.template(wanted, trace)
            for variant in variants:
                options = task.parse_options(dict(variant))
                scored = []
                for name, photos in found.items():
                    pool, view = pools[name]
                    ctx = TaskContext(
                        request=SelectionRequest(task="triptych", folders=tuple(roots[:1])),
                        photos=tuple(pool),
                        view=view,
                        models=models,
                        trace=trace,
                        limit=task.default_limit,
                        view_feature=self.engine.view_feature(),
                    )
                    scored.append(_finished_story(task, ctx, options, name, photos))
                label = "; ".join(f"{k}={v}" for k, v in variant.items()) or "defaults"
                reports.append(
                    FinishedReport(
                        template=template, variant=label, stories=scored, missing=missing
                    )
                )
        return finished_intervals(reports)

    def _story_task(self) -> TriptychTask:
        task = self.engine.task("triptych")
        if not isinstance(task, TriptychTask):
            raise CapabilityError("The 'triptych' task is not the built-in story task.")
        return task

    def _finished_pools(
        self, path: Path
    ) -> tuple[dict[str, list[Photo]], dict[str, tuple[list[Photo], FeatureView]], list[str]]:
        """The stories you finished (``path``, see ``evaluate_finished_stories``), each with
        its pool: every photo at its places. Stories with a photo the catalog lacks, or
        without features, are ``missing``."""
        stories = _read_finished(path)
        roots = [c.root for c in self.engine.settings.collections]
        catalogued = self.engine.catalog.photos_under(roots)
        everything = self.engine.unique_by_content(catalogued)
        by_path = {p.path: p for p in catalogued}
        by_place: dict[str, list[Photo]] = {}
        for photo in everything:
            by_place.setdefault(photo.location, []).append(photo)
        found: dict[str, list[Photo]] = {}
        missing = []
        for name, paths in stories.items():
            listed = [by_path.get(Path(p).expanduser()) for p in paths]
            if all(listed):
                found[name] = [p for p in listed if p is not None]
            else:
                missing.append(name)

        # A story's own photos are always in its pool: a picture filed at two places is
        # catalogued once, under its first path, which may be the other place.
        pools: dict[str, tuple[list[Photo], FeatureView]] = {}
        for name, photos in list(found.items()):
            keys = {p.content_key for p in photos}
            places = sorted({p.location for p in photos})
            others = [p for place in places for p in by_place[place] if p.content_key not in keys]
            pool, view = self.engine.feature_view([*photos, *others], Trace())
            if keys <= {p.content_key for p in pool}:
                pools[name] = (pool, view)
            else:  # a photo without features (it could not be read)
                del found[name]
                missing.append(name)
        return found, pools, missing

    def learn_stories(
        self,
        finished: Path | None = None,
        *,
        template: str = "general",
        l2: float = STORY_L2,
        swaps: bool = False,
        variants: Sequence[Mapping[str, Any]] = (),
    ) -> LearnedStories:
        """Would a scorer learned from your answers tell stories that work better than
        today's way? Every story rated works or doesn't in a session (not the hold-out's)
        and every story you ``finished`` (the file ``evaluate_finished_stories`` reads) is
        given every signal the story task reads of it over its own pool, ranked by
        ``template``; ``evaluation.learn_stories`` learns from them and judges each
        session (audit A10-14). With ``swaps``, each finished story also makes
        ``STORY_SWAPS_PER_SIDE`` stories at each side by swapping that photo for another
        candidate, learned from as stories that don't work (``LEARNED_SWAPS``). Each of
        ``variants`` (story options) scores the session stories today's way with those
        options, judged beside today."""
        if swaps and finished is None:
            raise InvalidRequestError(
                "Swapped stories are made from your finished stories: give their file, as "
                "calico eval finished-stories reads it."
            )
        task = self._story_task()
        labelled = {"; ".join(f"{k}={v}" for k, v in o.items()): o for o in variants if o}
        seen_by = self.engine.view_feature()  # as a search sees views, not the main embedding
        current = self.engine.ratings.current()
        held = self._held(False)
        trace = Trace()
        _, models = self.engine.template(template, trace)
        stories: list[SignalledStory] = []
        names: dict[str, str] = {}
        missing = unsure = 0
        for label_set in self.engine.labels.all():
            if label_set.kind != "stories" or label_set.set_id in held.story_sessions:
                continue
            labels = [story_label(s, current) for s in label_set.stories]
            unsure += sum(label == 1 for label in labels)
            judged = [
                (s, lab) for s, lab in zip(label_set.stories, labels, strict=True) if lab in {0, 2}
            ]
            if not judged:
                continue
            photos, view = self.engine.feature_view(
                self.engine.gather(list(label_set.folders), trace), trace
            )
            told = self._story_signals(
                task,
                photos,
                view,
                models,
                trace,
                stories=[story_keys(s) for s, _ in judged],
                variants=labelled,
                view_feature=seen_by,
            )
            names[label_set.set_id] = label_set.text
            places = frozenset(p.location for p in photos)
            for (_, label), signalled in zip(judged, told, strict=True):
                if signalled is None:
                    missing += 1
                    continue
                signals, today, scored = signalled
                stories.append(
                    SignalledStory(
                        session=label_set.set_id,
                        places=places,
                        signals=signals,
                        works=label == 2,
                        today=today,
                        variants=scored,
                    )
                )
        if finished is not None:
            found, pools, gone = self._finished_pools(finished)
            missing += len(gone)
            for name, story_photos in found.items():
                pool, view = pools[name]
                keys = [p.content_key for p in story_photos]
                _, roles = self._story_scorers(
                    task, pool, view, models, trace, include=keys, view_feature=seen_by
                )
                rows = [roles.row(k) for k in keys]
                places = frozenset(p.location for p in story_photos)
                stories.append(SignalledStory(None, places, roles.signals(*rows), works=True))
                stories.extend(
                    SignalledStory(None, places, roles.signals(*triple), works=False, swapped=True)
                    for triple in (_swapped(roles, rows, name) if swaps else ())
                )
        report = learn_stories(stories, names=names, l2=l2)
        return dataclasses.replace(
            report, missing=missing, unsure=unsure, notes=tuple(dict.fromkeys(trace.notes))
        )

    @staticmethod
    def _story_signals(
        task: TriptychTask,
        photos: Sequence[Photo],
        view: FeatureView,
        models: ModelProvider,
        trace: Trace,
        *,
        stories: Sequence[Sequence[str]],
        variants: Mapping[str, Mapping[str, Any]] | None = None,
        view_feature: str,
    ) -> list[tuple[dict[str, float], float, dict[str, float]] | None]:
        """Each story's signals (``StoryScorer.signals``, told by roles so that every role is
        read), its score told today's way, and today's way with each of ``variants``
        (story options, by label), over ``photos``; None for a story with a photo not among
        them."""
        if not photos:
            return [None for _ in stories]
        include = {k for keys in stories for k in keys}
        today, roles = Sessions._story_scorers(
            task, photos, view, models, trace, include=include, view_feature=view_feature
        )
        ctx = _story_context(task, photos, view, models, trace, view_feature=view_feature)
        others = {
            label: task.scorer(ctx, task.parse_options(dict(options)), include=include)
            for label, options in (variants or {}).items()
        }
        found: list[tuple[dict[str, float], float, dict[str, float]] | None] = []
        for keys in stories:
            if not all(k in today.keys and k in roles.keys for k in keys):
                found.append(None)
                continue
            signals = roles.signals(*[roles.row(k) for k in keys])
            scored = {
                label: other.score(*[other.row(k) for k in keys])
                for label, other in others.items()
                if all(k in other.keys for k in keys)
            }
            found.append((signals, today.score(*[today.row(k) for k in keys]), scored))
        return found

    @staticmethod
    def _story_scorers(
        task: TriptychTask,
        photos: Sequence[Photo],
        view: FeatureView,
        models: ModelProvider,
        trace: Trace,
        *,
        include: Collection[str],
        view_feature: str,
    ) -> tuple[StoryScorer, StoryScorer]:
        """Stories over ``photos`` (and the ``include``d keys) scored today's way, and told
        by roles, which reads every signal."""
        ctx = _story_context(task, photos, view, models, trace, view_feature=view_feature)
        return (
            task.scorer(ctx, task.parse_options({}), include=include),
            task.scorer(ctx, task.parse_options(dict(ROLE_STORIES)), include=include),
        )


def _story_context(
    task: TriptychTask,
    photos: Sequence[Photo],
    view: FeatureView,
    models: ModelProvider,
    trace: Trace,
    *,
    view_feature: str,
) -> TaskContext:
    """A story search's context over ``photos``, for scoring stories made elsewhere.
    ``view_feature`` is the engine's (``Engine.view_feature``): without it, what tells one
    view from another (distinct, fresh) would read the main embedding instead."""
    return TaskContext(
        request=SelectionRequest(task="triptych", folders=(photos[0].path.parent,)),
        photos=tuple(photos),
        view=view,
        models=models,
        trace=trace,
        limit=task.default_limit,
        view_feature=view_feature,
    )


def _swapped(scorer: StoryScorer, rows: Sequence[int], name: str) -> list[tuple[int, int, int]]:
    """A finished story with one photo swapped, ``STORY_SWAPS_PER_SIDE`` at each side: drawn
    from the candidates the task could tell it with (seeded by the story's name)."""
    seed = int.from_bytes(hashlib.blake2b(f"swaps\n{name}".encode(), digest_size=8).digest())
    rng = np.random.default_rng(seed)
    made = []
    for side in range(3):
        options = []
        for r in np.flatnonzero(scorer.shortlisted):
            if int(r) in rows:
                continue
            triple = list(rows)
            triple[side] = int(r)
            if scorer.allowed(*triple):
                options.append((triple[0], triple[1], triple[2]))
        count = min(STORY_SWAPS_PER_SIDE, len(options))
        made += [options[i] for i in sorted(rng.choice(len(options), count, replace=False))]
    return made


def _finished_story(
    task: TriptychTask,
    ctx: TaskContext,
    options: TriptychOptions,
    name: str,
    photos: Sequence[Photo],
) -> FinishedStory:
    """One finished story scored by the task, against putting each of the task's candidates
    in each of its photos' places (stories the task could not tell are not counted)."""
    keys = [p.content_key for p in photos]
    scorer = task.scorer(ctx, options, include=keys)
    rows = [scorer.row(k) for k in keys]
    swaps = []
    for side in range(3):
        scores = []
        for r in np.flatnonzero(scorer.shortlisted):
            if r in rows:
                continue
            triple = list(rows)
            triple[side] = int(r)
            if scorer.allowed(*triple):
                scores.append(scorer.score(*triple))
        swaps.append(tuple(scores))
    return FinishedStory(
        name=name,
        score=scorer.score(*rows),
        swaps=tuple(swaps),
        shortlisted=tuple(bool(scorer.shortlisted[r]) for r in rows),
        anchor_tried=bool(scorer.anchored[rows[1]]),
    )


def _read_finished(path: Path) -> dict[str, list[str]]:
    """Finished stories by name: each the paths of its left, middle and right photo."""
    how = (
        'Expected a JSON file like {"stories": {"A": {"paths": [left, middle, right]}}}, '
        "with the paths of the photos as catalogued."
    )
    try:
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        stories = {str(name): list(s["paths"]) for name, s in data["stories"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise InvalidRequestError(
            f"Cannot read finished stories from {path}: {exc}. {how}"
        ) from exc
    for name, paths in stories.items():
        if len(paths) != 3:
            raise InvalidRequestError(f"Story {name!r} lists {len(paths)} photos, not three. {how}")
    return stories


@dataclass(frozen=True)
class TasteTraining:
    """What ``Sessions.train_taste`` measured, and whether it saved the taste."""

    report: TasteReport
    saved: bool
    calibration: Calibration | None = None
    """How often you liked what it gave each probability, at places it never learned
    (``models.calibration``); None when there was too little to say."""


_DECODE_BATCH = 32
"""Photos decoded at once when a candidate measure is computed for rated photos."""


def _drawn_items(
    sample: Sequence[Drawn], photos: Sequence[Photo], folders: Sequence[Path], what: str
) -> tuple[LabelItem, ...]:
    """The session's items, each with how it was drawn. Refused when nothing is left: a
    session of photos already rated would only ask the same questions again."""
    if not sample:
        names = ", ".join(f.name for f in folders)
        raise InvalidRequestError(
            f"Every photo in {names} (one frame per burst) has already been rated {what}. "
            "Choose other folders to rate more."
        )
    by_key = {p.content_key: p for p in photos}
    return tuple(
        LabelItem(
            content_key=d.key, path=by_key[d.key].path, stratum=d.stratum, inclusion=d.inclusion
        )
        for d in sample
    )


def _strength_note(report: TasteReport, plain: bool) -> str:
    """Whether strength was scored as Strongest ranks it, over each place's whole pool, or
    among the rated photos alone (rule 5: say what the measurement did instead)."""
    places = len(report.rows)
    if report.scores is not None:
        return (
            "Strength is scored among the photos the file scores at each location, not over "
            "its whole pool: the file's numbers exist for those photos alone, and General and "
            "Yours are scored the same way. On the 10 November 2026 copy, General's strength "
            "read this way was about 0.03 higher among the photos a taste chose than as "
            "Strongest ranks it (decision 0011), so this is the harder comparison for the scores."
        )
    if report.pooled and report.pooled == places:
        return "Strength is scored over each location's whole pool, as Strongest ranks it."
    if report.pooled:
        return (
            f"Strength is scored over the whole pool at {report.pooled} of {places} locations, "
            "as Strongest ranks it, and among the rated photos alone at the rest."
        )
    if plain:
        return (
            "Strength is scored among the rated photos alone: the quality or aesthetic model "
            "could not run over each location's whole pool."
        )
    return (
        "Strength is scored among the rated photos alone: a candidate or a fit is read for "
        "those photos, and is compared with today's the same way."
    )


def _scores_given(path: Path | None, field: str, *, alongside: bool) -> Scores | None:
    """The scores in ``path``, if one was given; a file of scores is the one candidate measured
    at a time, so it is refused beside another (``alongside``)."""
    if path is None:
        return None
    if alongside:
        raise InvalidRequestError(
            "Measure one candidate at a time: a file of --scores, or --technical, "
            "--aesthetic, --prompts, --general-embedder, --yours-embedder or --fit."
        )
    return read_scores(path, field)


def _require_rated(given: Scores | None, rated: Mapping[str, bool], path: Path | None) -> None:
    """A file none of whose keys is a rated photo is the wrong file (or the wrong kind of key):
    say so, not that nothing could be judged."""
    if given is not None and not given.values.keys() & rated.keys():
        raise InvalidRequestError(
            f"None of the {len(given.values)} photos in {path.name if path else 'the file'} is "
            "one you rated for your taste. A photo's key in the file is its content key, as the "
            "catalog stores it."
        )


def _with_scores(photos: Sequence[TastePhoto], given: Scores) -> list[TastePhoto]:
    """Each photo with its number from the file; the photos the file does not score keep none,
    which leaves them to teach and not to be judged."""
    return [dataclasses.replace(p, candidate_score=given.values.get(p.content_key)) for p in photos]


def _scores_note(given: Scores, rated: Mapping[str, bool], photos: Sequence[TastePhoto]) -> str:
    """What the file covered (rule 5): how many rated photos have a score, which are the ones
    judged, and how many of its photos are not ones you rated."""
    covered = sum(p.candidate_score is not None for p in photos)
    strangers = len(given.values.keys() - rated.keys())
    return (
        f"The scores file {given.name} scores {covered} of {len(photos)} rated photos: those "
        "are the ones judged, at each location where they hold a like and a dislike (a photo "
        "thumbed on a result too, read with the photos a taste chose); every rating still "
        "teaches Yours. "
        + (
            f"{strangers} {'photo' if strangers == 1 else 'photos'} in the file "
            f"{'is' if strangers == 1 else 'are'} not "
            f"{'one' if strangers == 1 else 'ones'} you rated."
            if strangers
            else "Every photo in the file is one you rated."
        )
    )


def _same_frames(keys: Sequence[str], vectors: NDArray[np.float32]) -> dict[str, str]:
    """For each photo that is a copy of another (``selection.copies``), the key of the
    first in key order: the same whatever order the catalog returned them in."""
    order = sorted(range(len(keys)), key=keys.__getitem__)
    first = copy_of(vectors[order]) if order else np.zeros(0, dtype=np.intp)
    return {keys[order[i]]: keys[order[int(j)]] for i, j in enumerate(first.tolist()) if j != i}


def _marker_drawn(run: SelectionResult | None) -> int | None:
    """How many picks of Yours's list in a Compare session come before the line it drew
    where the list stops being good: read from its saved run, None where it drew none
    (no probabilities then, or an older run this version cannot read)."""
    groups = run.groups if run is not None else ()
    marker = groups[0].marker if groups else None
    return None if marker is None else marker.before


BURSTS_STAGE = "collapse bursts"
"""The stage of a Strongest run that says how many bursts its pool held (``tasks/best.py``)."""


def _bursts_held(run: SelectionResult | None) -> int | None:
    """How many bursts the pool a Compare session's Yours run searched held, before one pick
    per view: the size the roadmap asks 100 of a place read for the bar. None where the run
    cannot be read or has no such stage."""
    if run is None:
        return None
    return next((s.count_out for s in run.stages if s.name == BURSTS_STAGE), None)


def _seed(kind: str, folders: Sequence[Path]) -> int:
    """A session's sample depends only on its kind and folders: the same folders, the same
    sample."""
    text = "\n".join([kind, *sorted(str(f) for f in folders)])
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest())


def _count(n: int, noun: str) -> str:
    return f"{n} {noun if n == 1 else noun + 's'}"
