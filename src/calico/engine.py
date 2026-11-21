"""The one selection pipeline. Every interface calls ``Engine.select``.

``select`` is the only place that touches the catalog, extractors and models on
behalf of a request; tasks receive plain arrays. The steps, and why each exists,
are in ``docs/architecture.md`` ("The request path"). Its public steps
(``feature_view``, ``template``, ``patch_scores``...) are what ``calico.sessions``
builds labelling sessions and evaluations from; this module never imports that one.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np

from calico import __version__
from calico.catalog.indexer import Indexer, thumbnail_path
from calico.catalog.store import Catalog
from calico.criteria import (
    EXAMPLE_WEIGHT,
    CriteriaScorer,
    CriteriaScores,
    blend_examples,
    criteria_scorer,
    example_similarity,
    scorer_key,
)
from calico.domain import SelectionRequest, SelectionResult, StageStat
from calico.errors import (
    FolderNotFoundError,
    ImageDecodeError,
    InvalidRequestError,
    TasteUnreadableError,
)
from calico.features import build_extractors, describe_embedder_of, embedder_of
from calico.features.base import Embedder, PatchExtractor, feature_key
from calico.imaging import RAW_HINT, as_thumbnail, decode
from calico.labels import (
    LabelStore,
)
from calico.models.registry import Models, WithTaste
from calico.models.taste import TasteModel
from calico.patch_store import PatchStore, sweep
from calico.paths import missing_folder, normalise
from calico.plugins import TASKS_GROUP, Registry
from calico.ratings_migration import open_journal
from calico.ratings_store import RatingsJournal
from calico.reviews import Ratings
from calico.runs import RunStore
from calico.selection.clear_matches import mark_clear_matches, narrow_to_clear_matches
from calico.selection.context import FeatureView, TaskContext, Trace
from calico.selection.patch_match import PatchMeasure, pair_share, patch_shares
from calico.settings import Settings, load_settings
from calico.tasks.base import ReferenceUse, Task
from calico.tracks import Track, load_tracks, track_for
from calico.trained import TrainedModels

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    from numpy.typing import NDArray
    from PIL import Image

    from calico.catalog.indexer import ProgressFn
    from calico.domain import Photo
    from calico.features.base import ImageExtractor
    from calico.models.base import ModelProvider
    from calico.models.taste import TasteWeights

logger = logging.getLogger(__name__)

NAMED_FAILURES = 3
"""Unreadable files named in a run's note; ``calico index`` lists them all."""
PATCH_BATCH = 16
"""Photos whose patches are held at once: each is ~1.5 MB of DINOv2 patches."""
PAIR_SAVE = 2000
"""Pairs compared between saves to the catalog, so a long run that is stopped keeps what
it compared (about 8 s of work)."""
PATCH_MEMORY = 16
"""Reference sets whose patch shares the engine remembers. Comparing a site's shortlist
takes about a minute (600 photos at ~0.12 s), so labelling right after a search, or
re-running it with other options, reuses what the search compared."""

type TaskRegistry = Registry[type[Task[Any]]]


class Engine:
    def __init__(
        self,
        *,
        settings: Settings,
        catalog: Catalog,
        indexer: Indexer,
        extractors: Mapping[str, ImageExtractor],
        models: ModelProvider,
        tasks: TaskRegistry,
        runs: RunStore,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.indexer = indexer
        self.extractors = dict(extractors)
        self.embedder: Embedder = embedder_of(self.extractors, settings)
        self.describer: Embedder = describe_embedder_of(self.extractors, settings)
        self.criteria: CriteriaScorer = criteria_scorer(
            settings.criteria_scorer, self.describer, settings
        )
        self.models = models
        self.task_registry = tasks
        self.runs = runs
        self.labels = LabelStore(settings.labels_dir)
        self.journal = RatingsJournal(settings.ratings_dir)
        self.ratings = Ratings(self.labels, self.journal, runs)
        self.trained = TrainedModels(settings.models_dir)
        self.clock = clock
        self.tracks: dict[str, Track] = load_tracks()
        self._shares: OrderedDict[tuple[str, tuple[str, ...]], dict[str, float]] = OrderedDict()

    @classmethod
    def open(cls, settings: Settings | None = None) -> Engine:
        """The real thing: settings from disk, entry-point plugins, SQLite catalog."""
        settings = settings or load_settings()
        settings.check_home_apart()  # before the catalog is created, inside the archive
        sweep(settings.home)  # scratch space a killed Calico left: GBs, else never freed
        open_journal(settings)  # the old ratings are in the journal before any is read
        catalog = Catalog(settings.catalog_path)
        extractors = build_extractors(settings)
        embedder = embedder_of(extractors, settings)
        return cls(
            settings=settings,
            catalog=catalog,
            indexer=Indexer(settings, catalog, extractors),
            extractors=extractors,
            models=Models.from_settings(settings, embedder),
            tasks=Registry.from_entry_points(TASKS_GROUP, "task"),
            runs=RunStore(settings.runs_dir),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.catalog.close()

    # -- tasks -----------------------------------------------------------------

    def task(self, name: str) -> Task[Any]:
        return self.task_registry.get(name)()

    def tasks(self) -> list[Task[Any]]:
        return [self.task(name) for name in self.task_registry.names()]

    def track(self, task: str, taste: Literal["general", "yours"]) -> Track | None:
        """The track a search runs on, with the options it ships, or None for a task
        no track covers."""
        name = track_for(task, taste)
        return None if name is None else self.tracks[name]

    # -- selection -------------------------------------------------------------

    def select(
        self,
        request: SelectionRequest,
        progress: ProgressFn | None = None,
        *,
        taste: TasteWeights | None = None,
    ) -> SelectionResult:
        """Run a request. ``taste``: the weights Yours runs with for this search alone, in place
        of the saved ones (a taste fitted without a place, ``Sessions.cross_fitted_taste``);
        never saved, and said in the run's notes."""
        task = self.task(request.task)
        limit = request.limit or task.default_limit
        describe_weight = (
            task.default_describe_weight
            if request.describe_weight is None
            else request.describe_weight
        )
        # Saved runs show the real numbers.
        request = request.model_copy(update={"limit": limit, "describe_weight": describe_weight})
        trace = Trace()
        template, models = self.template(
            request.template or self.settings.template, trace, taste=taste
        )
        request = request.model_copy(update={"template": template})
        track = self.track(task.name, template)
        options = task.parse_options({**(track.options if track else {}), **request.options})
        folders = self.validated_folders(request.folders)
        example_paths = self._validated_examples(request.examples)
        request = request.model_copy(update={"examples": example_paths})
        self._check_references(task, request, trace)

        started = time.perf_counter()
        report = self.indexer.index(folders, progress=progress)
        trace.stages.append(
            StageStat(
                name="index",
                count_in=report.scanned,
                count_out=report.scanned - len(report.failed),
                seconds=round(time.perf_counter() - started, 4),
            )
        )
        if report.failed:
            names = [f"{p.parent.name}/{p.name}" for p, _ in report.failed[:NAMED_FAILURES]]
            more = len(report.failed) - len(names)
            trace.note(
                f"{len(report.failed)} files could not be read and were skipped: "
                + ", ".join(names)
                + (f" and {more} more" if more else "")
                + ". See why with: calico index FOLDER"
            )
        if report.raw_without_rendered:
            one = report.raw_without_rendered == 1
            trace.note(
                f"{report.raw_without_rendered} RAW file{'' if one else 's'} "
                f"{'has' if one else 'have'} no JPEG/HEIC alongside {'it' if one else 'them'} "
                f"and {'was' if one else 'were'} skipped. To read RAW files: {RAW_HINT}"
            )
        if report.links_skipped or report.linked_dirs:
            trace.note(
                f"{report.links_skipped} linked file(s) and {len(report.linked_dirs)} linked "
                "folder(s) (symlinks) were skipped: a link can point outside your "
                "collections. Add the real folder as a collection instead."
            )
        if report.unsupported:
            trace.note(f"Files Calico does not read were left out: {_kinds(report.unsupported)}.")

        example_keys, examples = self._examples(example_paths, folders, trace)
        photos, view = self._pool(folders, example_keys, trace)
        photos, view, criteria = self._describe(request, photos, view, examples, trace)

        ctx = TaskContext(
            request=request,
            photos=tuple(photos),
            view=view,
            models=models,
            trace=trace,
            limit=limit,
            criteria_scores=None if criteria is None else criteria.scores,
            view_feature=self.view_feature(),
            # A reference the task ignores is not read: unreadable, it would fail the run.
            **self.references(
                () if task.reference_use is ReferenceUse.UNUSED else request.references
            ),
        )
        if photos:
            ctx = self.compare_parts(task, options, ctx, progress)
        with trace.stage(f"task:{task.name}", len(photos)) as stage:
            groups = mark_clear_matches(task.run(ctx, options) if photos else [], photos, criteria)
            stage.count_out = sum(len(g.picks) for g in groups)

        now = self.clock()
        result = SelectionResult(
            run_id=self.runs.new_run_id(now),
            created_at=now,
            request=request,
            groups=tuple(groups),
            example_keys=example_keys,
            notes=tuple(trace.notes),
            stages=tuple(trace.stages),
            versions={
                **self._versions(models),
                **({f"track:{track.name}": track.version} if track else {}),
            },
        )
        self.runs.save(result, ctx.photos, trace.columns)
        logger.info("run %s: %d groups from %d photos", result.run_id, len(groups), len(photos))
        return result

    # -- steps -------------------------------------------------------------------

    def _pool(
        self, folders: Sequence[Path], example_keys: Collection[str], trace: Trace
    ) -> tuple[list[Photo], FeatureView]:
        """The photos a search chooses from, with their features: the folders' photos less
        the examples, which are part of the description, and a note if none is left."""
        with trace.stage("gather", 0) as stage:
            photos = self.gather(folders, trace)
            shown = [p for p in photos if p.content_key not in example_keys]
            if len(shown) < len(photos):
                trace.note(
                    "Your example photos were left out of the results: they are part of the "
                    "description."
                )
            all_examples = bool(photos) and not shown
            stage.count_out = len(shown)
        with trace.stage("features", len(shown)) as stage:
            photos, view = self.feature_view(shown, trace)
            stage.count_out = len(photos)
        if all_examples:
            trace.note(
                "All the photos in these folders are your examples, so none is left to choose "
                "from. Choose a wider folder, or fewer examples."
            )
        elif not photos:
            trace.note("No readable photos were found in the chosen folders.")
        return photos, view

    def validated_folders(self, folders: Sequence[Path]) -> list[Path]:
        resolved = []
        for folder in folders:
            path = normalise(Path(folder))
            if not path.is_dir():
                raise FolderNotFoundError(missing_folder(path))
            self.settings.collection_for(path)  # raises with a how-to-fix message
            resolved.append(path)
        return resolved

    def _describe(
        self,
        request: SelectionRequest,
        photos: list[Photo],
        view: FeatureView,
        examples: FeatureView | None,
        trace: Trace,
    ) -> tuple[list[Photo], FeatureView, CriteriaScores | None]:
        """ "Describe it" read into scores: the text, plus the examples; then, if asked, the
        pool narrowed to clear matches. ``request.describe_weight`` is already resolved."""
        criteria = self.criteria.score(request.criteria, photos, view, trace)
        if examples is not None:
            similarity = example_similarity(self.criteria.embedder, view, examples)
            criteria = blend_examples(criteria, similarity, EXAMPLE_WEIGHT)
        if criteria is not None and request.describe_weight == 0:
            trace.note(
                "The description counts for nothing (its weight is 0%), so it does not change "
                "the order; it only marks clear matches."
            )
        if request.only_clear_matches:
            with trace.stage("clear matches", len(photos)) as stage:
                photos, view, criteria = narrow_to_clear_matches(
                    request, photos, view, criteria, trace
                )
                stage.count_out = len(photos)
        else:
            photos, view, criteria = narrow_to_clear_matches(request, photos, view, criteria, trace)
        if examples is not None:
            trace.record(
                "example_similarity", example_similarity(self.criteria.embedder, view, examples)
            )
        return photos, view, criteria

    def _validated_examples(self, paths: Sequence[Path]) -> tuple[Path, ...]:
        """Examples are photos in your collections: chosen from results, not brought in."""
        resolved = tuple(normalise(Path(p)) for p in paths)
        for path in resolved:
            self.settings.collection_for(path)  # raises with a how-to-fix message
        return resolved

    def _examples(
        self, paths: Sequence[Path], folders: Sequence[Path], trace: Trace
    ) -> tuple[tuple[str, ...], FeatureView | None]:
        """The examples' content keys and features, noting any from outside the folders."""
        if not paths:
            return (), None
        for path in paths:
            if not any(path.is_relative_to(folder) for folder in folders):
                trace.note(
                    f"The example “{path.name}” is from outside the chosen folders. Examples "
                    "from another place bring its look along, not only its light, so they may "
                    "favour photos like that place."
                )
        keys, matrices = self.indexer.features_for_files(paths)
        return tuple(keys), self.as_view(matrices)

    @staticmethod
    def _check_references(task: Task[Any], request: SelectionRequest, trace: Trace) -> None:
        if task.reference_use is ReferenceUse.REQUIRED and not request.references:
            raise InvalidRequestError(f"{task.title!r} needs a reference photo.")
        if task.reference_use is ReferenceUse.UNUSED and request.references:
            trace.note(f"{task.title!r} does not use reference photos; they were ignored.")

    def gather(self, folders: Sequence[Path], trace: Trace) -> list[Photo]:
        """The photos a search or sample draws from: one path per content, and no
        screenshots where a collection leaves them out (``exclude_screenshots``)."""
        photos = self.unique_by_content(self.catalog.photos_under(folders))
        leave_out = {c.name for c in self.settings.collections if c.exclude_screenshots}
        kept = [p for p in photos if not (p.collection in leave_out and _screenshot(p))]
        if dropped := len(photos) - len(kept):
            trace.note(
                f"{dropped} screenshot{'' if dropped == 1 else 's'} (PNG with no camera "
                "recorded) left out, as the collection's exclude_screenshots says."
            )
        return kept

    @staticmethod
    def unique_by_content(photos: Sequence[Photo]) -> list[Photo]:
        """Exact copies share features and would crowd results; keep the first path."""
        seen: set[str] = set()
        unique = []
        for photo in photos:
            if photo.content_key not in seen:
                seen.add(photo.content_key)
                unique.append(photo)
        return unique

    def feature_view(self, photos: list[Photo], trace: Trace) -> tuple[list[Photo], FeatureView]:
        """Aligned feature matrices for ``photos``.

        Photos missing a feature (failed decode) or with a non-finite value are
        dropped here, once, so no task or model ever has to handle NaN.
        """
        keys = [p.content_key for p in photos]
        found = {
            name: self.catalog.get_features(feature_key(ex), keys)
            for name, ex in self.extractors.items()
        }
        complete = [p for p in photos if all(p.content_key in f for f in found.values())]
        valid = [
            p
            for p in complete
            if all(np.isfinite(vectors[p.content_key]).all() for vectors in found.values())
        ]
        if len(valid) < len(complete):
            trace.note(
                f"{len(complete) - len(valid)} photos had invalid (NaN/inf) features and were "
                "skipped. Re-index them, or report it: it means an extractor has a bug."
            )
        matrices = {
            name: _stack([vectors[p.content_key] for p in valid], self.extractors[name].dim)
            for name, vectors in found.items()
        }
        return valid, self.as_view(matrices)

    def as_view(self, matrices: Mapping[str, NDArray[np.float32]]) -> FeatureView:
        name = self.settings.embedder
        return FeatureView(
            embeddings=matrices[name],
            features={k: v for k, v in matrices.items() if k != name},
            columns={
                k: ex.columns
                for k, ex in self.extractors.items()
                if ex.columns is not None and k != name
            },
        )

    def references(self, paths: Sequence[Path]) -> dict[str, Any]:
        if not paths:
            return {}
        resolved = tuple(normalise(Path(p)) for p in paths)
        keys, matrices = self.indexer.features_for_files(resolved)
        return {
            "references": self.as_view(matrices),
            "reference_paths": resolved,
            "reference_keys": tuple(keys),
        }

    def view_feature(self) -> str:
        name = self.settings.view_embedder
        return "" if name == self.settings.embedder else name

    def copy_vectors(self, view: FeatureView) -> NDArray[np.float32]:
        """What tells one frame saved twice from frames of a burst: the view embedder's
        vectors, which ``COPY_SIMILARITY`` was measured on (the main embedding's when it is
        the view embedder, or was not computed)."""
        name = self.view_feature()
        return view.features.get(name, view.embeddings)

    def patch_extractor(self, trace: Trace) -> PatchExtractor | None:
        extractor = self.extractors.get(self.settings.view_embedder)
        if isinstance(extractor, PatchExtractor):
            return extractor
        trace.note(
            f"The view embedder {self.settings.view_embedder!r} cannot compare photos part by "
            'part, so same views are matched as whole photos. Set view_embedder = "dinov2" '
            "in calico.toml."
        )
        return None

    def patch_scores(
        self,
        ctx: TaskContext,
        shortlist: NDArray[np.intp],
        progress: ProgressFn | None = None,
    ) -> NDArray[np.float32] | None:
        """``ctx.patch_scores``: the shortlisted photos compared with the references part by
        part, from thumbnails (the drive is only read for a photo without one). Shares
        already compared for these references are reused (``PATCH_MEMORY``)."""
        extractor = self.patch_extractor(ctx.trace)
        if extractor is None:
            return None
        known = self._remembered_shares(extractor, ctx.reference_keys)
        todo = [int(i) for i in shortlist if ctx.photos[i].content_key not in known]
        before = len(shortlist) - len(todo)

        def compared(done: int) -> None:
            if progress:
                progress(before + done, len(shortlist), "comparing")

        scores = np.full(len(ctx.photos), np.nan, dtype=np.float32)
        with ctx.trace.stage("patch match", len(shortlist)) as stage:
            compared(0)
            if todo:
                refs = self.reference_images(ctx.reference_paths, ctx.reference_keys)
                found = self.patch_shares(extractor, [ctx.photos[i] for i in todo], refs, compared)
                for i, share in zip(todo, found, strict=True):
                    if np.isfinite(share):
                        known[ctx.photos[i].content_key] = float(share)
            for i in shortlist:
                scores[i] = known.get(ctx.photos[i].content_key, np.nan)
            stage.count_out = int(np.isfinite(scores).sum())
        unread = len(shortlist) - stage.count_out
        if unread:
            ctx.trace.note(
                f"{unread} photos could not be read to compare part by part, and count as no "
                "match. See why with: calico index FOLDER"
            )
        return scores

    def compare_parts(
        self,
        task: Task[Any],
        options: Any,
        ctx: TaskContext,
        progress: ProgressFn | None = None,
    ) -> TaskContext:
        """What the task asked to compare part by part: photos with the references
        (``patch_shortlist``), and photos with each other (``patch_pairs``). A step of
        ``select``, public so a session running a task in memory runs it as a search does."""
        shortlist = task.patch_shortlist(ctx, options)
        if shortlist is not None and ctx.references is not None:
            scores = self.patch_scores(ctx, shortlist, progress)
            ctx = dataclasses.replace(ctx, patch_scores=scores)
        pairs = task.patch_pairs(ctx, options)
        if pairs is not None:
            shares = self._pair_shares(ctx, pairs, progress)
            ctx = dataclasses.replace(ctx, compared_pairs=pairs, pair_shares=shares)
        return ctx

    def _pair_shares(
        self, ctx: TaskContext, pairs: NDArray[np.intp], progress: ProgressFn | None = None
    ) -> NDArray[np.float32] | None:
        """``ctx.pair_shares``: how much each pair of photos shares, part by part. Pairs
        compared before (by any run, with this view embedder) come from the catalog; the
        rest are compared now and remembered. ``None`` when the view embedder cannot
        compare patches (noted)."""
        extractor = self.patch_extractor(ctx.trace)
        if extractor is None:
            return None
        key = feature_key(extractor)
        photos = ctx.photos
        involved = sorted({int(i) for i in pairs.ravel()})
        known = self.catalog.pair_shares_among(key, [photos[i].content_key for i in involved])
        out = np.full(len(pairs), np.nan, dtype=np.float32)
        todo = []
        for n, (a, b) in enumerate(pairs):
            share = known.get(_pair_key(photos[int(a)], photos[int(b)]))
            if share is None:
                todo.append(n)
            else:
                out[n] = share
        with ctx.trace.stage("compare pairs", len(pairs)) as stage:
            if todo:
                self._compare_pairs(
                    extractor, key, photos, pairs, todo=todo, out=out, progress=progress
                )
            stage.count_out = int(np.isfinite(out).sum())
        unread = len(pairs) - stage.count_out
        if unread:
            ctx.trace.note(
                f"{unread} pairs of photos could not be compared part by part (a photo could "
                "not be read), and count as sharing nothing. See why with: calico index FOLDER"
            )
        return out

    def _compare_pairs(
        self,
        extractor: PatchExtractor,
        key: str,
        photos: Sequence[Photo],
        pairs: NDArray[np.intp],
        *,
        todo: list[int],
        out: NDArray[np.float32],
        progress: ProgressFn | None,
    ) -> None:
        """Compare ``pairs[todo]`` into ``out`` and the catalog. Each photo's patches are
        read once, into the run's scratch store (``calico.patch_store``), then every pair is
        matched."""
        need = sorted({int(i) for n in todo for i in pairs[n]})
        row = {i: k for k, i in enumerate(need)}
        with PatchStore(self.settings.home, len(need)) as store:
            for start in range(0, len(need), PATCH_BATCH):
                batch = need[start : start + PATCH_BATCH]
                images = [self._thumbnail(photos[i].content_key, photos[i].path) for i in batch]
                found = [(k, image) for k, image in enumerate(images) if image is not None]
                if found:
                    patches = extractor.extract_patches([image for _, image in found])
                    for (k, _), patch in zip(found, patches, strict=True):
                        store.put(start + k, patch)
                if progress:
                    progress(start + len(batch), len(need), "comparing")
            order = sorted(todo, key=lambda n: (row[int(pairs[n][0])], row[int(pairs[n][1])]))
            fresh: dict[tuple[str, str], float] = {}
            for done, n in enumerate(order, start=1):
                a, b = int(pairs[n][0]), int(pairs[n][1])
                first, second = store.get(row[a]), store.get(row[b])
                if first is not None and second is not None:
                    out[n] = pair_share(first, second)
                    fresh[_pair_key(photos[a], photos[b])] = float(out[n])
                if len(fresh) >= PAIR_SAVE or done == len(order):
                    self.catalog.put_pair_shares(key, fresh)
                    fresh = {}
                    if progress:
                        progress(done, len(order), "comparing")

    def _remembered_shares(
        self, extractor: PatchExtractor, reference_keys: Sequence[str]
    ) -> dict[str, float]:
        """The patch shares already compared with these references, by content key; filled
        in place. Only the ``PATCH_MEMORY`` most recently used reference sets are kept."""
        key = (feature_key(extractor), tuple(sorted(reference_keys)))
        known = self._shares.setdefault(key, {})
        self._shares.move_to_end(key)
        while len(self._shares) > PATCH_MEMORY:
            self._shares.popitem(last=False)
        return known

    def patch_shares(
        self,
        extractor: PatchExtractor,
        photos: Sequence[Photo],
        refs: Sequence[Image.Image],
        compared: Callable[[int], None] | None = None,
    ) -> NDArray[np.float32]:
        """Each photo's patch share with ``refs`` (NaN for a photo that cannot be read);
        ``compared(done)`` after each batch."""
        shares: NDArray[np.float32] = self.patch_measures(
            extractor, photos, refs, (patch_shares,), compared
        )[0]
        return shares

    def patch_measures(
        self,
        extractor: PatchExtractor,
        photos: Sequence[Photo],
        refs: Sequence[Image.Image],
        measures: Sequence[PatchMeasure],
        compared: Callable[[int], None] | None = None,
    ) -> NDArray[np.float32]:
        """Each photo measured against ``refs`` by each of ``measures``, from one reading of
        its patches: ``(len(measures), len(photos))``, NaN for a photo that cannot be read.
        ``compared(done)`` after each batch."""
        ref_patches = extractor.extract_patches(refs)
        out = np.full((len(measures), len(photos)), np.nan, dtype=np.float32)
        for start in range(0, len(photos), PATCH_BATCH):
            batch = photos[start : start + PATCH_BATCH]
            images = [self._thumbnail(p.content_key, p.path) for p in batch]
            readable = [(i, image) for i, image in enumerate(images) if image is not None]
            if readable:
                patches = extractor.extract_patches([image for _, image in readable])
                rows = [start + i for i, _ in readable]
                for m, measure in enumerate(measures):
                    out[m, rows] = measure(patches, ref_patches)
            if compared:
                compared(start + len(batch))
        return out

    def _thumbnail(self, content_key: str, path: Path) -> Image.Image | None:
        """The catalog's thumbnail, or one made the same way from the file."""
        from PIL import Image  # noqa: PLC0415 - PIL is only needed when comparing patches

        thumb = thumbnail_path(self.settings.thumbs_dir, content_key)
        try:
            if thumb.is_file():
                with Image.open(thumb) as image:
                    return image.convert("RGB")
            original, prepared = decode(path, self.settings.decode_max_side)
            original.close()
            return as_thumbnail(prepared, self.settings.thumbnail_side)
        except (OSError, ImageDecodeError):
            logger.warning("cannot read %s to compare patches", path)
            return None

    def reference_images(self, paths: Sequence[Path], keys: Sequence[str]) -> list[Image.Image]:
        images = [self._thumbnail(key, path) for path, key in zip(paths, keys, strict=True)]
        missing = [p.name for p, image in zip(paths, images, strict=True) if image is None]
        if missing:
            raise InvalidRequestError(f"Could not read the reference {', '.join(missing)}.")
        return [image for image in images if image is not None]

    def _versions(self, models: ModelProvider | None = None) -> dict[str, str]:
        versions = {"calico": __version__, "criteria": scorer_key(self.criteria)}
        versions.update({name: feature_key(ex) for name, ex in self.extractors.items()})
        versions.update({f"model:{k}": v for k, v in (models or self.models).versions().items()})
        return versions

    # -- templates -----------------------------------------------------------------

    def template(
        self, wanted: str, trace: Trace, taste: TasteWeights | None = None
    ) -> tuple[Literal["general", "yours"], ModelProvider]:
        """The template a request runs with, and its models. Yours falls back to General,
        saying why, until a taste is trained on the embedding these photos have. With
        ``taste``, Yours is that taste, not the saved one."""
        if wanted != "yours":
            if taste is not None:
                trace.note(
                    "A taste was given to this search, but it runs General's, which does not "
                    "read one."
                )
            return "general", self.models
        try:
            weights = taste if taste is not None else self.trained.load_taste()
        except TasteUnreadableError as exc:
            trace.note(f"{exc} General was used.")
            return "general", self.models
        if weights is None:
            trace.note(
                "The Yours template has not been trained yet, so General was used. Rate photos "
                "with Rate for your taste (the Label page), then run: calico train taste"
            )
            return "general", self.models
        now = self.taste_feature(weights.extractor)
        if now != weights.feature:
            trace.note(
                f"Your taste was learned from {weights.feature}, but photos now have {now}, so "
                "General was used. Retrain with: calico train taste"
            )
            return "general", self.models
        if weights.extractor == self.settings.embedder:
            # Learned on what is now the main embedding: its vectors are the view's
            # embeddings, no longer one of its other features.
            weights = dataclasses.replace(weights, extractor="")
        if taste is not None:
            trace.note(
                f"Yours was fitted for this search alone, from {len(taste.places)} "
                f"{'place' if len(taste.places) == 1 else 'places'}, not the saved taste "
                f"(version {taste.version})."
            )
        return "yours", WithTaste(self.models, TasteModel(weights))

    def taste_feature(self, extractor: str) -> str:
        """The exact version of the vectors a taste reads ("" = the main embedding)."""
        name = extractor or self.settings.embedder
        found = self.extractors.get(name)
        return feature_key(found) if found is not None else f"{name}@missing"


def _screenshot(photo: Photo) -> bool:
    """A PNG with no camera recorded: in a photo archive, a screenshot or a saved image.
    Phone screenshots come at screen sizes (1170x2532, 1179x2556, 1344x2992) and never
    carry a camera; a camera's own PNGs do."""
    return photo.path.suffix.lower() == ".png" and not photo.camera


def _stack(rows: Sequence[NDArray[np.float32]], dim: int) -> NDArray[np.float32]:
    if not rows:
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack(rows).astype(np.float32, copy=False)


def _pair_key(a: Photo, b: Photo) -> tuple[str, str]:
    """How a pair is remembered: by content, lower key first."""
    return (
        (a.content_key, b.content_key)
        if a.content_key <= b.content_key
        else (b.content_key, a.content_key)
    )


def _kinds(counts: Mapping[str, int], shown: int = 5) -> str:
    """'340 .mov, 12 .pef and 2 more kinds', most first."""
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    named = ", ".join(f"{n} {suffix}" for suffix, n in ranked[:shown])
    more = len(ranked) - shown
    return named + (f" and {more} more kinds" if more > 0 else "")
