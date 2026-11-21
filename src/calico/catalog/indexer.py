"""Bring the catalog up to date with the files on disk.

Each new or changed image is decoded **once**; every extractor runs on that
decode, and the thumbnail is cut from it. Unchanged files cost a ``stat``.
Per-file decode problems are recorded and reported but never abort a run: one
corrupt JPEG must not block indexing the other thousand.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from calico.catalog.locations import location_for
from calico.catalog.metadata import FileFacts, ImageFacts, file_facts, image_facts
from calico.catalog.scanner import nfc_path, scan
from calico.catalog.store import FACTS_VERSION, PhotoRecord
from calico.domain import Photo
from calico.errors import ExtractorError, ImageDecodeError
from calico.features.base import feature_key
from calico.imaging import decode, open_image, raw_support, thumbnail_jpeg
from calico.paths import normalise

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from PIL import Image

    from calico.catalog.store import Catalog
    from calico.features.base import ImageExtractor
    from calico.settings import Settings

logger = logging.getLogger(__name__)

type ProgressStep = Literal["reading", "comparing"]
"""What a long request is doing: reading photos into the catalog, or comparing them part
by part (same view: with a reference, about a minute for a large site; recurring views:
with each other, first reading each photo's details, then matching pairs)."""

ProgressFn = Callable[[int, int, ProgressStep], None]
"""Called as ``progress(done, total, step)``: after each file read, and as photos are
compared."""


@dataclass
class IndexReport:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: list[tuple[Path, str]] = field(default_factory=list)
    raw_without_rendered: int = 0
    links_skipped: int = 0
    linked_dirs: list[Path] = field(default_factory=list)
    """Symlinked folders, not entered: a link can point outside every collection."""
    unsupported: dict[str, int] = field(default_factory=dict)
    """Files not read, by extension: video, sidecars, formats Calico cannot decode."""
    unreadable_dirs: list[Path] = field(default_factory=list)
    seconds: float = 0.0


@dataclass(frozen=True)
class _Loaded:
    """One successfully decoded file, ready for extraction."""

    path: Path
    file: FileFacts
    image: ImageFacts
    prepared: Image.Image


def thumbnail_path(thumbs_dir: Path, content_key: str) -> Path:
    """Sharded by key prefix so no directory grows to hundreds of thousands of files."""
    return thumbs_dir / content_key[:2] / f"{content_key}.jpg"


def _load(path: Path, max_side: int) -> _Loaded:
    """Stat, hash and decode one file. Runs in a worker thread."""
    try:
        facts = file_facts(path)
    except OSError as exc:
        raise ImageDecodeError(f"{path}: {exc}") from exc
    original, prepared = decode(path, max_side)
    try:
        meta = image_facts(original)
    finally:
        original.close()
    return _Loaded(path, facts, meta, prepared)


def _read_facts(path: Path) -> ImageFacts | None:
    """A catalogued photo's EXIF facts, read again without decoding its pixels; ``None``
    (tried again next time) if the file cannot be opened. Runs in a worker thread."""
    try:
        with open_image(path) as image:
            return image_facts(image)
    except (ImageDecodeError, OSError) as exc:
        logger.warning("cannot re-read the EXIF of %s: %s", path, exc)
        return None


def _outermost(folders: Sequence[Path]) -> list[Path]:
    """Resolved folders with duplicates and folders nested inside another dropped.

    Indexing a folder covers its subtree, so a nested argument would only do the
    same work twice (and report it twice).
    """
    resolved = sorted({nfc_path(f.resolve()) for f in folders})
    return [f for f in resolved if not any(f != o and f.is_relative_to(o) for o in resolved)]


def _under_any(path: Path, folders: Sequence[Path]) -> bool:
    return any(path.is_relative_to(folder) for folder in folders)


def _batches[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class Indexer:
    def __init__(
        self,
        settings: Settings,
        catalog: Catalog,
        extractors: Mapping[str, ImageExtractor],
        *,
        workers: int = 4,
        batch_size: int = 32,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._extractors = dict(extractors)
        self._workers = workers
        self._batch_size = batch_size

    # -- public ---------------------------------------------------------

    def index(self, folders: Sequence[Path], progress: ProgressFn | None = None) -> IndexReport:
        """Index every folder; returns what changed. Unreadable files are reported, not raised."""
        self._settings.check_home_apart()
        started = time.perf_counter()
        report = IndexReport()
        for folder in _outermost(folders):
            self._index_folder(folder, report, progress)
        report.seconds = time.perf_counter() - started
        return report

    def features_for_files(
        self, paths: Sequence[Path]
    ) -> tuple[list[str], dict[str, NDArray[np.float32]]]:
        """Content keys and one feature matrix per extractor, for arbitrary files.

        Used for reference images, which need not be catalogued. Cached vectors
        are reused when the catalog already knows the content.
        """
        loaded = self._load_many(list(paths))
        keys = [item.file.content_key for item in loaded]
        self._extract_missing(loaded)
        matrices: dict[str, NDArray[np.float32]] = {}
        for name, extractor in self._extractors.items():
            have = self._catalog.get_features(feature_key(extractor), keys)
            rows = [have[key] for key in keys]
            matrices[name] = (
                np.stack(rows) if rows else np.zeros((0, extractor.dim), dtype=np.float32)
            )
        return keys, matrices

    # -- one folder -----------------------------------------------------

    def _index_folder(self, folder: Path, report: IndexReport, progress: ProgressFn | None) -> None:
        self._settings.collection_for(folder)  # raises with a how-to-fix message
        scanned = scan(
            folder,
            raw=raw_support() is not None,
            skip=(normalise(self._settings.home),),
            exclude_for=lambda d: self._settings.collection_for(d).exclude,
        )
        existing = self._catalog.records_under(folder)
        report.scanned += len(scanned.images)
        report.raw_without_rendered += len(scanned.raw_without_rendered)
        report.links_skipped += len(scanned.links)
        report.linked_dirs.extend(scanned.linked_dirs)
        for suffix, count in scanned.unsupported:
            report.unsupported[suffix] = report.unsupported.get(suffix, 0) + count

        unchanged = self._unchanged(scanned.images, existing)
        todo = [p for p in scanned.images if p not in unchanged]
        refreshed = self._refresh_derived(unchanged, existing)
        report.updated += refreshed
        report.unchanged += len(unchanged) - refreshed
        total = len(scanned.images)
        done = 0
        for _ in unchanged:
            done += 1
            if progress:
                progress(done, total, "reading")

        for batch in _batches(todo, self._batch_size):
            loaded = self._load_many(batch, report)
            committed = self._commit(loaded, existing, report)
            # A catalogued file that changed and can no longer be read: its row would keep
            # a content key and features that are no longer the file's.
            stale = [p for p in batch if p in existing and p not in committed]
            if stale:
                report.removed += self._catalog.remove(stale)
            for _ in batch:
                done += 1
                if progress:
                    progress(done, total, "reading")

        on_disk = set(scanned.images)
        for directory in scanned.unreadable:
            logger.warning("cannot read directory %s; keeping its catalogued photos", directory)
        report.unreadable_dirs.extend(scanned.unreadable)
        gone = [p for p in existing if p not in on_disk and not _under_any(p, scanned.unreadable)]
        report.removed += self._catalog.remove(gone)
        stale = [
            p
            for p in self._catalog.failure_paths_under(folder)
            if p not in on_disk and not _under_any(p, scanned.unreadable)
        ]
        self._catalog.clear_failures(stale)
        logger.info(
            "indexed %s: %d scanned, %d new/changed, %d unchanged, %d removed",
            folder,
            total,
            len(todo),
            len(unchanged),
            len(gone),
        )

    def _unchanged(self, images: Sequence[Path], existing: Mapping[Path, PhotoRecord]) -> set[Path]:
        """Files whose stat matches the catalog, whose features are all cached and whose
        thumbnail exists: ``thumbs/`` looks like a cache, so a deleted one is made again."""
        candidates: dict[Path, str] = {}
        for path in images:
            record = existing.get(path)
            if record is None:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if (stat.st_size, stat.st_mtime) == (record.size, record.mtime):
                candidates[path] = record.photo.content_key
        keys = sorted(set(candidates.values()))
        complete = set(keys)
        for extractor in self._extractors.values():
            complete &= self._catalog.has_features(feature_key(extractor), keys)
        thumbs = self._settings.thumbs_dir
        return {
            path
            for path, key in candidates.items()
            if key in complete and thumbnail_path(thumbs, key).is_file()
        }

    def _refresh_derived(self, unchanged: set[Path], existing: Mapping[Path, PhotoRecord]) -> int:
        """Re-derive collection and location for files that need no decode, and re-read
        the EXIF of those catalogued before ``FACTS_VERSION`` (no decode either).

        Settings can change between runs (location rules, a new nested
        collection); the stored values would otherwise stay stale forever.
        """
        stale = sorted(p for p in unchanged if existing[p].facts < FACTS_VERSION)
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            facts = dict(zip(stale, pool.map(_read_facts, stale), strict=True))
        fixed: list[PhotoRecord] = []
        for path in sorted(unchanged):
            record = existing[path]
            collection = self._settings.collection_for(path)
            location = location_for(collection, path, self._settings.location_aliases)
            photo, read = record.photo, facts.get(path)
            if read is not None:
                photo = replace(
                    photo,
                    captured_at=read.captured_at,
                    camera=read.camera,
                    utc_offset=read.utc_offset,
                )
            if read is not None or (photo.collection, photo.location) != (
                collection.name,
                location,
            ):
                updated = replace(photo, collection=collection.name, location=location)
                fixed.append(
                    PhotoRecord(
                        updated,
                        record.size,
                        record.mtime,
                        FACTS_VERSION if read is not None else record.facts,
                    )
                )
        if fixed:
            self._catalog.upsert(fixed)
        return len(fixed)

    def _load_many(self, paths: Sequence[Path], report: IndexReport | None = None) -> list[_Loaded]:
        """Decode in parallel. With a ``report``, failures are recorded; without, they raise."""
        max_side = self._settings.decode_max_side
        loaded: list[_Loaded] = []
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = [pool.submit(_load, path, max_side) for path in paths]
            for path, future in zip(paths, futures, strict=True):
                try:
                    loaded.append(future.result())
                except ImageDecodeError as exc:
                    if report is None:
                        raise
                    self._fail(path, str(exc), report)
        return loaded

    def _fail(self, path: Path, message: str, report: IndexReport, stage: str = "decode") -> None:
        logger.warning("cannot index %s: %s", path, message)
        self._catalog.record_failure(path, stage, message)
        report.failed.append((path, message))

    def _commit(
        self,
        loaded: Sequence[_Loaded],
        existing: Mapping[Path, PhotoRecord],
        report: IndexReport,
    ) -> set[Path]:
        """Store each photo under its own collection (a folder can hold a nested one), but
        not one an extractor could not read; the paths stored."""
        failed = self._extract_missing(loaded, report)
        records = []
        for item in loaded:
            if item.file.content_key in failed:
                continue
            self._write_thumbnail(item)
            collection = self._settings.collection_for(item.path)
            photo = Photo(
                path=item.path,
                content_key=item.file.content_key,
                collection=collection.name,
                location=location_for(collection, item.path, self._settings.location_aliases),
                captured_at=item.image.captured_at,
                width=item.image.width,
                height=item.image.height,
                camera=item.image.camera,
                utc_offset=item.image.utc_offset,
            )
            records.append(PhotoRecord(photo, item.file.size, item.file.mtime))
            if item.path in existing:
                report.updated += 1
            else:
                report.added += 1
        self._catalog.upsert(records)
        for record in records:
            self._catalog.clear_failure(record.photo.path)
        return {record.photo.path for record in records}

    def _extract_missing(
        self, loaded: Sequence[_Loaded], report: IndexReport | None = None
    ) -> set[str]:
        """Run each extractor only on content whose features are not cached yet. With a
        ``report``, a photo an extractor cannot read is recorded and left out (its content
        key is returned); without one, as for a reference photo, the error is raised."""
        by_key = {item.file.content_key: item for item in loaded}
        keys = list(by_key)
        failed: set[str] = set()
        for extractor in self._extractors.values():
            fkey = feature_key(extractor)
            have = self._catalog.has_features(fkey, keys)
            missing = [by_key[key] for key in keys if key not in have and key not in failed]
            if not missing:
                continue
            try:
                vectors = _vectors(extractor, missing)
            except Exception as exc:
                if report is None:
                    raise
                vectors = self._one_by_one(extractor, missing, exc, report, failed)
            self._catalog.put_features(fkey, vectors)
        return failed

    def _one_by_one(
        self,
        extractor: ImageExtractor,
        items: Sequence[_Loaded],
        batch_error: Exception,
        report: IndexReport,
        failed: set[str],
    ) -> dict[str, NDArray[np.float32]]:
        """After a batch failed: each photo alone, so one pathological photo costs only
        itself. An extractor that fails on all of them is broken (out of memory, missing
        weights, the wrong shape), and stops the run instead of failing every photo."""
        fkey = feature_key(extractor)
        vectors: dict[str, NDArray[np.float32]] = {}
        errors: list[tuple[_Loaded, Exception]] = []
        for item in items:
            try:
                vectors.update(_vectors(extractor, [item]))
            except Exception as exc:
                errors.append((item, exc))
        if len(items) > 1 and len(errors) == len(items):
            raise ExtractorError(
                f"The {fkey} extractor failed on every photo in a batch of {len(items)} "
                f"({type(batch_error).__name__}: {batch_error}), so indexing stopped rather "
                "than mark them all unreadable. Free memory or fix the model, then run "
                "calico index again."
            ) from batch_error
        for item, error in errors:
            failed.add(item.file.content_key)
            message = f"{fkey}: {type(error).__name__}: {error}"
            self._fail(item.path, message, report, f"extract:{fkey}")
        return vectors

    def _write_thumbnail(self, item: _Loaded) -> None:
        target = thumbnail_path(self._settings.thumbs_dir, item.file.content_key)
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_bytes(thumbnail_jpeg(item.prepared, self._settings.thumbnail_side))
            tmp.replace(target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


def _vectors(extractor: ImageExtractor, items: Sequence[_Loaded]) -> dict[str, NDArray[np.float32]]:
    """One feature vector per item, by content key; ``ExtractorError`` if the extractor
    does not return one row of its declared size per image."""
    rows = extractor.extract([item.prepared for item in items])
    if rows.shape != (len(items), extractor.dim):
        raise ExtractorError(
            f"The {feature_key(extractor)} extractor returned {rows.shape} for {len(items)} "
            f"images of {extractor.dim} values each; it is broken."
        )
    return {item.file.content_key: rows[i] for i, item in enumerate(items)}
