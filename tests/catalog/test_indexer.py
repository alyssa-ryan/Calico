from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from calico.catalog import indexer as indexer_module
from calico.catalog.indexer import Indexer, thumbnail_path
from calico.catalog.store import FACTS_VERSION, Catalog
from calico.errors import (
    CalicoError,
    FolderNotFoundError,
    HomeInsideArchiveError,
    ImageDecodeError,
    NotInCollectionError,
)
from calico.features.base import feature_key
from calico.settings import CollectionConfig, Settings
from tests.factories import fake_rawpy, save_jpeg, scene, solid, with_exif

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from PIL import Image


@dataclass
class MeanRGB:
    """Fake extractor: mean colour of the image, with a record of every call."""

    name: str = "mean"
    version: str = "1"
    dim: int = 3
    columns: tuple[str, ...] | None = None
    calls: list[int] = field(default_factory=list)

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        self.calls.append(len(images))
        rows = [np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(axis=0) for im in images]
        return np.stack(rows).astype(np.float32)

    @property
    def total(self) -> int:
        return sum(self.calls)


@dataclass
class Size2:
    """Second fake extractor: width and height of the prepared image."""

    name: str = "size"
    version: str = "1"
    dim: int = 2
    columns: tuple[str, ...] | None = ("w", "h")
    calls: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.calls)

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        self.calls.append(len(images))
        return np.array([im.size for im in images], dtype=np.float32)


@dataclass
class Env:
    settings: Settings
    catalog: Catalog
    root: Path
    mean: MeanRGB
    size: Size2

    def indexer(self, **kwargs: int) -> Indexer:
        return Indexer(
            self.settings,
            self.catalog,
            {"mean": self.mean, "size": self.size},
            **kwargs,
        )


@pytest.fixture
def env(tmp_path: Path, calico_home: Path) -> Env:
    root = tmp_path / "archive"
    root.mkdir()
    collection = CollectionConfig(name="trips", root=root.resolve())
    settings = Settings(home=calico_home, collections=(collection,), thumbnail_side=64)
    catalog = Catalog(settings.catalog_path)
    return Env(settings, catalog, root.resolve(), MeanRGB(), Size2())


def make_photos(root: Path) -> list[Path]:
    return [
        save_jpeg(scene(1), root / "Old Mill" / "a.jpg", captured_at=datetime(2020, 5, 1, 9, 0, 0)),
        save_jpeg(scene(2), root / "Old Mill" / "b.jpg"),
        save_jpeg(scene(3), root / "Harbour" / "c.jpg", camera=("Canon", "R5")),
    ]


def test_first_index_adds_everything_and_writes_thumbnails(env: Env) -> None:
    paths = make_photos(env.root)
    seen: list[tuple[int, int, str]] = []
    report = env.indexer().index([env.root], progress=lambda d, t, s: seen.append((d, t, s)))
    assert (report.scanned, report.added, report.updated, report.unchanged) == (3, 3, 0, 0)
    assert report.failed == []
    assert seen[-1] == (3, 3, "reading")
    assert len(seen) == 3
    photos = {p.path: p for p in env.catalog.photos_under([env.root])}
    assert set(photos) == set(paths)
    first = photos[paths[0]]
    assert first.captured_at == datetime(2020, 5, 1, 9, 0, 0)
    assert (first.collection, first.location) == ("trips", "Old Mill")
    assert photos[paths[2]].location == "Harbour"
    assert photos[paths[2]].camera == "Canon R5"
    for photo in photos.values():
        thumb = thumbnail_path(env.settings.thumbs_dir, photo.content_key)
        assert thumb.parent.name == photo.content_key[:2]
        assert thumb.is_file()
        assert not list(thumb.parent.glob("*.tmp"))
    assert env.catalog.stats().features == {"mean@1": 3, "size@1": 3}
    assert report.seconds >= 0


def test_second_index_is_all_unchanged_and_calls_no_extractor(env: Env) -> None:
    make_photos(env.root)
    env.indexer().index([env.root])
    calls = (len(env.mean.calls), len(env.size.calls))
    report = env.indexer().index([env.root])
    assert (report.added, report.updated, report.unchanged) == (0, 0, 3)
    assert (len(env.mean.calls), len(env.size.calls)) == calls


def test_batches_respect_batch_size(env: Env) -> None:
    for i in range(5):
        save_jpeg(scene(i), env.root / f"{i}.jpg")
    env.indexer(batch_size=2, workers=2).index([env.root])
    assert env.mean.calls == [2, 2, 1]


def test_modified_file_is_updated(env: Env) -> None:
    paths = make_photos(env.root)
    env.indexer().index([env.root])
    old_key = env.catalog.records_under(env.root)[paths[0]].photo.content_key
    save_jpeg(scene(99, (200, 100)), paths[0])
    os.utime(paths[0], (1_000_000, 1_000_000))
    report = env.indexer().index([env.root])
    assert (report.added, report.updated, report.unchanged) == (0, 1, 2)
    new = env.catalog.records_under(env.root)[paths[0]].photo
    assert new.content_key != old_key
    assert (new.width, new.height) == (200, 100)


def test_new_extractor_version_reprocesses(env: Env) -> None:
    make_photos(env.root)
    env.indexer().index([env.root])
    env.mean.version = "2"
    report = env.indexer().index([env.root])
    assert (report.updated, report.unchanged) == (3, 0)
    assert env.catalog.stats().features["mean@2"] == 3
    assert len(env.size.calls) == 1  # size@1 was already cached


def test_deleted_file_is_removed(env: Env) -> None:
    paths = make_photos(env.root)
    env.indexer().index([env.root])
    paths[1].unlink()
    report = env.indexer().index([env.root])
    assert (report.removed, report.unchanged) == (1, 2)
    assert paths[1] not in env.catalog.records_under(env.root)


def test_corrupt_file_fails_but_others_index(env: Env) -> None:
    make_photos(env.root)
    bad = env.root / "Old Mill" / "bad.jpg"
    bad.write_bytes(b"definitely not a jpeg")
    report = env.indexer().index([env.root])
    assert report.added == 3
    assert [p for p, _ in report.failed] == [bad]
    assert "bad.jpg" in report.failed[0][1]
    assert [(p, s) for p, s, _ in env.catalog.failures()] == [(bad, "decode")]
    bad.unlink()
    save_jpeg(scene(7), bad)
    env.indexer().index([env.root])
    assert env.catalog.failures() == []


def test_copied_file_reuses_features(env: Env) -> None:
    paths = make_photos(env.root)
    env.indexer().index([env.root])
    calls = (env.mean.total, env.size.total)
    copy = env.root / "Harbour" / "copy.jpg"
    shutil.copyfile(paths[2], copy)
    report = env.indexer().index([env.root])
    assert report.added == 1
    assert (env.mean.total, env.size.total) == calls
    records = env.catalog.records_under(env.root)
    assert records[copy].photo.content_key == records[paths[2]].photo.content_key


def test_duplicates_in_one_run_extract_once(env: Env) -> None:
    src = save_jpeg(scene(5), env.root / "x" / "a.jpg")
    shutil.copyfile(src, env.root / "x" / "b.jpg")
    env.indexer().index([env.root])
    assert env.mean.total == 1


def test_raw_sibling_handling(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rawpy", None)  # no RAW support
    save_jpeg(scene(1), env.root / "IMG_1.jpg")
    (env.root / "IMG_1.CR2").write_bytes(b"raw")
    (env.root / "IMG_2.NEF").write_bytes(b"raw")
    report = env.indexer().index([env.root])
    assert (report.scanned, report.added, report.raw_without_rendered) == (1, 1, 1)


def test_with_raw_support_a_lone_raw_is_catalogued_with_its_camera_and_date(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview = with_exif(
        scene(4, (2400, 1600)), camera=("Canon", "EOS RP"), captured_at=datetime(2021, 8, 6, 10, 31)
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy(preview))
    save_jpeg(scene(1), env.root / "IMG_1.jpg")
    (env.root / "IMG_1.CR3").write_bytes(b"raw beside its JPEG")
    (env.root / "IMG_2.CR3").write_bytes(b"a lone raw")
    report = env.indexer().index([env.root])
    assert (report.scanned, report.added, report.raw_without_rendered) == (2, 2, 0)
    raw = env.catalog.records_under(env.root)[env.root / "IMG_2.CR3"].photo
    assert (raw.camera, raw.captured_at) == ("Canon EOS RP", datetime(2021, 8, 6, 10, 31))


def test_photos_catalogued_before_time_zones_get_theirs_without_decoding_again(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = save_jpeg(
        scene(1), env.root / "a.jpg", captured_at=datetime(2026, 2, 26, 13, 2), utc_offset="+01:00"
    )
    env.indexer().index([env.root])
    with env.catalog._conn:  # as an older Calico left it
        env.catalog._conn.execute("UPDATE photos SET utc_offset = NULL, facts = 1")
    decoded: list[Path] = []
    monkeypatch.setattr(indexer_module, "decode", lambda p, side: decoded.append(p))
    report = env.indexer().index([env.root])
    record = env.catalog.records_under(env.root)[path]
    assert (record.photo.utc_offset, record.facts) == (60, FACTS_VERSION)
    assert decoded == []
    assert report.unchanged + report.updated == 1


def test_excluded_names_are_skipped(env: Env) -> None:
    save_jpeg(scene(1), env.root / "a.jpg")
    save_jpeg(scene(2), env.root / "._a.jpg")
    assert env.indexer().index([env.root]).added == 1


def test_subfolder_index_does_not_remove_siblings(env: Env) -> None:
    paths = make_photos(env.root)
    env.indexer().index([env.root])
    report = env.indexer().index([env.root / "Harbour"])
    assert (report.scanned, report.unchanged, report.removed) == (1, 1, 0)
    assert paths[0] in env.catalog.records_under(env.root)


def test_folder_outside_collections_raises(env: Env, tmp_path: Path) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(NotInCollectionError):
        env.indexer().index([other])


def test_missing_folder_raises(env: Env) -> None:
    with pytest.raises(FolderNotFoundError):
        env.indexer().index([env.root / "gone"])


def test_features_for_files_outside_any_collection(env: Env, tmp_path: Path) -> None:
    outside = [
        save_jpeg(solid((255, 0, 0)), tmp_path / "ref" / "r.jpg"),
        save_jpeg(solid((0, 0, 255), (30, 60)), tmp_path / "ref" / "b.jpg"),
    ]
    keys, matrices = env.indexer().features_for_files(outside)
    assert len(keys) == 2
    assert keys[0] != keys[1]
    assert set(matrices) == {"mean", "size"}
    assert matrices["mean"].shape == (2, 3)
    assert matrices["mean"].dtype == np.float32
    assert matrices["mean"][0, 0] > 200
    assert matrices["size"].tolist() == [[64.0, 48.0], [30.0, 60.0]]


def test_features_for_files_reuses_catalog_cache(env: Env) -> None:
    paths = make_photos(env.root)
    env.indexer().index([env.root])
    calls = (env.mean.total, env.size.total)
    keys, matrices = env.indexer().features_for_files([paths[0], paths[0]])
    assert keys[0] == keys[1]
    assert matrices["mean"].shape == (2, 3)
    assert (env.mean.total, env.size.total) == calls


def test_features_for_files_empty_and_bad(env: Env, tmp_path: Path) -> None:
    keys, matrices = env.indexer().features_for_files([])
    assert keys == []
    assert matrices["size"].shape == (0, 2)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    with pytest.raises(ImageDecodeError):
        env.indexer().features_for_files([bad])


def test_feature_key_is_name_at_version() -> None:
    assert feature_key(MeanRGB()) == "mean@1"


def test_unreadable_folder_keeps_its_records(env: Env) -> None:
    paths = make_photos(env.root)
    env.indexer().index([env.root])
    locked = env.root / "Harbour"
    locked.chmod(0)
    try:
        report = env.indexer().index([env.root])
    finally:
        locked.chmod(0o755)
    assert report.removed == 0
    assert report.unreadable_dirs == [locked]
    assert paths[2] in env.catalog.records_under(env.root)


def test_stale_failure_rows_are_cleared(env: Env) -> None:
    make_photos(env.root)
    bad = env.root / "bad.jpg"
    bad.write_bytes(b"nope")
    env.indexer().index([env.root])
    assert len(env.catalog.failures()) == 1
    bad.unlink()
    env.indexer().index([env.root])
    assert env.catalog.failures() == []


def test_changed_location_rule_updates_unchanged_files(env: Env) -> None:
    make_photos(env.root)
    env.indexer().index([env.root])
    deeper = CollectionConfig(name="trips", root=env.root, location_pattern=r"^(?P<location>Old)")
    env.settings = env.settings.model_copy(update={"collections": (deeper,)})
    calls = env.mean.total
    report = env.indexer().index([env.root])
    assert (report.updated, report.unchanged) == (2, 1)
    assert env.mean.total == calls
    locations = {p.location for p in env.catalog.photos_under([env.root])}
    assert locations == {"Old", "Harbour"}


def test_new_alias_renames_catalogued_photos_without_decoding(env: Env) -> None:
    make_photos(env.root)
    env.indexer().index([env.root])
    aliases = {"harbour": "Harbor"}
    env.settings = env.settings.model_copy(update={"location_aliases": aliases})
    calls = env.mean.total
    report = env.indexer().index([env.root])
    assert (report.updated, report.unchanged) == (1, 2)
    assert env.mean.total == calls
    locations = {p.location for p in env.catalog.photos_under([env.root])}
    assert locations == {"Old Mill", "Harbor"}


def test_new_nested_collection_reassigns_photos(env: Env) -> None:
    make_photos(env.root)
    env.indexer().index([env.root])
    inner = CollectionConfig(name="harbour", root=env.root / "Harbour")
    env.settings = env.settings.with_collection(inner)
    report = env.indexer().index([env.root])
    assert report.updated == 1
    by_name = {p.path.name: p for p in env.catalog.photos_under([env.root])}
    assert by_name["c.jpg"].collection == "harbour"
    assert by_name["c.jpg"].location == "harbour"


@pytest.mark.skipif(
    sys.platform != "darwin", reason="only macOS opens an NFD-named folder by its NFC name"
)
def test_nfd_folder_reindex_does_not_duplicate(env: Env) -> None:
    save_jpeg(scene(1), env.root / "Cafe\u0301" / "a.jpg")
    env.indexer().index([env.root])
    report = env.indexer().index([env.root / "Cafe\u0301"])
    assert report.unchanged == 1
    assert env.catalog.stats().photos == 1


@pytest.mark.skipif(
    sys.platform != "darwin", reason="only macOS opens an NFD-named file by its NFC name"
)
def test_nfd_file_name_stays_catalogued_across_runs(env: Env) -> None:
    save_jpeg(scene(1), env.root / "cafe\u0301.jpg")
    env.indexer().index([env.root])
    report = env.indexer().index([env.root])
    assert (report.unchanged, report.removed) == (1, 0)
    assert env.catalog.stats().photos == 1


def test_symlinked_files_are_counted_not_indexed(env: Env, tmp_path: Path) -> None:
    outside = save_jpeg(scene(1), tmp_path / "outside" / "secret.jpg")
    (env.root / "link.jpg").symlink_to(outside)
    report = env.indexer().index([env.root])
    assert (report.scanned, report.links_skipped) == (0, 1)
    assert env.catalog.stats().photos == 0


def test_nested_folder_arguments_are_indexed_once(env: Env) -> None:
    make_photos(env.root)
    report = env.indexer().index([env.root / "Harbour", env.root, env.root / "Old Mill"])
    assert (report.scanned, report.added, report.unchanged) == (3, 3, 0)


def test_thumbnail_failure_leaves_no_tmp_file(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    make_photos(env.root)

    def boom(*_: object) -> bytes:
        raise OSError("disk full")

    monkeypatch.setattr("calico.catalog.indexer.thumbnail_jpeg", boom)
    with pytest.raises(OSError, match="disk full"):
        env.indexer().index([env.root])
    assert not list(env.settings.thumbs_dir.rglob("*.tmp"))


def test_a_nested_collection_is_right_on_the_first_index(env: Env) -> None:
    """Photos in an inner collection are its own from the start, not the outer one's
    until the next run."""
    make_photos(env.root)
    inner = CollectionConfig(name="harbour", root=env.root / "Harbour")
    env.settings = env.settings.with_collection(inner)
    report = env.indexer().index([env.root])
    assert (report.added, report.updated) == (3, 0)
    by_name = {p.path.name: p for p in env.catalog.photos_under([env.root])}
    assert (by_name["c.jpg"].collection, by_name["c.jpg"].location) == ("harbour", "harbour")
    assert env.indexer().index([env.root]).updated == 0


def test_a_nested_collection_skips_what_it_excludes(env: Env) -> None:
    make_photos(env.root)
    save_jpeg(scene(4), env.root / "Harbour" / "draft.jpg")
    save_jpeg(scene(5), env.root / "Old Mill" / "draft.jpg")
    inner = CollectionConfig(name="harbour", root=env.root / "Harbour", exclude=("draft*",))
    env.settings = env.settings.with_collection(inner)
    env.indexer().index([env.root])
    names = sorted(str(p.path.relative_to(env.root)) for p in env.catalog.photos_under([env.root]))
    assert "Harbour/draft.jpg" not in names
    assert "Old Mill/draft.jpg" in names


def test_indexing_refuses_a_calico_home_inside_the_archive(env: Env, tmp_path: Path) -> None:
    """The catalog and thumbnails would be written into the archive, and indexed as photos."""
    outer = CollectionConfig(name="all", root=tmp_path)
    env.settings = env.settings.model_copy(update={"collections": (outer,)})
    with pytest.raises(HomeInsideArchiveError, match=r"calico\.toml"):
        env.indexer().index([tmp_path])


@dataclass
class Fragile(MeanRGB):
    """Fails on any image ``breaks`` picks: a pathological photo for one model."""

    name: str = "fragile"
    breaks: Any = None

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        if any(self.breaks(im) for im in images):
            raise RuntimeError("model blew up")
        return super().extract(images)


def test_one_photo_an_extractor_cannot_read_never_stops_the_index(env: Env) -> None:
    make_photos(env.root)
    save_jpeg(solid((123, 40, 40), (123, 77)), env.root / "Harbour" / "odd.jpg")
    fragile = Fragile(breaks=lambda im: im.width == 123)
    report = Indexer(env.settings, env.catalog, {"fragile": fragile}).index([env.root])
    assert report.added == 3
    assert [p.name for p, _ in report.failed] == ["odd.jpg"]
    ((path, stage, message),) = env.catalog.failures()
    assert (path.name, stage) == ("odd.jpg", "extract:fragile@1")
    assert "model blew up" in message


def test_an_extractor_that_fails_on_every_photo_stops_and_says_so(env: Env) -> None:
    """Out of memory or missing weights: marking every photo failed would hide it."""
    make_photos(env.root)
    fragile = Fragile(breaks=lambda _: True)
    with pytest.raises(CalicoError, match=r"fragile.*every photo"):
        Indexer(env.settings, env.catalog, {"fragile": fragile}).index([env.root])
    assert env.catalog.failures() == []


def test_a_photo_overwritten_with_an_unreadable_file_leaves_the_catalog(env: Env) -> None:
    """Its old row would keep a content key and features that are no longer the file's."""
    make_photos(env.root)
    env.indexer().index([env.root])
    damaged = env.root / "Old Mill" / "b.jpg"
    damaged.write_bytes(damaged.read_bytes()[:200])
    report = env.indexer().index([env.root])
    assert report.removed == 1
    assert "b.jpg" not in {p.path.name for p in env.catalog.photos_under([env.root])}
    assert [p.name for p, _, _ in env.catalog.failures()] == ["b.jpg"]


def test_missing_thumbnails_are_made_again(env: Env) -> None:
    """thumbs/ looks like a cache: deleting it must not leave the web pages blank."""
    made = make_photos(env.root)
    env.indexer().index([env.root])
    shutil.rmtree(env.settings.thumbs_dir)
    env.indexer().index([env.root])
    keys = {p.content_key for p in env.catalog.photos_under([env.root])}
    assert len(keys) == len(made)
    assert all(thumbnail_path(env.settings.thumbs_dir, k).is_file() for k in keys)
    assert env.mean.total == len(made)  # features were cached: nothing extracted again


def test_the_report_counts_files_not_read(env: Env) -> None:
    make_photos(env.root)
    (env.root / "Old Mill" / "clip.mov").write_bytes(b"")
    report = env.indexer().index([env.root])
    assert report.unsupported == {".mov": 1}
