from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from calico.catalog import store as store_module
from calico.catalog.store import FACTS_VERSION, Catalog, PhotoRecord
from calico.domain import Photo
from calico.errors import CatalogTooNewError

if TYPE_CHECKING:
    from collections.abc import Iterator


def record(path: str, key: str = "k", collection: str = "c", **kwargs: object) -> PhotoRecord:
    photo = Photo(
        path=Path(path),
        content_key=key,
        collection=collection,
        location=str(kwargs.get("location", "L")),
        captured_at=kwargs.get("captured_at"),  # type: ignore[arg-type]
        width=10,
        height=20,
        camera=kwargs.get("camera"),  # type: ignore[arg-type]
        utc_offset=kwargs.get("utc_offset"),  # type: ignore[arg-type]
    )
    return PhotoRecord(photo=photo, size=5, mtime=1.5)


@pytest.fixture
def catalog(tmp_path: Path) -> Iterator[Catalog]:
    cat = Catalog(tmp_path / "deep" / "dir" / "catalog.sqlite")
    yield cat
    cat.close()


def test_creates_parent_dirs_wal_and_version(tmp_path: Path) -> None:
    path = tmp_path / "x" / "y" / "c.sqlite"
    with Catalog(path) as cat:
        assert cat.stats().photos == 0
    assert path.exists()
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(store_module.MIGRATIONS)
    conn.close()


def test_reopening_does_not_reapply_migrations(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    with Catalog(path) as cat:
        cat.upsert([record("/a/1.jpg")])
    with Catalog(path) as cat:
        assert cat.stats().photos == 1


def test_opening_a_current_catalog_waits_for_no_writer(tmp_path: Path) -> None:
    """Opening reads the schema version and takes the write lock only to migrate, so a
    long write elsewhere never holds up a page that only reads."""
    path = tmp_path / "c.sqlite"
    Catalog(path).close()
    writer = sqlite3.connect(path, autocommit=True)
    writer.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        with Catalog(path) as cat:
            assert cat.stats().photos == 0
        assert time.monotonic() - started < 1
    finally:
        writer.execute("ROLLBACK")
        writer.close()


def test_a_catalog_from_a_newer_calico_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    Catalog(path).close()
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {len(store_module.MIGRATIONS) + 1}")
    conn.commit()
    conn.close()
    with pytest.raises(CatalogTooNewError, match="newer Calico"):
        Catalog(path)


def test_upsert_roundtrip_and_replace(catalog: Catalog) -> None:
    when = datetime(2021, 1, 2, 3, 4, 5)
    catalog.upsert([record("/a/1.jpg", "k1", captured_at=when, camera="Canon X")])
    got = catalog.records_under(Path("/a"))[Path("/a/1.jpg")]
    assert got.photo.captured_at == when
    assert got.photo.camera == "Canon X"
    assert (got.size, got.mtime) == (5, 1.5)
    catalog.upsert([record("/a/1.jpg", "k2")])
    again = catalog.records_under(Path("/a"))[Path("/a/1.jpg")]
    assert again.photo.content_key == "k2"
    assert again.photo.captured_at is None


def test_the_time_zone_round_trips_with_the_current_facts_version(catalog: Catalog) -> None:
    catalog.upsert([record("/a/1.jpg", utc_offset=60)])
    got = catalog.records_under(Path("/a"))[Path("/a/1.jpg")]
    assert (got.photo.utc_offset, got.facts) == (60, FACTS_VERSION)


def test_photos_catalogued_before_time_zones_were_read_say_so(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    conn = sqlite3.connect(path)
    for number, script in enumerate(store_module.MIGRATIONS[:2], start=1):
        conn.executescript(script)
        conn.execute(f"PRAGMA user_version = {number}")
    conn.execute(
        "INSERT INTO photos VALUES ('/a/1.jpg', 'k', 'c', 'L', NULL, 10, 20, NULL, 5, 1.5)"
    )
    conn.commit()
    conn.close()
    with Catalog(path) as cat:
        got = cat.records_under(Path("/a"))[Path("/a/1.jpg")]
    assert (got.photo.utc_offset, got.facts) == (None, 1)
    assert FACTS_VERSION > 1


def test_records_under_matches_exact_and_children_not_siblings(catalog: Catalog) -> None:
    catalog.upsert([record("/a/b/1.jpg"), record("/a/bc/2.jpg"), record("/a/b/s/3.jpg")])
    assert sorted(catalog.records_under(Path("/a/b"))) == [Path("/a/b/1.jpg"), Path("/a/b/s/3.jpg")]
    assert list(catalog.records_under(Path("/a/b/1.jpg"))) == [Path("/a/b/1.jpg")]


def test_records_under_escapes_like_wildcards(catalog: Catalog) -> None:
    catalog.upsert(
        [record("/x/50%_off/1.jpg"), record("/x/50abcoff/2.jpg"), record("/x/50%/3.jpg")]
    )
    assert list(catalog.records_under(Path("/x/50%_off"))) == [Path("/x/50%_off/1.jpg")]


def test_photos_under_sorted_without_duplicates(catalog: Catalog) -> None:
    catalog.upsert([record("/a/2.jpg"), record("/a/b/1.jpg"), record("/z/9.jpg")])
    photos = catalog.photos_under([Path("/a"), Path("/a/b")])
    assert [p.path for p in photos] == [Path("/a/2.jpg"), Path("/a/b/1.jpg")]


def test_locations_under_lists_each_place_once(catalog: Catalog) -> None:
    catalog.upsert(
        [
            record("/a/1.jpg", location="Mill"),
            record("/a/2.jpg", location="Mill"),
            record("/a/b/3.jpg", location="Harbor"),
            record("/z/9.jpg", location="Elsewhere"),
        ]
    )
    assert catalog.locations_under([Path("/a")]) == ["Harbor", "Mill"]
    assert catalog.locations_under([]) == ["Elsewhere", "Harbor", "Mill"]  # no folders: all


def test_keys_at_lists_each_frame_at_each_place_once(catalog: Catalog) -> None:
    catalog.upsert(
        [
            record("/a/1.jpg", key="m1", location="Mill"),
            record("/a/copy/1.jpg", key="m1", location="Mill"),  # one frame filed twice
            record("/a/2.jpg", key="m2", location="Mill"),
            record("/b/3.jpg", key="h3", location="Harbor"),
            record("/z/9.jpg", key="e9", location="Elsewhere"),
        ]
    )
    assert catalog.keys_at(["Mill", "Harbor", "Nowhere"]) == {
        "Mill": ["m1", "m2"],
        "Harbor": ["h3"],
    }


def test_remove_counts(catalog: Catalog) -> None:
    catalog.upsert([record("/a/1.jpg"), record("/a/2.jpg")])
    assert catalog.remove([Path("/a/1.jpg"), Path("/a/nope.jpg")]) == 1
    assert list(catalog.records_under(Path("/a"))) == [Path("/a/2.jpg")]
    assert catalog.remove([]) == 0


def test_photos_are_found_by_content_one_path_each(catalog: Catalog) -> None:
    catalog.upsert([record("/a/2.jpg", key="k1"), record("/a/1.jpg", key="k1"), record("/b/3.jpg")])
    found = catalog.photos_with_keys(["k1", "missing"])
    assert [p.path for p in found] == [Path("/a/1.jpg")]
    assert catalog.photos_with_keys([]) == []


def test_features_roundtrip(catalog: Catalog) -> None:
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    catalog.put_features("f@1", {"k1": vec, "k2": vec * 2})
    got = catalog.get_features("f@1", ["k1", "k2", "missing"])
    assert set(got) == {"k1", "k2"}
    assert got["k2"].dtype == np.float32
    np.testing.assert_array_equal(got["k2"], vec * 2)
    assert catalog.has_features("f@1", ["k1", "missing"]) == {"k1"}
    assert catalog.has_features("other@1", ["k1"]) == set()
    catalog.put_features("f@1", {"k1": vec * 5})
    np.testing.assert_array_equal(catalog.get_features("f@1", ["k1"])["k1"], vec * 5)


def test_pair_shares_are_remembered_whichever_way_round(catalog: Catalog) -> None:
    catalog.put_pair_shares("f@1", {("b", "a"): 0.4, ("a", "c"): 0.1})
    catalog.put_pair_shares("f@2", {("a", "b"): 0.9})
    assert catalog.pair_shares_among("f@1", ["a", "b"]) == {("a", "b"): pytest.approx(0.4)}
    assert catalog.pair_shares_among("f@1", ["c", "a", "b"]) == {
        ("a", "b"): pytest.approx(0.4),
        ("a", "c"): pytest.approx(0.1),
    }
    assert catalog.pair_shares_among("f@2", ["a", "b"]) == {("a", "b"): pytest.approx(0.9)}
    catalog.put_pair_shares("f@1", {("a", "b"): 0.5})
    assert catalog.pair_shares_among("f@1", ["a", "b"]) == {("a", "b"): pytest.approx(0.5)}
    assert catalog.pair_shares_among("f@1", []) == {}


def test_pair_shares_among_many_photos_are_read_in_chunks(catalog: Catalog) -> None:
    keys = [f"k{i:04d}" for i in range(1300)]
    catalog.put_pair_shares("f@1", {(keys[i], keys[i + 1]): 0.2 for i in range(1299)})
    assert len(catalog.pair_shares_among("f@1", keys)) == 1299
    assert len(catalog.pair_shares_among("f@1", keys[:10])) == 9


def test_put_features_rejects_non_vectors(catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="1-D"):
        catalog.put_features("f@1", {"k": np.zeros((2, 2), dtype=np.float32)})


def test_queries_chunk_large_key_lists(catalog: Catalog) -> None:
    keys = [f"k{i}" for i in range(1300)]
    catalog.put_features("f@1", {k: np.zeros(1, dtype=np.float32) for k in keys})
    assert len(catalog.has_features("f@1", keys)) == 1300
    assert len(catalog.get_features("f@1", keys)) == 1300
    catalog.upsert([record(f"/a/{i}.jpg") for i in range(1300)])
    assert catalog.remove([Path(f"/a/{i}.jpg") for i in range(1300)]) == 1300


def test_failures(catalog: Catalog) -> None:
    catalog.record_failure(Path("/a/bad.jpg"), "decode", "boom")
    catalog.record_failure(Path("/a/bad.jpg"), "decode", "boom2")
    assert catalog.failures() == [(Path("/a/bad.jpg"), "decode", "boom2")]
    assert catalog.stats().failures == 1
    catalog.clear_failure(Path("/a/bad.jpg"))
    assert catalog.failures() == []


def test_feedback(catalog: Catalog) -> None:
    catalog.add_feedback("r1", Path("/a/1.jpg"), "k1", 1, "lovely")
    catalog.add_feedback("r2", Path("/a/2.jpg"), "k2", -1)
    assert [f.run_id for f in catalog.feedback()] == ["r1", "r2"]
    only = catalog.feedback("r1")
    assert len(only) == 1
    assert (only[0].path, only[0].content_key, only[0].rating, only[0].note) == (
        Path("/a/1.jpg"),
        "k1",
        1,
        "lovely",
    )
    assert isinstance(only[0].created_at, datetime)
    assert catalog.stats().feedback == 2


def test_feedback_rating_validated(catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="rating"):
        catalog.add_feedback("r", Path("/a"), "k", 0)


def test_stats(catalog: Catalog) -> None:
    catalog.upsert(
        [
            record("/a/1.jpg", "k1", "one"),
            record("/a/2.jpg", "k2", "one"),
            record("/b/3.jpg", "k3", "two"),
        ]
    )
    vec = np.zeros(2, dtype=np.float32)
    catalog.put_features("f@1", {"k1": vec, "k2": vec, "orphan": vec})
    stats = catalog.stats()
    assert stats.photos == 3
    assert stats.by_collection == {"one": 2, "two": 1}
    assert stats.features == {"f@1": 2}


NFD = "Cafe\u0301"
NFC = "Caf\u00e9"


def test_nfd_and_nfc_paths_are_the_same_row(catalog: Catalog) -> None:
    catalog.upsert([record(f"/x/{NFD}/1.jpg")])
    catalog.upsert([record(f"/x/{NFC}/1.jpg", "k2")])
    assert catalog.stats().photos == 1
    assert list(catalog.records_under(Path(f"/x/{NFD}"))) == [Path(f"/x/{NFC}/1.jpg")]
    assert list(catalog.records_under(Path(f"/x/{NFC}"))) == [Path(f"/x/{NFC}/1.jpg")]
    assert [p.path for p in catalog.photos_under([Path(f"/x/{NFD}")])] == [Path(f"/x/{NFC}/1.jpg")]
    assert catalog.remove([Path(f"/x/{NFD}/1.jpg")]) == 1


def test_nfd_failures_and_feedback(catalog: Catalog) -> None:
    catalog.record_failure(Path(f"/x/{NFD}/b.jpg"), "decode", "e")
    catalog.record_failure(Path(f"/x/{NFC}/b.jpg"), "decode", "e2")
    assert catalog.failures() == [(Path(f"/x/{NFC}/b.jpg"), "decode", "e2")]
    catalog.clear_failure(Path(f"/x/{NFD}/b.jpg"))
    assert catalog.failures() == []
    catalog.add_feedback("r", Path(f"/x/{NFD}/b.jpg"), "k", 1)
    assert catalog.feedback()[0].path == Path(f"/x/{NFC}/b.jpg")


def test_failure_paths_under_and_clear_failures(catalog: Catalog) -> None:
    for name in ("/a/1.jpg", "/a/2.jpg", "/b/3.jpg"):
        catalog.record_failure(Path(name), "decode", "e")
    assert sorted(catalog.failure_paths_under(Path("/a"))) == [Path("/a/1.jpg"), Path("/a/2.jpg")]
    assert catalog.clear_failures([Path("/a/1.jpg"), Path("/a/zzz.jpg")]) == 1
    assert [p for p, _, _ in catalog.failures()] == [Path("/a/2.jpg"), Path("/b/3.jpg")]


def test_corrupt_database_raises_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "bad.sqlite"
    path.write_bytes(b"this is not a database" * 100)
    with pytest.raises(sqlite3.DatabaseError):
        Catalog(path)


def test_every_location_a_frame_is_filed_under_is_found_not_only_the_first(
    catalog: Catalog,
) -> None:
    catalog.upsert(
        [
            record("/a/Harbor/x.jpg", key="x", location="Harbor"),
            record("/a/Mill/x.jpg", key="x", location="Mill"),  # the same file filed again
            record("/a/Mill/y.jpg", key="y", location="Mill"),
            record("/a/Mill/z.jpg", key="z", location="Mill"),
        ]
    )
    assert catalog.locations_of(["x", "y", "nothing"]) == {
        "x": frozenset({"Harbor", "Mill"}),
        "y": frozenset({"Mill"}),
    }
    assert catalog.locations_of([]) == {}
