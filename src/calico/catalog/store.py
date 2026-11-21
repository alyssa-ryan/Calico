"""The SQLite catalog: photos, cached feature vectors, failures and feedback.

stdlib ``sqlite3`` only. The schema is versioned with ``PRAGMA user_version``:
``MIGRATIONS`` is an ordered list of scripts, and a database at version *n* has
run the first *n*. To change the schema, append a script -- never edit one.
Every public write commits before returning, so a crash loses at most the call
in flight.
"""

# ruff: noqa: S608 -- f-strings only splice constant column lists and "?" placeholders.
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from calico.catalog.scanner import nfc_path
from calico.domain import Photo
from calico.errors import CatalogTooNewError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from types import TracebackType

    from numpy.typing import NDArray

MIGRATIONS: list[str] = [
    """
    CREATE TABLE photos (
        path TEXT PRIMARY KEY,
        content_key TEXT NOT NULL,
        collection TEXT NOT NULL,
        location TEXT NOT NULL,
        captured_at TEXT,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        camera TEXT,
        size INTEGER NOT NULL,
        mtime REAL NOT NULL
    );
    CREATE INDEX photos_content_key ON photos (content_key);
    CREATE TABLE features (
        feature_key TEXT NOT NULL,
        content_key TEXT NOT NULL,
        dim INTEGER NOT NULL,
        value BLOB NOT NULL,
        PRIMARY KEY (feature_key, content_key)
    );
    CREATE TABLE failures (
        path TEXT PRIMARY KEY,
        stage TEXT NOT NULL,
        error TEXT NOT NULL,
        at TEXT NOT NULL
    );
    CREATE TABLE feedback (
        id INTEGER PRIMARY KEY,
        run_id TEXT NOT NULL,
        path TEXT NOT NULL,
        content_key TEXT NOT NULL,
        rating INTEGER NOT NULL,
        note TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX feedback_run ON feedback (run_id);
    """,
    """
    CREATE TABLE pair_shares (
        feature_key TEXT NOT NULL,
        key_a TEXT NOT NULL,
        key_b TEXT NOT NULL,
        share REAL NOT NULL,
        PRIMARY KEY (feature_key, key_a, key_b)
    );
    CREATE INDEX pair_shares_b ON pair_shares (feature_key, key_b);
    """,
    """
    ALTER TABLE photos ADD COLUMN utc_offset INTEGER;
    ALTER TABLE photos ADD COLUMN facts INTEGER NOT NULL DEFAULT 1;
    """,
]

FACTS_VERSION = 2
"""What a photo's stored facts were read with: 2 added the time zone. Photos with older
facts get them read again from the file's EXIF (no decode) the next time they are
indexed."""

MIGRATE_SECONDS = 30.0
"""How long opening a catalog that needs migrating waits for another process to finish."""
BUSY_SECONDS = 30.0
"""How long a write waits for another process's write: indexing batches are short, so
this is a stall, not a queue."""
_CHUNK = 500
_PHOTO_COLUMNS = (
    "path, content_key, collection, location, captured_at, width, height, camera, size, mtime, "
    "utc_offset, facts"
)


@dataclass(frozen=True)
class PhotoRecord:
    """A catalogued photo plus the file stat used to detect changes."""

    photo: Photo
    size: int
    mtime: float
    facts: int = FACTS_VERSION
    """``FACTS_VERSION`` when the photo's facts were read."""


@dataclass(frozen=True)
class FeedbackRow:
    run_id: str
    path: Path
    content_key: str
    rating: int
    note: str
    created_at: datetime


@dataclass(frozen=True)
class CatalogStats:
    photos: int
    by_collection: dict[str, int]
    features: dict[str, int]
    """Feature key -> number of catalogued photos whose content has that feature."""
    failures: int
    feedback: int


def _chunks(items: Sequence[Any]) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), _CHUNK):
        yield items[start : start + _CHUNK]


def _placeholders(count: int) -> str:
    return ",".join("?" * count)


def _folder_clause(folder: Path) -> tuple[str, tuple[str, str, str]]:
    """Match ``folder`` itself or anything below it.

    A range query (``folder/`` <= path < ``folder0``, since "0" follows "/") needs
    no LIKE escaping, so folders named ``50%_off`` are safe, and ``/a/bc`` is not
    below ``/a/b``.
    """
    name = _key(folder)
    base = name.rstrip("/")
    return "(path = ? OR (path >= ? AND path < ?))", (name, base + "/", base + "0")


def _key(path: Path) -> str:
    """The stored form of a path: NFC, so NFD and NFC spellings are one row."""
    return str(nfc_path(path))


def _migrate(path: Path) -> None:
    """Bring the schema up to date, under a write lock only when it is behind.

    A current schema is the usual case, and reading its version takes no lock, so
    opening the catalog never waits for another process's write. To migrate,
    ``BEGIN IMMEDIATE`` makes a second process opening a fresh catalog wait, then see
    the finished schema instead of applying migration 1 twice; the version is re-read
    after the lock is taken for the same reason.
    """
    conn = sqlite3.connect(path, timeout=MIGRATE_SECONDS, autocommit=True)
    try:
        if _schema_version(conn, path) == len(MIGRATIONS):
            return
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("BEGIN IMMEDIATE")
        try:
            version = _schema_version(conn, path)
            for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
                conn.executescript(script)
                conn.execute(f"PRAGMA user_version = {number}")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def _schema_version(conn: sqlite3.Connection, path: Path) -> int:
    version: int = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > len(MIGRATIONS):
        raise CatalogTooNewError(
            f"{path} was written by a newer Calico (catalog schema {version}; this one "
            f"knows {len(MIGRATIONS)}). Update Calico (git pull, then uv sync), or point "
            "CALICO_HOME at another home."
        )
    return version


def _row_to_record(row: sqlite3.Row) -> PhotoRecord:
    captured = row["captured_at"]
    photo = Photo(
        path=Path(row["path"]),
        content_key=row["content_key"],
        collection=row["collection"],
        location=row["location"],
        captured_at=datetime.fromisoformat(captured) if captured else None,
        width=row["width"],
        height=row["height"],
        camera=row["camera"],
        utc_offset=row["utc_offset"],
    )
    return PhotoRecord(photo=photo, size=row["size"], mtime=row["mtime"], facts=row["facts"])


class Catalog:
    """Persistent catalog. Use as a context manager or call ``close()``."""

    def __init__(self, path: Path, *, busy_seconds: float = BUSY_SECONDS) -> None:
        """``busy_seconds``: how long a write waits for another process's write to finish
        before ``sqlite3.OperationalError`` ("database is locked")."""
        path.parent.mkdir(parents=True, exist_ok=True)
        _migrate(path)
        self._conn = sqlite3.connect(path, timeout=busy_seconds)
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.Error:
            self._conn.close()
            raise

    def close(self) -> None:
        self._conn.close()

    def copy_to(self, path: Path) -> None:
        """A consistent copy of the whole catalog at ``path``, which must not exist. ``VACUUM
        INTO`` reads one snapshot, so a write by another process mid-copy is never half in it,
        as it could be in a copy of the file (and its WAL) beside a running Calico."""
        self._conn.execute("VACUUM INTO ?", (str(path),))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- photos ---------------------------------------------------------

    def upsert(self, records: Iterable[PhotoRecord]) -> None:
        rows = [
            (
                _key(r.photo.path),
                r.photo.content_key,
                r.photo.collection,
                r.photo.location,
                r.photo.captured_at.isoformat() if r.photo.captured_at else None,
                r.photo.width,
                r.photo.height,
                r.photo.camera,
                r.size,
                r.mtime,
                r.photo.utc_offset,
                r.facts,
            )
            for r in records
        ]
        with self._conn:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO photos ({_PHOTO_COLUMNS}) VALUES ({_placeholders(12)})",
                rows,
            )

    def records_under(self, folder: Path) -> dict[Path, PhotoRecord]:
        clause, params = _folder_clause(folder)
        rows = self._conn.execute(f"SELECT {_PHOTO_COLUMNS} FROM photos WHERE {clause}", params)
        records = (_row_to_record(row) for row in rows)
        return {r.photo.path: r for r in records}

    def photos_under(self, folders: Sequence[Path]) -> list[Photo]:
        found: dict[Path, Photo] = {}
        for folder in folders:
            found.update({p: r.photo for p, r in self.records_under(folder).items()})
        return [found[p] for p in sorted(found)]

    def photos_with_keys(self, content_keys: Sequence[str]) -> list[Photo]:
        """One photo per content key that is catalogued (the first path, if a frame is
        filed twice), in path order."""
        found: dict[str, Photo] = {}
        for chunk in _chunks(sorted(set(content_keys))):
            rows = self._conn.execute(
                f"SELECT {_PHOTO_COLUMNS} FROM photos "
                f"WHERE content_key IN ({_placeholders(len(chunk))}) ORDER BY path",
                tuple(chunk),
            )
            for row in rows:
                photo = _row_to_record(row).photo
                found.setdefault(photo.content_key, photo)
        return sorted(found.values(), key=lambda p: str(p.path))

    def locations_of(self, content_keys: Sequence[str]) -> dict[str, frozenset[str]]:
        """Every location each catalogued frame is filed under, not only the first (a frame
        filed twice is one frame at two locations); a key that is not catalogued is left out."""
        found: dict[str, set[str]] = {}
        for chunk in _chunks(sorted(set(content_keys))):
            rows = self._conn.execute(
                "SELECT DISTINCT content_key, location FROM photos "
                f"WHERE content_key IN ({_placeholders(len(chunk))})",
                tuple(chunk),
            )
            for row in rows:
                found.setdefault(row["content_key"], set()).add(row["location"])
        return {key: frozenset(places) for key, places in found.items()}

    def keys_at(self, locations: Sequence[str]) -> dict[str, list[str]]:
        """Each catalogued frame at each of ``locations``, once (a frame filed twice is one
        frame), in key order; a location with no photos is left out."""
        found: dict[str, set[str]] = {}
        for chunk in _chunks(sorted(set(locations))):
            rows = self._conn.execute(
                "SELECT DISTINCT location, content_key FROM photos "
                f"WHERE location IN ({_placeholders(len(chunk))})",
                tuple(chunk),
            )
            for row in rows:
                found.setdefault(row["location"], set()).add(row["content_key"])
        return {name: sorted(keys) for name, keys in found.items()}

    def locations_under(self, folders: Sequence[Path]) -> list[str]:
        """Distinct location names of the photos under ``folders`` (all, if none given)."""
        if not folders:
            rows = self._conn.execute("SELECT DISTINCT location FROM photos")
            return sorted(r["location"] for r in rows)
        found: set[str] = set()
        for folder in folders:
            clause, params = _folder_clause(folder)
            rows = self._conn.execute(
                f"SELECT DISTINCT location FROM photos WHERE {clause}", params
            )
            found.update(r["location"] for r in rows)
        return sorted(found)

    def remove(self, paths: Iterable[Path]) -> int:
        names = [_key(p) for p in paths]
        removed = 0
        with self._conn:
            for chunk in _chunks(names):
                cursor = self._conn.execute(
                    f"DELETE FROM photos WHERE path IN ({_placeholders(len(chunk))})", chunk
                )
                removed += cursor.rowcount
        return removed

    # -- features -------------------------------------------------------

    def put_features(self, feature_key: str, vectors: Mapping[str, NDArray[np.float32]]) -> None:
        rows = []
        for content_key, vector in vectors.items():
            if vector.ndim != 1:
                raise ValueError(f"feature vectors must be 1-D, got shape {vector.shape}")
            data = np.ascontiguousarray(vector, dtype=np.float32)
            rows.append((feature_key, content_key, data.shape[0], data.tobytes()))
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO features (feature_key, content_key, dim, value) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    def get_features(
        self, feature_key: str, content_keys: Sequence[str]
    ) -> dict[str, NDArray[np.float32]]:
        found: dict[str, NDArray[np.float32]] = {}
        for chunk in _chunks(content_keys):
            rows = self._conn.execute(
                "SELECT content_key, value FROM features WHERE feature_key = ? "
                f"AND content_key IN ({_placeholders(len(chunk))})",
                (feature_key, *chunk),
            )
            for row in rows:
                found[row["content_key"]] = np.frombuffer(row["value"], dtype=np.float32).copy()
        return found

    def has_features(self, feature_key: str, content_keys: Sequence[str]) -> set[str]:
        have: set[str] = set()
        for chunk in _chunks(content_keys):
            rows = self._conn.execute(
                "SELECT content_key FROM features WHERE feature_key = ? "
                f"AND content_key IN ({_placeholders(len(chunk))})",
                (feature_key, *chunk),
            )
            have.update(row["content_key"] for row in rows)
        return have

    # -- pair shares ----------------------------------------------------

    def put_pair_shares(self, feature_key: str, shares: Mapping[tuple[str, str], float]) -> None:
        """Remember how much each pair of photos shares part by part (``feature_key``: the
        extractor that compared them). Comparing costs ~0.2 s a photo and 4 ms a pair, so
        a place compared once is not compared again."""
        rows = [(feature_key, *sorted(pair), float(share)) for pair, share in shares.items()]
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO pair_shares (feature_key, key_a, key_b, share) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    def pair_shares_among(
        self, feature_key: str, content_keys: Sequence[str]
    ) -> dict[tuple[str, str], float]:
        """Every remembered share between two of ``content_keys``, keyed (lower, higher)."""
        wanted = set(content_keys)
        found: dict[tuple[str, str], float] = {}
        for chunk in _chunks(sorted(wanted)):
            rows = self._conn.execute(
                "SELECT key_a, key_b, share FROM pair_shares WHERE feature_key = ? "
                f"AND key_a IN ({_placeholders(len(chunk))})",
                (feature_key, *chunk),
            )
            for row in rows:
                if row["key_b"] in wanted:
                    found[(row["key_a"], row["key_b"])] = float(row["share"])
        return found

    # -- failures -------------------------------------------------------

    def record_failure(self, path: Path, stage: str, error: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO failures (path, stage, error, at) VALUES (?, ?, ?, ?)",
                (_key(path), stage, error, _now()),
            )

    def clear_failure(self, path: Path) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM failures WHERE path = ?", (_key(path),))

    def failure_paths_under(self, folder: Path) -> list[Path]:
        clause, params = _folder_clause(folder)
        rows = self._conn.execute(f"SELECT path FROM failures WHERE {clause}", params)
        return [Path(row["path"]) for row in rows]

    def clear_failures(self, paths: Iterable[Path]) -> int:
        names = [_key(p) for p in paths]
        cleared = 0
        with self._conn:
            for chunk in _chunks(names):
                cursor = self._conn.execute(
                    f"DELETE FROM failures WHERE path IN ({_placeholders(len(chunk))})", chunk
                )
                cleared += cursor.rowcount
        return cleared

    def failures(self) -> list[tuple[Path, str, str]]:
        rows = self._conn.execute("SELECT path, stage, error FROM failures ORDER BY path")
        return [(Path(r["path"]), r["stage"], r["error"]) for r in rows]

    # -- feedback -------------------------------------------------------

    def add_feedback(
        self, run_id: str, path: Path, content_key: str, rating: int, note: str = ""
    ) -> None:
        if rating not in (-1, 1):
            raise ValueError(f"rating must be -1 or 1, got {rating}")
        with self._conn:
            self._conn.execute(
                "INSERT INTO feedback (run_id, path, content_key, rating, note, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, _key(path), content_key, rating, note, _now()),
            )

    def feedback(self, run_id: str | None = None) -> list[FeedbackRow]:
        sql = "SELECT run_id, path, content_key, rating, note, created_at FROM feedback"
        params: tuple[str, ...] = ()
        if run_id is not None:
            sql += " WHERE run_id = ?"
            params = (run_id,)
        rows = self._conn.execute(sql + " ORDER BY id", params)
        return [
            FeedbackRow(
                run_id=r["run_id"],
                path=Path(r["path"]),
                content_key=r["content_key"],
                rating=r["rating"],
                note=r["note"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    # -- stats ----------------------------------------------------------

    def stats(self) -> CatalogStats:
        def scalar(sql: str) -> int:
            return int(self._conn.execute(sql).fetchone()[0])

        by_collection = {
            r["collection"]: r["n"]
            for r in self._conn.execute(
                "SELECT collection, COUNT(*) AS n FROM photos GROUP BY collection"
            )
        }
        features = {
            r["feature_key"]: r["n"]
            for r in self._conn.execute(
                "SELECT f.feature_key, COUNT(*) AS n FROM features f "
                "JOIN photos p ON p.content_key = f.content_key GROUP BY f.feature_key"
            )
        }
        return CatalogStats(
            photos=scalar("SELECT COUNT(*) FROM photos"),
            by_collection=by_collection,
            features=features,
            failures=scalar("SELECT COUNT(*) FROM failures"),
            feedback=scalar("SELECT COUNT(*) FROM feedback"),
        )


def _now() -> str:
    return datetime.now(UTC).isoformat()
