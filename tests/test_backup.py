"""``calico backup``: what cannot be recomputed, in one archive (plan, Workstream 2c)."""

from __future__ import annotations

import gzip
import io
import sqlite3
import tarfile
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from calico.backup import backup, last_backup, verify
from calico.catalog.store import Catalog
from calico.domain import SelectionRequest, SelectionResult
from calico.errors import BackupError
from calico.runs import RunStore
from calico.settings import CollectionConfig, Settings

if TYPE_CHECKING:
    from pathlib import Path

NOW = datetime(2026, 10, 29, 14, 30, 5, tzinfo=UTC)


@pytest.fixture
def settings(tmp_path: Path, calico_home: Path) -> Settings:
    """A home holding one of everything, and a collection beside it."""
    archive = tmp_path / "archive"
    archive.mkdir()
    home = calico_home
    for name, text in {
        "ratings/2026-10.jsonl": '{"use": "taste"}\n',
        "ratings/.lock": "",
        "labels/20261001-090000-tast.json": "{}",
        "labels/references/20261002-same.jpg": "jpeg",
        "labels/.trash/20261003-gone.20261004.json": "{}",
        "labels/.lock": "",
        "labels/.20261001-090000-tast.json.123.ab.tmp": "half",
        "models/taste.npz": "weights",
        "calico.toml": "[settings]\n",
        "looks.toml": "",
        "holdout.toml": "places = []\n",
        "experiments.jsonl": "{}\n",
        "taxonomies/mine.toml": "",
        "thumbs/ab/abcd.jpg": "thumb",
        "uploads/used/00-anchor.jpg": "jpeg",
        "uploads/stale/00-other.jpg": "jpeg",
        "uploads/moved/00-copied.jpg": "jpeg",
    }.items():
        path = home / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    runs = RunStore(home / "runs")
    for run_id, refs in (
        ("20261005-100000-trip", (home / "uploads" / "used" / "00-anchor.jpg",)),
        ("20261005-110000-pruned", (home / "uploads" / "pruned" / "00-x.jpg",)),
        # a reference photo from the archive, in a folder that happens to be called uploads
        ("20261005-113000-ref", (archive / "uploads" / "IMG_1.jpg",)),
        # saved when the home was elsewhere (it was copied or restored here since)
        ("20261005-120000-moved", (tmp_path / "old-home" / "uploads" / "moved" / "00-copied.jpg",)),
    ):
        runs.save(
            SelectionResult(
                run_id=run_id,
                created_at=NOW,
                request=SelectionRequest(task="triptych", folders=(archive,), references=refs),
                groups=(),
            ),
            [],
            {},
        )
    with Catalog(home / "catalog.sqlite") as catalog:
        catalog.add_feedback("20261005-100000-trip", archive / "a.jpg", "a", 1)
    return Settings(home=home, collections=(CollectionConfig(name="archive", root=archive),))


def names(path: Path) -> set[str]:
    with tarfile.open(path) as tar:
        return {m.name for m in tar.getmembers() if m.isfile()}


def test_a_backup_holds_what_cannot_be_recomputed(settings: Settings, tmp_path: Path) -> None:
    made = backup(settings, tmp_path / "backups", clock=lambda: NOW)
    assert made.path == tmp_path / "backups" / "calico-backup-20261029-143005.tar.gz"
    assert names(made.path) == {
        "manifest.json",
        "ratings/2026-10.jsonl",
        "labels/20261001-090000-tast.json",
        "labels/references/20261002-same.jpg",
        "labels/.trash/20261003-gone.20261004.json",
        "runs/20261005-100000-trip/result.json",
        "runs/20261005-100000-trip/signals.csv",
        "runs/20261005-110000-pruned/result.json",
        "runs/20261005-110000-pruned/signals.csv",
        "runs/20261005-113000-ref/result.json",
        "runs/20261005-113000-ref/signals.csv",
        "runs/20261005-120000-moved/result.json",
        "runs/20261005-120000-moved/signals.csv",
        "uploads/moved/00-copied.jpg",
        "models/taste.npz",
        "calico.toml",
        "looks.toml",
        "holdout.toml",
        "experiments.jsonl",
        "taxonomies/mine.toml",
        "uploads/used/00-anchor.jpg",  # a saved run points to it; the other upload is not kept
    }
    assert made.uploads_gone == 1  # pruned after a week, before this backup
    assert not made.catalog
    assert {p.name: p.files for p in made.parts}["labels"] == 3


def test_the_catalog_is_added_through_vacuum_into_when_asked(
    settings: Settings, tmp_path: Path
) -> None:
    made = backup(settings, tmp_path / "backups", with_catalog=True, clock=lambda: NOW)
    assert made.catalog
    with tarfile.open(made.path) as tar:
        tar.extract("catalog.sqlite", tmp_path / "out", filter="data")
    with sqlite3.connect(tmp_path / "out" / "catalog.sqlite") as db:
        assert db.execute("SELECT count(*) FROM feedback").fetchone() == (1,)
    assert not list((tmp_path / "backups").glob(".*"))  # no scratch left behind


def test_a_second_backup_never_replaces_the_first(settings: Settings, tmp_path: Path) -> None:
    first = backup(settings, tmp_path / "b", clock=lambda: NOW)
    second = backup(settings, tmp_path / "b", clock=lambda: NOW)
    assert first.path != second.path
    assert first.path.is_file()
    assert second.path.is_file()


@pytest.mark.parametrize("where", ["home", "archive"])
def test_a_destination_inside_the_home_or_a_collection_is_refused(
    settings: Settings, tmp_path: Path, where: str
) -> None:
    dest = (settings.home if where == "home" else tmp_path / "archive") / "backups"
    with pytest.raises(BackupError, match="outside"):
        backup(settings, dest, clock=lambda: NOW)
    assert not dest.exists()


def test_verify_lists_what_a_restore_brings_back(settings: Settings, tmp_path: Path) -> None:
    made = backup(settings, tmp_path / "b", clock=lambda: NOW)
    checked = verify(made.path)
    assert checked.ok
    assert checked.created_at == NOW
    assert {p.name: p.files for p in checked.parts} == {p.name: p.files for p in made.parts}


def test_verify_finds_a_damaged_archive(settings: Settings, tmp_path: Path) -> None:
    made = backup(settings, tmp_path / "b", clock=lambda: NOW)
    data = made.path.read_bytes()
    made.path.write_bytes(data[: len(data) // 2])
    with pytest.raises(BackupError, match="cannot be read"):
        verify(made.path)


def test_verify_finds_a_file_the_manifest_lists_but_the_archive_lacks(
    settings: Settings, tmp_path: Path
) -> None:
    made = backup(settings, tmp_path / "b", clock=lambda: NOW)
    with tarfile.open(made.path) as tar:
        kept = [(m, tar.extractfile(m).read()) for m in tar.getmembers() if m.isfile()]  # type: ignore[union-attr]
    short = tmp_path / "short.tar.gz"
    with gzip.open(short, "wb") as raw, tarfile.open(fileobj=raw, mode="w") as tar:
        for member, content in kept:
            if member.name != "models/taste.npz":
                tar.addfile(member, fileobj=io.BytesIO(content))
    checked = verify(short)
    assert not checked.ok
    assert any("models" in problem for problem in checked.problems)


def test_the_last_backup_is_noted_in_the_home(settings: Settings, tmp_path: Path) -> None:
    assert last_backup(settings) is None
    made = backup(settings, tmp_path / "b", clock=lambda: NOW)
    noted = last_backup(settings)
    assert noted is not None
    assert (noted.at, noted.path) == (NOW, made.path)


def test_a_damaged_note_reads_as_no_backup(settings: Settings) -> None:
    (settings.home / "last_backup.json").write_text("{nope", encoding="utf-8")
    assert last_backup(settings) is None


def test_the_note_is_the_newest_backup(settings: Settings, tmp_path: Path) -> None:
    backup(settings, tmp_path / "b", clock=lambda: NOW - timedelta(days=40))
    backup(settings, tmp_path / "b", clock=lambda: NOW)
    noted = last_backup(settings)
    assert noted is not None
    assert noted.at == NOW


def archive_of(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return path


def test_verify_flags_a_file_that_would_land_outside_the_home(tmp_path: Path) -> None:
    made = archive_of(tmp_path / "evil.tar.gz", {"manifest.json": b"{}", "../outside": b"x"})
    checked = verify(made)
    assert any("outside the home" in problem for problem in checked.problems)


def test_verify_says_when_there_is_no_manifest(tmp_path: Path) -> None:
    made = archive_of(tmp_path / "bare.tar.gz", {"labels/a.json": b"{}"})
    checked = verify(made)
    assert not checked.ok
    assert checked.created_at is None
    assert any("manifest.json is missing" in problem for problem in checked.problems)
