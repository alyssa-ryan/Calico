from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from calico.paths import missing_folder, normalise


def nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def case_insensitive(folder: Path) -> bool:
    probe = folder / "CaseProbe"
    probe.mkdir()
    try:
        return (folder / "caseprobe").exists()
    finally:
        probe.rmdir()


def test_a_decomposed_name_is_spelt_as_the_catalog_spells_it(tmp_path: Path) -> None:
    """Finder and some tools hand out NFD; the catalog and collection roots are NFC."""
    (tmp_path / "Zoë").mkdir()
    assert str(normalise(tmp_path / nfd("Zoë"))) == unicodedata.normalize(
        "NFC", str(tmp_path / "Zoë")
    )


def test_a_folder_typed_in_another_case_is_spelt_as_the_disk_spells_it(tmp_path: Path) -> None:
    if not case_insensitive(tmp_path):
        pytest.skip("this volume tells capitals apart, so 'old mill' is another folder")
    (tmp_path / "Old Mill" / "March").mkdir(parents=True)
    assert normalise(tmp_path / "old mill" / "MARCH") == (tmp_path / "Old Mill" / "March").resolve()


def test_a_folder_that_does_not_exist_is_kept_as_typed(tmp_path: Path) -> None:
    """An unmounted drive: nothing to look up, and the caller says it is missing."""
    missing = tmp_path / "Unmounted Drive" / nfd("Zoë")
    assert normalise(missing) == (tmp_path / "Unmounted Drive").resolve() / "Zoë"


def test_links_and_home_are_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert normalise(tmp_path / "link") == (tmp_path / "real").resolve()
    assert normalise(Path("~/real")) == (tmp_path / "real").resolve()


def test_a_folder_whose_name_ends_in_spaces_is_named_exactly_when_typed_without_them(
    tmp_path: Path,
) -> None:
    """7 November 2026: the folder's name ended in two spaces, so the path typed
    without them was "not mounted", though the drive was mounted and the folder there."""
    (tmp_path / "Old Mill - 2016  ").mkdir()
    message = missing_folder(tmp_path / "Old Mill - 2016")
    assert f"{tmp_path / 'Old Mill - 2016  '}" in message
    assert "2 spaces at the end" in message
    assert "not mounted" not in message


def test_the_rest_of_the_path_is_kept_after_the_folder_spelt_otherwise(tmp_path: Path) -> None:
    (tmp_path / " Trip  2016" / "Tilden").mkdir(parents=True)
    message = missing_folder(tmp_path / "Trip 2016" / "Tilden")
    assert str(tmp_path / " Trip  2016" / "Tilden") in message
    assert "other spaces" in message


def test_a_folder_with_no_near_name_is_missing_or_not_mounted(tmp_path: Path) -> None:
    (tmp_path / "Harbor").mkdir()
    assert (
        missing_folder(tmp_path / "Mill")
        == f"{tmp_path / 'Mill'} does not exist or is not mounted."
    )
    unmounted = Path("/Volumes/No Such Drive For Calico/EXPLORE")
    assert missing_folder(unmounted) == f"{unmounted} does not exist or is not mounted."


def test_a_name_typed_with_spaces_the_folder_lacks_is_named_too(tmp_path: Path) -> None:
    (tmp_path / "Mill").mkdir()
    message = missing_folder(tmp_path / "Mill ")
    assert f"{tmp_path / 'Mill'} does" in message
    assert "1 fewer space at the end" in message
