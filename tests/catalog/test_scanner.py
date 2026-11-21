from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from calico.catalog.scanner import scan
from calico.errors import FolderNotFoundError

if TYPE_CHECKING:
    from pathlib import Path


def touch(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def rel(root: Path, paths: tuple[Path, ...]) -> list[str]:
    return [p.relative_to(root).as_posix() for p in paths]


def test_scan_is_recursive_sorted_and_case_insensitive(tmp_path: Path) -> None:
    touch(tmp_path, "b/z.JPG", "a.png", "b/y.Heic", "notes.txt", "c/d/e.webp")
    result = scan(tmp_path)
    assert rel(tmp_path, result.images) == ["a.png", "b/y.Heic", "b/z.JPG", "c/d/e.webp"]


def test_scan_excludes_files_and_directories(tmp_path: Path) -> None:
    touch(tmp_path, "keep.jpg", "._junk.jpg", ".hidden/a.jpg", "Cache/b.jpg", "x/c.jpg")
    result = scan(tmp_path, ["._*", ".*", "Cache"])
    assert rel(tmp_path, result.images) == ["keep.jpg", "x/c.jpg"]


def test_scan_does_not_follow_symlinked_directories(tmp_path: Path) -> None:
    real = tmp_path / "real"
    touch(real, "a.jpg")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(real, target_is_directory=True)
    assert scan(root).images == ()


def test_raw_with_rendered_sibling_is_dropped(tmp_path: Path) -> None:
    touch(tmp_path, "IMG_1.CR2", "img_1.jpg", "IMG_2.NEF", "sub/IMG_1.cr2")
    result = scan(tmp_path)
    assert rel(tmp_path, result.images) == ["img_1.jpg"]
    assert rel(tmp_path, result.raw_without_rendered) == ["IMG_2.NEF", "sub/IMG_1.cr2"]


def test_with_raw_support_a_lone_raw_is_a_photo_and_one_beside_a_jpeg_is_not(
    tmp_path: Path,
) -> None:
    touch(tmp_path, "IMG_1.CR3", "img_1.jpg", "IMG_2.CR3", "._IMG_3.CR3")
    result = scan(tmp_path, ["._*"], raw=True)
    assert rel(tmp_path, result.images) == ["IMG_2.CR3", "img_1.jpg"]
    assert result.raw_without_rendered == ()


def test_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FolderNotFoundError, match="does not exist or is not mounted"):
        scan(tmp_path / "gone")


def test_file_instead_of_folder_raises(tmp_path: Path) -> None:
    touch(tmp_path, "a.jpg")
    with pytest.raises(FolderNotFoundError):
        scan(tmp_path / "a.jpg")


def test_unreadable_directory_is_reported(tmp_path: Path) -> None:
    touch(tmp_path, "ok/a.jpg", "locked/b.jpg")
    locked = tmp_path / "locked"
    locked.chmod(0)
    try:
        result = scan(tmp_path)
    finally:
        locked.chmod(0o755)
    assert rel(tmp_path, result.images) == ["ok/a.jpg"]
    assert result.unreadable == (locked,)


def test_paths_are_nfc_normalised(tmp_path: Path) -> None:
    nfd = "Cafe\u0301"
    touch(tmp_path, f"{nfd}/a.jpg")
    (path,) = scan(tmp_path).images
    assert "Caf\u00e9" in path.parts


def test_file_names_are_nfc_normalised_too(tmp_path: Path) -> None:
    touch(tmp_path, "café.jpg")
    (path,) = scan(tmp_path).images
    assert path.name == "café.jpg"


def test_symlinked_files_are_reported_not_listed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    touch(outside, "secret.jpg")
    root = tmp_path / "root"
    touch(root, "real.jpg")
    (root / "link.jpg").symlink_to(outside / "secret.jpg")
    result = scan(root)
    assert rel(root, result.images) == ["real.jpg"]
    assert rel(root, result.links) == ["link.jpg"]


def test_scan_never_lists_what_is_inside_a_skipped_folder(tmp_path: Path) -> None:
    """A calico home inside the archive: its thumbnails are not photos."""
    touch(tmp_path, "a.jpg", "calico/thumbs/ab/abcd.jpg")
    found = scan(tmp_path, skip=[tmp_path / "calico"])
    assert [p.name for p in found.images] == ["a.jpg"]


def test_scan_takes_each_folders_exclusions_from_its_own_collection(tmp_path: Path) -> None:
    touch(tmp_path, "Old Mill/draft1.jpg", "Harbour/draft2.jpg", "Harbour/keep.jpg")

    def exclude_for(folder: Path) -> tuple[str, ...]:
        return ("draft*",) if folder.is_relative_to(tmp_path / "Harbour") else ()

    found = scan(tmp_path, exclude_for=exclude_for)
    assert {p.name for p in found.images} == {"draft1.jpg", "keep.jpg"}


def test_scan_counts_what_it_does_not_read_and_the_links_it_does_not_follow(
    tmp_path: Path,
) -> None:
    touch(tmp_path, "a.jpg", "clip.MOV", "notes.txt", "b.mov", "sidecar.xmp")
    (tmp_path / "elsewhere").mkdir()
    touch(tmp_path / "elsewhere", "c.jpg")
    (tmp_path / "Trip").symlink_to(tmp_path / "elsewhere")
    found = scan(tmp_path)
    assert dict(found.unsupported) == {".mov": 2, ".txt": 1, ".xmp": 1}
    assert found.linked_dirs == (tmp_path / "Trip",)


def test_scan_reads_jpeg_spellings_and_raw_formats_it_used_to_miss(tmp_path: Path) -> None:
    touch(tmp_path, "a.JPE", "b.jfif", "c.PEF", "d.x3f")
    found = scan(tmp_path, raw=True)
    assert {p.name for p in found.images} == {"a.JPE", "b.jfif", "c.PEF", "d.x3f"}
