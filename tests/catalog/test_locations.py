from __future__ import annotations

from pathlib import Path

import pytest

from calico.catalog.locations import location_for
from calico.domain import place_key
from calico.errors import NotInCollectionError
from calico.settings import CollectionConfig

ROOT = Path("/archive")


def collection(**kwargs: object) -> CollectionConfig:
    return CollectionConfig.model_validate({"name": "trips", "root": ROOT, **kwargs})


def test_photo_in_root_uses_collection_name() -> None:
    assert location_for(collection(), ROOT / "a.jpg") == "trips"


def test_default_depth_is_first_folder() -> None:
    assert location_for(collection(), ROOT / "Old Mill" / "sub" / "a.jpg") == "Old Mill"


def test_depth_two_and_clamped_to_available_folders() -> None:
    c = collection(location_depth=2)
    assert location_for(c, ROOT / "2021" / "Old Mill" / "x" / "a.jpg") == "Old Mill"
    assert location_for(c, ROOT / "2021" / "a.jpg") == "2021"


def test_pattern_extracts_location_group() -> None:
    c = collection(location_pattern=r"^(?P<location>.+?) - \w+ \d{4}$")
    assert location_for(c, ROOT / "Old Mill - March 2021" / "a.jpg") == "Old Mill"


def test_pattern_without_match_falls_back_to_folder_name() -> None:
    c = collection(location_pattern=r"^(?P<location>\d+)$")
    assert location_for(c, ROOT / " Old Mill " / "a.jpg") == "Old Mill"


def test_pattern_empty_group_falls_back() -> None:
    c = collection(location_pattern=r"^(?P<location>x*)-")
    assert location_for(c, ROOT / "-Mill" / "a.jpg") == "-Mill"


def test_outside_root_raises() -> None:
    with pytest.raises(NotInCollectionError):
        location_for(collection(), Path("/elsewhere/a.jpg"))


def test_runs_of_spaces_collapse_so_untidy_folder_names_name_one_place() -> None:
    c = collection(location_pattern=r"^(?P<location>.+?)\s*-\s*\w+\s+\d{4}\s*$")
    tidy = location_for(c, ROOT / "Stone Church - Alderton, NY - April 2019" / "a.jpg")
    untidy = location_for(c, ROOT / "Stone  Church - Alderton, NY -  April 2019 " / "a.jpg")
    assert tidy == untidy == "Stone Church - Alderton, NY"


def test_ignored_text_is_removed_before_the_pattern() -> None:
    c = collection(
        location_pattern=r"^(?P<location>.+?)\s*-\s*\w+\s+\d{4}$",
        location_ignore=[r"\([^)]*\)"],
    )
    after = ROOT / "State Hospital (West Wing) - Alderton, NY - May 2020" / "a.jpg"
    before = ROOT / "State Hospital - Alderton, NY - March 2019" / "a.jpg"
    assert location_for(c, after) == location_for(c, before) == "State Hospital - Alderton, NY"


def test_a_folder_that_is_only_ignored_text_keeps_its_name() -> None:
    c = collection(location_ignore=[r"\([^)]*\)"])
    assert location_for(c, ROOT / "(misc)" / "a.jpg") == "(misc)"


def test_ignore_patterns_must_be_valid_regexes() -> None:
    with pytest.raises(ValueError, match="location_ignore"):
        collection(location_ignore=["(unclosed"])


def test_aliases_rename_a_place_matching_regardless_of_capitals_and_spaces() -> None:
    aliases = {"cobrett  building": "Corbett Building"}
    assert location_for(collection(), ROOT / "Cobrett Building" / "a.jpg", aliases) == (
        "Corbett Building"
    )
    assert location_for(collection(), ROOT / "Other" / "a.jpg", aliases) == "Other"


def test_place_key_ignores_capitals_and_spaces() -> None:
    assert place_key("  Grove Catholic HIgh   School ") == place_key("grove catholic high school")


# -- the deepest folder that names a place ------------------------------------------------

TRIPS = {
    "location_depth": 2,
    "location_deepest": True,
    "location_not_places": [r"(Un)?edited", r"Lightroom.*", r"Originals", r"Unsorted.*"],
    "location_ignore": [r"\bDay\b.*$"],
}


def test_the_deepest_folder_that_names_a_place_is_the_location() -> None:
    c = collection(**TRIPS)
    trip = ROOT / "Abroad" / "Lowlands & Coast - June 2015"
    assert location_for(c, trip / "Unedited" / "Calder" / "a.jpg") == "Calder"
    assert location_for(c, trip / "Calder" / "Unedited" / "a.jpg") == "Calder"


def test_folders_that_only_organise_are_passed_over_up_to_the_trip() -> None:
    c = collection(**TRIPS)
    trip = ROOT / "Abroad" / "Riverton & Hill Town, Farland"
    assert location_for(c, trip / "Lightroom Edits" / "a.jpg") == "Riverton & Hill Town, Farland"
    assert location_for(c, trip / "a.jpg") == "Riverton & Hill Town, Farland"


def test_a_folder_is_cleaned_before_it_names_a_place() -> None:
    c = collection(**TRIPS)
    trip = ROOT / "Abroad" / "Calder & Vane, Farland" / "Calder Day 2 (5:8)"
    assert location_for(c, trip / "Lightroom Edits" / "a.jpg") == "Calder"


def test_a_folder_cleaned_to_nothing_is_passed_over() -> None:
    c = collection(**TRIPS)
    assert location_for(c, ROOT / "Abroad" / "Farland 2019" / "Day 3" / "a.jpg") == "Farland 2019"


def test_not_places_match_whole_names_regardless_of_case() -> None:
    c = collection(**TRIPS)
    trip = ROOT / "Abroad" / "Farland 2019"
    assert location_for(c, trip / "EDITED" / "a.jpg") == "Farland 2019"
    assert location_for(c, trip / "Edited Calder" / "a.jpg") == "Edited Calder"


def test_with_no_place_below_it_the_folder_at_the_depth_is_the_location_as_before() -> None:
    c = collection(**TRIPS)
    assert location_for(c, ROOT / "Abroad" / "Unedited" / "a.jpg") == "Unedited"


def test_without_deepest_the_folder_at_the_depth_is_the_location_as_before() -> None:
    c = collection(location_depth=2, location_not_places=[r"Originals"])
    assert location_for(c, ROOT / "Abroad" / "Rome 2019" / "Colosseum" / "a.jpg") == "Rome 2019"


def test_a_not_place_that_is_not_a_regex_is_refused() -> None:
    with pytest.raises(ValueError, match="location_not_places"):
        collection(location_not_places=["(unclosed"])


def test_an_alias_typed_in_decomposed_unicode_still_renames_the_place() -> None:
    """Pasted from Finder, a name is often NFD; the scanner hands out NFC."""
    aliases = {"Cafe\u0301 Lumie\u0300re": "Coffee"}
    assert (
        location_for(collection(), ROOT / "Caf\u00e9 Lumi\u00e8re" / "a.jpg", aliases) == "Coffee"
    )
    assert place_key("Cafe\u0301") == place_key("Caf\u00e9")
