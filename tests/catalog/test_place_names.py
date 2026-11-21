from __future__ import annotations

import pytest

from calico.catalog.place_names import (
    SamePlace,
    WordSpellings,
    edit_distance,
    has_stray_spaces,
    similar_places,
    word_spellings,
)


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [
        ("", "", 0),
        ("abc", "abc", 0),
        ("Corbett", "Cobrett", 1),  # two letters swapped is one slip, not two
        ("Museum", "Musuem", 1),
        ("Sanatorium", "Sanitarium", 2),
        ("kitten", "sitting", 3),
        ("abc", "", 3),
    ],
)
def test_edit_distance_counts_a_swap_as_one_edit(a: str, b: str, distance: int) -> None:
    assert edit_distance(a, b) == distance
    assert edit_distance(b, a) == distance


def test_a_misspelt_place_is_grouped_with_the_spelling_that_has_more_photos() -> None:
    counts = {
        "Corbett Building - Marlow, OH": 40,
        "Cobrett Building - Marlow, OH": 12,
        "Lyric Theatre - Marlow, OH": 30,
    }
    assert similar_places(counts) == [
        SamePlace(
            names=("Corbett Building - Marlow, OH", "Cobrett Building - Marlow, OH"),
            photos=(40, 12),
            reasons=("spelling",),
        )
    ]


def test_capitals_spacing_and_bracketed_notes_are_named_as_the_reason() -> None:
    counts = {
        "Grove Catholic High School": 5,
        "Grove Catholic HIgh School": 9,
        "State Hospital (West Wing) - Alderton, NY": 7,
        "State Hospital - Alderton, NY": 20,
    }
    groups = {g.names[0]: g for g in similar_places(counts)}
    # Irregular capitals are never the suggested name, even with more photos.
    assert groups["Grove Catholic High School"].reasons == ("capitals or spacing",)
    assert groups["State Hospital - Alderton, NY"].reasons == ("a note in brackets",)


def test_names_that_differ_in_their_numbers_are_different_places() -> None:
    counts = {"Kestrel 21 Body Plant": 5, "Kestrel 18 Body Plant": 5, "Pier 1": 2, "Pier 2": 2}
    assert similar_places(counts) == []


def test_unrelated_short_names_are_not_grouped() -> None:
    counts = {"Mill": 3, "Hill": 3, "Church - Brenton, NJ": 4, "Church - Kenton, MI": 4}
    assert similar_places(counts) == []


def test_variants_chain_into_one_group_ordered_by_photos_then_name() -> None:
    counts = {"Sanatorium Hill": 2, "Sanitorium Hill": 2, "Sanitarium Hill": 9}
    (group,) = similar_places(counts)
    assert group.names == ("Sanitarium Hill", "Sanatorium Hill", "Sanitorium Hill")
    assert group.photos == (9, 2, 2)


def test_words_spelled_more_than_one_way_across_different_places() -> None:
    names = [
        "Ashby Sanatorium - Linden, OH",
        "Pine Ridge Sanatorium - Linden, KY",
        "Mount Sanitarium - Carver, NY",
        "Lake Sanitorium - Erie, PA",
        "Parish Musuem - Fairview, OH",
        "Art Museum - Riverton, MI",
    ]
    found = {w.spellings[0]: w for w in word_spellings(names)}
    assert found["Sanatorium"] == WordSpellings(
        spellings=("Sanatorium", "Sanitarium", "Sanitorium"),
        places=(2, 1, 1),
        examples=(names[0], names[2], names[3]),
    )
    assert set(found["Museum"].spellings) == {"Museum", "Musuem"}


def test_plurals_and_ordinary_capitals_are_not_spelling_variants() -> None:
    names = ["Grain Mills", "Grain Mill", "Old Station", "old station", "STATION ROAD"]
    assert word_spellings(names) == []


def test_irregular_capitals_in_a_word_are_reported() -> None:
    names = ["Grove Catholic HIgh School", "Hillcrest High School"]
    (found,) = word_spellings(names)
    assert found.spellings == ("High", "HIgh")


@pytest.mark.parametrize(
    ("name", "stray"),
    [("Old Mill", False), ("Old Mill ", True), (" Old Mill", True), ("Old  Mill", True)],
)
def test_stray_spaces(name: str, stray: bool) -> None:
    assert has_stray_spaces(name) is stray


def test_a_word_linked_earlier_keeps_its_capitals_reason() -> None:
    names = ["Old Musuem", "Old Museum", "Old MuSeum"]
    (found,) = word_spellings(names)
    assert set(found.spellings) == {"Museum", "Musuem", "MuSeum"}
    assert found.spellings[-1] == "MuSeum"


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Power Plant", "Tower Plant"),  # a different first letter is a different word
        ("Bridge Street", "Ridge Street"),
        ("Silver Planet", "Plane Hangar"),  # one word plus an ending is another word
        ("State Hospital", "Glen Estate"),
        ("Developmental Center", "Development Office"),
    ],
)
def test_different_words_that_happen_to_be_close_are_not_spelling_variants(a: str, b: str) -> None:
    assert word_spellings([a, b]) == []


def test_a_name_cut_off_mid_word_is_grouped_with_the_full_name() -> None:
    counts = {
        "Harbor Terminal - Deerfield, MI": 80,
        "Harbor Terminal - De": 6,
        "Farland": 10,
        "Farland, Vale & Coast": 40,  # a whole word, then more: a different place
    }
    assert similar_places(counts) == [
        SamePlace(
            names=("Harbor Terminal - Deerfield, MI", "Harbor Terminal - De"),
            photos=(80, 6),
            reasons=("name cut short",),
        )
    ]


def test_a_very_short_name_is_not_taken_as_the_start_of_longer_ones() -> None:
    counts = {"ME": 55, "Mercy Medical Center": 59, "Medical Arts Building": 3}
    assert similar_places(counts) == []
