from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from calico.looks import BUILT_IN, load_looks
from calico.reading import Hemisphere, Reading, read_description

PLACES = [
    "Marlow River State Hospital - Alderton, NY",
    "Linden State Hospital - Linden, NY",
    "St. Ambrose",
    "Church - Brenton, NJ",
    "Union Central Station - Riverton, MI",
    "Kestrel 21 Body Plant - Riverton, MI",
    "Ashby Sanatorio - Calder, Farland",
]


def read(text: str, hemisphere: Hemisphere = "north") -> Reading:
    return read_description(text, PLACES, hemisphere=hemisphere)


# -- what to avoid ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "looks_for", "avoids"),
    [
        ("a sky without clouds", "a sky", ("clouds",)),
        ("a room without people", "a room", ("people",)),
        ("a room with no people", "a room", ("people",)),
        ("stained glass, no people, no reflections", "stained glass", ("people", "reflections")),
        ("a corridor without debris and with light", "a corridor with light", ("debris",)),
        ("not crowded", "", ("crowded",)),
    ],
)
def test_without_and_no_become_things_to_avoid(
    text: str, looks_for: str, avoids: tuple[str, ...]
) -> None:
    reading = read(text)
    assert reading.looks_for == looks_for
    assert reading.avoids == avoids


def test_words_that_merely_contain_no_are_left_alone() -> None:
    reading = read("a nook by the north window")
    assert reading.looks_for == "a nook by the north window"
    assert reading.avoids == ()


# -- when ----------------------------------------------------------------------------------


def test_seasons_become_months_for_the_hemisphere() -> None:
    assert read("fog in winter").months == frozenset({12, 1, 2})
    assert read("fog in winter").looks_for == "fog"
    assert read("winter fog", hemisphere="south").months == frozenset({6, 7, 8})


def test_fall_and_spring_are_seasons_only_when_they_read_as_time() -> None:
    assert read("the hospital in the fall").months == frozenset({9, 10, 11})
    assert read("autumn leaves").months == frozenset({9, 10, 11})
    waterfall = read("a waterfall")
    assert waterfall.months == frozenset()
    assert waterfall.looks_for == "a waterfall"
    assert read("a spring in the woods").months == frozenset()


def test_months_including_the_ambiguous_may_and_march() -> None:
    assert read("fog in October").months == frozenset({10})
    assert read("october fog").months == frozenset({10})
    assert read("Sept and Oct").months == frozenset({9, 10})
    assert read("light in May").months == frozenset({5})
    assert read("light that may glow").months == frozenset()
    assert read("soldiers march").months == frozenset()


@pytest.mark.parametrize(
    ("text", "years"),
    [
        ("snow in 2021", (2021, 2021)),
        ("the station 2019-2021", (2019, 2021)),
        ("from 2018 to 2020", (2018, 2020)),
        ("nurses in the 1970s", (1970, 1979)),
    ],
)
def test_years_and_ranges(text: str, years: tuple[int, int]) -> None:
    reading = read(text)
    assert reading.years == years
    assert not any(ch.isdigit() for ch in reading.looks_for)


def test_numbers_that_are_not_years_stay_in_the_text() -> None:
    reading = read("room 237")
    assert reading.years is None
    assert reading.looks_for == "room 237"


def test_times_of_day_are_left_to_the_image_model() -> None:
    """Camera clocks are often on another time zone, so the hour cannot be trusted."""
    reading = read("golden hour in the morning")
    assert reading.looks_for == "golden hour in the morning"
    assert reading.months == frozenset()


# -- where ---------------------------------------------------------------------------------


def test_a_place_after_at_in_or_from_narrows_to_matching_locations() -> None:
    reading = read("nursing at St. Ambrose")
    assert reading.places == ("St. Ambrose",)
    assert reading.looks_for == "nursing"
    assert read("a staircase at marlow river state hospital").places == (
        "Marlow River State Hospital - Alderton, NY",
    )


def test_a_city_matches_every_place_in_it() -> None:
    reading = read("graffiti in Riverton")
    assert reading.places == (
        "Kestrel 21 Body Plant - Riverton, MI",
        "Union Central Station - Riverton, MI",
    )
    assert reading.place_words == ("Riverton",)


def test_place_words_without_a_preposition_stay_as_description() -> None:
    """ "Church - Brenton, NJ" is one place; "church interior" means any church."""
    reading = read("church interior")
    assert reading.places == ()
    assert reading.looks_for == "church interior"


def test_unknown_places_and_ordinary_phrases_stay_as_description() -> None:
    assert read("light in a window").looks_for == "light in a window"
    assert read("a castle in Narnia").places == ()


def test_state_codes_alone_are_not_places() -> None:
    assert read("a barn in NY").places == ()


# -- the whole reading ------------------------------------------------------------------------


def test_everything_at_once_and_a_readable_summary() -> None:
    reading = read("fog without people at Linden State Hospital in winter 2021")
    assert reading.looks_for == "fog"
    assert reading.avoids == ("people",)
    assert reading.places == ("Linden State Hospital - Linden, NY",)
    assert reading.months == frozenset({12, 1, 2})
    assert reading.years == (2021, 2021)
    assert reading.summary() == [
        ("looks for", "fog"),
        ("without", "people"),
        ("when", "winter (Dec, Jan, Feb)"),
        ("years", "2021"),
        ("where", "Linden State Hospital"),
    ]
    assert reading.uses_metadata


def test_a_plain_description_is_read_as_is() -> None:
    reading = read("light through broken windows")
    assert reading.looks_for == "light through broken windows"
    assert reading.summary() == [("looks for", "light through broken windows")]
    assert not reading.uses_metadata


@given(st.text(max_size=80))
def test_any_text_reads_without_error_into_tidy_parts(text: str) -> None:
    reading = read(text)
    assert reading.looks_for == reading.looks_for.strip()
    assert all(a == a.strip() and a for a in reading.avoids)
    assert set(reading.places) <= set(PLACES)
    assert reading.months <= set(range(1, 13))


# -- looks ---------------------------------------------------------------------------------

LOOKS = tuple(load_looks(BUILT_IN))


def looks_read(text: str) -> Reading:
    return read_description(text, PLACES, looks=LOOKS)


def test_look_words_are_taken_out_for_measuring() -> None:
    golden = looks_read("golden hour")
    assert [look.name for look in golden.looks] == ["golden hour"]
    assert golden.looks_for == ""
    moody = looks_read("a moody hallway")
    assert [look.name for look in moody.looks] == ["moody"]
    assert moody.looks_for == "a hallway"
    assert moody.summary() == [
        ("looks for", "a hallway"),
        ("look", "moody (darker, more muted)"),
    ]


def test_the_longest_look_words_win_and_case_does_not_matter() -> None:
    reading = looks_read("Harsh Light on a staircase")
    assert [look.name for look in reading.looks] == ["contrasty"]
    assert reading.looks_for == "on a staircase"


def test_avoiding_a_look_reverses_it() -> None:
    reading = looks_read("a hallway, not moody")
    assert reading.avoids == ()
    (look,) = reading.looks
    assert look.name == "not moody"
    assert look.signals == (("brightness", 1), ("saturation", 1))


def test_words_inside_other_words_are_not_looks() -> None:
    assert looks_read("dimension stones").looks == ()
    assert looks_read("a brightly lit room").looks == ()


def test_a_look_word_that_matches_nothing_cannot_hang_the_reader() -> None:
    """An empty word matched the empty string everywhere, so the text grew forever."""
    from calico.looks import Look  # noqa: PLC0415

    empty = Look("x", ("",), (("warmth", 1),))
    assert read_description("a red door", [], looks=[empty]).looks_for == "a red door"


def test_a_list_after_without_says_which_part_is_still_looked_for() -> None:
    """Read as before (measured before it changes): but say that "stairs" is searched for,
    and how to avoid it too."""
    reading = read_description("a sky without fog and stairs", [])
    assert reading.avoids == ("fog",)
    (note,) = [text for kind, text in reading.summary() if kind == "note"]
    assert "stairs" in note
    assert "without stairs" in note


def test_a_long_description_is_read_in_time_proportional_to_it() -> None:
    import time  # noqa: PLC0415

    started = time.perf_counter()
    read_description("in " * 166, ["Old Mill"])
    assert time.perf_counter() - started < 0.2
