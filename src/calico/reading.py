"""Reading a "Describe it" text into parts. Pure functions; no I/O, no models.

CLIP reads a description as one blurred meaning: it cannot tell "a sky
*without* clouds" from "a sky with clouds", and it knows nothing of when a
photo was taken or which of your places it shows. So before CLIP sees the
text, this module takes out what other information answers better:

* **what to avoid** -- "without X", "no X", "not X" -- scored as a penalty;
* **when** -- months, seasons (for the configured hemisphere) and years,
  matched against capture dates;
* **where** -- a place name after "at", "in", "from" or "near", matched
  against your location names.

What remains is what CLIP looks for. Times of day ("golden hour", "morning")
deliberately stay with CLIP: camera clocks are often on another time zone, so
the recorded hour cannot be trusted. Ambiguous words are only taken when they
clearly mean a date or place: "May" and "March" need a capital or "in";
"fall" and "spring" need "in"/"during"; a one-word place needs a capital
(so "a wedding in church" never narrows to a place named "Church").
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from calico.looks import Look

type Hemisphere = Literal["north", "south"]

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
SHORT_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
AMBIGUOUS_MONTHS = frozenset({"may", "march"})
"""Ordinary words too ("that may glow", "soldiers march"); also every abbreviation."""

NORTHERN_SEASONS: dict[str, tuple[int, ...]] = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
    "fall": (9, 10, 11),
}
AMBIGUOUS_SEASONS = frozenset({"spring", "fall"})
"""A waterfall, a spring in the woods: seasons only after "in"/"during"."""

_TIME_PREP = r"(?:\b(?:in|during|of|from|between)\s+(?:the\s+)?)"
_NEGATION = re.compile(
    r"\b(?:without|with\s+no|no|not|excluding|except(?:\s+for)?)\s+(?P<phrase>.+?)"
    r"(?P<tail>\s*(?:,|;|\band\b)|(?=\s+(?:but|with|at|in|from|near)\b)|\s*$)",
    re.IGNORECASE,
)
_YEAR_RANGE = re.compile(
    rf"{_TIME_PREP}?\b(?P<a>(?:19|20)\d\d)\s*(?:-|to|through|and)\s*(?P<b>(?:19|20)\d\d)\b",
    re.IGNORECASE,
)
_DECADE = re.compile(rf"{_TIME_PREP}?\b(?:the\s+)?(?P<d>(?:19|20)\d0)s\b", re.IGNORECASE)
_YEAR = re.compile(rf"{_TIME_PREP}?\b(?P<y>(?:19|20)\d\d)\b", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z]+")
_TOKEN = re.compile(r"\S+")
_PLACE_PREP = re.compile(r"\b(?:at|in|from|near)\s+(?:the\s+)?", re.IGNORECASE)
_CONNECTORS = r"(?:and|with|but|at|in|from|near|of|during|between)"
MAX_PLACE_WORDS = 8
MIN_PLACE_PART = 4
"""Shorter parts of a location name (NY, MI) are never places on their own."""


@dataclass(frozen=True)
class Reading:
    """How a description was read. ``looks_for`` is what is left for the image model."""

    text: str
    looks_for: str
    avoids: tuple[str, ...] = ()
    months: frozenset[int] = frozenset()
    seasons: tuple[str, ...] = ()
    years: tuple[int, int] | None = None
    places: tuple[str, ...] = ()
    """Location names (as catalogued) that the place words matched."""
    place_words: tuple[str, ...] = ()
    """The place words as typed."""
    looks: tuple[Look, ...] = ()
    """Light, colour and mood words, answered by measuring photos (``calico.looks``)."""
    cautions: tuple[str, ...] = ()
    """What a person might expect that this reading does not do, and how to say it."""
    _season_months: dict[str, tuple[int, ...]] = field(default_factory=dict, repr=False)

    @property
    def uses_metadata(self) -> bool:
        return bool(self.months or self.years or self.places)

    def summary(self) -> list[tuple[str, str]]:
        """(kind, text) pairs, in the order a person would say them."""
        parts: list[tuple[str, str]] = []
        if self.looks_for:
            parts.append(("looks for", self.looks_for))
        if self.avoids:
            parts.append(("without", ", ".join(self.avoids)))
        parts += [("look", f"{look.name} ({look.describe()})") for look in self.looks]
        if self.months:
            parts.append(("when", self._when()))
        if self.years:
            a, b = self.years
            parts.append(("years", str(a) if a == b else f"{a} to {b}"))
        if self.place_words:
            parts.append(("where", ", ".join(self.place_words)))
        parts += [("note", caution) for caution in self.cautions]
        return parts

    def _when(self) -> str:
        described: list[str] = []
        covered: set[int] = set()
        for season in self.seasons:
            months = self._season_months[season]
            covered.update(months)
            described.append(f"{season} ({', '.join(MONTH_NAMES[m - 1] for m in months)})")
        described += [MONTH_NAMES[m - 1] for m in sorted(self.months - covered)]
        return ", ".join(described)


def read_description(
    text: str,
    places: Iterable[str],
    *,
    hemisphere: Hemisphere = "north",
    looks: Sequence[Look] = (),
) -> Reading:
    """Split ``text`` into what to look for, avoid, and when and where it was taken.

    ``places`` are the location names that may be meant (usually those of the
    photos being searched); ``looks`` the light, colour and mood vocabulary.
    """
    rest = " ".join(text.split())
    rest, avoids, cautions = _take_negations(rest)
    rest, years = _take_years(rest)
    seasons_by_name = _seasons(hemisphere)
    rest, seasons = _take_seasons(rest, seasons_by_name)
    rest, months = _take_months(rest)
    rest, matched, words = _take_places(rest, places)
    avoids, negated = _negated_looks(avoids, looks)
    rest, found_looks = _take_looks(rest, looks)
    for season in seasons:
        months |= set(seasons_by_name[season])
    return Reading(
        text=text,
        looks_for=_tidy(rest),
        avoids=avoids,
        months=frozenset(months),
        seasons=seasons,
        years=years,
        places=matched,
        place_words=words,
        looks=(*found_looks, *negated),
        cautions=cautions,
        _season_months={s: seasons_by_name[s] for s in seasons},
    )


def _take_negations(text: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """What to avoid, and a caution for each list a negation stops short of: "without fog
    and stairs" avoids fog and looks for stairs. Carrying the negation on is a reading to
    measure with calico eval criteria before it changes; until then, say so."""
    avoids: list[str] = []
    cautions: list[str] = []
    while match := _NEGATION.search(text):
        phrase = _tidy(match.group("phrase"))
        if phrase:
            avoids.append(phrase)
        if phrase and match.group("tail").strip():
            following = _tidy(" ".join(text[match.end() :].split()[:3]))
            if following:
                cautions.append(
                    f"“{following}” after “{phrase}” is looked for, not avoided; to avoid "
                    f"it too, say “without {phrase}, without {following}”"
                )
        text = text[: match.start()] + " " + text[match.end() :]
    return text, tuple(avoids), tuple(cautions)


def _take_years(text: str) -> tuple[str, tuple[int, int] | None]:
    if match := _YEAR_RANGE.search(text):
        a, b = sorted((int(match.group("a")), int(match.group("b"))))
        return _cut(text, match), (a, b)
    if match := _DECADE.search(text):
        start = int(match.group("d"))
        return _cut(text, match), (start, start + 9)
    found: list[int] = []
    while match := _YEAR.search(text):
        found.append(int(match.group("y")))
        text = _cut(text, match)
    return text, (min(found), max(found)) if found else None


def _seasons(hemisphere: Hemisphere) -> dict[str, tuple[int, ...]]:
    if hemisphere == "north":
        return NORTHERN_SEASONS
    shift = {m: (m + 5) % 12 + 1 for m in range(1, 13)}  # six months on
    return {name: tuple(shift[m] for m in months) for name, months in NORTHERN_SEASONS.items()}


def _take_seasons(text: str, seasons: dict[str, tuple[int, ...]]) -> tuple[str, tuple[str, ...]]:
    found: list[str] = []
    for name in seasons:
        plain = rf"{_TIME_PREP}?\b{name}\b"
        pattern = rf"{_TIME_PREP}\b{name}\b" if name in AMBIGUOUS_SEASONS else plain
        while match := re.search(pattern, text, re.IGNORECASE):
            if name not in found:
                found.append(name)
            text = _cut(text, match)
    return text, tuple(found)


def _take_months(text: str) -> tuple[str, set[int]]:
    found: set[int] = set()
    for word, month in [*MONTHS.items(), *SHORT_MONTHS.items()]:
        ambiguous = word in AMBIGUOUS_MONTHS or word in SHORT_MONTHS
        pattern = re.compile(rf"(?P<prep>{_TIME_PREP})?\b(?P<w>{word})\b\.?", re.IGNORECASE)
        position = 0
        while match := pattern.search(text, position):
            capitalised = match.group("w")[0].isupper()
            if ambiguous and not (match.group("prep") or capitalised):
                position = match.end()
                continue
            found.add(month)
            text = _cut(text, match)
    return text, found


def _place_key(name: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", name).split()).casefold()


def _candidates(places: Iterable[str]) -> dict[str, set[str]]:
    """Ways to name each place: the whole name, its head and each part ("Riverton")."""
    by_key: dict[str, set[str]] = {}
    for place in places:
        parts = {place, place.split(" - ")[0], *re.split(r"\s+-\s+|,", place)}
        for part in parts:
            key = _place_key(part)
            if len(key) >= MIN_PLACE_PART:
                by_key.setdefault(key, set()).add(place)
    return by_key


def _take_places(text: str, places: Iterable[str]) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    candidates = _candidates(places)
    matched: set[str] = set()
    words: list[str] = []
    position = 0
    while prep := _PLACE_PREP.search(text, position):
        # Only the next few words can be a place: reading them all, at every "in", made a
        # long description quadratic (60,000 characters took 46 s).
        tokens = list(itertools.islice(_TOKEN.finditer(text, prep.end()), MAX_PLACE_WORDS))
        hit = None
        for n in range(len(tokens), 0, -1):
            span = text[prep.end() : tokens[n - 1].end()].rstrip(".,;:")
            key = _place_key(span)
            proper = n > 1 or span[:1].isupper()
            if key in candidates and proper:
                hit = (span, key, tokens[n - 1].end())
                break
        if hit is None:
            position = prep.end()
            continue
        span, key, end = hit
        matched |= candidates[key]
        words.append(span)
        text = text[: prep.start()] + " " + text[end:]
    return text, tuple(sorted(matched)), tuple(words)


def _look_pattern(word: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)


def _take_looks(text: str, looks: Sequence[Look]) -> tuple[str, tuple[Look, ...]]:
    """Take out look words, longest first ("harsh light" before "light")."""
    words = sorted(((w, look) for look in looks for w in look.words), key=lambda t: -len(t[0]))
    found: list[Look] = []
    for word, look in words:
        pattern = _look_pattern(word)
        while (match := pattern.search(text)) and match.end() > match.start():
            if look not in found:
                found.append(look)
            text = _cut(text, match)
    return text, tuple(found)


def _negated_looks(
    avoids: tuple[str, ...], looks: Sequence[Look]
) -> tuple[tuple[str, ...], tuple[Look, ...]]:
    """ "not moody": an avoided look word becomes that look reversed."""
    by_word = {w.casefold(): look for look in looks for w in look.words}
    kept: list[str] = []
    negated: list[Look] = []
    for phrase in avoids:
        look = by_word.get(phrase.casefold())
        if look is None:
            kept.append(phrase)
        else:
            negated.append(look.negated())
    return tuple(kept), tuple(negated)


def _cut(text: str, match: re.Match[str]) -> str:
    return text[: match.start()] + " " + text[match.end() :]


def _tidy(text: str) -> str:
    """Collapse spaces and drop connecting words and punctuation left dangling at the ends."""
    text = " ".join(text.split())
    text = re.sub(r"\s+([,;])", r"\1", text)
    text = re.sub(r"([,;])(?:\s*[,;])+", r"\1", text)
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"^(?:[,;]\s*|(?:and|but)\b\s*)", "", text, flags=re.IGNORECASE)
        text = re.sub(rf"(?:\s*[,;.]|\s+{_CONNECTORS}\b)$", "", text, flags=re.IGNORECASE)
        text = text.strip()
    return text
