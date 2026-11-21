"""Find location names that are probably one place spelt more than one way.

Folder names are typed by hand over years, so one place picks up typos, stray
capitals and notes like "(after the fire)". Calico never renames folders (the
archive is read-only), and merging names automatically would wrongly join
places that merely look alike ("Mill Pond", "Mill Road"). So this module only
*suggests*: ``calico locations`` shows the groups, and the user confirms each
one as a ``location_aliases`` entry or a collection's ``location_ignore``.
Pure functions; no I/O.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from calico.domain import place_key

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

CAPITALS = "capitals or spacing"
BRACKETS = "a note in brackets"
SPELLING = "spelling"
CUT_SHORT = "name cut short"

_BRACKETED = re.compile(r"[(\[][^)\]]*[)\]]")
_DIGITS = re.compile(r"\d+")
_WORD = re.compile(r"[^\W\d_]+")
MIN_SPELT_WORD = 5
"""Shorter words are too easily one letter apart by chance (Mill, Hill)."""
MIN_PLACE_NAME = 8
"""Shorter place names are never taken as misspelt or cut short ("ME" is not "Memorial")."""


@dataclass(frozen=True)
class SamePlace:
    """Location names that probably mean one place; the suggested name first."""

    names: tuple[str, ...]
    photos: tuple[int, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class WordSpellings:
    """One word spelt several ways across place names; the most used spelling first."""

    spellings: tuple[str, ...]
    places: tuple[int, ...]
    """How many place names use each spelling."""
    examples: tuple[str, ...]
    """A place name using each spelling."""


def edit_distance(a: str, b: str) -> int:
    """Insertions, deletions, substitutions and swaps of two neighbouring letters.

    A swap counts once, so "Cobrett" is one slip from "Corbett", as a person
    would judge it (optimal string alignment distance).
    """
    previous2: list[int] = []
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            )
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                current[j] = min(current[j], previous2[j - 2] + 1)
        previous2, previous = previous, current
    return previous[-1]


def has_stray_spaces(name: str) -> bool:
    """Leading, trailing or doubled spaces, which Calico ignores but a folder still has."""
    return name != " ".join(name.split())


def similar_places(counts: Mapping[str, int]) -> list[SamePlace]:
    """Groups of location names that are probably one place, given photos per name."""
    names = sorted(counts)
    keys = {n: place_key(n) for n in names}
    unbracketed = {n: " ".join(_BRACKETED.sub(" ", keys[n]).split()) for n in names}

    def reason(a: str, b: str) -> str | None:
        ka, kb = keys[a], keys[b]
        if ka == kb:
            return CAPITALS
        if unbracketed[a] and unbracketed[a] == unbracketed[b]:
            return BRACKETS
        shorter, longer = sorted((ka, kb), key=len)
        mid_word = longer.startswith(shorter) and longer[len(shorter)].isalpha()
        if mid_word and shorter[-1].isalpha() and len(shorter) >= MIN_PLACE_NAME:
            return CUT_SHORT  # "Terminal - De" for "Terminal - Deerfield, MI", but not "ME"
        if _DIGITS.findall(ka) != _DIGITS.findall(kb):
            return None  # "Pier 1" and "Pier 2" are different places
        limit = _allowed_slips(len(shorter), shortest=MIN_PLACE_NAME, long_from=16)
        return SPELLING if _within(ka, kb, limit) else None

    groups = _group(names, reason)
    found = []
    for members, reasons in groups:
        ordered = sorted(members, key=lambda n: (_irregular(n), -counts[n], n))
        found.append(
            SamePlace(
                names=tuple(ordered),
                photos=tuple(counts[n] for n in ordered),
                reasons=tuple(sorted(reasons)),
            )
        )
    return sorted(found, key=lambda g: (-sum(g.photos), g.names[0]))


def word_spellings(place_names: Iterable[str]) -> list[WordSpellings]:
    """Words spelt more than one way across place names (Sanatorium, Sanitarium)."""
    places: Counter[str] = Counter()
    example: dict[str, str] = {}
    for name in dict.fromkeys(place_names):
        for word in dict.fromkeys(_WORD.findall(name)):
            places[word] += 1
            example.setdefault(word, name)
    by_key: defaultdict[str, list[str]] = defaultdict(list)
    for word in places:
        by_key[word.casefold()].append(word)

    def reason(a: str, b: str) -> str | None:
        if a == b:  # one key, several capitalisations: only irregular ones are slips
            return CAPITALS if any(_irregular(w) for w in by_key[a]) else None
        shorter, longer = sorted((a, b), key=len)
        if a[0] != b[0] or longer.startswith(shorter) or longer.endswith(shorter):
            return None  # Tower/Power, Change/Chang, Estate/State: different words
        limit = _allowed_slips(len(shorter), shortest=MIN_SPELT_WORD, long_from=9)
        return SPELLING if _within(a, b, limit) else None

    found = []
    for keys, _ in _group(sorted(by_key), reason, self_pairs=True):
        words = [w for k in keys for w in by_key[k]]
        if len(words) < 2:
            continue
        ordered = sorted(words, key=lambda w: (_irregular(w), -places[w], w))
        found.append(
            WordSpellings(
                spellings=tuple(ordered),
                places=tuple(places[w] for w in ordered),
                examples=tuple(example[w] for w in ordered),
            )
        )
    return sorted(found, key=lambda w: (-sum(w.places), w.spellings[0]))


def _allowed_slips(length: int, *, shortest: int, long_from: int) -> int:
    """No slips in a short text, one in a medium one, two in a long one."""
    if length < shortest:
        return 0
    return 2 if length >= long_from else 1


def _within(a: str, b: str, limit: int) -> bool:
    """``edit_distance(a, b) <= limit``, skipping the full computation when it cannot be."""
    if limit == 0 or abs(len(a) - len(b)) > limit:
        return False
    letters = Counter(a)
    letters.subtract(b)
    if sum(abs(n) for n in letters.values()) > 2 * limit:
        return False
    return edit_distance(a, b) <= limit


def _irregular(text: str) -> bool:
    """A word with capitals where nobody would put them, like "HIgh"."""
    return any(not (w.islower() or w.isupper() or w.istitle()) for w in _WORD.findall(text))


def _group(
    items: list[str], reason: Callable[[str, str], str | None], *, self_pairs: bool = False
) -> list[tuple[list[str], set[str]]]:
    """Connected groups of ``items`` linked by any pair with a reason, with those reasons."""
    parent = {item: item for item in items}

    def root(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    reasons: defaultdict[str, set[str]] = defaultdict(set)
    linked: set[str] = set()
    for i, a in enumerate(items):
        if self_pairs and (why := reason(a, a)):
            reasons[root(a)].add(why)
            linked.add(a)
        for b in items[i + 1 :]:
            why = reason(a, b)
            if why:
                ra, rb = root(a), root(b)
                if ra != rb:
                    parent[rb] = ra
                    reasons[ra] |= reasons.pop(rb, set())
                reasons[ra].add(why)
                linked |= {a, b}
    members: defaultdict[str, list[str]] = defaultdict(list)
    for item in items:
        if item in linked:
            members[root(item)].append(item)
    return [(group, reasons[r]) for r, group in members.items()]
