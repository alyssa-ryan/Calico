"""Light, colour and mood words, answered by measuring photos rather than by CLIP.

On the user's labels CLIP ranked "golden hour" and "brooding" no better than
chance, while plain measurements (warmth and saturation; darkness and muted
colour) separated them almost perfectly. A *look* maps words to the technical
measurements they imply. The vocabulary is data (``looks.toml``, overridable in
``<calico home>/looks.toml``), written from photographic convention.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from calico.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Sequence

BUILT_IN = Path(__file__).with_name("looks.toml")
MEASUREMENTS = (
    "brightness",
    "contrast",
    "warmth",
    "saturation",
    "colorfulness",
    "highlight_clip",
    "shadow_clip",
    "sharpness",
)
"""The ``technical`` extractor's columns a look may use."""
_WORDS = {
    ("brightness", 1): "brighter",
    ("brightness", -1): "darker",
    ("contrast", 1): "more contrast",
    ("contrast", -1): "softer contrast",
    ("warmth", 1): "warmer",
    ("warmth", -1): "cooler",
    ("saturation", 1): "more saturated",
    ("saturation", -1): "more muted",
    ("colorfulness", 1): "more colourful",
    ("colorfulness", -1): "less colourful",
}


@dataclass(frozen=True)
class Look:
    name: str
    words: tuple[str, ...]
    signals: tuple[tuple[str, int], ...]
    """(measurement, +1 for more or -1 for less), in file order."""

    def describe(self) -> str:
        return ", ".join(
            _WORDS.get((m, s), f"{'more' if s > 0 else 'less'} {m.replace('_', ' ')}")
            for m, s in self.signals
        )

    def negated(self) -> Look:
        """ "not moody": the same words, every measurement reversed."""
        return Look(f"not {self.name}", (), tuple((m, -s) for m, s in self.signals))


def load_looks(path: Path) -> list[Look]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    looks = []
    for name, body in dict(data.get("looks", {})).items():
        words = body.get("words") if isinstance(body, dict) else None
        if not isinstance(words, list) or not words or not all(isinstance(w, str) for w in words):
            raise ConfigError(f'{path}: look {name!r} needs words = ["...", ...].')
        if not all(w.strip() for w in words):
            raise ConfigError(f"{path}: look {name!r} has an empty word; remove it.")
        signals = body.get("signals", {})
        if not isinstance(signals, dict) or not signals:
            raise ConfigError(f"{path}: look {name!r} needs signals = {{ warmth = 1, ... }}.")
        for measurement, sign in signals.items():
            if measurement not in MEASUREMENTS:
                raise ConfigError(
                    f"{path}: look {name!r} uses {measurement!r}; "
                    f"use one of: {', '.join(MEASUREMENTS)}."
                )
            if type(sign) is not int or sign not in (1, -1):  # True is an int, but not a sign
                raise ConfigError(f"{path}: look {name!r}: {measurement} must be +1 or -1.")
        looks.append(Look(name, tuple(words), tuple(signals.items())))
    return looks


def discover_looks(paths: Sequence[Path]) -> tuple[Look, ...]:
    """Looks from each existing file; a later file overrides an earlier one by name."""
    found: dict[str, Look] = {}
    for path in paths:
        if path.is_file():
            found.update({look.name: look for look in load_looks(path)})
    return tuple(found.values())
