"""User settings: where data lives, which collections exist, which embedder to use.

Settings are a TOML file in the calico home directory. The home directory is
``$CALICO_HOME`` if set, otherwise the platform's user-data directory. Every
derived path (catalog, thumbnails, runs) hangs off ``home``, so pointing
``CALICO_HOME`` at a temp dir gives a fully isolated instance (the tests do this).
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Annotated, Literal

import platformdirs
import tomli_w
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from calico.domain import place_key
from calico.errors import ConfigError, HomeInsideArchiveError, NotInCollectionError
from calico.paths import normalise

HOME_ENV = "CALICO_HOME"
CONFIG_FILENAME = "calico.toml"


class CollectionConfig(BaseModel):
    """A folder tree of photos, plus the rule for naming each photo's location.

    The location of a photo is the name of the folder ``location_depth`` levels
    below ``root`` on its path. If ``location_pattern`` is set, it is a regex
    applied to that folder name and its ``location`` named group becomes the
    location -- so ``Old Mill - March 2021`` and ``Old Mill - May 2024`` can both
    map to ``Old Mill``. Photos directly in ``root`` get the collection name. With
    ``location_deepest``, a deeper folder names the place when it can: ``Trip/Calder``
    is Calder, while ``Trip/Originals`` (a ``location_not_places`` name) is the trip.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, Field(min_length=1, pattern=r"^[\w.-]+$")]
    root: Path
    location_depth: Annotated[int, Field(ge=1)] = 1
    location_pattern: str | None = None
    location_ignore: tuple[str, ...] = Field(
        default=(),
        description="Regexes removed from the folder name before location_pattern, e.g. "
        r"'\([^)]*\)' so 'Mill (after the fire)' is still 'Mill'.",
    )
    location_deepest: bool = Field(
        default=False,
        description="Name the place by the deepest folder at or below location_depth that "
        "names one (a city inside a trip), passing over location_not_places.",
    )
    location_not_places: tuple[str, ...] = Field(
        default=(),
        description="Regexes for folder names that organise rather than name a place, "
        "matched whole and ignoring case, e.g. 'Originals', 'Lightroom.*'.",
    )
    exclude: tuple[str, ...] = Field(
        default=("._*", ".*"), description="Glob patterns; matching files and folders are skipped."
    )
    exclude_screenshots: bool = Field(
        default=False,
        description="Leave PNG files with no camera recorded out of searches: screenshots and "
        "images saved from apps or the web. Off by default because scans are often "
        "camera-less PNGs too.",
    )

    @field_validator("root")
    @classmethod
    def _root_is_absolute(cls, root: Path) -> Path:
        if not root.is_absolute():
            raise ValueError(f"collection root must be an absolute path, got {root}")
        # Links resolved (an unmounted drive still validates), spelt as the disk spells
        # it and in NFC, so every path derived from the root compares equal.
        return normalise(root)

    @field_validator("location_not_places")
    @classmethod
    def _not_places_are_regexes(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"location_not_places {pattern!r} is not a regex: {exc}") from exc
        return patterns

    @field_validator("location_pattern")
    @classmethod
    def _pattern_has_location_group(cls, pattern: str | None) -> str | None:
        if pattern is None:
            return None
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"location_pattern is not a valid regex: {exc}") from exc
        if "location" not in compiled.groupindex:
            raise ValueError("location_pattern needs a named group: (?P<location>...)")
        return pattern

    @field_validator("location_ignore")
    @classmethod
    def _ignore_patterns_compile(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"location_ignore {pattern!r} is not a valid regex: {exc}"
                ) from exc
        return patterns

    def contains(self, path: Path) -> bool:
        return path.is_relative_to(self.root)


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    home: Path
    collections: tuple[CollectionConfig, ...] = ()
    embedder: str = Field(
        default="clip",
        description="Extractor used as the semantic embedding. 'clip' needs the ml extra; "
        "'tiny' works everywhere but cannot read the criteria box.",
    )
    describe_embedder: str = Field(
        default="siglip",
        description="Embedder that reads 'Describe it'. 'siglip' ranked labelled searches "
        "better than 'clip' and can tell a clear match. See: calico eval criteria",
    )
    template: Literal["general", "yours"] = Field(
        default="general",
        description="Whose taste ranks photos when a search does not say: 'general' (the "
        "zero-shot aesthetic) or 'yours' (learned from your ratings; see: calico train taste).",
    )
    view_embedder: str = Field(
        default="dinov2",
        description="Image model that tells the same view from places like it, for Same view. "
        "'dinov2' (448 px, patches averaged) kept more same views than 'clip' and fewer "
        "look-alikes. Computed for every photo. See: calico eval same-view",
    )
    extractor_options: dict[str, dict[str, JsonValue]] = Field(
        default_factory=dict, description="Keyword arguments passed to each extractor by name."
    )
    location_aliases: dict[str, str] = Field(
        default_factory=dict,
        description="Location name -> the name to use instead, for misspelt or renamed "
        "places. Matched ignoring capitals and spacing. See: calico locations",
    )
    hemisphere: Literal["north", "south"] = Field(
        default="north",
        description="Where the photos were taken, for reading seasons ('winter') as months.",
    )
    criteria_scorer: str = Field(
        default="parts",
        description="How the 'Describe it' text is matched to photos. See: calico eval criteria",
    )
    decode_max_side: Annotated[int, Field(ge=256)] = 1024
    thumbnail_side: Annotated[int, Field(ge=64)] = 480

    @field_validator("collections")
    @classmethod
    def _unique_collection_names(
        cls, collections: tuple[CollectionConfig, ...]
    ) -> tuple[CollectionConfig, ...]:
        names = [c.name for c in collections]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate collection names: {', '.join(duplicates)}")
        return collections

    @field_validator("location_aliases")
    @classmethod
    def _aliases_do_not_chain(cls, aliases: dict[str, str]) -> dict[str, str]:
        renamed: dict[str, str] = {}
        for old in aliases:
            first = renamed.setdefault(place_key(old), old)
            if first != old:
                raise ValueError(
                    f"location_aliases names one place twice: {first!r} and {old!r} differ "
                    "only in capitals, spacing or Unicode form, so one would silently win. "
                    "Keep one."
                )
        for old, new in aliases.items():
            if place_key(new) in renamed and place_key(new) != place_key(old):
                raise ValueError(
                    f"location_aliases renames {old!r} to {new!r}, which is itself renamed "
                    f"(as {renamed[place_key(new)]!r}). Point {old!r} at the final name."
                )
        return aliases

    @property
    def config_path(self) -> Path:
        return self.home / CONFIG_FILENAME

    @property
    def catalog_path(self) -> Path:
        return self.home / "catalog.sqlite"

    @property
    def thumbs_dir(self) -> Path:
        return self.home / "thumbs"

    @property
    def looks_files(self) -> tuple[Path, ...]:
        """Built-in light, colour and mood words, then ``<home>/looks.toml`` overriding by name."""
        return (Path(__file__).parent / "looks.toml", self.home / "looks.toml")

    @property
    def labels_dir(self) -> Path:
        return self.home / "labels"

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"

    @property
    def ratings_dir(self) -> Path:
        """The ratings journal (``calico.ratings_store``): every answer you gave."""
        return self.home / "ratings"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def experiments_path(self) -> Path:
        """Every setting a ``calico eval`` measured (``calico.experiments``)."""
        return self.home / "experiments.jsonl"

    @property
    def controls_path(self) -> Path:
        """The pool of control photos asked again in Compare sessions (``calico.controls``)."""
        return self.home / "controls.toml"

    @property
    def holdout_path(self) -> Path:
        """The ratings no tuning may see (``calico.holdout``)."""
        return self.home / "holdout.toml"

    @property
    def feature_extractors(self) -> tuple[str, ...]:
        """Extractors computed for every catalogued photo. Tasks may rely on all of them."""
        return tuple(
            dict.fromkeys(
                (
                    "technical",
                    "layout",
                    "palette",
                    self.embedder,
                    self.describe_embedder,
                    self.view_embedder,
                )
            )
        )

    @property
    def taxonomy_dirs(self) -> tuple[Path, ...]:
        """Built-in vocabularies first; files in ``<home>/taxonomies`` override by name."""
        return (Path(__file__).parent / "models" / "taxonomies", self.home / "taxonomies")

    def check_home_apart(self) -> None:
        """``HomeInsideArchiveError`` if the calico home is inside a collection root, or
        holds one: Calico would write its catalog and thumbnails into the archive (rule 8)
        and index them as photos, or index its own home."""
        home = normalise(self.home)
        for c in self.collections:
            if home.is_relative_to(c.root) or c.root.is_relative_to(home):
                raise HomeInsideArchiveError(
                    f"The calico home ({self.home}) and the {c.name!r} collection ({c.root}) "
                    "are inside one another, so Calico would write its catalog and "
                    "thumbnails into your archive, or index its own files as photos. Keep "
                    "them apart: set CALICO_HOME to a folder outside the archive, or change "
                    f"the collection in {self.config_path} (calico collections remove "
                    f"{c.name})."
                )

    def collection_for(self, path: Path) -> CollectionConfig:
        """The collection whose root contains ``path`` (the deepest one if roots nest)."""
        matches = [c for c in self.collections if c.contains(path)]
        if not matches:
            roots = ", ".join(str(c.root) for c in self.collections) or "none configured"
            raise NotInCollectionError(
                f"{path} is not inside any collection (roots: {roots}). "
                "Add one with: calico collections add NAME ROOT"
            )
        return max(matches, key=lambda c: len(c.root.parts))

    def with_collection(self, collection: CollectionConfig) -> Settings:
        others = tuple(c for c in self.collections if c.name != collection.name)
        return self.model_copy(update={"collections": (*others, collection)})


def default_home() -> Path:
    env = os.environ.get(HOME_ENV)
    return Path(env).expanduser() if env else platformdirs.user_data_path("calico")


def load_settings(home: Path | None = None) -> Settings:
    """Read settings from ``home`` (default: ``default_home()``). A missing file means defaults."""
    home = (home or default_home()).resolve()
    path = home / CONFIG_FILENAME
    data: dict[str, object] = {}
    if path.exists():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    try:
        return Settings.model_validate({**data, "home": home})
    except ValidationError as exc:
        raise ConfigError(f"{path} is invalid:\n{exc}") from exc


def save_settings(settings: Settings) -> Path:
    """Write settings atomically. ``home`` is implied by the file's location, so not stored.

    Only settings that were chosen are written: a default frozen into the file would
    stop the user from ever receiving a better default.
    """
    settings.home.mkdir(parents=True, exist_ok=True)
    data = settings.model_dump(mode="json", exclude={"home"}, exclude_none=True, exclude_unset=True)
    tmp = settings.config_path.with_suffix(".toml.tmp")
    tmp.write_text(tomli_w.dumps(data), encoding="utf-8")
    tmp.replace(settings.config_path)
    return settings.config_path
