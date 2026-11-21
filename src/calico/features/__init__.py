"""Extractor lookup and construction from settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from calico.errors import ConfigError
from calico.features.base import Embedder, ImageExtractor
from calico.plugins import EXTRACTORS_GROUP, Registry

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from calico.settings import Settings

type ExtractorFactory = Callable[..., ImageExtractor]


def extractor_registry() -> Registry[ExtractorFactory]:
    return Registry.from_entry_points(EXTRACTORS_GROUP, "extractor")


def build_extractors(
    settings: Settings, registry: Registry[ExtractorFactory] | None = None
) -> dict[str, ImageExtractor]:
    """Instantiate every extractor the settings call for, with its configured options."""
    registry = registry or extractor_registry()
    built: dict[str, ImageExtractor] = {
        name: build_extractor(name, settings, registry) for name in settings.feature_extractors
    }
    for setting, name in (
        ("embedder", settings.embedder),
        ("describe_embedder", settings.describe_embedder),
    ):
        if not isinstance(built[name], Embedder):
            raise ConfigError(
                f"{setting} {name!r} is not an embedder (it has no encode_texts/"
                f'supports_text). Set {setting} = "clip", "siglip" or "tiny" in calico.toml.'
            )
    return built


def build_extractor(
    name: str, settings: Settings, registry: Registry[ExtractorFactory] | None = None
) -> ImageExtractor:
    """One extractor by name, with its configured options, whether or not the settings
    compute it for every photo (measuring a candidate builds it this way)."""
    registry = registry or extractor_registry()
    return registry.get(name)(**settings.extractor_options.get(name, {}))


def embedder_of(extractors: Mapping[str, ImageExtractor], settings: Settings) -> Embedder:
    """The main embedding: same view, stories, kinds of shot."""
    return _embedder(extractors, "embedder", settings.embedder)


def describe_embedder_of(extractors: Mapping[str, ImageExtractor], settings: Settings) -> Embedder:
    """The embedder that reads "Describe it"."""
    return _embedder(extractors, "describe_embedder", settings.describe_embedder)


def _embedder(extractors: Mapping[str, ImageExtractor], setting: str, name: str) -> Embedder:
    embedder = extractors[name]
    if not isinstance(embedder, Embedder):
        raise ConfigError(f"{setting} {name!r} is not an embedder.")
    return embedder
