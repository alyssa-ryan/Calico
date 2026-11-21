"""The model provider: code-defined models plus zero-shot taxonomies, by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from calico.errors import CapabilityError, UnknownPluginError
from calico.models.taxonomy import Taxonomy, discover_taxonomies
from calico.models.zero_shot import ZeroShotLabelModel
from calico.plugins import MODELS_GROUP, Registry

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from calico.features.base import Embedder
    from calico.models.base import Model, ModelProvider
    from calico.settings import Settings


class Models:
    """Code models win over taxonomies of the same name, so a trained model can
    replace a zero-shot stand-in without any task noticing."""

    def __init__(
        self,
        embedder: Embedder,
        taxonomies: Mapping[str, Taxonomy],
        code_models: Mapping[str, Callable[[], Model]],
    ) -> None:
        self._embedder = embedder
        self._taxonomies = dict(taxonomies)
        self._code = dict(code_models)
        self._built: dict[str, Model] = {}

    @classmethod
    def from_settings(cls, settings: Settings, embedder: Embedder) -> Models:
        registry: Registry[Callable[[], Model]] = Registry.from_entry_points(MODELS_GROUP, "model")
        code = {name: registry.get(name) for name in registry.names()}
        return cls(embedder, discover_taxonomies(settings.taxonomy_dirs), code)

    def names(self) -> list[str]:
        return sorted(set(self._code) | set(self._taxonomies))

    def get(self, name: str) -> Model:
        if name in self._built:
            return self._built[name]
        model: Model
        if name in self._code:
            model = self._code[name]()
        elif name in self._taxonomies:
            if not self._embedder.supports_text:
                raise CapabilityError(
                    f"Model {name!r} is zero-shot and needs a text-capable embedder: "
                    'install the ml extra and set embedder = "clip".'
                )
            model = ZeroShotLabelModel(self._taxonomies[name], self._embedder)
        else:
            raise UnknownPluginError(
                f"No model named {name!r}. Known models: {', '.join(self.names()) or 'none'}."
            )
        self._built[name] = model
        return model

    def versions(self) -> dict[str, str]:
        return {name: model.version for name, model in self._built.items()}


class WithTaste:
    """The models, except that "aesthetic" is your taste: the Yours template. Tasks ask
    for "aesthetic" as before and never know which template ranked them."""

    def __init__(self, base: ModelProvider, taste: Model) -> None:
        self._base = base
        self._taste = taste
        self._asked = False

    def names(self) -> list[str]:
        return self._base.names()

    def get(self, name: str) -> Model:
        if name == "aesthetic":
            self._asked = True
            return self._taste
        return self._base.get(name)

    def versions(self) -> dict[str, str]:
        versions = self._base.versions()
        if self._asked:
            versions["aesthetic"] = self._taste.version
        return versions
