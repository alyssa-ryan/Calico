"""One registry type for every pluggable thing: tasks and feature extractors.

Built-ins are declared as entry points in ``pyproject.toml`` exactly like a
third-party plugin would be, so there is a single path by which anything
becomes available. Tests use ``Registry.from_mapping`` to swap in fakes
without touching installed metadata.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from calico.errors import UnknownPluginError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

TASKS_GROUP = "calico.tasks"
EXTRACTORS_GROUP = "calico.extractors"
MODELS_GROUP = "calico.models"


class Registry[T]:
    """Name -> lazily loaded plugin object (a class or factory).

    Loading is lazy so that listing tasks never imports PyTorch.
    """

    def __init__(self, kind: str, loaders: Mapping[str, Callable[[], T]]) -> None:
        self._kind = kind
        self._loaders = dict(loaders)
        self._loaded: dict[str, T] = {}

    @classmethod
    def from_entry_points(cls, group: str, kind: str) -> Registry[T]:
        loaders: dict[str, Callable[[], T]] = {ep.name: ep.load for ep in entry_points(group=group)}
        return cls(kind, loaders)

    @classmethod
    def from_mapping(cls, kind: str, plugins: Mapping[str, T]) -> Registry[T]:
        def constant(plugin: T) -> Callable[[], T]:
            return lambda: plugin

        return cls(kind, {name: constant(plugin) for name, plugin in plugins.items()})

    def names(self) -> list[str]:
        return sorted(self._loaders)

    def get(self, name: str) -> T:
        if name not in self._loaded:
            try:
                loader = self._loaders[name]
            except KeyError:
                known = ", ".join(self.names()) or "none"
                raise UnknownPluginError(
                    f"No {self._kind} named {name!r}. Known {self._kind}s: {known}."
                ) from None
            self._loaded[name] = loader()
        return self._loaded[name]

    def __contains__(self, name: object) -> bool:
        return name in self._loaders
