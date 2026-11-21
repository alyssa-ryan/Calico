"""Calico: pick photographs from an archive.

See ``docs/architecture.md`` for how the pieces fit together.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("calico")
except PackageNotFoundError:  # pragma: no cover - only when run from an uninstalled checkout
    __version__ = "0.0.0"
