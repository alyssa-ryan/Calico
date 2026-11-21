from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

from calico.settings import HOME_ENV

if TYPE_CHECKING:
    from pathlib import Path

pytest_plugins = ["pytester", "tests.skip_limit"]

HAS_ML = all(importlib.util.find_spec(m) is not None for m in ("torch", "open_clip"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if HAS_ML:
        return
    skip = pytest.mark.skip(reason="needs the ml extra: uv sync --extra ml")
    for item in items:
        if "ml" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def calico_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own isolated calico home; nothing touches the real one.

    Output is plain text too. On GitHub Actions Typer forces a colour terminal (decided
    at import), and escape codes inside words made assertions fail only there; a dumb
    terminal prints none. ``FORCE_COLOR`` is cleared because Rich sizes a forced dumb
    terminal at 80 columns, ignoring the ``COLUMNS`` tests set for wide tables.
    """
    home = tmp_path / "calico-home"
    monkeypatch.setenv(HOME_ENV, str(home))
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    return home
