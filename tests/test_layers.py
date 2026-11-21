"""The module map, enforced: which parts of Calico may import which (audit A10-1, A10-2).

Tasks, selection helpers and models are pure (CLAUDE.md, rule 2), so they never import
what does I/O or holds state. The engine is the selection pipeline and knows nothing of
the sessions built on it; only the interfaces reach either. Imports under
``TYPE_CHECKING`` are left out: they name a type and load nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "calico"

PURE = ("tasks", "selection", "models")
"""Packages that receive arrays and return values."""
STATEFUL = (
    "catalog",
    "engine",
    "sessions",
    "runs",
    "settings",
    "labels",
    "ratings_store",
    "ratings_migration",
    "reviews",
    "evaluation",
    "storage",
    "trained",
    "patch_store",
    "backup",
    "holdout",
    "experiments",
    "web",
    "cli",
)
"""What the pure packages may not import: I/O, stored state, and everything above them."""
INTERFACES = ("cli", "web")


def _part(module: str) -> str:
    """``calico.tasks.best`` -> ``tasks``; ``calico`` -> ``""``."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


def _runtime_imports(path: Path) -> set[str]:
    """The calico modules ``path`` imports when it runs."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    type_only = {
        id(inner)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test)
        for inner in ast.walk(node)
    }
    found = set()
    for node in ast.walk(tree):
        if id(node) in type_only:
            continue
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
    return {m for m in found if m == "calico" or m.startswith("calico.")}


def _graph() -> dict[str, set[str]]:
    """Each calico module's part (``tasks``, ``engine``...) and the parts it imports, by file."""
    return {
        str(path.relative_to(SRC)): {_part(m) for m in _runtime_imports(path)}
        for path in sorted(SRC.rglob("*.py"))
    }


def _owner(file: str) -> str:
    return Path(file).parts[0].removesuffix(".py")


def test_pure_packages_import_nothing_that_does_io_or_holds_state() -> None:
    broken = {
        file: sorted(parts & set(STATEFUL))
        for file, parts in _graph().items()
        if _owner(file) in PURE and parts & set(STATEFUL)
    }
    assert broken == {}


def test_the_engine_knows_nothing_of_sessions_or_interfaces() -> None:
    parts = _graph()["engine.py"]
    assert not parts & {"sessions", *INTERFACES}


def test_only_the_interfaces_reach_the_engine_and_sessions() -> None:
    reach = {
        file
        for file, parts in _graph().items()
        if parts & {"engine", "sessions"} and _owner(file) not in {"engine", "sessions"}
    }
    assert {_owner(file) for file in reach} <= set(INTERFACES)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from calico.catalog.store import Catalog\n", {"calico.catalog.store"}),
        ("import calico.runs\n", {"calico.runs"}),
        (
            (
                "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
                "    from calico.settings import Settings\n"
            ),
            set(),
        ),
        ("import numpy as np\n", set()),
    ],
)
def test_runtime_imports_leave_out_type_only_ones(
    tmp_path: Path, source: str, expected: set[str]
) -> None:
    module = tmp_path / "m.py"
    module.write_text(source, encoding="utf-8")
    assert _runtime_imports(module) == expected
