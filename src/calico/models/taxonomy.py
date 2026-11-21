"""Vocabularies for zero-shot models, as TOML files.

A taxonomy is data, not code: labels with the text prompts that describe them.
Label ids may nest with ``:`` (``establishing:wide``) so ``rollup`` can answer
coarse or fine questions from one prediction.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from calico.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_LABEL_ID = re.compile(r"^[a-z0-9_]+(:[a-z0-9_-]+)*$")


@dataclass(frozen=True)
class Taxonomy:
    name: str
    description: str
    version: str
    labels: tuple[str, ...]
    prompts: tuple[tuple[str, ...], ...]


def load_taxonomy(path: Path) -> Taxonomy:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    raw = data.get("labels")
    if not isinstance(raw, dict) or len(raw) < 2:
        raise ConfigError(f"{path}: a taxonomy needs at least 2 [labels.<id>] tables.")
    labels: list[str] = []
    prompts: list[tuple[str, ...]] = []
    for label, body in raw.items():
        if not _LABEL_ID.fullmatch(label):
            raise ConfigError(
                f"{path}: label id {label!r} must be lowercase letters, digits and '_', "
                "optionally followed by ':'-separated sub-labels that may also contain '-' "
                "(e.g. 'establishing:wide-angle')."
            )
        texts = body.get("prompts") if isinstance(body, dict) else None
        if (
            not isinstance(texts, list)
            or not texts
            or not all(isinstance(t, str) and t.strip() for t in texts)
        ):
            raise ConfigError(f"{path}: label {label!r} needs prompts = [...] with non-empty text.")
        labels.append(label)
        prompts.append(tuple(texts))
    return Taxonomy(
        name=path.stem,
        description=str(data.get("description", path.stem)),
        version=str(data.get("version", "1")),
        labels=tuple(labels),
        prompts=tuple(prompts),
    )


def discover_taxonomies(dirs: Sequence[Path]) -> dict[str, Taxonomy]:
    """``*.toml`` from each directory; later directories override earlier by name."""
    found: dict[str, Taxonomy] = {}
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.toml")):
            found[path.stem] = load_taxonomy(path)
    return found
