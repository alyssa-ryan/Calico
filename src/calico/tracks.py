"""The four tracks and the task options each one ships with.

A track is a task under a taste: Strongest or Story, under General or Yours. Each has a
settings file, ``tracks/<track>.toml``, holding the task options its champion runs with
(the improvement loop's ``tracks/<track>/champion.toml``, kept the same by ``loop status``),
so a track can be improved without the others moving. A request's own options win over
the track's, and the track's version is recorded in every run.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from calico.errors import ConfigError

BUILT_IN = Path(__file__).with_name("tracks")
TRACKS = ("strongest-general", "strongest-yours", "story-general", "story-yours")
_TASKS = {"best": "strongest", "triptych": "story"}


@dataclass(frozen=True)
class Track:
    name: str
    version: str
    options: dict[str, Any]


def track_for(task: str, taste: Literal["general", "yours"]) -> str | None:
    """The track a task runs on under a taste, or None for a task no track covers."""
    kind = _TASKS.get(task)
    return None if kind is None else f"{kind}-{taste}"


def load_track(name: str, directory: Path = BUILT_IN) -> Track:
    path = directory / f"{name}.toml"
    if not path.is_file():
        raise ConfigError(f"Track {name!r} has no file at {path}.")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    extra = sorted(set(data) - {"version", "options"})
    if extra:
        raise ConfigError(
            f"{path} holds {', '.join(extra)}; a track file holds only version and [options]."
        )
    if "version" not in data:
        raise ConfigError(f"{path} has no version; add one, and bump it when options change.")
    options = data.get("options", {})
    if not isinstance(options, dict):
        raise ConfigError(f"{path}: [options] must be a table of task options.")
    return Track(name, str(data["version"]), dict(options))


def load_tracks(directory: Path = BUILT_IN) -> dict[str, Track]:
    return {name: load_track(name, directory) for name in TRACKS}
