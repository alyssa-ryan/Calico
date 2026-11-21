"""Per-track settings: each of the four tracks ships its champion's task options."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from calico.errors import ConfigError
from calico.tasks.best import BestTask
from calico.tasks.triptych import TriptychTask
from calico.tracks import TRACKS, Track, load_track, load_tracks, track_for

if TYPE_CHECKING:
    from pathlib import Path


def test_a_track_is_a_task_under_a_taste() -> None:
    assert track_for("best", "general") == "strongest-general"
    assert track_for("best", "yours") == "strongest-yours"
    assert track_for("triptych", "general") == "story-general"
    assert track_for("triptych", "yours") == "story-yours"
    assert track_for("same_view", "yours") is None


def test_every_built_in_track_loads_and_its_options_parse() -> None:
    found = load_tracks()
    assert set(found) == set(TRACKS)
    for name, track in found.items():
        assert track.name == name
        assert track.version
        task = BestTask() if name.startswith("strongest") else TriptychTask()
        task.parse_options(track.options)


def test_story_yours_ships_its_champion_sy_sw25() -> None:
    assert load_track("story-yours").options == {"spend_alike": 0.871, "strength_weight": 0.25}
    assert load_track("story-yours").version == "3"
    assert load_track("strongest-yours").options == {}


def test_a_track_file_holds_a_version_and_options(tmp_path: Path) -> None:
    path = tmp_path / "strongest-yours.toml"
    path.write_text('version = "3"\n\n[options]\ndiversity = 0.1\n')
    assert load_track("strongest-yours", tmp_path) == Track(
        "strongest-yours", "3", {"diversity": 0.1}
    )


def test_a_track_file_without_a_version_or_with_more_than_options_is_refused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "story-general.toml"
    path.write_text("[options]\n")
    with pytest.raises(ConfigError, match="version"):
        load_track("story-general", tmp_path)
    path.write_text('version = "1"\ntaste = "yours"\n[options]\n')
    with pytest.raises(ConfigError, match="taste"):
        load_track("story-general", tmp_path)
    with pytest.raises(ConfigError, match="no file"):
        load_track("strongest-general", tmp_path)
