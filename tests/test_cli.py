"""The command line: arguments in, Engine/RunStore/Catalog calls, rendered text out."""

from __future__ import annotations

import csv
import http.server
import importlib.util
import io
import json
import math
import os
import socket
import sys
import threading
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner

from calico import cli, evaluation
from calico import sessions as sessions_module
from calico.catalog.metadata import file_facts
from calico.catalog.store import Catalog
from calico.cli import Check, app, parse_option, run_checks
from calico.controls import Controls, write_controls
from calico.domain import Marker, SelectionRequest
from calico.engine import Engine
from calico.errors import InvalidRequestError
from calico.evaluation.controls import Control
from calico.evaluation.holdout import HoldOut
from calico.experiments import record
from calico.features import extractor_registry
from calico.features.tiny import TinyEmbedder
from calico.holdout import write_holdout
from calico.labels import LabelItem, LabelSet, LabelStore
from calico.models import calibration as calibration_module
from calico.plugins import Registry
from calico.ratings_migration import MigrationReport, UseCount
from calico.ratings_store import Answer, RatingsJournal
from calico.reviews import Ratings
from calico.runs import RunStore
from calico.settings import CollectionConfig, Settings, load_settings, save_settings
from tests import factories as f

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from numpy.typing import NDArray
    from PIL import Image

T0 = datetime(2021, 3, 6, 10, 0, 0)
HAS_TORCH = importlib.util.find_spec("torch") is not None
runner = CliRunner()


def invoke(*args: str | Path) -> Any:
    return runner.invoke(app, [str(a) for a in args])


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    for seed in (1, 2, 3):
        f.save_jpeg(
            f.scene(seed),
            root / "Old Mill - March 2021" / f"view{seed}.jpg",
            captured_at=T0 + timedelta(hours=seed),
        )
    for seed in (10, 11):
        f.save_jpeg(
            f.scene(seed),
            root / "Harbor - June 2022" / f"h{seed}.jpg",
            captured_at=datetime(2022, 6, 1, seed),
        )
    return root


@pytest.fixture
def configured(archive: Path, calico_home: Path) -> Settings:
    settings = Settings(
        home=calico_home,
        embedder="tiny",
        describe_embedder="tiny",
        view_embedder="tiny",
        collections=(
            CollectionConfig(
                name="archive", root=archive, location_pattern=r"^(?P<location>.+?) - "
            ),
        ),
    )
    save_settings(settings)
    return settings


@pytest.fixture
def run_id(configured: Settings, archive: Path) -> str:
    result = invoke("select", "best", archive, "--limit", "3", "--json")
    assert result.exit_code == 0, result.output
    run: str = json.loads(result.stdout)["run_id"]
    return run


# -- global behaviour -----------------------------------------------------------


def test_help_lists_every_command() -> None:
    result = invoke("--help")
    assert result.exit_code == 0
    for command in (
        "collections",
        "config",
        "index",
        "tasks",
        "models",
        "select",
        "runs",
        "doctor",
        "serve",
        "feedback",
        "locations",
        "eval",
    ):
        assert command in result.stdout


@pytest.mark.parametrize(
    "command",
    [
        ["locations"],
        ["eval", "new"],
        ["eval", "list"],
        ["eval", "criteria"],
        ["collections", "add"],
        ["collections", "list"],
        ["config", "show"],
        ["config", "set-embedder"],
        ["index"],
        ["tasks"],
        ["models"],
        ["select"],
        ["runs", "list"],
        ["runs", "show"],
        ["runs", "explain"],
        ["doctor"],
        ["serve"],
        ["feedback", "export"],
    ],
)
def test_every_command_has_help(command: list[str]) -> None:
    result = invoke(*command, "--help")
    assert result.exit_code == 0
    assert len(result.stdout.strip().splitlines()) > 3


def test_calico_errors_print_a_red_message_without_traceback(tmp_path: Path) -> None:
    result = invoke("collections", "add", "x", tmp_path / "missing")
    assert result.exit_code == 1
    assert "error:" in result.stderr
    assert "not mounted" in result.stderr
    assert "Traceback" not in result.output
    assert result.stdout == ""


def test_other_exceptions_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: object, **__: object) -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr(cli, "load_settings", boom)
    result = invoke("config", "show")
    assert isinstance(result.exception, RuntimeError)


def test_verbose_flags_configure_logging(
    monkeypatch: pytest.MonkeyPatch, configured: Settings
) -> None:
    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(cli, "configure_logging", lambda v, j: calls.append((v, j)))
    invoke("-vv", "--log-json", "collections", "list")
    invoke("collections", "list")
    assert calls == [(2, True), (0, False)]


# -- collections & config ---------------------------------------------------------


def test_collections_add_saves_settings(tmp_path: Path, calico_home: Path) -> None:
    root = tmp_path / "photos"
    root.mkdir()
    result = invoke(
        "collections",
        "add",
        "mine",
        root,
        "--location-depth",
        "2",
        "--location-pattern",
        r"^(?P<location>.+?) - ",
        "--exclude",
        "*.tmp",
        "--exclude",
        "Trash",
    )
    assert result.exit_code == 0, result.output
    (saved,) = load_settings().collections
    assert saved.name == "mine"
    assert saved.root == root.resolve()
    assert saved.location_depth == 2
    assert saved.exclude == ("*.tmp", "Trash")
    assert "mine" in result.stdout


def test_collections_add_takes_location_ignore_patterns(tmp_path: Path) -> None:
    (tmp_path / "p").mkdir()
    result = invoke("collections", "add", "c", tmp_path / "p", "--location-ignore", r"\([^)]*\)")
    assert result.exit_code == 0, result.output
    assert load_settings().collections[0].location_ignore == (r"\([^)]*\)",)
    assert "ignore" in result.stdout


def test_collections_add_takes_the_deepest_place_and_folders_that_are_not_places(
    tmp_path: Path,
) -> None:
    (tmp_path / "p").mkdir()
    result = invoke(
        "collections",
        "add",
        "travel",
        tmp_path / "p",
        "--location-depth",
        "2",
        "--location-deepest",
        "--not-place",
        "Originals",
        "--not-place",
        "Lightroom.*",
    )
    assert result.exit_code == 0, result.output
    (saved,) = load_settings().collections
    assert saved.location_deepest
    assert saved.location_not_places == ("Originals", "Lightroom.*")
    assert "deepest place, not Originals, Lightroom.*" in result.stdout


def test_collections_add_can_leave_out_screenshots(tmp_path: Path) -> None:
    (tmp_path / "p").mkdir()
    result = invoke("collections", "add", "c", tmp_path / "p", "--exclude-screenshots")
    assert result.exit_code == 0, result.output
    assert load_settings().collections[0].exclude_screenshots
    assert "screenshots (PNG with no camera)" in result.stdout
    assert invoke("collections", "add", "d", tmp_path / "p").exit_code == 0
    assert not load_settings().collections[1].exclude_screenshots


def test_collections_add_warns_when_another_collection_has_that_root(tmp_path: Path) -> None:
    (tmp_path / "p").mkdir()
    assert invoke("collections", "add", "first", tmp_path / "p").exit_code == 0
    result = invoke("collections", "add", "second", tmp_path / "p")
    assert result.exit_code == 0, result.output
    assert "'first' already has that root" in result.stderr
    assert [c.name for c in load_settings().collections] == ["first", "second"]


def test_collections_add_defaults_keep_default_excludes(tmp_path: Path) -> None:
    (tmp_path / "p").mkdir()
    assert invoke("collections", "add", "c", tmp_path / "p").exit_code == 0
    assert load_settings().collections[0].exclude == ("._*", ".*")


def test_collections_add_relative_root_is_made_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rel").mkdir()
    monkeypatch.chdir(tmp_path)
    assert invoke("collections", "add", "c", "rel").exit_code == 0
    assert load_settings().collections[0].root == (tmp_path / "rel").resolve()


def test_collections_add_invalid_pattern_is_a_config_error(tmp_path: Path) -> None:
    result = invoke("collections", "add", "c", tmp_path, "--location-pattern", "no-group")
    assert result.exit_code == 1
    assert "location" in result.stderr
    assert load_settings().collections == ()


def test_collections_add_same_name_replaces(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    invoke("collections", "add", "c", tmp_path / "a")
    invoke("collections", "add", "c", tmp_path / "b")
    (only,) = load_settings().collections
    assert only.root == (tmp_path / "b").resolve()


def test_collections_list_shows_mount_state(tmp_path: Path, calico_home: Path) -> None:
    here = tmp_path / "here"
    here.mkdir()
    settings = Settings(
        home=calico_home,
        embedder="tiny",
        describe_embedder="tiny",
        view_embedder="tiny",
        collections=(
            CollectionConfig(name="here", root=here, location_depth=2),
            CollectionConfig(name="gone", root=tmp_path / "gone"),
        ),
    )
    save_settings(settings)
    out = invoke("collections", "list").stdout
    assert "here" in out
    assert "gone" in out
    assert "yes" in out
    assert "no" in out
    assert "depth 2" in out


def test_collections_list_empty_says_how_to_add() -> None:
    assert "calico collections add" in invoke("collections", "list").stdout


def test_config_show_prints_path_and_settings(configured: Settings) -> None:
    out = invoke("config", "show").stdout
    assert "calico.toml" in out
    assert 'embedder = "tiny"' in out
    assert "archive" in out


def test_config_set_embedder_saves_a_valid_embedder(configured: Settings) -> None:
    save_settings(configured.model_copy(update={"embedder": "clip"}))
    result = invoke("config", "set-embedder", "tiny")
    assert result.exit_code == 0, result.output
    assert load_settings().embedder == "tiny"


def test_config_set_embedder_for_describe_it(configured: Settings) -> None:
    save_settings(configured.model_copy(update={"describe_embedder": "clip"}))
    result = invoke("config", "set-embedder", "tiny", "--describe")
    assert result.exit_code == 0, result.output
    assert "“Describe it”" in result.stdout
    assert "cannot read text" in result.stdout
    assert load_settings().describe_embedder == "tiny"
    assert load_settings().embedder == "tiny"


def test_config_set_embedder_for_same_view(configured: Settings) -> None:
    save_settings(configured.model_copy(update={"view_embedder": "palette"}))
    result = invoke("config", "set-embedder", "tiny", "--view")
    assert result.exit_code == 0, result.output
    assert "Same view" in result.stdout
    assert load_settings().view_embedder == "tiny"


def test_config_set_embedder_takes_one_role_at_a_time(configured: Settings) -> None:
    result = invoke("config", "set-embedder", "tiny", "--view", "--describe")
    assert result.exit_code != 0
    assert "--describe or --view" in result.output


def test_cli_output_is_plain_text_even_where_ci_asks_for_colour(
    configured: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GitHub Actions sets GITHUB_ACTIONS, which makes Rich colour its output; escape codes
    # then split text such as "--describe", so tests reading it failed only on CI.
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = invoke("config", "set-embedder", "tiny", "--view", "--describe")
    assert "\x1b[" not in result.output
    assert "--describe or --view" in result.output


def test_config_set_embedder_rejects_unknown_name(configured: Settings) -> None:
    result = invoke("config", "set-embedder", "nonsense")
    assert result.exit_code == 1
    assert "nonsense" in result.stderr
    assert load_settings().embedder == "tiny"


def test_config_set_embedder_rejects_non_embedder(configured: Settings) -> None:
    result = invoke("config", "set-embedder", "palette")
    assert result.exit_code == 1
    assert load_settings().embedder == "tiny"


@pytest.mark.skipif(HAS_TORCH, reason="needs torch to be missing")
def test_config_set_embedder_clip_without_torch_shows_hint_and_does_not_save(
    configured: Settings,
) -> None:
    result = invoke("config", "set-embedder", "clip")
    assert result.exit_code == 1
    assert "uv sync --extra ml" in result.stderr
    assert load_settings().embedder == "tiny"


# -- index -------------------------------------------------------------------------


def test_index_reports_counts_and_is_incremental(configured: Settings, archive: Path) -> None:
    first = invoke("index", archive, "--workers", "2")
    assert first.exit_code == 0, first.output
    assert "added 5" in first.stdout
    second = invoke("index", archive)
    assert "added 0" in second.stdout
    assert "unchanged 5" in second.stdout


def test_index_lists_failures(configured: Settings, archive: Path) -> None:
    (archive / "Harbor - June 2022" / "broken.jpg").write_bytes(b"not a jpeg")
    out = invoke("index", archive).stdout
    assert "failed" in out
    assert "broken.jpg" in out


def test_index_missing_folder_is_an_error(configured: Settings, archive: Path) -> None:
    result = invoke("index", archive / "nope")
    assert result.exit_code == 1
    assert "not mounted" in result.stderr


@pytest.mark.parametrize("command", [("index",), ("eval", "new-compare"), ("eval", "new-taste")])
def test_a_folder_typed_without_its_trailing_spaces_is_named_as_the_disk_has_it(
    configured: Settings, archive: Path, command: tuple[str, ...]
) -> None:
    """7 November 2026: "does not exist or is not mounted" for a mounted folder
    whose name ended in two spaces."""
    (archive / "Old Mill - 2016  ").mkdir()
    result = invoke(*command, archive / "Old Mill - 2016")
    assert result.exit_code == 1
    assert "2 spaces at the end" in result.stderr
    assert "not mounted" not in result.stderr


def test_new_list_makes_a_session_of_the_photos_a_file_names(
    configured: Settings, archive: Path, tmp_path: Path
) -> None:
    assert invoke("index", archive).exit_code == 0
    photos = sorted((archive / "Harbor - June 2022").glob("*.jpg"))
    sheet = tmp_path / "2026-11-14-picks.csv"
    sheet.write_text("place,photo\n" + "".join(f"Harbor,{p}\n" for p in photos))
    made = invoke("eval", "new-list", sheet)
    assert made.exit_code == 0, made.output
    assert f"{len(photos)} photos to rate" in made.stdout
    assert "/label/" in made.stdout
    (label_set,) = LabelStore(configured.labels_dir).all()
    assert label_set.text == "Your taste: 2026-11-14-picks"
    assert sorted(i.path for i in label_set.items) == photos


def test_new_list_says_what_is_wrong_with_a_file_it_cannot_use(
    configured: Settings, archive: Path, tmp_path: Path
) -> None:
    assert invoke("index", archive).exit_code == 0
    missing = invoke("eval", "new-list", tmp_path / "none.csv")
    assert missing.exit_code == 1
    assert "none.csv" in missing.stderr
    sheet = tmp_path / "picks.csv"
    sheet.write_text("place,file\nHarbor,a.jpg\n")
    unnamed = invoke("eval", "new-list", sheet)
    assert unnamed.exit_code == 1
    assert "column named photo" in " ".join(unnamed.stderr.split())
    sheet.write_text(f"place,photo\nHarbor,{archive / 'Harbor - June 2022' / 'gone.jpg'}\n")
    absent = invoke("eval", "new-list", sheet)
    assert absent.exit_code == 1
    assert "gone.jpg" in absent.stderr
    assert not LabelStore(configured.labels_dir).all()


def test_index_folder_outside_collections_is_an_error(configured: Settings, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    result = invoke("index", other)
    assert result.exit_code == 1
    assert "calico collections add" in result.stderr


# -- tasks & models ------------------------------------------------------------------


def test_tasks_lists_descriptions_and_options() -> None:
    out = invoke("tasks").stdout
    assert "best" in out
    assert "reference" in out.lower()
    assert "diversity" in out
    assert "0.3" in out
    assert "Weight of technical quality" in out
    assert "description counts: 50% (--describe-weight)" in out


@pytest.mark.parametrize("name", ["same_view", "triptych"])
def test_tasks_lists_the_other_tasks(name: str) -> None:
    assert name in invoke("tasks").stdout


def test_models_lists_kind_version_and_labels(configured: Settings) -> None:
    out = invoke("models").stdout
    assert "quality" in out
    assert "score" in out


def test_models_marks_unavailable_ones(configured: Settings) -> None:
    out = invoke("models").stdout
    assert "unavailable" in out
    assert "text-capable embedder" in out


# -- select / runs ---------------------------------------------------------------------


def test_select_renders_a_table_and_run_footer(configured: Settings, archive: Path) -> None:
    result = invoke("select", "best", archive, "--limit", "3")
    assert result.exit_code == 0, result.output
    assert ".jpg" in result.stdout
    assert "quality" in result.stdout or "aesthetic" in result.stdout
    assert "saved" in result.stdout
    assert "calico runs show" in result.stdout


def test_select_json_is_exactly_the_result_on_stdout(configured: Settings, archive: Path) -> None:
    result = invoke("select", "best", archive, "--limit", "2", "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["request"]["task"] == "best"
    assert data["request"]["limit"] == 2
    assert result.stdout.strip().startswith("{")


def test_select_passes_criteria_options_and_shows_notes(
    configured: Settings, archive: Path
) -> None:
    result = invoke(
        "select", "best", archive, "--criteria", "snow", "-o", "diversity=0.1", "--json"
    )
    data = json.loads(result.stdout)
    assert data["request"]["criteria"] == "snow"
    assert data["request"]["options"] == {"diversity": 0.1}
    text = invoke("select", "best", archive, "--criteria", "snow").stdout
    assert "cannot read text" in text


def test_select_only_clear_matches(configured: Settings, archive: Path) -> None:
    result = invoke("select", "best", archive, "--only-clear-matches", "--json")
    data = json.loads(result.stdout)
    assert data["request"]["only_clear_matches"] is True
    assert any("needs a description" in n for n in data["notes"])


LIKED_FOR_TASTE = {"view3.jpg", "h11.jpg"}
"""What these tests always rated liked at the mill and the harbor, by name: a session at a small
location is in a shuffled order now, so the position no longer says which photo it was."""


DISAGREE = {"view3.jpg", "h11.jpg"}
"""The photos liked at the mill and the harbor when General and Yours must disagree somewhere:
they are named, because a session at a small location is in a shuffled order now, so a position
no longer says which photo it was."""


def rate_for_taste(
    configured: Settings, archive: Path, liked: Collection[str] | None = None
) -> None:
    """Rate a session at each place: the photos named in ``liked``, or by default every other
    one in the order shown (both answers at every place, whatever the order)."""
    for place in ("Old Mill - March 2021", "Harbor - June 2022"):
        assert invoke("eval", "new-taste", archive / place).exit_code == 0
    store = LabelStore(configured.labels_dir)
    for label_set in store.all():
        for i, item in enumerate(label_set.items):
            like = item.path.name in liked if liked is not None else i % 2 == 1
            rate(store).rate_photo(label_set.set_id, item.content_key, 2 if like else 0)


def test_eval_taste_compares_yours_with_general_place_by_place(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    assert "No taste ratings yet" in invoke("eval", "taste").stdout
    rate_for_taste(configured, archive)
    text = invoke("eval", "taste")
    assert text.exit_code == 0, text.output
    assert "Old Mill" in text.stdout
    assert "Harbor" in text.stdout
    (report,) = json.loads(invoke("eval", "taste", "--json").stdout)["reports"]
    assert {r["location"] for r in report["places"]} == {"Old Mill", "Harbor"}
    assert set(report) >= {"general_strength", "yours_strength", "yours_wins", "feature", "l2"}
    assert report["first_at"] == [20, 40]
    assert set(report["general_first"][0]) == {"rated", "liked", "share"}
    assert set(report["places"][0]["yours_first"][1]) == {"rated", "liked"}
    assert "first 20" in text.stdout
    assert "average of places" in text.stdout
    assert "(2 places, " in text.stdout  # every headline says what it rests on
    assert "Yours is not shown to beat General" in text.stdout
    assert "at least 5" in text.stdout
    assert "wins" not in text.stdout
    strength = report["strength"]
    assert set(strength) == {"a", "b", "difference"}
    assert set(strength["difference"]) >= {"value", "lo", "hi", "places", "rated"}
    assert strength["difference"]["places"] == 2
    assert set(report["macro"]) >= {"general_strength", "yours_strength"}
    assert len(report["first"]) == 2
    assert report["left_out"] == {"copies": 0, "same_place": 0, "same_places": []}
    assert "Never learned from where they could give the answer away: 0 copies" in text.stdout
    both = json.loads(invoke("eval", "taste", "--l2", "1", "--l2", "30", "--json").stdout)
    assert [r["l2"] for r in both["reports"]] == [1.0, 30.0]


def test_eval_taste_says_how_many_ratings_were_drawn_for_their_value(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    # Sessions of 2, so the mill's three bursts are drawn by value, not taken whole.
    monkeypatch.setattr(evaluation, "TASTE_RANDOM", 1)
    monkeypatch.setattr(evaluation, "VALUE_QUOTAS", (("first: General only", 1),))
    monkeypatch.setattr(sessions_module, "VALUE_DEPTH", 1)
    made = invoke("eval", "new-taste", archive / "Old Mill - March 2021")
    assert "drawn by value" in made.stdout
    rate_for_taste(configured, archive)
    text = invoke("eval", "taste")
    assert text.exit_code == 0, text.output
    assert (
        "Chosen by a taste's ranking (drawn for their value, or shown in Compare "
        "Strongest): 1 rated photo." in (" ".join(text.stdout.split()))
    )
    (report,) = json.loads(invoke("eval", "taste", "--json").stdout)["reports"]
    mill = next(r for r in report["places"] if r["location"] == "Old Mill")
    assert mill["for_value"] == 1


def test_eval_taste_measures_a_candidate_sharpness_at_each_quality_weight(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    rate_for_taste(configured, archive)
    args = (
        "eval",
        "taste",
        "--technical",
        "3",
        "--quality-weight",
        "0.2",
        "--quality-weight",
        "0.5",
    )
    text = invoke(*args)
    assert text.exit_code == 0, text.output
    assert "quality weight 0.2" in text.stdout
    assert "General's strength with technical@3" in text.stdout
    assert "minus today's" in text.stdout
    assert "Strength minus the aesthetic alone, at quality weight 0.2: General" in text.stdout
    (first, second) = json.loads(invoke(*args, "--json").stdout)["reports"]
    assert (first["quality_weight"], second["quality_weight"]) == (0.2, 0.5)
    assert first["technical"] == "technical@3"
    assert set(first["candidate"]) == {"a", "b", "difference"}
    assert set(first["with_quality"]) == {"general", "yours"}
    assert set(first["with_quality"]["general"]) == {"a", "b", "difference"}
    assert first["unread"] == 0
    assert invoke("eval", "taste", "--technical", "4").exit_code != 0


def test_eval_taste_reads_a_file_of_scores_beside_general_and_yours(
    configured: Settings, archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    rate_for_taste(configured, archive)
    judge = tmp_path / "judge.jsonl"
    judge.write_text(
        "".join(  # a judge that is always right: rate_for_taste likes every other photo shown
            json.dumps({"key": item.content_key, "logit_yes_no": float(i % 2), "p_yes": 0.5}) + "\n"
            for label_set in LabelStore(configured.labels_dir).all()
            for i, item in enumerate(label_set.items)
        )
    )
    text = invoke("eval", "taste", "--scores", judge, "--score-field", "logit_yes_no")
    assert text.exit_code == 0, text.output
    assert "judge@" in text.stdout
    assert "On the whole range" in text.stdout
    assert "Among the photos a taste chose to show" in text.stdout
    assert "minus General" in text.stdout
    assert "Yours plus judge@" in text.stdout
    (report,) = json.loads(
        invoke("eval", "taste", "--scores", judge, "--score-field", "logit_yes_no", "--json").stdout
    )["reports"]
    assert report["scores"]["name"].startswith("judge@")
    assert set(report["scores"]["whole"]) == {"versus_general", "blend_versus_yours"}
    assert set(report["scores"]["whole"]["versus_general"]) == {"a", "b", "difference"}
    missing = invoke("eval", "taste", "--scores", judge)  # the number is not called "score"
    assert missing.exit_code == 1
    assert "no number called 'score'" in missing.output
    assert invoke("eval", "taste", "--scores", tmp_path / "nowhere.jsonl").exit_code != 0


def test_eval_taste_says_what_a_taste_chose_to_show_is_not(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    rate_for_taste(configured, archive)
    assert invoke("train", "taste", "--force").exit_code == 0
    assert "Among the photos a taste chose" not in invoke("eval", "taste").stdout
    for place in ("Old Mill - March 2021", "Harbor - June 2022"):
        assert invoke("eval", "new-compare", archive / place).exit_code == 0
    store = LabelStore(configured.labels_dir)
    for compare in (s for s in store.all() if s.kind == "compare"):
        for i, item in enumerate(compare.items):
            rate(store).rate_photo(compare.set_id, item.content_key, 2 if i % 2 else 0)
    text = invoke("eval", "taste")
    assert text.exit_code == 0, text.output
    assert "Among the photos a taste chose to show" in text.stdout
    assert "chose these photos" in text.stdout
    assert "not a test of which" in text.stdout
    (report,) = json.loads(invoke("eval", "taste", "--json").stdout)["reports"]
    assert set(report["chosen"]) == {"a", "b", "difference"}


def test_eval_taste_measures_a_candidate_aesthetic_for_general(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    real = extractor_registry()
    plugins = {name: real.get(name) for name in real.names()} | {"tagged": OneNumber}
    monkeypatch.setattr(
        "calico.features.extractor_registry", lambda: Registry.from_mapping("extractor", plugins)
    )
    rate_for_taste(configured, archive)
    text = invoke("eval", "taste", "--aesthetic", "tagged")
    assert text.exit_code == 0, text.output
    assert "General with tagged@1 as its aesthetic, by strength" in text.stdout
    assert "By the aesthetic alone" in text.stdout
    (report,) = json.loads(invoke("eval", "taste", "--aesthetic", "tagged", "--json").stdout)[
        "reports"
    ]
    assert report["aesthetic_candidate"] == "tagged@1"
    assert set(report["candidate_general"]) == {"a", "b", "difference"}
    assert set(report["candidate_general_alone"]) == {"a", "b", "difference"}
    unknown = invoke("eval", "taste", "--aesthetic", "nope")
    assert unknown.exit_code != 0
    assert "nope" in unknown.output
    blended = invoke("eval", "taste", "--aesthetic", "tagged", "--blend")
    assert blended.exit_code == 0, blended.output
    assert "General with tagged@1 blended half and half with today's" in blended.stdout
    assert "top quarter of each place's rated photos" in blended.stdout
    (report,) = json.loads(
        invoke("eval", "taste", "--aesthetic", "tagged", "--blend", "--json").stdout
    )["reports"]
    assert report["general_blend"] is True
    assert set(report["top_candidate_general"]) == {"a", "b", "difference"}
    alone = invoke("eval", "taste", "--blend")
    assert alone.exit_code == 1
    assert "--aesthetic, --prompts or --general-embedder" in alone.stderr


class Reader(TinyEmbedder):
    """A larger embedding that reads text, registered but not configured (clip_large)."""

    name = "reader"
    version = "1"
    supports_text = True

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.eye(1, self.dim, dtype=np.float32).repeat(len(texts), axis=0)


def test_eval_taste_measures_both_templates_on_an_embedding_for_the_rated_photos_alone(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    real = extractor_registry()
    plugins = {name: real.get(name) for name in real.names()} | {"reader": Reader}
    monkeypatch.setattr(
        "calico.features.extractor_registry", lambda: Registry.from_mapping("extractor", plugins)
    )
    rate_for_taste(configured, archive)
    yours = invoke("eval", "taste", "--embedder", "reader")
    assert yours.exit_code == 0, yours.output
    assert "First picks not measured on 'reader'" in yours.stdout
    general = invoke("eval", "taste", "--general-embedder", "reader", "--json")
    assert general.exit_code == 0, general.output
    (report,) = json.loads(general.stdout)["reports"]
    assert report["aesthetic_candidate"].endswith("-reader@1")
    assert set(report["candidate_general"]) == {"a", "b", "difference"}
    assert "candidate_general_by_collection" in report
    paired = invoke("eval", "taste", "--yours-embedder", "reader")
    assert paired.exit_code == 0, paired.output
    assert "Yours learned on reader@1, by strength" in paired.stdout
    assert "yours=reader@1" in configured.experiments_path.read_text()  # each is a try


class OneNumber:
    """A one-number aesthetic, registered but not configured."""

    name = "tagged"
    version = "1"
    dim = 1
    columns = ("score",)

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        return np.asarray([[float(im.width % 7)] for im in images], dtype=np.float32)


def test_eval_taste_measures_yours_fitted_another_way_against_todays(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    rate_for_taste(configured, archive)
    text = invoke("eval", "taste", "--fit", "centred")
    assert text.exit_code == 0, text.output
    assert "Yours fitted centred on each place" in text.stdout
    (report,) = json.loads(invoke("eval", "taste", "--fit", "centred", "--json").stdout)["reports"]
    assert report["fit"] == "centred"
    assert set(report["fitted"]) == {"a", "b", "difference"}
    assert report["uncentred"] == []
    assert invoke("eval", "taste", "--fit", "upside-down").exit_code != 0
    offset = invoke("eval", "taste", "--fit", "offset")
    assert offset.exit_code == 0, offset.output
    assert "Yours fitted with General as an offset" in offset.stdout
    leaning = invoke("eval", "taste", "--fit", "leaning")
    assert leaning.exit_code == 0, leaning.output
    assert "Yours leaning on General where it has seen little" in leaning.stdout
    header = leaning.stdout.split("┡")[0]  # each place, beside today's (headers wrap)
    assert "leaning" in header
    assert "General's" in header  # how much it leaned there
    (report,) = json.loads(invoke("eval", "taste", "--fit", "leaning", "--json").stdout)["reports"]
    assert report["fitted_by_collection"] == {}  # one collection rated: nothing to split
    assert all("leaned" in place for place in report["places"])
    scaled = invoke("eval", "taste", "--fit", "by_scale")
    assert scaled.exit_code == 1  # reading a photo's scale needs a text encoder
    assert "clip" in scaled.stderr


def test_eval_taste_shows_where_it_compares_views_and_that_it_is_once_per_place(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first picks compare each rated place's views part by part (one pick per view):
    minutes at a place no search has compared, which a run must not spend in silence."""
    rate_for_taste(configured, archive)
    heard: list[object] = []
    real = sessions_module.Sessions.evaluate_taste

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        heard.append(kwargs.get("progress"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(sessions_module.Sessions, "evaluate_taste", spy)
    assert invoke("eval", "taste").exit_code == 0
    assert heard
    assert callable(heard[0])
    shown = io.StringIO()
    monkeypatch.setenv("TERM", "xterm-256color")  # a terminal that draws the bar
    monkeypatch.setattr(cli, "_err", lambda: Console(file=shown, force_terminal=True, width=120))
    with cli._place_progress() as progress:
        progress("Old Mill", 5, 40)
    assert "Comparing views at Old Mill" in shown.getvalue()
    assert "once per place" in shown.getvalue()


def test_eval_taste_measures_a_candidate_wording_for_general(
    configured: Settings, archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rate_for_taste(configured, archive)
    candidate = tmp_path / "phone.toml"
    candidate.write_text(
        '[labels.strong]\nprompts = ["a fine photograph"]\n'
        '[labels.weak]\nprompts = ["a phone snapshot"]\n'
    )
    refused = invoke("eval", "taste", "--prompts", str(candidate))
    assert refused.exit_code == 1  # the test embedder reads no text
    assert "clip" in refused.stderr
    heard: list[object] = []
    real = sessions_module.Sessions.evaluate_taste

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        heard.append(kwargs.get("prompts"))
        return real(self, *args, **{**kwargs, "prompts": None})

    monkeypatch.setattr(sessions_module.Sessions, "evaluate_taste", spy)
    assert invoke("eval", "taste", "--prompts", str(candidate)).exit_code == 0
    assert heard == [candidate]
    both = invoke("eval", "taste", "--prompts", str(candidate), "--aesthetic", "laion_aesthetic")
    assert heard[-1] == candidate  # passed on; Sessions refuses the pair (test_engine)
    assert both.exit_code == 0


def test_eval_disagreements_lists_and_shows_where_the_tastes_disagree_most(
    configured: Settings, archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S5: look at the rated photos General and Yours rank furthest apart, by whom you sided
    with, on a contact sheet that opens anywhere."""
    monkeypatch.setenv("COLUMNS", "200")
    assert "No taste ratings yet" in invoke("eval", "disagreements").stdout
    rate_for_taste(configured, archive, liked=DISAGREE)
    text = invoke("eval", "disagreements")
    assert text.exit_code == 0, text.output
    assert "Liked, and Yours ranked it far above General" in text.stdout
    report = json.loads(invoke("eval", "disagreements", "--json").stdout)
    assert set(report) >= {"general_missed", "general_wrong", "yours_missed", "yours_wrong"}
    assert report["places"] == 2
    shown = [d for group in ("general_missed", "general_wrong") for d in report[group]]
    shown += [d for group in ("yours_missed", "yours_wrong") for d in report[group]]
    assert shown  # two or three rated photos a place, ranked 0 to 1: they disagree somewhere
    assert all(Path(d["path"]).is_file() for d in shown)
    sheet = tmp_path / "sheet.html"
    made = invoke("eval", "disagreements", "--sheet", str(sheet))
    assert made.exit_code == 0, made.output
    page = sheet.read_text()
    assert page.count("data:image/jpeg;base64,") == len(shown)  # opens offline, anywhere
    assert "General misses" in page
    inside = invoke("eval", "disagreements", "--sheet", str(archive / "sheet.html"))
    assert inside.exit_code == 1
    assert "never writes into your archive" in inside.stderr
    assert not (archive / "sheet.html").exists()


def test_the_hold_out_is_drawn_once_shown_and_left_out_of_every_eval(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    rate_for_taste(configured, archive)
    assert "Nothing is held out yet" in invoke("holdout", "show").stdout
    assert "Nothing is held out yet" in invoke("eval", "taste").stdout
    refused = invoke("eval", "taste", "--holdout")
    assert refused.exit_code == 1
    assert "calico holdout draw" in refused.stderr
    drawn = invoke("holdout", "draw")
    assert drawn.exit_code == 0, drawn.output
    assert "Held out 0 places, 0 same-view references, 0 story sessions" in drawn.stdout
    assert invoke("holdout", "draw").exit_code == 1  # drawn once
    shown = json.loads(invoke("holdout", "show", "--json").stdout)
    assert set(shown) == {"places", "references", "story_sessions", "seed", "drawn_at"}


def test_eval_taste_leaves_a_held_out_place_out_and_reports_it_alone_when_asked(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    rate_for_taste(configured, archive)
    write_holdout(
        configured.holdout_path,
        HoldOut(("Harbor",), (), (), seed=1, drawn_at=datetime(2026, 10, 29, tzinfo=UTC)),
    )
    text = invoke("eval", "taste")
    assert "Left out: the hold-out (1 place, 0 same-view references" in text.stdout
    (report,) = json.loads(invoke("eval", "taste", "--json").stdout)["reports"]
    assert "Harbor" not in {r["location"] for r in report["places"]}
    alone = invoke("eval", "taste", "--holdout", "--json")
    assert alone.exit_code == 0, alone.output
    (report,) = json.loads(alone.stdout)["reports"]
    assert {r["location"] for r in report["places"]} | set(report["one_sided"]) == {"Harbor"}
    for command in ("compare", "anchors", "stories", "ratings", "same-view"):
        result = invoke("eval", command, "--holdout")
        assert result.exit_code == 0, (command, result.output)
        assert "On the hold-out alone" in result.stdout


def test_each_eval_counts_the_settings_tried_so_a_best_is_read_as_a_best_of_n(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    rate_for_taste(configured, archive)
    first = json.loads(invoke("eval", "taste", "--json").stdout)
    assert first["tried"] == {"settings": 1, "holdout_looks": 0}
    assert "Settings tried" not in invoke("eval", "taste").stdout  # still one setting
    text = invoke("eval", "taste", "--l2", "30")
    assert "Settings tried with calico eval taste so far: 2" in text.stdout
    assert "best of 2" in text.stdout
    lines = configured.experiments_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["command"] for line in lines] == ["taste"] * 3
    write_holdout(
        configured.holdout_path,
        HoldOut(("Harbor",), (), (), seed=1, drawn_at=datetime(2026, 10, 29, tzinfo=UTC)),
    )
    looked = invoke("eval", "taste", "--holdout")
    assert "Looks at the hold-out with calico eval taste so far: 1" in looked.stdout


def test_a_hold_out_drawn_again_starts_with_no_looks(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    rate_for_taste(configured, archive)
    for day in (30, 31):  # two looks at the first hold-out, which was drawn on 29 October
        at = datetime(2026, 10, day, tzinfo=UTC)
        record(configured.experiments_path, "taste", ["defaults"], holdout=True, at=at)
    write_holdout(  # retired and drawn again
        configured.holdout_path,
        HoldOut(("Harbor",), (), (), seed=2, drawn_at=datetime(2026, 11, 1, tzinfo=UTC)),
    )
    looked = invoke("eval", "taste", "--holdout")
    assert "Looks at the hold-out with calico eval taste so far: 1" in looked.stdout


def test_holdout_draw_refuses_a_left_out_place_it_cannot_find(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    rate_for_taste(configured, archive)
    refused = invoke("holdout", "draw", "--leave-out", "Harbour")
    assert refused.exit_code == 1
    assert "no location named 'Harbour'" in refused.stderr
    assert "did you mean 'Harbor'" in refused.stderr
    assert not configured.holdout_path.exists()


def test_eval_stories_re_scores_the_stories_you_rated(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    assert "No rated stories yet" in invoke("eval", "stories").stdout
    assert "calico eval new-stories" in invoke("eval", "stories").stdout
    run = json.loads(invoke("select", "triptych", archive, "--limit", "1", "--json").stdout)
    ratings = rate(LabelStore(configured.labels_dir))
    for pick, rating in zip(run["groups"][0]["picks"], (1, -1, -1), strict=True):
        ratings.rate_result(run["run_id"], pick["content_key"], rating)
    text = invoke("eval", "stories", "--too-alike", "0.8", "--variety", "0.5")
    assert text.exit_code == 0, text.output
    assert "no variety" in text.stdout
    assert "on story results" in text.stdout
    assert "rated whole in sessions" not in text.stdout  # none yet
    assert "0 liked, 1 rejected, 0 mixed" in text.stdout
    assert "minus no variety" in text.stdout
    report = json.loads(invoke("eval", "stories", "--too-alike", "0.8", "--json").stdout)
    assert report["sessions"]["rows"][0]["auc"] is None
    report = report["results"]
    assert [(r["too_alike"], r["weight"]) for r in report["rows"]] == [
        (None, 0.0),
        (0.8, 0.25),
        (0.8, 0.5),
    ]
    assert (report["liked"], report["rejected"]) == (0, 1)
    assert report["rows"][0]["versus"] is None
    assert report["rows"][1]["versus"]["unit"] == "run or session"


def test_eval_finished_stories_compares_templates_and_story_options(
    configured: Settings, archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    assert invoke("index", archive).exit_code == 0
    story = [str(archive / "Old Mill - March 2021" / f"view{i}.jpg") for i in (1, 2, 3)]
    story[2] = str(archive / "Harbor - June 2022" / "h10.jpg")
    path = tmp_path / "finished.json"
    path.write_text(json.dumps({"stories": {"A": {"paths": story}}}), encoding="utf-8")
    args = ("eval", "finished-stories", path, "--template", "general")
    text = invoke(*args, "--variant", "variety_weight=0; candidates=300")
    assert text.exit_code == 0, text.output
    assert "variety_weight=0; candidates=300" in text.stdout
    assert "defaults" in text.stdout
    (report, varied) = json.loads(invoke(*args, "--variant", "variety_weight=0", "--json").stdout)[
        "reports"
    ]
    assert (report["template"], report["variant"]) == ("general", "defaults")
    assert varied["variant"] == "variety_weight=0"
    assert set(report) >= {"mean_win", "shortlisted", "all_three_shortlisted", "stories", "missing"}
    assert report["anchors_tried"] == 1.0
    assert report["stories"][0]["anchor_tried"] is True
    assert "anchors tried" in text.stdout
    assert report["versus"] is None
    assert report["win_interval"]["unit"] == "story"
    assert varied["versus"]["places"] == 1
    assert "minus the first" in text.stdout
    assert "wins" not in text.stdout
    (scored,) = report["stories"]
    assert scored["name"] == "A"
    assert len(scored["wins"]) == 3


def test_eval_new_stories_sets_a_named_candidate_beside_todays_way(
    configured: Settings, archive: Path
) -> None:
    made = invoke(
        "eval",
        "new-stories",
        archive,
        "--variant",
        "sy-vw55",
        "--template",
        "general",
        "-o",
        "variety_weight=0.55",
    )
    assert made.exit_code == 0, made.stdout
    (label_set,) = LabelStore(configured.labels_dir).all()
    assert label_set.scorer.endswith('candidate:sy-vw55 {"variety_weight": 0.55}')
    alone = invoke("eval", "new-stories", archive, "-o", "variety_weight=0.55")
    assert alone.exit_code != 0
    assert "--variant" in alone.output


def test_eval_learned_stories_judges_a_scorer_learned_from_your_answers_against_today(
    configured: Settings, archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    empty = invoke("eval", "learned-stories")
    assert "No stories rated works or doesn't yet" in empty.stdout
    assert "calico eval new-stories" in empty.stdout
    assert invoke("eval", "new-stories", archive).exit_code == 0
    (label_set,) = LabelStore(configured.labels_dir).all()
    rate(LabelStore(configured.labels_dir)).rate_story(label_set.set_id, 0, 0, doesnt_fit=[])
    story = [str(archive / "Old Mill - March 2021" / f"view{i}.jpg") for i in (1, 2, 3)]
    path = tmp_path / "finished.json"
    path.write_text(json.dumps({"stories": {"A": {"paths": story}}}), encoding="utf-8")
    text = invoke("eval", "learned-stories", path, "--l2", "3")
    assert text.exit_code == 0, text.output
    assert "Stories: archive" in text.stdout
    assert "learned: not measured" in text.stdout  # one session: nowhere else to learn
    assert "learned from as stories that work and never judged: 1." in text.stdout
    assert "strength" in text.stdout  # the weights it learned, signal by signal
    assert "with swaps" not in text.stdout  # only when asked for
    alone = invoke("eval", "learned-stories", path, "--swaps")  # the mill has three photos
    assert "No swapped stories could be made" in alone.stdout
    story[2] = str(archive / "Harbor - June 2022" / "h10.jpg")  # now the harbor's can swap in
    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps({"stories": {"A": {"paths": story}}}), encoding="utf-8")
    swapped = invoke("eval", "learned-stories", mixed, "--swaps")
    assert swapped.exit_code == 0, swapped.output
    assert "learned, with swaps" in swapped.stdout
    assert "Swapped stories, learned from as stories that don't work:" in swapped.stdout
    assert invoke("eval", "learned-stories", "--swaps").exit_code != 0  # needs the file
    varied = invoke("eval", "learned-stories", path, "--variant", "distinct_weight=0.3")
    assert varied.exit_code == 0, varied.output
    assert "distinct_weight=0.3 minus today" in varied.stdout
    report = json.loads(invoke("eval", "learned-stories", path, "--json").stdout)
    assert (report["finished"], report["l2"], report["swaps"]) == (1, 1.0, 0)
    (session,) = report["sessions"]
    assert (session["works"], session["doesnt"]) == (0, 1)
    assert "bond" in report["signals"]


def test_eval_finished_stories_says_what_the_file_must_hold(
    configured: Settings, tmp_path: Path
) -> None:
    result = invoke("eval", "finished-stories", tmp_path / "nope.json")
    assert result.exit_code == 1
    assert "paths" in result.output


def test_train_taste_saves_only_a_taste_that_wins_unless_forced(
    configured: Settings, archive: Path
) -> None:
    rate_for_taste(configured, archive)
    plain = invoke("train", "taste")
    assert plain.exit_code == 0, plain.output
    assert "not shown to beat General" in plain.stdout
    assert "Not saved" in plain.stdout
    assert "--force" in plain.stdout
    forced = invoke("train", "taste", "--force")
    assert forced.exit_code == 0, forced.output
    assert "Saved" in forced.stdout
    assert "no probability per pick yet" in forced.stdout  # two places cannot calibrate it
    result = json.loads(invoke("select", "best", archive, "--template", "yours", "--json").stdout)
    assert result["request"]["template"] == "yours"
    assert all(p["probability"] is None for p in result["groups"][0]["picks"])
    assert result["groups"][0]["marker"] is None


def test_train_taste_says_what_a_probability_means_once_it_can_give_one(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr(calibration_module, "MIN_FORECASTS", 4)
    monkeypatch.setattr(calibration_module, "MIN_PLACES", 2)
    monkeypatch.setattr(calibration_module, "SURE", -1e9)
    rate_for_taste(configured, archive)
    assert invoke("train", "taste", "--force").exit_code == 0
    for place in ("Old Mill - March 2021", "Harbor - June 2022"):
        assert invoke("eval", "new-compare", archive / place).exit_code == 0
    store = LabelStore(configured.labels_dir)
    for compare in (s for s in store.all() if s.kind == "compare"):
        for i, item in enumerate(compare.items):  # the photos Strongest showed first, rated
            rate(store).rate_photo(compare.set_id, item.content_key, 2 if i % 2 else 0)
    trained = invoke("train", "taste", "--force")
    assert trained.exit_code == 0, trained.output
    assert "how likely you are to like them" in trained.stdout
    assert "one it gives 90% is one you liked" in trained.stdout


def test_a_compare_session_can_show_fewer_or_more_picks_of_each_list(
    configured: Settings, archive: Path
) -> None:
    harbor = archive / "Harbor - June 2022"
    rate_for_taste(configured, archive)
    assert invoke("train", "taste", "--force").exit_code == 0
    store = LabelStore(configured.labels_dir)
    assert invoke("eval", "new-compare", harbor, "--picks", "2").exit_code == 0
    (short,) = [s for s in store.all() if s.kind == "compare"]
    assert len([i for i in short.items if not i.control]) <= 4  # 2 of each list
    assert max(s.rank for i in short.items for s in i.shown_by) == 1
    refused = invoke("eval", "new-compare", harbor, "--picks", "1")
    assert refused.exit_code != 0  # a list of one has no order to read


def test_compare_strongest_rates_both_templates_first_picks_blind(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    harbor = archive / "Harbor - June 2022"
    untrained = invoke("eval", "new-compare", harbor)
    assert untrained.exit_code == 1
    assert "calico train taste" in untrained.output
    assert "No Compare Strongest ratings yet" in invoke("eval", "compare").stdout

    rate_for_taste(configured, archive)
    assert invoke("train", "taste", "--force").exit_code == 0
    made = invoke("eval", "new-compare", harbor)
    assert made.exit_code == 0, made.output
    assert "photos to rate" in made.stdout
    store = LabelStore(configured.labels_dir)
    (compare,) = [s for s in store.all() if s.kind == "compare"]
    for i, item in enumerate(compare.items):
        rate(store).rate_photo(compare.set_id, item.content_key, 2 if i % 2 else 0)
    text = invoke("eval", "compare")
    assert text.exit_code == 0, text.output
    assert "Harbor" in text.stdout
    assert "In-sample" in text.stdout  # the taste learned from every place, Harbor too
    report = json.loads(invoke("eval", "compare", "--json").stdout)
    assert report["rows"] == []
    apart = report["in_sample"]
    assert [r["place"] for r in apart["rows"]] == ["Harbor"]
    assert apart["rows"][0]["rated"] == len(compare.items)
    assert set(apart["general"]) == {"rated", "liked", "share"}
    assert apart["shown"]["difference"]["places"] == 1
    assert set(apart["only"]) == {"a", "b", "difference"}
    assert set(apart["macro"]) == {"general", "yours"}
    assert "Yours minus General" in text.stdout
    assert "(1 place, " in text.stdout
    assert "average of places" in text.stdout


def test_compare_shows_each_places_bursts_and_sets_the_small_ones_apart_when_asked(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    rate_for_taste(configured, archive)
    assert invoke("train", "taste", "--force").exit_code == 0
    quarry = archive / "Quarry - July 2023"
    for seed in (20, 21, 22):
        f.save_jpeg(
            f.scene(seed),
            quarry / f"q{seed}.jpg",
            captured_at=datetime(2022, 5, 6, 10, 0, 0) + timedelta(hours=seed),
        )
    assert invoke("index", archive).exit_code == 0
    assert invoke("eval", "new-compare", quarry, "--picks", "2").exit_code == 0
    store = LabelStore(configured.labels_dir)
    (compare,) = [s for s in store.all() if s.kind == "compare"]
    for i, item in enumerate(compare.items):
        rate(store).rate_photo(compare.set_id, item.content_key, 2 if i % 2 else 0)

    plain = invoke("eval", "compare")
    assert plain.exit_code == 0, plain.output
    assert "bursts" in plain.stdout
    assert "Fewer than" not in plain.stdout  # no size was asked for
    apart = invoke("eval", "compare", "--min-bursts", "1000000")
    assert apart.exit_code == 0, apart.output
    assert "Fewer than 1000000 bursts" in apart.stdout
    report = json.loads(invoke("eval", "compare", "--json", "--min-bursts", "1000000").stdout)
    assert report["rows"] == []
    assert [r["place"] for r in report["small"]["rows"]] == ["Quarry"]
    assert report["small"]["rows"][0]["bursts"] >= 1
    assert json.loads(invoke("eval", "compare", "--json").stdout)["small"] is None


def test_a_place_whose_size_is_unknown_is_said_to_stay_in_the_table_when_a_size_is_asked() -> None:
    def session(place: str, bursts: int | None) -> tuple[str, list[evaluation.ComparedPhoto]]:
        photos = [
            evaluation.ComparedPhoto(True, frozenset({"yours"}), (("yours", 0),), bursts=bursts),
            evaluation.ComparedPhoto(
                False, frozenset({"general"}), (("general", 0),), bursts=bursts
            ),
        ]
        return place, photos

    report = evaluation.measure_compare(
        [session("Mill", None), session("Harbor", 500)], min_bursts=100, resamples=10
    )
    said = io.StringIO()
    cli._print_unsized(Console(file=said, width=200), report, 100)
    assert "No burst count for Mill" in said.getvalue()
    assert "Harbor" not in said.getvalue()
    quiet = io.StringIO()
    cli._print_unsized(
        Console(file=quiet, width=200), report, None
    )  # no size asked: nothing to say
    assert quiet.getvalue() == ""


def test_new_compare_can_show_yours_fitted_without_the_place_and_compare_reports_it_apart(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    rate_for_taste(configured, archive)
    assert invoke("train", "taste", "--force").exit_code == 0
    harbor = archive / "Harbor - June 2022"
    made = invoke("eval", "new-compare", harbor, "--cross-fit", "--picks", "2")
    assert made.exit_code == 0, made.output
    assert "without" in made.stdout  # says whose Yours it is showing
    store = LabelStore(configured.labels_dir)
    (compare,) = [s for s in store.all() if s.kind == "compare"]
    assert compare.taste is not None
    assert compare.taste.cross_fit
    for i, item in enumerate(compare.items):
        rate(store).rate_photo(compare.set_id, item.content_key, 2 if i % 2 else 0)

    text = invoke("eval", "compare")
    assert text.exit_code == 0, text.output
    assert "fitted without the place" in text.stdout
    assert "In-sample" not in text.stdout  # not judged on what it learned
    report = json.loads(invoke("eval", "compare", "--json").stdout)
    assert report["rows"] == []
    assert [r["place"] for r in report["cross_fitted"]["rows"]] == ["Harbor"]
    assert json.loads(invoke("eval", "compare", "--json", "--min-bursts", "1000000").stdout)[
        "cross_fitted"
    ]["small"]["rows"]


def test_new_compare_cross_fit_is_refused_with_a_way_to_fix_it_when_nothing_else_is_rated(
    configured: Settings, archive: Path
) -> None:
    harbor = archive / "Harbor - June 2022"
    assert invoke("eval", "new-taste", harbor).exit_code == 0
    store = LabelStore(configured.labels_dir)
    (taste,) = [s for s in store.all() if s.kind == "taste"]
    for i, item in enumerate(taste.items):
        rate(store).rate_photo(taste.set_id, item.content_key, 2 if i % 2 else 0)
    refused = invoke("eval", "new-compare", harbor, "--cross-fit")
    assert refused.exit_code == 1
    assert "Rate more places" in refused.output


def test_eval_list_counts_a_story_sessions_stories(configured: Settings, archive: Path) -> None:
    assert invoke("eval", "new-stories", archive).exit_code == 0
    assert "0 of 1" in invoke("eval", "list").stdout  # the CLI-test archive tells one story


def test_select_says_when_yours_is_not_trained(configured: Settings, archive: Path) -> None:
    result = invoke("select", "best", archive, "--template", "yours")
    assert result.exit_code == 0, result.output
    assert "calico train taste" in result.stdout


def test_select_describe_weight(configured: Settings, archive: Path) -> None:
    result = invoke("select", "best", archive, "--describe-weight", "0.2", "--json")
    assert json.loads(result.stdout)["request"]["describe_weight"] == 0.2


def test_select_retired_criteria_weight_points_to_the_flag(
    configured: Settings, archive: Path
) -> None:
    result = invoke("select", "best", archive, "-o", "criteria_weight=2")
    assert result.exit_code == 1
    assert "--describe-weight" in result.stderr


def test_select_bad_option_is_a_clean_error(configured: Settings, archive: Path) -> None:
    result = invoke("select", "best", archive, "-o", "colour=red")
    assert result.exit_code == 1
    assert "colour" in result.stderr


def test_select_malformed_option_is_a_clean_error(configured: Settings, archive: Path) -> None:
    result = invoke("select", "best", archive, "-o", "noequals")
    assert result.exit_code == 1
    assert "KEY=VALUE" in result.stderr


def test_select_unknown_task(configured: Settings, archive: Path) -> None:
    result = invoke("select", "nope", archive)
    assert result.exit_code == 1
    assert "best" in result.stderr


def test_select_reference_is_forwarded(configured: Settings, archive: Path, tmp_path: Path) -> None:
    ref = f.save_jpeg(f.scene(2), tmp_path / "ref.jpg")
    result = invoke("select", "best", archive, "-r", ref, "--json")
    data = json.loads(result.stdout)
    assert data["request"]["references"] == [str(ref)]
    assert any("does not use reference" in n for n in data["notes"])


def test_select_examples_are_forwarded(configured: Settings, archive: Path) -> None:
    example = next(archive.rglob("*.jpg"))
    result = invoke("select", "best", archive, "--example", example, "--json")
    data = json.loads(result.stdout)
    assert data["request"]["examples"] == [str(example.resolve())]
    assert len(data["example_keys"]) == 1


def test_select_same_view_with_reference(
    configured: Settings, archive: Path, tmp_path: Path
) -> None:
    ref = f.save_jpeg(f.scene(2), tmp_path / "ref.jpg")
    result = invoke("select", "same_view", archive, "-r", ref, "--limit", "2")
    assert result.exit_code == 0, result.output
    assert "view2.jpg" in result.stdout


def test_select_triptych(configured: Settings, archive: Path) -> None:
    result = invoke("select", "triptych", archive, "--limit", "1")
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("diversity=0.5", ("diversity", 0.5)),
        ("flag=true", ("flag", True)),
        ("n=3", ("n", 3)),
        ('x=["a", "b"]', ("x", ["a", "b"])),
        ("connections=mood, color", ("connections", ["mood", "color"])),
        ("kind=portrait", ("kind", "portrait")),
        ("expr=a=b", ("expr", "a=b")),
    ],
)
def test_parse_option(raw: str, expected: tuple[str, object]) -> None:
    assert parse_option(raw) == expected


def test_parse_option_requires_equals() -> None:
    with pytest.raises(InvalidRequestError, match="KEY=VALUE"):
        parse_option("oops")


def test_runs_list_and_show(configured: Settings, run_id: str) -> None:
    listing = invoke("runs", "list", "--limit", "5").stdout
    assert run_id in listing
    assert "best" in listing
    shown = invoke("runs", "show", run_id)
    assert shown.exit_code == 0
    assert ".jpg" in shown.stdout
    as_json = invoke("runs", "show", run_id, "--json")
    assert json.loads(as_json.stdout)["run_id"] == run_id


def test_runs_list_names_runs_it_cannot_read(configured: Settings, run_id: str) -> None:
    path = configured.runs_dir / run_id / "result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["request"]["from_a_newer_calico"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    result = invoke("runs", "list")
    assert result.exit_code == 0
    assert run_id in result.stderr
    assert "another version" in result.stderr


def test_runs_list_empty(configured: Settings) -> None:
    assert "No runs" in invoke("runs", "list").stdout


def test_runs_show_unknown_run(configured: Settings) -> None:
    result = invoke("runs", "show", "nope")
    assert result.exit_code == 1
    assert "calico runs list" in result.stderr


def test_runs_explain_shows_signals_and_placement(
    configured: Settings, archive: Path, run_id: str
) -> None:
    picked = json.loads(invoke("runs", "show", run_id, "--json").stdout)["groups"][0]["picks"][0]
    name = picked["path"].rsplit("/", 1)[1]
    out = invoke("runs", "explain", run_id, name).stdout
    assert name in out
    assert "picked" in out
    assert "score" in out
    other = invoke("runs", "explain", run_id, ".jpg").stdout
    assert "not picked" in other or "picked" in other


def test_runs_explain_no_match_suggests_checking_the_name(
    configured: Settings, run_id: str
) -> None:
    result = invoke("runs", "explain", run_id, "zzz-nothing")
    assert result.exit_code == 1
    assert "zzz-nothing" in result.stderr
    assert "check" in result.stderr.lower()


# -- doctor ------------------------------------------------------------------------------


def rate(store: LabelStore) -> Ratings:
    """Answers to ``store``'s sets, in the journal of the home that holds them."""
    home = store.labels_dir.parent
    return Ratings(store, RatingsJournal(home / "ratings"), RunStore(home / "runs"))


def by_name(checks: list[Check]) -> dict[str, Check]:
    return {c.name: c for c in checks}


def test_doctor_checks_on_a_healthy_setup(configured: Settings, archive: Path) -> None:
    invoke("index", archive)
    checks = by_name(run_checks(load_settings()))
    assert checks["python"].status == "ok"
    assert checks["calico"].status == "ok"
    assert checks["settings"].status == "ok"
    assert checks["collection archive"].status == "ok"
    assert checks["embedder"].status in {"ok", "warn"}
    assert "tiny" in checks["embedder"].detail
    assert "5 photos" in checks["catalog"].detail
    assert checks["catalog"].status == "ok"
    assert checks["runs"].detail.startswith("0")


def test_doctor_counts_runs_and_feature_coverage(configured: Settings, run_id: str) -> None:
    checks = by_name(run_checks(load_settings()))
    assert checks["runs"].detail.startswith("1")
    assert "100%" in checks["catalog"].detail
    assert "tiny@" in checks["catalog"].detail


def test_ratings_migrate_writes_the_journal_and_prints_its_counts(
    configured: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    LabelStore(configured.labels_dir).save(
        LabelSet(
            set_id="20261001-090000-tast",
            created_at=datetime(2026, 10, 1, 9, tzinfo=UTC),
            text="Your taste: Mill",
            folders=(configured.home,),
            scorer="test",
            items=(
                LabelItem(content_key="a", path=Path("/a/a.jpg"), label=2),
                LabelItem(content_key="b", path=Path("/a/b.jpg"), label=0),
                LabelItem(content_key="c", path=Path("/a/c.jpg")),
            ),
            kind="taste",
        )
    )
    with Catalog(configured.catalog_path) as catalog:
        catalog.add_feedback("20261002-100000-gone", Path("/a/w.jpg"), "w", 1)
    result = invoke("ratings", "migrate")
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "1 labelling set (1 taste)" in out
    assert "1 👍/👎" in out
    rows = {line.split("│")[1].strip(): line for line in out.splitlines() if "│" in line}
    assert set(rows) >= {"describe", "same_view", "taste", "compare", "anchor", "result"}
    assert rows["compare"].split("│")[2].strip() == "0"  # no session of it, shown as 0
    assert "All 3 answers are in the journal" in out
    assert "1 👍/👎 is on a run that cannot be read" in out
    assert (configured.home / "ratings").is_dir()
    again = invoke("ratings", "migrate", "--json")
    report = json.loads(again.stdout)
    assert report["complete"] is True
    assert report["counts"]["taste"] == {"found": 2, "written": 0, "in_journal": 2}
    assert report["sets"] == {"taste": 1}
    assert report["feedback_rows"] == 1


def test_ratings_migrate_fails_when_the_journal_misses_an_answer(
    configured: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def short(*_: object) -> MigrationReport:
        return MigrationReport(
            journal=configured.home / "ratings",
            counts=(UseCount("taste", found=2, written=2, in_journal=1),),
            sets={"taste": 1},
            feedback_rows=0,
            lost_runs=0,
            unreadable=(),
        )

    monkeypatch.setattr(cli, "migrate", short)
    result = invoke("ratings", "migrate")
    assert result.exit_code == 1
    assert "1 of 2 answers are not in the journal" in result.stderr


def test_backup_writes_an_archive_and_verify_reads_it_back(
    configured: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    (configured.home / "labels").mkdir(parents=True, exist_ok=True)
    (configured.home / "labels" / "20261001-090000-tast.json").write_text("{}", encoding="utf-8")
    made = invoke("backup", tmp_path / "backups")
    assert made.exit_code == 0, made.output
    (path,) = (tmp_path / "backups").glob("calico-backup-*.tar.gz")
    assert str(path) in made.stdout
    assert "labels" in made.stdout
    assert "catalog is not in it" in made.stdout
    checked = invoke("backup", "--verify", path)
    assert checked.exit_code == 0, checked.output
    assert "Every file in it reads" in checked.stdout
    assert "labels" in checked.stdout


def test_backup_refuses_a_folder_inside_the_home(configured: Settings) -> None:
    result = invoke("backup", configured.home / "backups")
    assert result.exit_code == 1
    assert "inside the calico home" in result.stderr


def test_backup_needs_a_destination_or_a_file_to_verify(configured: Settings) -> None:
    result = invoke("backup")
    assert result.exit_code == 2
    assert "calico backup <folder>" in result.output  # not a word to type as it stands


def test_backup_without_a_destination_names_the_folder_the_last_one_went_into(
    configured: Settings,
) -> None:
    note = configured.home / "last_backup.json"
    when = datetime.now(UTC).isoformat()
    note.write_text(json.dumps({"at": when, "path": "/Volumes/Spare/Calico Backups/b.tar.gz"}))
    result = invoke("backup")
    assert result.exit_code == 2
    assert "'/Volumes/Spare/Calico Backups'" in result.output.replace("\n", " ")
    assert "calico backup DEST" not in result.output


def test_doctor_warns_until_there_is_a_recent_backup(configured: Settings) -> None:
    check = by_name(run_checks(load_settings()))["backup"]
    assert (check.status, check.detail) == ("warn", "no backup made yet")
    assert "calico backup" in check.hint
    note = configured.home / "last_backup.json"
    recent = datetime.now(UTC) - timedelta(days=3)
    note.write_text(json.dumps({"at": recent.isoformat(), "path": "/b/x.tar.gz"}))
    check = by_name(run_checks(load_settings()))["backup"]
    assert check.status == "ok"
    assert "3 days ago" in check.detail
    old = datetime.now(UTC) - timedelta(days=31)
    note.write_text(json.dumps({"at": old.isoformat(), "path": "/b/x.tar.gz"}))
    check = by_name(run_checks(load_settings()))["backup"]
    assert check.status == "warn"
    assert "31 days ago" in check.detail


def test_doctor_says_where_the_last_backup_went_and_warns_on_ratings_made_since(
    configured: Settings,
) -> None:
    note = configured.home / "last_backup.json"
    saved = datetime.now(UTC) - timedelta(days=3)
    note.write_text(json.dumps({"at": saved.isoformat(), "path": "/b/old folder/x.tar.gz"}))
    journal = RatingsJournal(configured.ratings_dir)

    def rated(when: datetime, count: int) -> None:
        journal.append(
            *[
                Answer.model_validate(
                    {"at": when, "use": "taste", "subject": (f"{when:%d%H}-{i}",), "value": 2}
                )
                for i in range(count)
            ]
        )

    rated(saved - timedelta(hours=1), cli.BACKUP_STALE_ANSWERS + 50)  # in the backup already
    check = by_name(run_checks(load_settings()))["backup"]
    assert check.status == "ok"
    assert "since" not in check.detail
    rated(saved + timedelta(hours=1), cli.BACKUP_STALE_ANSWERS + 1)
    check = by_name(run_checks(load_settings()))["backup"]
    assert check.status == "warn"
    assert f"{cli.BACKUP_STALE_ANSWERS + 1} ratings since" in check.detail
    assert check.hint.endswith("calico backup '/b/old folder'")  # the folder to run it with


def test_doctor_reports_scratch_space_in_use_and_left_behind(configured: Settings) -> None:
    home = configured.home
    assert by_name(run_checks(load_settings()))["scratch"].status == "ok"
    in_use = home / f"patches-{os.getpid()}-abc"
    in_use.mkdir()
    (in_use / "patches.npy").write_bytes(b"x" * 8192)
    check = by_name(run_checks(load_settings()))["scratch"]
    assert check.status == "ok"
    assert "in use by a running Calico" in check.detail
    left = home / "patches-999999999-def"  # no process has an id this large
    older = home / "patches-k2j3h4"
    for folder in (left, older):
        folder.mkdir()
        (folder / "patches.npy").write_bytes(b"x" * 8192)
    check = by_name(run_checks(load_settings()))["scratch"]
    assert check.status == "warn"
    assert "1 left by a stopped Calico" in check.detail
    assert "1 left by an older Calico" in check.detail
    assert "kB" in check.detail
    assert str(older) in check.hint
    assert in_use.is_dir()
    assert left.is_dir()  # doctor reports; it never deletes


def test_doctor_says_whether_raw_files_can_be_read(
    configured: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "rawpy", f.fake_rawpy())
    check = by_name(run_checks(load_settings()))["raw"]
    assert (check.status, check.detail) == ("ok", "RAW files are read by LibRaw 0.22.1")
    monkeypatch.setitem(sys.modules, "rawpy", None)
    check = by_name(run_checks(load_settings()))["raw"]
    assert check.status == "warn"
    assert check.hint == "uv sync --extra raw"


def test_doctor_warns_on_unmounted_collection(configured: Settings, archive: Path) -> None:
    for child in list(archive.iterdir()):
        for img in child.iterdir():
            img.unlink()
        child.rmdir()
    archive.rmdir()
    check = by_name(run_checks(load_settings()))["collection archive"]
    assert check.status == "warn"
    assert "not mounted" in check.detail
    assert check.hint


def test_doctor_warns_when_two_collections_share_a_root(
    configured: Settings, archive: Path
) -> None:
    again = CollectionConfig(name="again", root=archive)
    save_settings(configured.with_collection(again))
    checks = by_name(run_checks(load_settings()))
    assert checks["collection archive"].status == "ok"
    check = checks["collection again"]
    assert check.status == "warn"
    assert "'archive' has the same root" in check.detail
    assert check.hint
    assert "calico.toml" in check.hint


def test_doctor_without_settings_file_warns(calico_home: Path) -> None:
    checks = by_name(run_checks(load_settings()))
    assert checks["settings"].status == "warn"
    assert "calico collections add" in checks["settings"].hint


def test_doctor_counts_failures_and_answers(configured: Settings, archive: Path) -> None:
    (archive / "Harbor - June 2022" / "bad.jpg").write_bytes(b"nope")
    invoke("index", archive)
    RatingsJournal(configured.ratings_dir).append(
        Answer.model_validate(
            {"at": "2026-10-01T09:00:00Z", "use": "taste", "subject": ["k"], "value": 2}
        )
    )
    checks = by_name(run_checks(load_settings()))
    assert checks["catalog"].status == "warn"
    assert "1 failure" in checks["catalog"].detail
    assert checks["ratings"].status == "ok"
    assert "1 answer in" in checks["ratings"].detail


@pytest.mark.skipif(HAS_TORCH, reason="needs torch to be missing")
def test_doctor_fails_when_clip_is_unavailable(configured: Settings) -> None:
    save_settings(configured.model_copy(update={"embedder": "clip"}))
    check = by_name(run_checks(load_settings()))["embedder"]
    assert check.status == "fail"
    assert "uv sync --extra ml" in check.hint
    result = invoke("doctor")
    assert result.exit_code == 1


def test_doctor_command_renders_and_exits_zero_without_failures(configured: Settings) -> None:
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "python" in result.stdout
    assert "embedder" in result.stdout


def test_doctor_json(configured: Settings) -> None:
    result = invoke("doctor", "--json")
    rows = json.loads(result.stdout)
    assert {"name", "status", "detail", "hint"} <= set(rows[0])
    assert {r["name"] for r in rows} >= {"python", "calico", "settings", "embedder", "runs"}


def test_doctor_exit_code_one_on_fail(
    configured: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "run_checks", lambda _s: [Check("x", "fail", "broken", "fix it")])
    result = invoke("doctor")
    assert result.exit_code == 1
    assert "fix it" in result.stdout


# -- serve -------------------------------------------------------------------------------


def free_port() -> str:
    """A port nothing listens on, so a Calico server already running here does not count."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return str(probe.getsockname()[1])


def fake_web(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    module = types.ModuleType("calico.web.app")

    def create_app(settings: Settings) -> str:
        seen["settings"] = settings
        return "the-app"

    module.create_app = create_app  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "calico.web.app", module)

    import uvicorn  # noqa: PLC0415

    monkeypatch.setattr(uvicorn, "run", lambda app_, **kw: seen.update(app=app_, **kw))
    monkeypatch.setattr(cli, "open_browser_soon", lambda url: seen.update(opened=url))
    return seen


def test_serve_runs_uvicorn_and_opens_the_browser(
    monkeypatch: pytest.MonkeyPatch, configured: Settings
) -> None:
    seen = fake_web(monkeypatch)
    result = invoke("serve", "--port", "9999")
    assert result.exit_code == 0, result.output
    assert seen["app"] == "the-app"
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 9999
    assert seen["opened"] == "http://127.0.0.1:9999"
    assert seen["settings"].embedder == "tiny"


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "my-mac.local"])  # noqa: S104
def test_serve_refuses_an_address_other_computers_can_reach(
    monkeypatch: pytest.MonkeyPatch, configured: Settings, host: str
) -> None:
    """Calico has no login, so it serves only this computer."""
    seen = fake_web(monkeypatch)
    result = invoke("serve", "--no-browser", "--host", host)
    assert result.exit_code == 1
    assert "only this computer" in result.output
    assert "ssh -L 8765:127.0.0.1:8765" in result.output
    assert "app" not in seen


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_serve_listens_on_this_computers_own_addresses(
    monkeypatch: pytest.MonkeyPatch, configured: Settings, host: str
) -> None:
    seen = fake_web(monkeypatch)
    result = invoke("serve", "--no-browser", "--host", host, "--port", free_port())
    assert result.exit_code == 0, result.output
    assert seen["host"] == host


def test_serve_no_browser(monkeypatch: pytest.MonkeyPatch, configured: Settings) -> None:
    seen = fake_web(monkeypatch)
    result = invoke("serve", "--no-browser", "--port", free_port())
    assert result.exit_code == 0, result.output
    assert "opened" not in seen


def test_serve_refuses_a_port_another_server_holds(
    monkeypatch: pytest.MonkeyPatch, configured: Settings
) -> None:
    """Otherwise the old server keeps answering and the browser shows yesterday's code."""
    seen = fake_web(monkeypatch)
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        result = invoke("serve", "--port", str(port))
    assert result.exit_code == 1
    assert f"Port {port} is already in use" in result.output
    assert f"lsof -ti tcp:{port}" in result.output
    assert "app" not in seen
    assert "opened" not in seen


def test_serve_says_where_calico_is_already_served(
    monkeypatch: pytest.MonkeyPatch, configured: Settings
) -> None:
    """The always-on service restarts when stopped, so "stop it" is the wrong advice."""

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps({"engine": "ready", "error": None, "api": "abc"}).encode()
            self.send_response(200 if self.path == "/api/health" else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    seen = fake_web(monkeypatch)
    with http.server.HTTPServer(("127.0.0.1", 0), Health) as running:
        port = running.server_address[1]
        threading.Thread(target=running.serve_forever, daemon=True).start()
        try:
            result = invoke("serve", "--port", str(port))
        finally:
            running.shutdown()
    assert result.exit_code == 1
    assert f"Calico is already served at http://127.0.0.1:{port}" in result.output
    assert "a service that restarts it" in result.output
    assert "install-mac" not in result.output
    assert "--port" in result.output
    assert "lsof" not in result.output
    assert "app" not in seen


def test_open_browser_soon_schedules_a_daemon_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[Any] = []

    class FakeTimer:
        def __init__(self, delay: float, fn: Any, args: Any) -> None:
            self.daemon = False
            started.append((delay, fn, args))

        def start(self) -> None:
            started.append("start")

    monkeypatch.setattr(threading, "Timer", FakeTimer)
    cli.open_browser_soon("http://x")
    assert started[-1] == "start"
    assert started[0][2] == ["http://x"]


# -- feedback --------------------------------------------------------------------------------


def test_feedback_export_writes_csv(configured: Settings, archive: Path) -> None:
    first, second = (
        json.loads(invoke("select", "best", archive, "--json").stdout) for _ in range(2)
    )
    ratings = rate(LabelStore(configured.labels_dir))
    pick = first["groups"][0]["picks"][0]["content_key"]
    ratings.rate_result(first["run_id"], pick, 1, "lovely")
    ratings.rate_result(second["run_id"], pick, -1)
    ratings.rate_result(second["run_id"], pick, None)  # cleared
    rows = list(csv.DictReader(io.StringIO(invoke("feedback", "export").stdout)))
    assert [r["run_id"] for r in rows] == [first["run_id"], second["run_id"], second["run_id"]]
    assert rows[0]["note"] == "lovely"
    assert [r["rating"] for r in rows] == ["1", "-1", ""]
    assert rows[0]["task"] == "best"
    only = invoke("feedback", "export", "--run", second["run_id"]).stdout
    assert [r["run_id"] for r in csv.DictReader(io.StringIO(only))] == [second["run_id"]] * 2


# -- rendering details ----------------------------------------------------------------------


def make_result(picks: int, with_evidence: bool = True) -> Any:
    from calico.domain import Evidence, Group, Pick, SelectionRequest, SelectionResult  # noqa: PLC0415

    def pick(i: int) -> Pick:
        return Pick(
            path=f"/a/p{i}.jpg",
            content_key=f"k{i}",
            location="L",
            captured_at=None,
            score=0.5,
        )

    group = Group(
        title="Story",
        score=0.7,
        picks=tuple(pick(i) for i in range(picks)),
        evidence=(
            Evidence(signal="theme", value=0.82, weight=0.2, detail="same doorway"),
            Evidence(signal="mood", value=0.64, weight=0.2),
        )
        if with_evidence
        else (),
    )
    return SelectionResult(
        run_id="r",
        created_at=T0,
        request=SelectionRequest(task="best", folders=("/a",)),
        groups=(group,),
    )


def capture(result: Any) -> str:
    from rich.console import Console  # noqa: PLC0415

    buffer = io.StringIO()
    console = Console(file=buffer, width=200, highlight=False)
    cli_console = cli._out
    cli._out = lambda: console
    try:
        cli.render_result(result, footer=False)
    finally:
        cli._out = cli_console
    return buffer.getvalue()


def test_eval_compare_prints_the_liked_share_down_each_list_in_tens() -> None:
    def session(place: str) -> tuple[str, list[evaluation.ComparedPhoto]]:
        photos = []
        for rank in range(30):
            liked = rank % 10 < (9, 7, 5)[rank // 10]  # Yours stops being good down the list
            photos.append(evaluation.ComparedPhoto(liked, frozenset({"yours"}), (("yours", rank),)))
        return place, photos

    report = evaluation.measure_compare(
        [session("Mill"), session("Harbor"), session("Fisher")], resamples=50
    )
    shown = io.StringIO()
    cli._print_compare(Console(file=shown, width=200), report, "Fresh places")
    text = shown.getvalue()
    for heading in ("1-10", "11-20", "21-30", "first minus last"):
        assert heading in text
    assert "27/30 (0.90)" in text
    assert "15/30 (0.50)" in text
    assert "+0.40" in text  # the first ten minus the last
    assert "variety chose" in text  # said once, so a flat curve is not read as a result
    none = io.StringIO()
    cli._print_compare(
        Console(file=none, width=200), evaluation.measure_compare([]), "Fresh places"
    )
    assert "11-20" not in none.getvalue()  # no ranks, no curve


def test_eval_compare_prints_precision_where_the_share_falls_and_the_marker() -> None:
    def session(place: str) -> tuple[str, list[evaluation.ComparedPhoto]]:
        photos = [  # Yours is liked down to its marker and not after; General is liked throughout
            evaluation.ComparedPhoto(rank < 20, frozenset({"yours"}), (("yours", rank),), marker=20)
            for rank in range(30)
        ]
        photos += [
            evaluation.ComparedPhoto(True, frozenset({"general"}), (("general", rank),))
            for rank in range(30)
        ]
        return place, photos

    report = evaluation.measure_compare(
        [session("Mill"), session("Harbor"), session("Fisher")], resamples=50
    )
    shown = io.StringIO()
    cli._print_compare(Console(file=shown, width=220), report, "Fresh places")
    text = shown.getvalue()
    for heading in ("first 10", "first 20", "first 30"):
        assert heading in text
    assert "first 60" not in text  # no list is that long
    assert "Yours minus General" in text
    assert "first falls below 80%" in text
    assert "Yours at pick 26" in text
    assert "General never fell" in text
    assert "above it 60/60" in text
    assert "below it 0/30" in text
    none = io.StringIO()
    cli._print_compare(
        Console(file=none, width=200), evaluation.measure_compare([]), "Fresh places"
    )
    assert "first 10" not in none.getvalue()


def test_eval_compare_json_carries_the_curve_precision_the_fall_and_the_marker() -> None:
    photos = [
        evaluation.ComparedPhoto(rank < 15, frozenset({"yours"}), (("yours", rank),), marker=15)
        for rank in range(30)
    ] + [
        evaluation.ComparedPhoto(True, frozenset({"general"}), (("general", rank),))
        for rank in range(30)
    ]
    report = evaluation.measure_compare([("Mill", photos), ("Harbor", photos)], resamples=20)
    out = cli._compare_json(report)
    assert set(out) >= {"curve", "fall", "precision", "drop", "marker"}
    assert sorted(out["precision"]) == [10, 20, 30]
    assert out["drop"]["yours"]["positions"] == [19, 19]
    assert out["marker"]["at"] == [15, 15]


def test_group_evidence_is_rendered() -> None:
    out = capture(make_result(3))
    assert "theme 0.82\u00d70.20" in out
    assert "mood 0.64\u00d70.20" in out
    assert "same doorway" in out


def test_a_result_with_probabilities_prints_them_and_the_line_where_the_list_stops_being_good() -> (
    None
):
    result = make_result(4, with_evidence=False)
    picks = tuple(
        pick.model_copy(update={"probability": value})
        for pick, value in zip(result.groups[0].picks, (0.93, 0.84, 0.61, 0.42), strict=True)
    )
    group = result.groups[0].model_copy(
        update={"picks": picks, "marker": Marker(before=2, probability=0.8)}
    )
    out = capture(result.model_copy(update={"groups": (group,)}))
    assert "likely" in out
    for shown in ("93%", "84%", "61%", "42%"):
        assert shown in out
    lines = out.splitlines()
    line = next(i for i, text in enumerate(lines) if "less than 80% likely" in text)
    third = next(i for i, text in enumerate(lines) if "61%" in text)
    second = next(i for i, text in enumerate(lines) if "84%" in text)
    assert second < line < third  # between the second pick and the third
    plain = capture(make_result(4, with_evidence=False))
    assert "likely" not in plain  # General's result has neither


def test_three_pick_groups_use_positions() -> None:
    out = capture(make_result(3))
    assert "position" in out
    for label in ("left", "anchor", "right"):
        assert label in out


def test_other_groups_keep_ranks() -> None:
    out = capture(make_result(2, with_evidence=False))
    assert "rank" in out
    assert "anchor" not in out


def test_runs_list_prints_the_full_run_id(configured: Settings, run_id: str) -> None:
    result = runner.invoke(app, ["runs", "list"], env={"COLUMNS": "60"})
    assert run_id in result.stdout


def test_verbose_is_a_flag_in_help() -> None:
    out = invoke("--help").stdout
    assert "<int>" not in out
    assert "-v" in out


def test_doctor_announces_clip_loading_on_stderr(
    configured: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_settings(configured.model_copy(update={"embedder": "clip"}))
    monkeypatch.setattr(cli, "run_checks", lambda _s: [Check("x", "ok", "fine")])
    result = invoke("doctor")
    assert "Loading the clip embedder" in result.stderr
    assert "Loading" not in result.stdout


def test_sandbox_builds_an_isolated_archive_that_select_can_use(tmp_path: Path) -> None:
    target = tmp_path / "sandbox"
    made = invoke("sandbox", target)
    assert made.exit_code == 0, made.output
    assert f"export CALICO_HOME={target / 'home'}" in made.stdout

    settings = load_settings(target / "home")
    assert settings.collections[0].root == (target / "archive").resolve()
    with Engine.open(settings) as engine:
        result = engine.select(SelectionRequest(task="best", folders=(target / "archive",)))
    assert result.groups


def test_sandbox_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    target = tmp_path / "sandbox"
    target.mkdir()
    refused = invoke("sandbox", target)
    assert refused.exit_code == 1
    assert "--force" in refused.output
    assert invoke("sandbox", target, "--force").exit_code == 0


def test_sandbox_force_replaces_a_sandbox_it_made(tmp_path: Path) -> None:
    target = tmp_path / "sandbox"
    assert invoke("sandbox", target).exit_code == 0
    (target / "archive" / "stray.txt").write_text("from an experiment")
    assert invoke("sandbox", target, "--force").exit_code == 0
    assert not (target / "archive" / "stray.txt").exists()
    assert (target / ".calico-sandbox").is_file()


def test_sandbox_force_replaces_a_sandbox_made_before_it_was_marked(tmp_path: Path) -> None:
    target = tmp_path / "sandbox"
    assert invoke("sandbox", target).exit_code == 0
    (target / ".calico-sandbox").unlink()
    assert invoke("sandbox", target, "--force").exit_code == 0


def test_sandbox_force_never_deletes_a_folder_it_did_not_make(tmp_path: Path) -> None:
    """A typo for calico sandbox ./sandbox must not wipe the folder holding your photos."""
    precious = tmp_path / "precious"
    (precious / "sub").mkdir(parents=True)
    (precious / "photo.jpg").write_bytes(b"irreplaceable")
    refused = invoke("sandbox", precious, "--force")
    assert refused.exit_code == 1
    assert "did not make" in refused.output
    assert (precious / "photo.jpg").read_bytes() == b"irreplaceable"
    assert (precious / "sub").is_dir()


def test_sandbox_force_rebuilds_the_sandbox_whose_home_is_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLAUDE.md's habit: export CALICO_HOME=/tmp/cal/home, later rebuild /tmp/cal."""
    target = tmp_path / "sandbox"
    assert invoke("sandbox", target).exit_code == 0
    monkeypatch.setenv("CALICO_HOME", str(target / "home"))
    assert invoke("sandbox", target, "--force").exit_code == 0


def test_sandbox_force_never_deletes_a_sandbox_holding_another_home_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "sandbox"
    assert invoke("sandbox", target).exit_code == 0
    real = target / "archive" / "my calico"
    real.mkdir()
    monkeypatch.setenv("CALICO_HOME", str(real))
    refused = invoke("sandbox", target, "--force")
    assert refused.exit_code == 1
    assert "in use" in refused.output
    assert real.is_dir()


def test_sandbox_force_never_deletes_a_folder_holding_a_collection(
    configured: Settings, tmp_path: Path
) -> None:
    """Even one that looks like a sandbox: a collection of yours lives in its archive/."""
    marked = tmp_path / "marked"
    photos = marked / "archive"
    f.save_jpeg(f.scene(1), photos / "Old Mill" / "a.jpg")
    assert invoke("collections", "add", "mine", photos).exit_code == 0
    (marked / ".calico-sandbox").write_text("")
    refused = invoke("sandbox", marked, "--force")
    assert refused.exit_code == 1
    assert "collection" in refused.output
    assert (photos / "Old Mill" / "a.jpg").is_file()


def test_doctor_names_label_sets_it_cannot_read(configured: Settings) -> None:
    configured.labels_dir.mkdir(parents=True)
    (configured.labels_dir / "20990101-000000-bad0.json").write_text("not json")
    labels = {c.name: c for c in run_checks(configured)}["labels"]
    assert labels.status == "warn"
    assert "20990101-000000-bad0.json" in labels.detail


def test_every_eval_says_which_label_sets_it_left_out(configured: Settings) -> None:
    configured.labels_dir.mkdir(parents=True)
    (configured.labels_dir / "20990101-000000-bad0.json").write_text("not json")
    result = invoke("eval", "list")
    assert result.exit_code == 0, result.output
    assert "1 labelling set could not be read and is left out" in result.stderr
    assert "20990101-000000-bad0.json" in result.stderr


def test_train_taste_refuses_to_learn_without_label_sets_it_cannot_read(
    configured: Settings,
) -> None:
    configured.labels_dir.mkdir(parents=True)
    (configured.labels_dir / "20990101-000000-bad0.json").write_text("not json")
    refused = invoke("train", "taste", "--force")
    assert refused.exit_code == 1
    assert "--accept-missing" in refused.output


def test_select_without_limit_uses_the_task_default(configured: Settings, archive: Path) -> None:
    result = invoke("select", "best", archive, "--json")
    assert json.loads(result.stdout)["request"]["limit"] == 24


# -- locations ----------------------------------------------------------------------------


def misspell(archive: Path) -> None:
    """A swapped-letter copy of a place, in a folder with a trailing space."""
    f.save_jpeg(f.scene(4), archive / "Old Mlil - May 2024 " / "a.jpg")


def test_locations_suggests_an_alias_for_a_misspelt_place(
    configured: Settings, archive: Path
) -> None:
    misspell(archive)
    result = invoke("locations")
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "3 locations" in out
    assert "Old Mlil" in out
    assert "spelling" in out
    assert "[location_aliases]" in out
    assert '"Old Mlil" = "Old Mill"' in out
    assert "“Old Mlil - May 2024 ”" in out  # the stray space is made visible


def test_locations_json_lists_counts_groups_and_untidy_folders(
    configured: Settings, archive: Path
) -> None:
    misspell(archive)
    result = invoke("locations", archive, "--json")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["locations"] == {"Harbor": 2, "Old Mill": 3, "Old Mlil": 1}
    (group,) = report["same_place"]
    assert group == {"names": ["Old Mill", "Old Mlil"], "photos": [3, 1], "reasons": ["spelling"]}
    assert report["stray_spaces"] == ["Old Mlil - May 2024 "]
    assert report["words"] == []


def test_locations_applies_aliases_already_configured(configured: Settings, archive: Path) -> None:
    misspell(archive)
    save_settings(configured.model_copy(update={"location_aliases": {"Old Mlil": "Old Mill"}}))
    report = json.loads(invoke("locations", "--json").stdout)
    assert report["locations"] == {"Harbor": 2, "Old Mill": 4}
    assert report["same_place"] == []


def test_locations_needs_a_mounted_collection(calico_home: Path) -> None:
    result = invoke("locations")
    assert result.exit_code == 1
    assert "calico collections add" in result.output


# -- eval -----------------------------------------------------------------------------------


@pytest.fixture
def reads_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tiny embedder cannot read text; give the baseline scorer a fake reader."""
    from calico.criteria import SCORERS, BaselineScorer, PartsScorer  # noqa: PLC0415

    monkeypatch.setitem(SCORERS, "baseline", lambda e, _s: BaselineScorer(f.FirstAxisReader(e)))
    monkeypatch.setitem(
        SCORERS, "parts", lambda e, s: PartsScorer(f.FirstAxisReader(e), s.hemisphere)
    )


def test_eval_new_creates_a_set_and_list_shows_progress(
    configured: Settings, archive: Path, reads_text: None
) -> None:
    result = invoke("eval", "new", "fog over water", archive)
    assert result.exit_code == 0, result.output
    assert "5 photos to label" in result.stdout
    assert "/label/" in result.stdout
    listed = invoke("eval", "list")
    assert "fog over water" in listed.stdout
    assert "0 of 5" in listed.stdout


def test_eval_new_without_a_text_reader_says_how_to_get_one(
    configured: Settings, archive: Path
) -> None:
    result = invoke("eval", "new", "fog", archive)
    assert result.exit_code == 1
    assert "cannot read text" in result.output


def test_eval_criteria_reports_each_scorer_on_labelled_searches(
    configured: Settings, archive: Path, reads_text: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    assert "No labelled searches yet" in invoke("eval", "criteria").stdout
    invoke("eval", "new", "fog", archive)
    store = LabelStore(configured.labels_dir)
    (made,) = store.all()
    for i, item in enumerate(made.items):
        rate(store).rate_photo(made.set_id, item.content_key, 2 if i < 2 else 0)
    text = invoke("eval", "criteria")
    assert text.exit_code == 0, text.output
    assert "baseline@1 (tiny)" in text.stdout
    assert "parts@1 (tiny)" in text.stdout
    assert "fog" in text.stdout
    report = json.loads(invoke("eval", "criteria", "--json").stdout)
    assert {r["scorer"] for r in report["searches"]} == {"baseline@1 (tiny)", "parts@1 (tiny)"}
    (row,) = [r for r in report["searches"] if r["scorer"] == "baseline@1 (tiny)"]
    assert row["labelled"] == 5
    assert row["matches"] == 2
    assert row["clear"] is None
    mean = report["mean"]["baseline@1 (tiny)"]
    assert set(mean) >= {"precision_at_10", "ndcg_at_10", "auc"}
    assert report["baseline"] == "baseline@1 (tiny)"
    assert mean["versus"] is None
    assert report["mean"]["parts@1 (tiny)"]["versus"]["unit"] == "search"
    assert "parts@1 (tiny) minus baseline@1 (tiny), AUC" in text.stdout
    assert "(1 search, 5 rated; too few searches to trust a range)" in text.stdout


def test_eval_examples_reports_text_beside_text_with_examples(
    configured: Settings, archive: Path, reads_text: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    assert "No labelled searches" in invoke("eval", "examples").stdout
    invoke("eval", "new", "fog", archive)
    store = LabelStore(configured.labels_dir)
    (made,) = store.all()
    for i, item in enumerate(made.items):
        rate(store).rate_photo(made.set_id, item.content_key, 2 if i % 2 else 0)
    text = invoke("eval", "examples", "--k", "1", "--weight", "0.5")
    assert text.exit_code == 0, text.output
    assert "same search" in text.stdout
    assert "text+examples×0.5" in text.stdout
    report = json.loads(invoke("eval", "examples", "--k", "1", "--weight", "0.5", "--json").stdout)
    assert set(report["mean"]["same search"]) == {"text", "text+examples×0.5", "examples"}
    assert report["mean"]["same search"]["text"]["searches"] == 1
    assert report["mean"]["same search"]["text"]["versus_text"] is None
    assert report["mean"]["same search"]["examples"]["versus_text"]["places"] == 1
    assert "minus text alone" in text.stdout
    (row,) = [r for r in report["searches"] if r["variant"] == "examples"]
    assert row["set_id"] == made.set_id
    assert row["source"] == made.set_id
    assert set(row) >= {"labelled", "precision_at_10", "ndcg_at_10", "auc", "site"}
    assert set(row["left_out"]) == {"copies", "nearby"}
    assert "Never scored, so no example counts for itself" in text.stdout
    assert "same search: 0 saved again, " in text.stdout


def test_eval_new_taste_starts_a_rating_session(configured: Settings, archive: Path) -> None:
    made = invoke("eval", "new-taste", archive)
    assert made.exit_code == 0, made.output
    assert "to rate for “Your taste: archive”" in made.stdout
    (label_set,) = LabelStore(configured.labels_dir).all()
    assert label_set.kind == "taste"
    assert f"/label/{label_set.set_id}" in made.stdout


def test_eval_new_anchors_starts_a_session_and_eval_anchors_measures_it(
    configured: Settings, archive: Path
) -> None:
    made = invoke("eval", "new-anchors", archive)
    assert made.exit_code == 0, made.output
    assert "to rate for “Could it anchor a story: archive”" in made.stdout
    (label_set,) = LabelStore(configured.labels_dir).all()
    assert label_set.kind == "anchor"
    store = LabelStore(configured.labels_dir)
    for i, item in enumerate(label_set.items):
        rate(store).rate_photo(label_set.set_id, item.content_key, 2 if i % 2 else 0)
    could = len(label_set.items) // 2
    report = invoke("eval", "anchors")
    assert report.exit_code == 0, report.output
    told = f"{could} could anchor, {len(label_set.items) - could} could not, at 1 place"
    assert told in report.stdout
    assert "general strength" in report.stdout
    assert "(1 place, " in report.stdout
    data = json.loads(invoke("eval", "anchors", "--json").stdout)
    assert data["could"] == could
    assert data["measured"]["baseline"] == "general strength"
    assert "general strength" in data["measured"]["auc"]


def test_eval_new_stories_starts_a_story_session_and_eval_stories_counts_it(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    made = invoke("eval", "new-stories", archive)
    assert made.exit_code == 0, made.output
    (label_set,) = LabelStore(configured.labels_dir).all()
    assert label_set.kind == "stories"
    assert "1 story to rate for “Stories: archive”" in made.stdout  # a small archive
    assert f"/label/{label_set.set_id}" in made.stdout
    rate(LabelStore(configured.labels_dir)).rate_story(label_set.set_id, 0, 2, doesnt_fit=[])
    text = invoke("eval", "stories", "--too-alike", "0.8", "--variety", "0.5")
    assert text.exit_code == 0, text.output
    assert "1 liked, 0 rejected, 0 mixed, 0 not sure" in text.stdout
    assert "1 run or session" in text.stdout
    assert "no variety" in text.stdout
    assert "Stories rated in sessions" in text.stdout  # how the session's stories were told
    report = json.loads(invoke("eval", "stories", "--json").stdout)
    assert {(a["template"], a["variety"], a["way"]) for a in report["sessions"]["arms"]} <= {
        ("general", True, "today"),
        ("general", True, "likely_anchors"),
    }


def test_eval_same_view_labels_and_recommends_a_threshold(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    assert "No labelled same-view searches" in invoke("eval", "same-view").stdout
    reference = next(archive.rglob("*.jpg"))
    made = invoke("eval", "new-view", reference, archive)
    assert made.exit_code == 0, made.output
    assert "to label for “Same view as" in made.stdout
    store = LabelStore(configured.labels_dir)
    (label_set,) = store.all()
    for i, item in enumerate(label_set.items):
        rate(store).rate_photo(label_set.set_id, item.content_key, 2 if i % 2 else 0)
    text = invoke("eval", "same-view")
    assert text.exit_code == 0, text.output
    assert "Same view as" in text.stdout
    assert "min_similarity" in text.stdout
    (report,) = json.loads(invoke("eval", "same-view", "--json").stdout)["reports"]
    (row,) = report["searches"]
    assert row["set_id"] == label_set.set_id
    assert set(row["held_out"]) >= {"threshold", "precision", "recall"}
    assert set(report["pooled"]) >= {"threshold", "precision", "recall"}
    assert (report["default"], report["embedder"], report["layout_weight"]) == (0.75, "tiny", 0.5)
    assert report["auc"]["unit"] == "reference"
    assert set(report["recall"]) >= {"value", "lo", "hi"}
    assert "(1 reference, " in text.stdout
    mixes = invoke("eval", "same-view", "--layout-weight", "0", "--layout-weight", "1", "--json")
    reports = json.loads(mixes.stdout)["reports"]
    assert [r["layout_weight"] for r in reports] == [0.0, 1.0]
    assert reports[0]["versus"] is None
    assert reports[1]["versus"]["places"] == 1
    compared = invoke("eval", "same-view", "--layout-weight", "0", "--layout-weight", "1")
    assert "held out" in compared.stdout
    assert "minus the first" in compared.stdout


def test_eval_same_view_compares_matches_in_place_with_the_share(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    reference = next(archive.rglob("*.jpg"))
    assert invoke("eval", "new-view", reference, archive).exit_code == 0
    store = LabelStore(configured.labels_dir)
    (label_set,) = store.all()
    for i, item in enumerate(label_set.items):
        rate(store).rate_photo(label_set.set_id, item.content_key, 2 if i % 2 else 0)
    text = invoke("eval", "same-view", "--geometry")
    assert text.exit_code == 0, text.output
    assert "Over every labelled photo" in text.stdout
    assert "Where the share is 0.12 to 0.30" in text.stdout
    assert "tiny (patches in place)" in text.stdout
    report = json.loads(invoke("eval", "same-view", "--geometry", "--json").stdout)
    assert [r["embedder"] for r in report["whole"]] == ["tiny (patches)", "tiny (patches in place)"]
    assert report["band"] == [0.12, 0.30]
    assert len(report["within_band"]) == 2
    assert isinstance(report["in_band"], int)
    refused = invoke("eval", "same-view", "--geometry", "--layout-weight", "0")
    assert refused.exit_code == 1
    assert "--geometry" in refused.stderr


def test_eval_ratings_measures_past_ratings_within_each_location(
    configured: Settings, archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    assert "No ratings" in invoke("eval", "ratings").stdout
    invoke("index", archive)
    mill = sorted((archive / "Old Mill - March 2021").glob("*.jpg"))
    legacy = tmp_path / "feedback.csv"
    lines = ["image_path,stars,vote"]
    lines += [f"{p.resolve()},{5 if i % 2 else 1}," for i, p in enumerate(mill)]
    lines.append("/gone/x.jpg,,up")
    legacy.write_text("\n".join(lines) + "\n")
    text = invoke("eval", "ratings", "--legacy", legacy)
    assert text.exit_code == 0, text.output
    assert "Old Mill" in text.stdout
    assert "1 rated photo is not in the catalog" in text.stdout
    assert "Counted: 0 👍/👎 on Strongest results" in text.stdout
    assert "3 star and 1 thumb ratings from --legacy files" in text.stdout
    report = json.loads(invoke("eval", "ratings", "--legacy", legacy, "--json").stdout)
    (row,) = report["locations"]
    assert (row["location"], row["liked"], row["disliked"]) == ("Old Mill", 1, 2)
    assert "strength" in row["auc"]
    assert (report["rated"], report["missing"]) == (3, 1)
    assert report["measured"]["auc"]["strength"]["places"] == 1
    assert report["measured"]["baseline"] == "strength"
    assert "average of places" in text.stdout
    assert "quality minus strength" in text.stdout


def test_eval_ratings_refuses_a_legacy_file_it_cannot_use(
    configured: Settings, tmp_path: Path
) -> None:
    missing = invoke("eval", "ratings", "--legacy", tmp_path / "nowhere.csv")
    assert missing.exit_code == 2
    assert "does not exist" in missing.output
    assert "Traceback" not in missing.output
    binary = tmp_path / "binary.csv"
    binary.write_bytes(bytes(range(256)))
    result = invoke("eval", "ratings", "--legacy", binary)
    assert result.exit_code == 1
    assert "not UTF-8 text" in result.stderr
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("a,b\n1,2\n", encoding="utf-8")
    result = invoke("eval", "ratings", "--legacy", wrong)
    assert result.exit_code == 1
    assert "wrong.csv has no rating column" in result.stderr
    assert "No ratings yet" not in result.stdout


class Tagged(TinyEmbedder):
    """A candidate embedder with an option, versioned by it."""

    name = "tagged"

    def __init__(self, tag: str = "a") -> None:
        super().__init__()
        self.version = f"tag-{tag}"


def test_eval_same_view_options_a_candidate_embedder(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = extractor_registry()
    plugins = {name: real.get(name) for name in real.names()} | {"tagged": Tagged}
    monkeypatch.setattr(
        "calico.features.extractor_registry", lambda: Registry.from_mapping("extractor", plugins)
    )
    invoke("eval", "new-view", next(archive.rglob("*.jpg")), archive)
    store = LabelStore(configured.labels_dir)
    (label_set,) = store.all()
    for i, item in enumerate(label_set.items):
        rate(store).rate_photo(label_set.set_id, item.content_key, 2 if i % 2 else 0)
    result = invoke("eval", "same-view", "--embedder", "tagged", "-o", "tagged.tag=b", "--json")
    assert result.exit_code == 0, result.output
    (report,) = json.loads(result.stdout)["reports"]
    assert report["embedder"] == "tagged"
    keys = [i.content_key for i in label_set.items]
    with Catalog(configured.catalog_path) as catalog:
        assert catalog.has_features("tagged@tag-b", keys) == set(keys)


@pytest.mark.parametrize(
    ("option", "message"),
    [("tiny.x=1", "computes for every photo"), ("nodot=1", "NAME.KEY=VALUE")],
)
def test_eval_same_view_refuses_options_it_cannot_apply(
    configured: Settings, option: str, message: str
) -> None:
    result = invoke("eval", "same-view", "-o", option)
    assert result.exit_code == 1
    assert message in result.stderr


class SureReader(f.FirstAxisReader):
    """Certain that every photo matches."""

    def match_probability(self, similarities: NDArray[np.float32]) -> NDArray[np.float32] | None:
        return np.ones_like(similarities)


def test_eval_criteria_shows_how_often_a_clear_match_was_right(
    configured: Settings, archive: Path, reads_text: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from calico.criteria import SCORERS, BaselineScorer  # noqa: PLC0415

    monkeypatch.setenv("COLUMNS", "160")
    invoke("eval", "new", "fog", archive)
    store = LabelStore(configured.labels_dir)
    (made,) = store.all()
    for i, item in enumerate(made.items):
        rate(store).rate_photo(made.set_id, item.content_key, 2 if i < 2 else 1 if i == 2 else 0)
    monkeypatch.setitem(SCORERS, "baseline", lambda e, _s: BaselineScorer(SureReader(e)))
    text = invoke("eval", "criteria", "--scorer", "baseline")
    assert text.exit_code == 0, text.output
    assert "2 of 5 right, 2 wrong" in text.stdout
    report = json.loads(invoke("eval", "criteria", "--scorer", "baseline", "--json").stdout)
    (row,) = report["searches"]
    assert row["clear"] == {"matches": 2, "partly": 1, "wrong": 2}
    assert report["mean"]["baseline@1 (tiny)"]["clear"] == {"matches": 2, "partly": 1, "wrong": 2}


def test_eval_criteria_rejects_an_unknown_embedder(configured: Settings, reads_text: None) -> None:
    result = invoke("eval", "criteria", "--embedder", "nope")
    assert result.exit_code == 1
    assert "tiny" in result.output


def test_eval_criteria_json_has_null_for_undefined_metrics(
    configured: Settings, archive: Path, reads_text: None
) -> None:
    invoke("eval", "new", "fog", archive)
    store = LabelStore(configured.labels_dir)
    (made,) = store.all()
    rate(store).rate_photo(made.set_id, made.items[0].content_key, 2)  # no non-match: AUC undefined
    out = invoke("eval", "criteria", "--json").stdout
    assert "NaN" not in out
    assert json.loads(out)["searches"][0]["auc"] is None


def test_eval_criteria_rejects_an_unknown_scorer(configured: Settings, reads_text: None) -> None:
    result = invoke("eval", "criteria", "--scorer", "nope")
    assert result.exit_code == 1
    assert "baseline" in result.output


def test_eval_same_view_says_when_a_frame_saved_twice_was_counted_once(
    configured: Settings, archive: Path
) -> None:
    made = invoke("eval", "new-view", archive / "Old Mill - March 2021" / "view1.jpg", archive)
    assert made.exit_code == 0, made.output
    store = LabelStore(configured.labels_dir)
    (label_set,) = store.all()
    first = label_set.items[0]
    # The same pixels saved again later. The sampler draws one photo per band, so the copy
    # is added to the set by hand.
    seed = int("".join(c for c in first.path.stem if c.isdigit()))
    export = f.save_jpeg(f.scene(seed), archive / "Exports - 2023" / "again.jpg")
    copy = LabelItem(content_key=file_facts(export).content_key, path=export)
    grown = label_set.model_copy(update={"items": (*label_set.items, copy)})
    (configured.labels_dir / f"{label_set.set_id}.json").write_text(grown.model_dump_json())
    for item in store.load(label_set.set_id).items:
        rate(store).rate_photo(
            label_set.set_id, item.content_key, 2 if item.path == first.path else 0
        )
    invoke("index", archive)
    text = invoke("eval", "same-view")
    assert "Counted 1 copy of a frame saved twice" in text.stdout
    (report,) = json.loads(invoke("eval", "same-view", "--json").stdout)["reports"]
    assert report["copies"] == 1
    assert report["searches"][0]["same"] == 1


def test_eval_same_view_can_match_patches(configured: Settings, archive: Path) -> None:
    reference = archive / "Old Mill - March 2021" / "view1.jpg"
    assert invoke("eval", "new-view", reference, archive).exit_code == 0
    store = LabelStore(configured.labels_dir)
    (label_set,) = store.all()
    for i, item in enumerate(label_set.items):
        rate(store).rate_photo(label_set.set_id, item.content_key, 2 if i % 2 else 0)
    result = invoke("eval", "same-view", "--patches", "--json")
    assert result.exit_code == 0, result.output
    (report,) = json.loads(result.stdout)["reports"]
    assert report["embedder"] == "tiny (patches)"
    text = invoke("eval", "same-view", "--patches")
    assert text.exit_code == 0, text.output
    assert "patches" in text.stdout


def test_collections_remove_forgets_a_collection_and_touches_no_file(
    configured: Settings, archive: Path
) -> None:
    removed = invoke("collections", "remove", "archive")
    assert removed.exit_code == 0, removed.output
    assert load_settings().collections == ()
    assert archive.is_dir()
    unknown = invoke("collections", "remove", "nope")
    assert unknown.exit_code == 1
    assert "calico collections list" in unknown.output


def test_collections_list_json(configured: Settings, archive: Path) -> None:
    result = invoke("collections", "list", "--json")
    assert result.exit_code == 0, result.output
    (found,) = json.loads(result.stdout)
    assert (found["name"], found["root"], found["mounted"]) == ("archive", str(archive), True)


def test_adding_a_collection_inside_another_says_which_one_its_photos_belong_to(
    configured: Settings, archive: Path
) -> None:
    inner = archive / "Harbor - June 2022"
    result = invoke("collections", "add", "harbor", inner)
    assert result.exit_code == 0, result.output
    assert "inside 'archive'" in result.stderr
    assert "belong to 'harbor'" in result.stderr


def test_a_home_inside_the_archive_is_refused_but_settings_can_still_fix_it(
    configured: Settings, calico_home: Path
) -> None:
    assert invoke("collections", "add", "everything", calico_home.parent).exit_code == 0
    refused = invoke("index", calico_home.parent)
    assert refused.exit_code == 1
    assert "inside" in refused.output
    assert invoke("collections", "list").exit_code == 0
    assert invoke("collections", "remove", "everything").exit_code == 0


def test_eval_taste_says_when_general_could_not_run(
    configured: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audit A01-7: tiny reads no text, so General is a constant there."""
    monkeypatch.setenv("COLUMNS", "160")
    rate_for_taste(configured, archive)
    text = invoke("eval", "taste")
    assert "General's aesthetic could not run" in text.stdout
    (report,) = json.loads(invoke("eval", "taste", "--json").stdout)["reports"]
    assert any("could not run" in n for n in report["notes"])


def test_eval_anchors_names_the_sessions_it_left_out(
    configured: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audit A01-4: a session whose folders hold no photos now is left out, and named."""
    report = evaluation.AnchorReport(
        could=0, could_not=0, places=0, signals=[], skipped=["Anchors: Mill"]
    )
    monkeypatch.setattr("calico.cli.Sessions.evaluate_anchors", lambda self, **_: report)
    text = invoke("eval", "anchors")
    assert "Left out, since their folders hold no catalogued photos now: Anchors: Mill" in (
        text.stdout
    )


def test_serve_refuses_a_port_out_of_range_without_a_traceback() -> None:
    """Audit A07-2: 70000 reached bind() and gave an OverflowError."""
    result = invoke("serve", "--port", "70000", "--no-browser")
    assert result.exit_code == 2
    assert "65535" in result.output
    assert "Traceback" not in result.output


def test_label_hints_name_the_address_serve_listens_on_by_default() -> None:
    """Audit A07-7: one constant for serve's default and every hint that names it."""
    assert f"http://{cli.DEFAULT_HOST}:{cli.DEFAULT_PORT}" == cli.DEFAULT_URL
    assert f"[default: {cli.DEFAULT_PORT}]" in invoke("serve", "--help").output


def test_json_holds_no_nan_or_infinity_even_inside_tuples() -> None:
    """Audit A07-6: tuples were not walked, so --json could print NaN, which is not JSON."""
    out = cli._nan_to_null({"a": (1.0, math.nan), "b": [math.inf, {"c": -math.inf}]})
    assert out == {"a": [1.0, None], "b": [None, {"c": None}]}
    json.dumps(out, allow_nan=False)


def test_log_json_stderr_is_json_lines_only(configured: Settings, archive: Path) -> None:
    """Audit A07-8: off a terminal the progress bar left a blank line in stderr."""
    result = invoke("--log-json", "-v", "index", archive)
    assert result.exit_code == 0, result.output
    lines = result.stderr.splitlines()
    assert lines
    for line in lines:
        json.loads(line)


def test_a_fit_by_scale_says_where_the_photos_you_liked_rank_at_each_scale() -> None:
    from tests.test_evaluation import scaled_places  # noqa: PLC0415

    report = evaluation.measure_taste(
        scaled_places(), feature="clip@1", quality_weight=0.0, resamples=10, fit="by_scale"
    )
    shown = io.StringIO()
    cli._print_taste_alternatives(Console(file=shown, width=200), report)
    text = shown.getvalue()
    assert "Yours fitted by scale, by strength" in text
    assert "Photos you liked, macro: ranked" in text
    assert text.index("wide:") < text.index("close:") < text.index("macro:")  # far to near


def asked_control(key: str, liked: bool, when: datetime, session: str) -> Answer:
    return Answer.model_validate(
        {
            "at": when,
            "use": "compare",
            "scope": {"control": "yes"},
            "subject": (key,),
            "value": 2 if liked else 0,
            "session": session,
        }
    )


def test_controls_show_and_retest_say_what_is_missing_until_the_pool_is_drawn(
    configured: Settings,
) -> None:
    assert "Nothing is drawn yet" in invoke("controls", "show").output
    assert json.loads(invoke("controls", "show", "--json").output) is None
    result = invoke("eval", "retest")
    assert result.exit_code == 1
    assert "calico controls draw" in result.stderr


def test_controls_show_lists_the_pool_and_retest_reads_what_was_asked_again(
    configured: Settings,
) -> None:
    drawn = datetime(2026, 11, 11, 9, tzinfo=UTC)
    write_controls(
        configured.controls_path,
        Controls(photos=(Control("k1", True), Control("k2", False)), seed=1, drawn_at=drawn),
    )
    shown = json.loads(invoke("controls", "show", "--json").output)
    assert [(p["key"], p["liked"]) for p in shown["photos"]] == [("k1", True), ("k2", False)]
    RatingsJournal(configured.ratings_dir).append(
        asked_control("k1", True, drawn + timedelta(days=1), "s1"),
        asked_control("k2", True, drawn + timedelta(days=1), "s1"),  # not liked, now liked
    )
    out = json.loads(invoke("eval", "retest", "--json").output)
    assert (out["asked"], out["photos"], out["sessions"]) == (2, 2, 1)
    assert out["against_baseline"] == {"kept": 1, "n": 2}
    assert out["from_liked"] == {"kept": 1, "n": 1}
    text = invoke("eval", "retest").output
    assert "1 of 2" in text
    assert "ceiling" in text


def test_holdout_draw_can_leave_out_the_places_designed_against(
    configured: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    heard: list[tuple[str, ...]] = []

    def spy(self: Any, seed: int = 0, leave_out: Any = (), places: Any = None, **_: Any) -> HoldOut:
        heard.append(tuple(leave_out))
        return HoldOut(
            places=("Mill",),
            references=(),
            story_sessions=(),
            seed=1,
            drawn_at=datetime(2026, 11, 11, tzinfo=UTC),
        )

    monkeypatch.setattr(sessions_module.Sessions, "draw_holdout", spy)
    result = invoke("holdout", "draw", "--leave-out", "Harbor", "--leave-out", "Quarry")
    assert result.exit_code == 0, result.output
    assert heard == [("Harbor", "Quarry")]
    assert "Held out 1 place" in result.stdout


def test_holdout_draw_can_hold_out_more_places_and_be_previewed_first(
    configured: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    heard: list[tuple[Any, bool]] = []

    def spy(
        self: Any, seed: int = 0, leave_out: Any = (), places: Any = None, *, write: bool = True
    ) -> HoldOut:
        heard.append((places, write))
        return HoldOut(
            places=("Mill",),
            references=(),
            story_sessions=(),
            seed=1,
            drawn_at=datetime(2026, 11, 11, tzinfo=UTC),
        )

    monkeypatch.setattr(sessions_module.Sessions, "draw_holdout", spy)
    preview = invoke("holdout", "draw", "--places", "8", "--dry-run")
    assert preview.exit_code == 0, preview.output
    assert heard == [(8, False)]
    assert "Would hold out 1 place" in preview.stdout
    assert "nothing was written" in preview.stdout
    drawn = invoke("holdout", "draw", "--places", "8")
    assert heard[-1] == (8, True)
    assert "Held out 1 place" in drawn.stdout
    assert invoke("holdout", "draw", "--places", "0").exit_code == 2  # at least one


def test_eval_taste_offers_every_taste_fit_and_nothing_else() -> None:
    assert get_args(cli.FitChoice) == evaluation.TASTE_FITS
