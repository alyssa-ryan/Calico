"""End-to-end: real JPEGs on disk -> catalog -> features -> task -> saved run.

Uses the ``tiny`` embedder so it runs without PyTorch. A small synthetic
archive with two locations, a few views revisited under different light, and
a burst of near-identical frames.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import unicodedata
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pytest
from PIL import Image as PILImage

from calico import evaluation
from calico import sessions as sessions_module
from calico.controls import read_controls
from calico.criteria import BaselineScorer
from calico.domain import Group, Marker, SelectionRequest
from calico.engine import Engine
from calico.errors import (
    CapabilityError,
    ControlsError,
    FolderNotFoundError,
    HoldoutError,
    HomeInsideArchiveError,
    InvalidRequestError,
    NotInCollectionError,
    UnknownPluginError,
)
from calico.evaluation import (
    TASTE_FIRST,
    VALUE_STRATA,
    AnchorRating,
    AnchorReport,
    FirstPicks,
    measure_anchors,
    shipped_strength,
    taste_apart,
)
from calico.evaluation import controls as controls_module
from calico.evaluation import holdout as holdout_module
from calico.evaluation.controls import CONTROLS_PER_SESSION
from calico.evaluation.holdout import HoldOut
from calico.features import extractor_registry
from calico.features.base import PatchExtractor, feature_key
from calico.features.tiny import TinyEmbedder
from calico.holdout import write_holdout
from calico.labels import LabelItem, LabelSet
from calico.models import calibration as calibration_module
from calico.models.registry import Models
from calico.models.taste import fit_taste
from calico.patch_store import scratch_folders
from calico.plugins import Registry
from calico.ratings import Rating
from calico.ratings_store import Answer
from calico.reviews import labels_of, taste_labels
from calico.selection.anchors import SCALES
from calico.selection.context import Trace
from calico.sessions import Sessions, StoryVariant
from calico.settings import CollectionConfig, Settings
from calico.tasks.same_view import SURE_SHARE, SameViewOptions, SameViewTask
from calico.tasks.triptych import StoryScorer, TriptychTask
from calico.tracks import Track
from tests import factories as f

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from PIL import Image

    from calico.domain import Photo
    from calico.evaluation import Drawn
    from calico.features.base import Embedder
    from calico.selection.context import FeatureView, TaskContext

T0 = datetime(2021, 3, 6, 10, 0, 0)


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """
    archive/
      Old Mill - March 2021/   seeds 1..4, plus a 3-frame burst of seed 1
      Old Mill - May 2024/     seeds 1 and 2 again, warmer and darker light
      Harbor - June 2022/      seeds 10..12
    """
    root = tmp_path / "archive"
    march = root / "Old Mill - March 2021"
    for seed in (1, 2, 3, 4):
        f.save_jpeg(
            f.scene(seed),
            march / f"view{seed}.jpg",
            captured_at=T0 + timedelta(hours=seed),
            camera=f.SANDBOX_CAMERA,
        )
    for i in range(3):
        f.save_jpeg(
            f.scene(1, shift=(i, 0)),
            march / f"burst{i}.jpg",
            captured_at=T0 + timedelta(hours=1, seconds=2 * i + 1),
            camera=f.SANDBOX_CAMERA,
        )
    may = root / "Old Mill - May 2024"
    for seed in (1, 2):
        f.save_jpeg(
            f.scene(seed, brightness=0.7, tint=(1.1, 1.0, 0.85)),
            may / f"revisit{seed}.jpg",
            captured_at=datetime(2024, 5, 1, 12) + timedelta(hours=seed),
            camera=f.SANDBOX_CAMERA,
        )
    harbor = root / "Harbor - June 2022"
    for seed in (10, 11, 12):
        f.save_jpeg(f.scene(seed), harbor / f"h{seed}.jpg", captured_at=datetime(2022, 6, 1, seed))
    return root


@pytest.fixture
def engine(archive: Path, calico_home: Path) -> Iterator[Engine]:
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
    with Engine.open(settings) as eng:
        yield eng


def request(archive: Path, task: str, **kw: object) -> SelectionRequest:
    return SelectionRequest.model_validate({"task": task, "folders": [archive], **kw})


def screenshot_engine(archive: Path, calico_home: Path, *, exclude: bool) -> Engine:
    """The archive plus a phone screenshot (a PNG with no camera) and a camera-less JPEG."""
    harbor = archive / "Harbor - June 2022"
    f.scene(20).save(harbor / "IMG_0001.PNG")
    f.save_jpeg(f.scene(21), harbor / "scan.jpg")
    collection = CollectionConfig(
        name="archive",
        root=archive,
        location_pattern=r"^(?P<location>.+?) - ",
        exclude_screenshots=exclude,
    )
    settings = Settings(
        home=calico_home,
        embedder="tiny",
        describe_embedder="tiny",
        view_embedder="tiny",
        collections=(collection,),
    )
    return Engine.open(settings)


def test_screenshots_are_left_out_where_the_collection_says_so(
    archive: Path, calico_home: Path
) -> None:
    with screenshot_engine(archive, calico_home, exclude=True) as engine:
        result = engine.select(request(archive, "best", limit=50, options={"diversity": 0}))
        picked = {p.path.name for g in result.groups for p in g.picks}
        assert "IMG_0001.PNG" not in picked
        assert "scan.jpg" in picked  # no camera, but not a PNG
        assert any("1 screenshot" in n and "exclude_screenshots" in n for n in result.notes)
        rated = Sessions(engine).label_taste([archive / "Harbor - June 2022"])
        assert "IMG_0001.PNG" not in {i.path.name for i in rated.items}


def test_screenshots_are_kept_unless_the_collection_says_otherwise(
    archive: Path, calico_home: Path
) -> None:
    with screenshot_engine(archive, calico_home, exclude=False) as engine:
        result = engine.select(request(archive, "best", limit=50, options={"diversity": 0}))
        assert "IMG_0001.PNG" in {p.path.name for g in result.groups for p in g.picks}
        assert not any("screenshot" in n for n in result.notes)


def test_strongest_in_memory_compares_views_as_a_search_does(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Taste sessions draw, and eval taste counts, what Strongest shows first: that must be
    the Strongest a search runs, one pick per view included."""
    compared: list[str] = []
    real = engine.compare_parts

    def spy(task: Any, options: Any, ctx: Any, progress: Any = None) -> Any:
        compared.append(task.name)
        return real(task, options, ctx, progress)

    monkeypatch.setattr(engine, "compare_parts", spy)
    Sessions(engine).label_taste([archive / "Harbor - June 2022"])
    assert compared == ["best"]
    compared.clear()
    Sessions(engine)._first_picks()("Harbor", None, 3)
    assert compared == ["best"]


def test_best_returns_one_group_of_distinct_photos(engine: Engine, archive: Path) -> None:
    result = engine.select(request(archive, "best", limit=5))
    (group,) = result.groups
    assert 0 < len(group.picks) <= 5
    assert len({p.content_key for p in group.picks}) == len(group.picks)


def test_burst_frames_come_back_as_one_pick_with_alternates(engine: Engine, archive: Path) -> None:
    result = engine.select(request(archive, "best", limit=20))
    picked = [p.path.name for g in result.groups for p in g.picks]
    assert sum(name.startswith(("burst", "view1")) for name in picked) <= 1


def test_a_tracks_options_reach_the_task_and_its_version_is_in_the_run(
    engine: Engine, archive: Path
) -> None:
    engine.tracks["strongest-general"] = Track(
        "strongest-general",
        "7",
        {"burst_window_seconds": 0, "burst_similarity": 1.0, "one_per_view": False},
    )
    result = engine.select(request(archive, "best", limit=20))
    picked = [p.path.name for g in result.groups for p in g.picks]
    assert sum(name.startswith("burst") for name in picked) > 1
    assert result.versions["track:strongest-general"] == "7"
    assert result.request.options == {}


def test_the_requests_own_options_win_over_the_tracks(engine: Engine, archive: Path) -> None:
    engine.tracks["strongest-general"] = Track(
        "strongest-general",
        "7",
        {"burst_window_seconds": 0, "burst_similarity": 1.0, "one_per_view": False},
    )
    result = engine.select(
        request(
            archive,
            "best",
            limit=20,
            options={"burst_window_seconds": 120, "burst_similarity": 0.9, "one_per_view": True},
        )
    )
    picked = [p.path.name for g in result.groups for p in g.picks]
    assert sum(name.startswith("burst") for name in picked) <= 1


def test_the_track_follows_the_taste_a_search_runs_with(engine: Engine, archive: Path) -> None:
    engine.tracks["strongest-yours"] = Track("strongest-yours", "9", {})
    result = engine.select(request(archive, "best", template="yours"))
    assert result.request.template == "general"
    assert result.versions["track:strongest-general"] == "1"
    assert "track:strongest-yours" not in result.versions


def test_locations_come_from_the_collection_pattern(engine: Engine, archive: Path) -> None:
    result = engine.select(request(archive, "best", limit=20))
    assert {p.location for g in result.groups for p in g.picks} <= {"Old Mill", "Harbor"}


def test_criteria_with_a_text_blind_embedder_is_noted_not_ignored(
    engine: Engine, archive: Path
) -> None:
    result = engine.select(request(archive, "best", criteria="snow"))
    assert any("cannot read text" in note for note in result.notes)


def test_unreadable_files_are_named_in_the_note(engine: Engine, archive: Path) -> None:
    for i in range(5):
        (archive / "Harbor - June 2022" / f"broken{i}.jpg").write_bytes(b"not a jpeg")
    result = engine.select(request(archive, "best"))
    (note,) = [n for n in result.notes if "could not be read" in n]
    assert "5 files could not be read" in note
    assert "Harbor - June 2022/broken0.jpg" in note
    assert "broken3.jpg" not in note
    assert "and 2 more" in note
    assert "calico index" in note


def test_without_raw_support_lone_raw_files_are_skipped_with_how_to_add_it(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "rawpy", None)
    (archive / "Harbor - June 2022" / "IMG_9.CR3").write_bytes(b"raw")
    result = engine.select(request(archive, "best", limit=50))
    (note,) = [n for n in result.notes if "RAW" in n]
    assert note.startswith("1 RAW file has no JPEG/HEIC alongside it")
    assert "uv sync --extra raw" in note


def test_symlinked_files_are_skipped_with_a_note(
    engine: Engine, archive: Path, tmp_path: Path
) -> None:
    outside = f.save_jpeg(f.scene(50), tmp_path / "outside" / "private.jpg")
    (archive / "Harbor - June 2022" / "link.jpg").symlink_to(outside)
    result = engine.select(request(archive, "best", limit=50))
    assert "link.jpg" not in {p.path.name for g in result.groups for p in g.picks}
    assert any("1 linked file" in note for note in result.notes)


class FirstWordsReader:
    """The engine's embedder, plus a text encoder that reads only the first three words."""

    supports_text = True

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.name, self.version, self.dim, self.columns = (
            inner.name,
            inner.version,
            inner.dim,
            inner.columns,
        )

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        return self.inner.extract(images)

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.full((len(texts), self.dim), self.dim**-0.5, dtype=np.float32)

    def readable_prefix(self, text: str) -> str:
        return " ".join(text.split()[:3])

    def match_probability(self, similarities: NDArray[np.float32]) -> NDArray[np.float32] | None:
        return None


def test_criteria_longer_than_the_text_encoder_reads_are_noted(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine, "criteria", BaselineScorer(FirstWordsReader(engine.embedder)))
    result = engine.select(request(archive, "best", criteria="fog over still water at dawn"))
    (note,) = [n for n in result.notes if "longer than" in n]
    assert "“fog over still”" in note
    assert "“water at dawn”" in note
    short = engine.select(request(archive, "best", criteria="fog over water"))
    assert not [n for n in short.notes if "longer than" in n]


@pytest.fixture
def reading(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Engine:
    monkeypatch.setattr(engine, "criteria", BaselineScorer(f.FirstAxisReader(engine.embedder)))
    monkeypatch.setattr(
        "calico.sessions.criteria_scorer", lambda _, e, _s: BaselineScorer(f.FirstAxisReader(e))
    )
    return engine


def test_a_labelling_set_samples_the_folders_and_is_saved(reading: Engine, archive: Path) -> None:
    made = Sessions(reading).label_set("fog", [archive])
    assert made.text == "fog"
    assert made.scorer == "baseline@1 (tiny)"
    assert made.folders == (archive.resolve(),)
    pool = {p.content_key for p in reading.catalog.photos_under([archive])}
    assert {item.content_key for item in made.items} == pool  # a small pool is taken whole
    assert all(item.label is None for item in made.items)
    assert reading.labels.load(made.set_id) == made
    assert made == Sessions(reading).label_set("fog", [archive]).model_copy(
        update={"set_id": made.set_id, "created_at": made.created_at}
    )  # the same search samples the same photos


def test_a_labelling_set_needs_text_the_embedder_can_read(engine: Engine, archive: Path) -> None:
    with pytest.raises(InvalidRequestError, match="what to look for"):
        Sessions(engine).label_set("  ", [archive])
    with pytest.raises(CapabilityError, match="cannot read text"):
        Sessions(engine).label_set("fog", [archive])


def test_evaluation_scores_every_scorer_on_every_labelled_search(
    reading: Engine, archive: Path
) -> None:
    made = Sessions(reading).label_set("fog", [archive])
    assert Sessions(reading).evaluate_criteria(["baseline"]) == []  # nothing labelled yet
    for i, item in enumerate(made.items):
        reading.ratings.rate_photo(made.set_id, item.content_key, 2 if i % 2 else 0)
    (row,) = Sessions(reading).evaluate_criteria(["baseline"])
    assert (row.set_id, row.text, row.scorer) == (made.set_id, "fog", "baseline@1 (tiny)")
    assert row.score.labelled == len(made.items)
    assert row.missing == 0
    assert row.clear is None  # this reader only ranks


def test_evaluation_compares_the_models_that_can_read_text(
    reading: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = Sessions(reading).label_set("fog", [archive])
    for i, item in enumerate(made.items):
        reading.ratings.rate_photo(made.set_id, item.content_key, 2 if i % 2 else 0)
    other = ClearAxisReader(reading.embedder)
    other.name = "other"
    keys = [p.content_key for p in reading.catalog.photos_under([archive])]
    tiny = reading.catalog.get_features(feature_key(reading.embedder), keys)
    reading.catalog.put_features(feature_key(other), tiny)  # the same vectors, stored as its own
    monkeypatch.setitem(reading.extractors, "other", other)
    rows = Sessions(reading).evaluate_criteria(["baseline"])
    assert [r.scorer for r in rows] == ["baseline@1 (tiny)", "baseline@1 (other)"]
    assert rows[1].clear is not None
    assert rows[1].clear.called > 0
    (only,) = Sessions(reading).evaluate_criteria(["baseline"], embedders=["other"])
    assert only.scorer == "baseline@1 (other)"


def label_halves(engine: Engine, set_id: str) -> None:
    """Every other photo of a set labelled a match, the rest not."""
    made = engine.labels.load(set_id)
    for i, item in enumerate(made.items):
        engine.ratings.rate_photo(set_id, item.content_key, 2 if i % 2 else 0)


def test_examples_are_measured_on_the_rest_of_their_own_search(
    reading: Engine, archive: Path
) -> None:
    made = Sessions(reading).label_set("fog", [archive])
    label_halves(reading, made.set_id)
    rows = Sessions(reading).evaluate_examples(k=2, draws=3, weights=(0.5,))
    assert {r.protocol for r in rows} == {"same search"}
    assert [r.variant for r in rows] == ["text", "text+examples×0.5", "examples"]
    assert all((r.set_id, r.source, r.text) == (made.set_id, made.set_id, "fog") for r in rows)
    assert all(r.score.labelled <= len(made.items) - 2 for r in rows)  # examples left out
    assert rows == Sessions(reading).evaluate_examples(k=2, draws=3, weights=(0.5,))  # repeatable


def test_a_photo_labelled_for_the_same_text_twice_counts_once_at_its_newest(
    reading: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    days = iter(datetime(2026, 10, d, tzinfo=UTC) for d in (1, 2))
    monkeypatch.setattr(reading, "clock", lambda: next(days))  # sets made a day apart
    mill = Sessions(reading).label_set("fog", [archive / "Old Mill - March 2021"])
    whole = Sessions(reading).label_set("fog", [archive])
    shared = mill.items[0].content_key
    reading.ratings.rate_photo(mill.set_id, shared, 2)
    reading.ratings.rate_photo(whole.set_id, shared, 0)  # the same question, asked again
    other = next(i.content_key for i in whole.items if i.content_key != shared)
    reading.ratings.rate_photo(whole.set_id, other, 2)
    (row,) = Sessions(reading).evaluate_criteria(["baseline"])
    assert row.set_id == whole.set_id  # the older set's one answer was replaced
    assert row.score.labelled == 2


def test_examples_from_another_site_are_measured_on_the_same_words(
    reading: Engine, archive: Path
) -> None:
    mill = Sessions(reading).label_set("fog", [archive / "Old Mill - March 2021"])
    harbor = Sessions(reading).label_set("Fog ", [archive / "Harbor - June 2022"])
    boats = Sessions(reading).label_set("boats", [archive / "Harbor - June 2022"])
    # Not "fog": the same text would be the same question, and its newer answers would
    # replace the Old Mill set's. "FOG" is the same words to the examples' matching.
    overlapping = Sessions(reading).label_set("FOG", [archive])
    for made in (mill, harbor, boats, overlapping):
        label_halves(reading, made.set_id)
    rows = [
        r
        for r in Sessions(reading).evaluate_examples(k=1, weights=(1.0,))
        if r.protocol == "other site"
    ]
    assert {(r.set_id, r.source) for r in rows} == {  # never from folders that overlap
        (mill.set_id, harbor.set_id),
        (harbor.set_id, mill.set_id),
    }
    (text,) = [r for r in rows if r.set_id == mill.set_id and r.variant == "text"]
    assert text.site == "Old Mill - March 2021"
    assert text.score.labelled == len(mill.items)  # examples from elsewhere hide nothing


def test_photos_near_an_example_are_found_on_one_clock(
    reading: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = Sessions(reading).label_set("fog", [archive])
    label_halves(reading, made.set_id)
    labelled = Sessions._labelled_photos
    noon = datetime(2022, 6, 1, 12)

    def two_cameras(
        sessions: Sessions, label_set: LabelSet, labels: Mapping[str, int], trace: Trace
    ) -> tuple[list[Photo], FeatureView]:
        # The matches on a camera set to +01:00, the rest a minute later on a phone at UTC:
        # an hour apart by their clocks, together in fact.
        photos, view = labelled(sessions, label_set, labels, trace)
        moved = [
            dataclasses.replace(p, captured_at=noon, utc_offset=60, camera="Canon")
            if labels[p.content_key] == 2
            else dataclasses.replace(
                p, captured_at=noon - timedelta(minutes=59), utc_offset=0, camera="iPhone"
            )
            for p in photos
        ]
        return moved, view

    monkeypatch.setattr(Sessions, "_labelled_photos", two_cameras)
    rows = Sessions(reading).evaluate_examples(k=1, draws=1, weights=())
    assert rows
    assert all(r.nearby + r.copies == len(made.items) - 1 for r in rows)  # all but the example


def test_searches_with_too_few_matches_give_no_examples(reading: Engine, archive: Path) -> None:
    made = Sessions(reading).label_set("fog", [archive])
    reading.ratings.rate_photo(made.set_id, made.items[0].content_key, 2)
    reading.ratings.rate_photo(made.set_id, made.items[1].content_key, 0)
    assert Sessions(reading).evaluate_examples(k=1) == []  # one match: nothing left to find


def test_evaluation_skips_labelled_photos_no_longer_in_the_catalog(
    reading: Engine, archive: Path
) -> None:
    made = Sessions(reading).label_set("fog", [archive])
    gone = made.items[0]
    reading.ratings.rate_photo(made.set_id, gone.content_key, 2)
    reading.ratings.rate_photo(made.set_id, made.items[1].content_key, 0)
    gone.path.unlink()
    reading.indexer.index([archive])
    (row,) = Sessions(reading).evaluate_criteria(["baseline"])
    assert (row.score.labelled, row.missing) == (1, 1)


class ClearAxisReader(f.FirstAxisReader):
    """Reads any text as the first axis, and is sure of the photos leaning on it most."""

    def match_probability(self, similarities: NDArray[np.float32]) -> NDArray[np.float32] | None:
        return (similarities >= np.median(similarities)).astype(np.float32)


@pytest.fixture
def telling(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Engine:
    monkeypatch.setattr(engine, "criteria", BaselineScorer(ClearAxisReader(engine.embedder)))
    return engine


def test_picks_say_whether_they_clearly_match(telling: Engine, archive: Path) -> None:
    result = telling.select(request(archive, "best", criteria="fog", limit=50))
    marks = [p.clear_match for g in result.groups for p in g.picks]
    assert True in marks
    assert False in marks
    assert result.versions["criteria"] == "baseline@1 (tiny)"
    assert not [n for n in result.notes if "clearly match" in n]


def test_only_clear_matches_leaves_out_the_rest(telling: Engine, archive: Path) -> None:
    result = telling.select(
        request(archive, "best", criteria="fog", limit=50, only_clear_matches=True)
    )
    picks = [p for g in result.groups for p in g.picks]
    assert picks
    assert all(p.clear_match for p in picks)
    assert any(n.startswith("Only the ") and "“fog”" in n for n in result.notes)
    assert any(s.name == "clear matches" for s in result.stages)
    signals = telling.runs.signals(result.run_id)
    assert {row["criteria_match"] for row in signals} == {"1.0000"}


def test_the_switch_without_a_description_is_noted(engine: Engine, archive: Path) -> None:
    result = engine.select(request(archive, "best", only_clear_matches=True))
    assert any("needs a description" in n for n in result.notes)
    assert all(p.clear_match is None for g in result.groups for p in g.picks)


def test_same_view_returns_every_close_match_oldest_first(
    engine: Engine, archive: Path, tmp_path: Path
) -> None:
    reference = f.save_jpeg(f.scene(2, brightness=0.9), tmp_path / "ref.jpg")
    result = engine.select(request(archive, "same_view", references=[reference]))
    assert result.request.limit == 200
    names = [p.path.name for g in result.groups for p in g.picks]
    assert names == ["view2.jpg", "revisit2.jpg"]
    stamps = [p.captured_at for g in result.groups for p in g.picks]
    assert stamps == sorted(s for s in stamps if s is not None)


def test_same_view_without_reference_is_rejected(engine: Engine, archive: Path) -> None:
    with pytest.raises(InvalidRequestError, match="reference"):
        engine.select(request(archive, "same_view"))


def test_triptych_returns_stories_of_three(engine: Engine, archive: Path) -> None:
    result = engine.select(
        request(archive, "triptych", limit=2, options={"connections": ["theme", "location"]})
    )
    assert result.groups
    assert all(len(g.picks) == 3 for g in result.groups)


def test_every_run_is_saved_and_explainable(engine: Engine, archive: Path) -> None:
    result = engine.select(request(archive, "best", limit=3))
    assert engine.runs.load(result.run_id) == result
    rows = engine.runs.explain(result.run_id, "view3.jpg")
    assert len(rows) == 1
    assert "score" in rows[0]
    assert engine.runs.recent()[0].run_id == result.run_id


def test_result_records_versions_and_stages(engine: Engine, archive: Path) -> None:
    result = engine.select(request(archive, "best"))
    assert result.versions["calico"]
    assert result.versions["tiny"].startswith("tiny@")
    assert [s.name for s in result.stages][:3] == ["index", "gather", "features"]


def test_second_request_reuses_the_index(engine: Engine, archive: Path) -> None:
    engine.select(request(archive, "best"))
    report = engine.indexer.index([archive])
    assert report.added == report.updated == 0
    assert report.unchanged == report.scanned


def test_folder_outside_collections_is_rejected(engine: Engine, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(NotInCollectionError):
        engine.select(SelectionRequest(task="best", folders=(elsewhere,)))


def test_missing_folder_is_reported_as_not_mounted(engine: Engine, archive: Path) -> None:
    with pytest.raises(FolderNotFoundError, match="not mounted"):
        engine.select(SelectionRequest(task="best", folders=(archive / "nope",)))


def test_unknown_option_is_rejected(engine: Engine, archive: Path) -> None:
    with pytest.raises(InvalidRequestError, match="colour"):
        engine.select(request(archive, "best", options={"colour": "red"}))


def test_photos_with_non_finite_features_are_dropped_with_a_note(
    engine: Engine, archive: Path
) -> None:
    engine.indexer.index([archive])
    victim = next(p for p in engine.catalog.photos_under([archive]) if p.path.name == "h10.jpg")
    key = feature_key(engine.extractors["technical"])
    vector = engine.catalog.get_features(key, [victim.content_key])[victim.content_key].copy()
    vector[0] = np.nan
    engine.catalog.put_features(key, {victim.content_key: vector})

    result = engine.select(request(archive, "best", limit=20))

    assert "h10.jpg" not in {p.path.name for g in result.groups for p in g.picks}
    assert any("invalid (NaN/inf) features" in note for note in result.notes)


@pytest.mark.parametrize(("task", "expected"), [("best", 24), ("triptych", 6)])
def test_unset_limit_resolves_to_the_tasks_default_and_is_recorded(
    engine: Engine, archive: Path, task: str, expected: int
) -> None:
    result = engine.select(request(archive, task))
    assert result.request.limit == expected
    assert engine.runs.load(result.run_id).request.limit == expected


def test_an_explicit_limit_is_kept(engine: Engine, archive: Path) -> None:
    assert engine.select(request(archive, "best", limit=3)).request.limit == 3


@pytest.mark.parametrize(("task", "expected"), [("best", 0.5), ("triptych", 1 / 3)])
def test_unset_describe_weight_resolves_to_the_tasks_default_and_is_recorded(
    engine: Engine, archive: Path, task: str, expected: float
) -> None:
    result = engine.select(request(archive, task))
    assert result.request.describe_weight == pytest.approx(expected)
    assert engine.runs.load(result.run_id).request.describe_weight == pytest.approx(expected)


def test_describe_weight_is_the_descriptions_share(telling: Engine, archive: Path) -> None:
    result = telling.select(request(archive, "best", criteria="fog", describe_weight=0.8))
    weights = {e.signal: e.weight for e in result.groups[0].picks[0].evidence}
    assert weights["criteria"] == pytest.approx(0.8, abs=1e-3)
    assert result.request.describe_weight == 0.8


def test_a_description_weighted_zero_is_noted(telling: Engine, archive: Path) -> None:
    result = telling.select(request(archive, "best", criteria="fog", describe_weight=0))
    assert any("counts for nothing" in n for n in result.notes)
    assert all(e.signal != "criteria" for g in result.groups for p in g.picks for e in p.evidence)
    assert any(p.clear_match is not None for g in result.groups for p in g.picks)


# -- measuring against your ratings ------------------------------------------------------


def test_ratings_are_measured_within_each_location(engine: Engine, archive: Path) -> None:
    mill = archive / "Old Mill - March 2021"
    liked = [mill / "view1.jpg", mill / "view2.jpg"]
    disliked = [mill / "view3.jpg", mill / "view4.jpg"]
    ratings = [Rating(p.resolve(), 1.0, "stars") for p in liked]
    ratings += [Rating(p.resolve(), 0.0, "stars") for p in disliked]
    ratings.append(Rating(archive.parent / "gone" / "x.jpg", 1.0, "stars"))
    engine.indexer.index([archive])
    report = Sessions(engine).evaluate_ratings(ratings)
    (row,) = report.rows
    assert (row.location, row.rated, row.liked, row.disliked) == ("Old Mill", 4, 2, 2)
    assert set(row.auc) >= {"quality", "strength"}
    assert all(0.0 <= v <= 1.0 for v in row.auc.values())
    assert report.missing == 1
    assert report.rated == 4
    mixes = Sessions(engine).evaluate_ratings(ratings, quality_weights=(0.0, 1.0)).rows[0].auc
    assert set(mixes) >= {"strength", "strength (quality 0)", "strength (quality 1)"}
    assert mixes["strength (quality 1)"] == mixes["quality"]


def test_locations_without_both_verdicts_are_counted_not_measured(
    engine: Engine, archive: Path
) -> None:
    harbor = (archive / "Harbor - June 2022").glob("*.jpg")
    only_liked = [Rating(p.resolve(), 1.0, "stars") for p in harbor]
    engine.indexer.index([archive])
    report = Sessions(engine).evaluate_ratings(only_liked)
    assert report.rows == []
    assert report.one_sided == 1


# -- calibrating same view ---------------------------------------------------------------


def test_a_same_view_labelling_set_samples_around_the_reference(
    engine: Engine, archive: Path
) -> None:
    reference = archive / "Old Mill - March 2021" / "view1.jpg"
    made = Sessions(engine).label_same_view(reference, [archive])
    assert (made.kind, made.reference) == ("same_view", reference.resolve())
    assert made.text == "Same view as view1.jpg"
    names = {item.path.name for item in made.items}
    assert "view1.jpg" not in names  # the reference itself is not asked about
    assert {"burst0.jpg", "revisit1.jpg"} <= names
    again = Sessions(engine).label_same_view(reference, [archive])
    assert [i.content_key for i in again.items] == [i.content_key for i in made.items]


def test_a_same_view_labelling_set_is_drawn_by_patch_share(engine: Engine, archive: Path) -> None:
    reference = archive / "Old Mill - March 2021" / "view1.jpg"
    made = Sessions(engine).label_same_view(reference, [archive])
    assert made.scorer == "same_view patches (tiny)"
    names = {item.path.name for item in made.items}
    assert {"burst0.jpg", "burst1.jpg", "burst2.jpg"} <= names  # surely the same view
    assert "view1.jpg" not in names


def test_a_taste_session_rates_one_frame_per_burst_from_every_strength(
    engine: Engine, archive: Path
) -> None:
    made = Sessions(engine).label_taste([archive])
    assert made.kind == "taste"
    assert made.reference is None
    assert made.text == "Your taste: archive"
    names = [item.path.name for item in made.items]
    # view1 and burst0-2 are one burst: rated once. The other 8 photos are all rated.
    assert len({"view1.jpg", "burst0.jpg", "burst1.jpg", "burst2.jpg"} & set(names)) == 1
    assert len(names) == len(set(names)) == 9
    assert engine.labels.load(made.set_id).kind == "taste"
    assert (
        Sessions(engine).label_taste([archive]).items == made.items
    )  # the same folders, the same sample


def test_an_anchor_session_rates_one_frame_per_burst_and_says_what_it_could_not_read(
    engine: Engine, archive: Path
) -> None:
    made = Sessions(engine).label_anchors([archive])
    assert (made.kind, made.text) == ("anchor", "Could it anchor a story: archive")
    names = [item.path.name for item in made.items]
    assert len({"view1.jpg", "burst0.jpg", "burst1.jpg", "burst2.jpg"} & set(names)) == 1
    assert len(names) == len(set(names)) == 9
    assert "without scale or subject" in made.scorer  # the tiny embedder reads no text
    assert Sessions(engine).label_anchors([archive]).items == made.items
    assert engine.labels.load(made.set_id).kind == "anchor"


BURST = {"view1.jpg", "burst0.jpg", "burst1.jpg", "burst2.jpg"}


def rated_by_hand(engine: Engine, archive: Path, name: str, kind: str) -> None:
    """``name`` rated in a session of its own, as any frame of a burst can be (Compare shows
    another frame than a taste session does)."""
    (photo,) = [p for p in engine.catalog.photos_under([archive]) if p.path.name == name]
    made = LabelSet(
        set_id="by-hand",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        text="by hand",
        folders=(archive,),
        scorer="by hand",
        items=(LabelItem(content_key=photo.content_key, path=photo.path),),
        kind=kind,
    )
    engine.labels.save(made)
    engine.ratings.rate_photo(made.set_id, photo.content_key, 0)


@pytest.mark.parametrize("kind", ["taste", "anchor"])
def test_a_new_session_draws_only_photos_not_yet_rated_for_its_use(
    engine: Engine, archive: Path, kind: str
) -> None:
    label = Sessions(engine).label_taste if kind == "taste" else Sessions(engine).label_anchors
    first = label([archive])
    shown = {i.path.name for i in first.items}
    (frame,) = BURST & shown
    other_frame = min(BURST - shown)  # the burst's other frames were never sampled
    rated_by_hand(engine, archive, other_frame, kind)
    engine.ratings.rate_photo(
        first.set_id, next(i.content_key for i in first.items if i.path.name == "view2.jpg"), 2
    )
    again = label([archive])
    names = {i.path.name for i in again.items}
    assert not names & (BURST | {"view2.jpg"})  # a rated frame's burst is reviewed whole
    assert len(names) == 7
    assert all(i.stratum and i.inclusion == 1.0 for i in again.items)  # a small pool: all of it
    assert frame not in names


@pytest.mark.parametrize("kind", ["taste", "anchor"])
def test_a_place_with_nothing_left_to_rate_says_so(
    engine: Engine, archive: Path, kind: str
) -> None:
    label = Sessions(engine).label_taste if kind == "taste" else Sessions(engine).label_anchors
    first = label([archive])
    for item in first.items:
        engine.ratings.rate_photo(first.set_id, item.content_key, 1)  # not sure is an answer too
    with pytest.raises(InvalidRequestError, match="already been rated"):
        label([archive])


def test_anchor_ratings_are_measured_against_each_signal(engine: Engine, archive: Path) -> None:
    made = Sessions(engine).label_anchors([archive])
    for i, item in enumerate(made.items):
        engine.ratings.rate_photo(made.set_id, item.content_key, 2 if i % 2 else 0)
    report = Sessions(engine).evaluate_anchors()
    assert (report.could, report.could_not) == (4, 5)
    assert "general strength" in {s.signal for s in report.signals}


class WithVocabularies:
    """The engine's models, plus scale and subject vocabularies read from palette columns:
    the tiny embedder reads no text, so it has neither."""

    def __init__(self, models: Any) -> None:
        self._models = models

    def names(self) -> list[str]:
        return [*self._models.names(), "scale", "subject"]

    def get(self, name: str) -> Any:
        if name == "scale":
            return f.FeatureLabels("scale", "palette", SCALES)
        if name == "subject":
            return f.FeatureLabels("subject", "palette", ("living", "none"))
        return self._models.get(name)

    def versions(self) -> dict[str, str]:
        return dict(self._models.versions())


def test_anchor_ratings_measure_the_prior_that_roles_try_first(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One definition of the anchor prior (``selection.anchors``): if the task's changes,
    ``calico eval anchors`` measures the new one, not a copy."""
    monkeypatch.setattr(engine, "models", WithVocabularies(engine.models))
    rows: list[AnchorRating] = []

    def kept(rated: Sequence[AnchorRating]) -> AnchorReport:
        rows.extend(rated)
        return measure_anchors(rated)

    marker = np.float32(0.123)

    def prior(strength: NDArray[np.float32], **_: Any) -> NDArray[np.float32]:
        return np.full(len(strength), marker)

    monkeypatch.setattr("calico.sessions.measure_anchors", kept)
    monkeypatch.setattr("calico.sessions.anchor_prior", prior)
    made = Sessions(engine).label_anchors([archive])
    for i, item in enumerate(made.items):
        engine.ratings.rate_photo(made.set_id, item.content_key, 2 if i % 2 else 0)
    report = Sessions(engine).evaluate_anchors()
    signals = {s.signal for s in report.signals}
    assert {"person or animal", "wide view", "tried first by roles"} <= signals
    assert rows
    assert all(r.signals["tried first by roles"] == pytest.approx(marker) for r in rows)


def test_a_taste_learned_on_what_is_now_the_main_embedding_still_runs(
    engine: Engine, archive: Path
) -> None:
    """Trained with --embedder X, then X made the main embedder: its vectors moved from the
    view's features to its embeddings, and the taste failed with a KeyError."""
    tiny = engine.extractors["tiny"]
    x = f.unit(np.random.default_rng(0).normal(size=(20, tiny.dim)))
    weights = fit_taste(
        x, (x[:, 0] > 0).astype(np.int8), feature=feature_key(tiny), extractor="tiny"
    )
    engine.trained.save_taste(weights)
    result = engine.select(SelectionRequest(task="best", folders=(archive,), template="yours"))
    assert result.groups
    assert result.request.template == "yours"


def test_a_folder_named_in_decomposed_unicode_is_the_catalogued_folder(
    engine: Engine, archive: Path
) -> None:
    """Finder hands out NFD; the collection roots and the catalog are NFC."""
    folder = archive / unicodedata.normalize("NFD", "Café - June 2023")
    f.save_jpeg(f.scene(7), archive / "Café - June 2023" / "a.jpg")
    f.save_jpeg(f.scene(8), archive / "Café - June 2023" / "b.jpg")
    result = engine.select(SelectionRequest(task="best", folders=(folder,)))
    assert {p.path.name for g in result.groups for p in g.picks} == {"a.jpg", "b.jpg"}
    example = folder / "a.jpg"
    again = engine.select(SelectionRequest(task="best", folders=(folder,), examples=(example,)))
    assert again.groups


def test_a_folder_typed_in_other_capitals_is_the_catalogued_folder(
    engine: Engine, archive: Path
) -> None:
    probe = archive / "CaseProbe"
    probe.mkdir()
    insensitive = (archive / "caseprobe").exists()
    probe.rmdir()
    if not insensitive:
        pytest.skip("this volume tells capitals apart, so it is another folder")
    result = engine.select(
        SelectionRequest(task="best", folders=(archive / "old mill - march 2021",))
    )
    assert result.groups


@pytest.mark.parametrize("inside", ["home", "root"])
def test_a_calico_home_and_an_archive_inside_one_another_are_refused(
    tmp_path: Path, inside: str
) -> None:
    """Calico would write its catalog and thumbnails into the archive (or index its own
    home as photos); nothing is written."""
    outer = tmp_path / "outer"
    home, root = (outer / "calico", outer) if inside == "home" else (outer, outer / "photos")
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings(home=home, collections=(CollectionConfig(name="all", root=root),))
    with pytest.raises(HomeInsideArchiveError, match=r"calico\.toml"):
        Engine.open(settings)
    assert not (home / "catalog.sqlite").exists()


def test_a_search_says_which_files_it_did_not_read(engine: Engine, archive: Path) -> None:
    """A lone .pef used to be invisible; now it is read, and a video is named as left out."""
    (archive / "Harbor - June 2022" / "clip.MOV").write_bytes(b"")
    result = engine.select(SelectionRequest(task="best", folders=(archive,)))
    assert any("left out: 1 .mov" in n for n in result.notes)


def test_a_thumb_whose_run_could_not_be_read_is_left_out_of_taste(
    engine: Engine, archive: Path
) -> None:
    """A 👍 on a run saved by another version was migrated with its task unknown (it may
    have been on a story): taste counts only thumbs on Strongest."""
    result = engine.select(SelectionRequest(task="best", folders=(archive,)))
    first, second = result.groups[0].picks[:2]
    engine.journal.append(
        Answer.model_validate(
            {
                "at": "2026-10-01T09:00:00Z",
                "use": "result",
                "scope": {"task": None},
                "subject": [second.content_key],
                "value": 1,
                "source": "catalog feedback",
            }
        )
    )
    engine.ratings.rate_result(result.run_id, first.content_key, 1)
    labels = taste_labels(engine.ratings.current())
    photos = Sessions(engine)._taste_photos(engine.settings.embedder, labels)
    assert [p.content_key for p in photos] == [first.content_key]


def test_a_damaged_taste_falls_back_to_general_and_says_to_retrain(
    engine: Engine, archive: Path
) -> None:
    x = f.unit(np.random.default_rng(0).normal(size=(20, 4)))
    engine.trained.save_taste(
        fit_taste(x, (x[:, 0] > 0).astype(np.int8), feature="tiny@2", extractor="")
    )
    (engine.settings.models_dir / "taste.npz").write_bytes(b"damaged")
    result = engine.select(SelectionRequest(task="best", folders=(archive,), template="yours"))
    assert result.groups
    (note,) = [n for n in result.notes if "could not be read" in n]
    assert "calico train taste" in note


def test_without_patches_a_labelling_set_is_drawn_by_whole_photo_match(
    engine: Engine, archive: Path
) -> None:
    settings = engine.settings.model_copy(update={"view_embedder": "palette"})
    with Engine.open(settings) as whole:
        made = Sessions(whole).label_same_view(
            archive / "Old Mill - March 2021" / "view1.jpg", [archive]
        )
    assert made.scorer == "same_view match (layout 0.5)"
    assert {"burst0.jpg", "revisit1.jpg"} <= {item.path.name for item in made.items}


def test_same_view_labels_measure_the_threshold(engine: Engine, archive: Path) -> None:
    reference = archive / "Old Mill - March 2021" / "view1.jpg"
    made = Sessions(engine).label_same_view(reference, [archive])
    same = {"burst0.jpg", "burst1.jpg", "burst2.jpg", "revisit1.jpg"}
    for item in made.items:
        engine.ratings.rate_photo(made.set_id, item.content_key, 2 if item.path.name in same else 0)
    report = Sessions(engine).evaluate_same_view()
    (row,) = report.rows
    assert (row.set_id, row.labelled, row.same) == (made.set_id, len(made.items), 4)
    assert row.auc == pytest.approx(1.0)
    assert row.at_default.threshold == 0.75
    assert 0 < report.pooled.threshold <= 1
    assert report.pooled.recall == pytest.approx(1.0)
    assert report.skipped == []


class Mirror(TinyEmbedder):
    """A second embedder, registered but not configured: built only when measured."""

    name = "mirror"


def test_same_view_can_be_measured_with_another_embedder_and_mix(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "calico.features.extractor_registry",
        lambda: Registry.from_mapping("extractor", {"mirror": Mirror}),
    )
    reference = archive / "Old Mill - March 2021" / "view1.jpg"
    made = Sessions(engine).label_same_view(reference, [archive])
    same = {"burst0.jpg", "burst1.jpg", "burst2.jpg", "revisit1.jpg"}
    for item in made.items:
        engine.ratings.rate_photo(made.set_id, item.content_key, 2 if item.path.name in same else 0)
    report = Sessions(engine).evaluate_same_view(embedder="mirror", layout_weight=0.0)
    assert (report.embedder, report.layout_weight) == ("mirror", 0.0)
    (row,) = report.rows
    assert row.labelled == len(made.items)
    keys = [i.content_key for i in made.items]
    assert engine.catalog.has_features(feature_key(Mirror()), keys) == set(keys)  # cached
    default = Sessions(engine).evaluate_same_view()
    assert (default.embedder, default.layout_weight) == ("tiny", 0.5)


def test_measuring_with_an_unknown_embedder_names_the_ones_there_are(
    engine: Engine, archive: Path
) -> None:
    with pytest.raises(UnknownPluginError, match="clip"):
        Sessions(engine).evaluate_same_view(embedder="nonsense")


def test_a_same_view_set_whose_reference_is_gone_is_reported_as_skipped(
    engine: Engine, archive: Path, tmp_path: Path
) -> None:
    reference = f.save_jpeg(f.scene(1), archive / "Old Mill - March 2021" / "copy.jpg")
    made = Sessions(engine).label_same_view(reference, [archive / "Harbor - June 2022"])
    engine.ratings.rate_photo(made.set_id, made.items[0].content_key, 0)
    reference.unlink()
    report = Sessions(engine).evaluate_same_view()
    assert report.rows == []
    assert report.skipped == [made.set_id]


def test_an_uploaded_reference_is_kept_with_the_labels(
    engine: Engine, archive: Path, tmp_path: Path
) -> None:
    upload = f.save_jpeg(f.scene(1), tmp_path / "uploads" / "abc" / "mine.jpg")
    made = Sessions(engine).label_same_view(upload, [archive])
    assert made.reference is not None
    assert made.reference.is_relative_to(engine.settings.labels_dir)
    assert made.reference.read_bytes() == upload.read_bytes()
    assert made.text == "Same view as mine.jpg"
    engine.labels.delete(made.set_id)
    assert not made.reference.exists()


def test_describe_evaluations_leave_same_view_sets_alone(reading: Engine, archive: Path) -> None:
    made = Sessions(reading).label_same_view(
        archive / "Old Mill - March 2021" / "view1.jpg", [archive]
    )
    label_halves(reading, made.set_id)
    assert Sessions(reading).evaluate_criteria(["baseline"]) == []
    assert Sessions(reading).evaluate_examples(k=1) == []


# -- describing by example --------------------------------------------------------------


def test_examples_alone_rank_the_photos_most_like_them_first(engine: Engine, archive: Path) -> None:
    example = archive / "Old Mill - March 2021" / "view1.jpg"
    result = engine.select(
        request(archive, "best", examples=[example], describe_weight=1.0, options={"diversity": 0})
    )
    picks = result.groups[0].picks
    assert picks[0].path.name in {"burst0.jpg", "burst1.jpg", "burst2.jpg", "revisit1.jpg"}
    assert result.groups[0].title.endswith("like your examples")


def test_example_photos_are_left_out_of_the_results_and_recorded(
    engine: Engine, archive: Path
) -> None:
    example = archive / "Harbor - June 2022" / "h10.jpg"
    result = engine.select(request(archive, "best", examples=[example], limit=50))
    picked = {p.path.name for g in result.groups for p in g.picks}
    picked |= {a.name for g in result.groups for p in g.picks for a in p.alternates}
    assert "h10.jpg" not in picked
    assert any("example" in n and "left out" in n for n in result.notes)
    keys = {p.path.name: p.content_key for p in engine.catalog.photos_under([archive])}
    assert result.example_keys == (keys["h10.jpg"],)
    assert engine.runs.load(result.run_id).example_keys == (keys["h10.jpg"],)
    signals = engine.runs.signals(result.run_id)
    assert all(row["example_similarity"] for row in signals)


def test_examples_add_to_the_text_they_describe(telling: Engine, archive: Path) -> None:
    example = archive / "Old Mill - March 2021" / "view1.jpg"
    plain = telling.select(request(archive, "best", criteria="fog", describe_weight=1.0))
    steered = telling.select(
        request(archive, "best", criteria="fog", examples=[example], describe_weight=1.0)
    )
    assert [p.path for p in plain.groups[0].picks] != [p.path for p in steered.groups[0].picks]
    assert "“fog”" in steered.groups[0].title
    assert any(p.clear_match is not None for g in steered.groups for p in g.picks)


def test_an_example_from_outside_the_chosen_folders_is_noted(engine: Engine, archive: Path) -> None:
    example = archive / "Old Mill - March 2021" / "view1.jpg"
    harbor = archive / "Harbor - June 2022"
    result = engine.select(SelectionRequest(task="best", folders=(harbor,), examples=(example,)))
    assert any("view1.jpg" in n and "outside the chosen folders" in n for n in result.notes)


def test_examples_must_be_photos_in_a_collection(
    engine: Engine, archive: Path, tmp_path: Path
) -> None:
    stray = tmp_path / "elsewhere" / "x.jpg"
    f.save_jpeg(f.scene(5), stray)
    with pytest.raises(NotInCollectionError):
        engine.select(request(archive, "best", examples=[stray]))


def test_the_retired_criteria_weight_option_says_what_replaced_it(
    engine: Engine, archive: Path
) -> None:
    with pytest.raises(InvalidRequestError, match="describe_weight"):
        engine.select(request(archive, "best", options={"criteria_weight": 2}))


# -- matching part by part ------------------------------------------------------------


def label_old_mill_view(engine: Engine, reference: Path, archive: Path) -> LabelSet:
    made = Sessions(engine).label_same_view(reference, [archive])
    same = {"burst0.jpg", "burst1.jpg", "burst2.jpg", "revisit1.jpg"}
    for item in made.items:
        engine.ratings.rate_photo(made.set_id, item.content_key, 2 if item.path.name in same else 0)
    return made


def test_same_view_labels_can_be_measured_by_matching_patches(
    engine: Engine, archive: Path
) -> None:
    made = label_old_mill_view(engine, archive / "Old Mill - March 2021" / "view1.jpg", archive)
    report = Sessions(engine).evaluate_same_view(patches=True)
    assert report.embedder == "tiny (patches)"
    (row,) = report.rows
    assert row.labelled == len(made.items)
    assert row.auc == pytest.approx(1.0)
    assert row.at_default.threshold == SURE_SHARE


def test_same_view_labels_can_be_measured_by_matches_in_place(
    engine: Engine, archive: Path
) -> None:
    made = label_old_mill_view(engine, archive / "Old Mill - March 2021" / "view1.jpg", archive)
    shares = Sessions(engine).evaluate_same_view(patches=True)
    measured = Sessions(engine).evaluate_view_geometry()
    today, in_place = measured.whole
    assert [r.auc for r in today.rows] == [r.auc for r in shares.rows]
    assert (today.embedder, in_place.embedder) == ("tiny (patches)", "tiny (patches in place)")
    (row,) = in_place.rows
    assert row.labelled == len(made.items)
    assert in_place.versus is not None
    assert in_place.versus.places == 1
    # In place is at most the share, photo by photo.
    (search,), (placed,) = today.searches, in_place.searches
    assert all(placed.match[k] <= search.match[k] for k in search.match)


def test_matches_in_place_on_the_hold_out_alone_are_measured_only_after_a_draw(
    engine: Engine, archive: Path
) -> None:
    label_old_mill_view(engine, archive / "Old Mill - March 2021" / "view1.jpg", archive)
    with pytest.raises(HoldoutError, match="calico holdout draw"):
        Sessions(engine).evaluate_view_geometry(only_holdout=True)
    assert Sessions(engine).draw_holdout().references == ()  # one reference: none held out
    measured = Sessions(engine).evaluate_view_geometry(only_holdout=True)
    assert [r.rows for r in measured.whole] == [[], []]


def test_a_frame_saved_twice_is_counted_once_and_a_burst_is_not(
    engine: Engine, archive: Path
) -> None:
    # burst0's pixels re-exported later: different bytes and date, so a different photo.
    export = archive / "Old Mill - May 2024" / "burst0-export.jpg"
    f.save_jpeg(f.scene(1, shift=(0, 0)), export, captured_at=datetime(2024, 5, 9, 12))
    made = label_old_mill_view(engine, archive / "Old Mill - March 2021" / "view1.jpg", archive)
    (item,) = [i for i in made.items if i.path.name == "burst0-export.jpg"]
    engine.ratings.rate_photo(made.set_id, item.content_key, 2)
    for report in (
        Sessions(engine).evaluate_same_view(),
        Sessions(engine).evaluate_same_view(patches=True),
    ):
        (row,) = report.rows
        assert report.copies == 1
        # burst0-2 are three frames, not copies; revisit1 is the fourth same view.
        assert (row.labelled, row.same) == (len(made.items) - 1, 4)


def test_a_reference_outside_the_catalog_is_matched_patch_by_patch(
    engine: Engine, archive: Path, tmp_path: Path
) -> None:
    upload = f.save_jpeg(f.scene(1), tmp_path / "uploads" / "abc" / "mine.jpg")
    made = label_old_mill_view(engine, upload, archive)
    report = Sessions(engine).evaluate_same_view(patches=True)
    (row,) = report.rows
    assert report.copies == 1  # burst0 is view1's pixels again
    assert row.labelled == len(made.items) - 1
    assert row.auc > 0.5  # read and scored; the tiny embedder's 4x4 grid is coarse (DINOv2: 32x32)


# -- your taste --------------------------------------------------------------------

LIKED = {"view1.jpg", "view2.jpg", "revisit1.jpg", "revisit2.jpg", "h10.jpg"}
LIKED |= {f"burst{i}.jpg" for i in range(3)}


def rate(engine: Engine, folders: list[Path]) -> None:
    made = Sessions(engine).label_taste(folders)
    for item in made.items:
        engine.ratings.rate_photo(
            made.set_id, item.content_key, 2 if item.path.name in LIKED else 0
        )


def old_mill(archive: Path) -> list[Path]:
    return [archive / "Old Mill - March 2021", archive / "Old Mill - May 2024"]


def test_taste_is_judged_at_each_place_by_a_model_trained_elsewhere(
    engine: Engine, archive: Path
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    (report,) = Sessions(engine).evaluate_taste()
    assert [r.location for r in report.rows] == ["Harbor", "Old Mill"]
    assert report.trained_on == 9  # 6 at the mill (one frame of the burst), 3 at the harbor
    assert report.feature.startswith("tiny@")
    assert [r.l2 for r in Sessions(engine).evaluate_taste(l2s=[1.0, 30.0])] == [1.0, 30.0]


def test_taste_counts_your_ratings_among_the_photos_strongest_would_show_first(
    engine: Engine, archive: Path
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    (report,) = Sessions(engine).evaluate_taste()
    assert engine.runs.recent() == []  # measured without saving a run
    assert report.first_at == TASTE_FIRST
    mill = next(r for r in report.rows if r.location == "Old Mill")

    shown = engine.select(request(archive, "best", folders=old_mill(archive), limit=20))
    current = engine.ratings.current()
    verdicts = {
        i.content_key: given[i.content_key] == 2
        for s in engine.labels.all()
        for given in [labels_of(s, current)]
        for i in s.items
        if i.path.parent.name.startswith("Old Mill") and i.content_key in given
    }
    rated = [verdicts[p.content_key] for p in shown.groups[0].picks if p.content_key in verdicts]
    assert mill.general_first[0] == FirstPicks(rated=len(rated), liked=sum(rated))
    assert len(mill.yours_first) == len(TASTE_FIRST)
    assert all(first.rated <= k for first, k in zip(mill.yours_first, TASTE_FIRST, strict=True))
    assert report.general_first[0].rated == sum(r.general_first[0].rated for r in report.rows)


class SpreadReader(f.FirstAxisReader):
    """Reads each text as an axis of its own (by its length), so General's aesthetic, which
    asks the same photo whether it is strong or weak, varies from photo to photo."""

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i, len(text) % self.dim] = 1.0
        return out


def general_that_reads(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the engine a General whose aesthetic can run (the test embedder reads no text)."""
    reader = SpreadReader(engine.embedder)
    monkeypatch.setattr(engine, "models", Models.from_settings(engine.settings, reader))


def test_the_evaluations_pool_gives_each_photo_the_strength_strongest_gives_it(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `eval taste` scored strength among the rated photos, and the search scores it over the
    # whole location (10 November 2026 audit, B-7, SG-1): same arrays, same answer.
    general_that_reads(engine, monkeypatch)
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    sessions = Sessions(engine)
    options = engine.task("best").parse_options({})
    share = options.quality_weight / (options.quality_weight + options.aesthetic_weight)
    photos = sessions._taste_photos(
        engine.settings.embedder, taste_labels(engine.ratings.current())
    )
    pools = sessions._taste_pools(photos, engine.settings.embedder)
    assert set(pools) == {"Old Mill", "Harbor"}
    for location, pool in pools.items():
        everything = engine.catalog.photos_with_keys(engine.catalog.keys_at([location])[location])
        kept, view = engine.feature_view(everything, Trace())
        shipped = sessions._place_strength(kept, view, engine.models)
        assert list(pool.keys) == [p.content_key for p in kept]
        assert shipped_strength(pool.quality, pool.general, share) == pytest.approx(
            shipped, abs=1e-5
        )
        assert {p.content_key for p in photos if p.location == location} <= set(pool.keys)


def test_the_report_says_what_strength_was_scored_over(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    (unable,) = Sessions(engine).evaluate_taste()  # the test embedder reads no text
    assert unable.pooled == 0
    assert any("could not run" in note for note in unable.notes)
    general_that_reads(engine, monkeypatch)
    (over_pool,) = Sessions(engine).evaluate_taste()
    assert over_pool.pooled == 2
    assert any("whole pool" in note for note in over_pool.notes)
    (among_rated,) = Sessions(engine).evaluate_taste(fit="centred")
    assert among_rated.pooled == 0
    assert any("rated photos alone" in note for note in among_rated.notes)


def small_sessions(monkeypatch: pytest.MonkeyPatch, depth: int) -> None:
    """Sessions of 5 drawn by value, from each template's first ``depth``: the archive's
    places are smaller than a real session, which would take each one whole."""
    monkeypatch.setattr(evaluation, "TASTE_RANDOM", 1)
    monkeypatch.setattr(
        evaluation,
        "VALUE_QUOTAS",
        (
            ("first: General only", 1),
            ("first: Yours only", 1),
            ("first: both", 1),
            ("top quarter", 1),
        ),
    )
    monkeypatch.setattr(sessions_module, "VALUE_DEPTH", depth)


def test_a_taste_session_draws_from_what_each_template_shows_first(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[list[str], list[str], list[str]]] = []
    real = evaluation.sample_by_value

    def spy(
        ranked: Sequence[str],
        general_first: Sequence[str],
        yours_first: Sequence[str],
        seed: int,
        *,
        exclude: Collection[str] = frozenset(),
    ) -> list[Drawn]:
        seen.append((list(ranked), list(general_first), list(yours_first)))
        return real(ranked, general_first, yours_first, seed, exclude=exclude)

    monkeypatch.setattr(sessions_module, "sample_by_value", spy)
    made = Sessions(engine).label_taste([archive])
    ranked, general, yours = seen[-1]
    shown = engine.select(request(archive, "best", limit=evaluation.VALUE_DEPTH)).groups[0].picks
    assert len(general) == len(shown)
    assert set(general) <= set(ranked)  # a burst's frame is the one the session would show
    assert yours == []
    assert "General's first picks" in made.scorer
    assert "Yours is not trained" in made.scorer

    rate(engine, [archive / "Harbor - June 2022"])
    rate(engine, [archive / "Old Mill - May 2024"])
    assert Sessions(engine).train_taste(force=True).saved
    made = Sessions(engine).label_taste([archive / "Old Mill - March 2021"])
    _, general, yours = seen[-1]
    assert yours
    assert "General's and Yours's first picks" in made.scorer


def test_ratings_drawn_for_their_value_are_counted_apart_from_the_whole_range(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small_sessions(monkeypatch, depth=2)
    rate(engine, old_mill(archive))  # 6 bursts: more than a session of 5, so drawn by value
    rate(engine, [archive / "Harbor - June 2022"])  # 3: taken whole
    mill_sets = [s for s in engine.labels.all() if "Old Mill" in s.text]
    (mill_set,) = mill_sets
    for_value = sum(i.stratum in VALUE_STRATA for i in mill_set.items)
    assert for_value == 4  # all but the one drawn at random
    (report,) = Sessions(engine).evaluate_taste()
    mill = next(r for r in report.rows if r.location == "Old Mill")
    assert (mill.rated, mill.for_value) == (5, for_value)
    harbor = next(r for r in report.rows if r.location == "Harbor")
    assert harbor.for_value == 0


def stated(answer: Answer) -> bool | None:
    """What an answer said of a photo for taste: liked, not liked, or nothing."""
    if answer.use == "result":
        return answer.value == 1
    if answer.use in ("taste", "compare") and answer.value in (0, 2):
        return answer.value == 2
    return None


def test_every_label_yours_learns_from_is_one_of_your_answers(
    engine: Engine, archive: Path
) -> None:
    """Nothing makes a label for Yours: each training photo's label is an answer you gave,
    to a taste or Compare session or as a 👍/👎 on Strongest (plan, decision 1)."""
    rate(engine, old_mill(archive))
    best = engine.select(request(archive, "best"))
    for n, pick in enumerate(best.groups[0].picks[:3]):
        engine.ratings.rate_result(best.run_id, pick.content_key, 1 if n % 2 else -1)
    given: dict[str, set[bool]] = {}
    for answer in engine.journal.answers():
        if (liked := stated(answer)) is not None:
            given.setdefault(answer.subject[0], set()).add(liked)
    labels = taste_labels(engine.ratings.current())
    photos = Sessions(engine)._taste_photos(engine.settings.embedder, labels)
    assert photos
    assert all(p.liked in given.get(p.content_key, set()) for p in photos)


def test_likes_on_strongest_results_train_but_likes_on_stories_do_not(
    engine: Engine, archive: Path
) -> None:
    rate(engine, old_mill(archive))
    best = engine.select(request(archive, "best")).run_id
    story = engine.select(request(archive, "triptych")).run_id
    keys = {p.path.name: p for p in engine.catalog.photos_under([archive])}
    for name, rating, run in (("h10.jpg", 1, best), ("h11.jpg", -1, best), ("h12.jpg", -1, story)):
        engine.ratings.rate_result(run, keys[name].content_key, rating)
    (report,) = Sessions(engine).evaluate_taste()
    assert report.trained_on == 6 + 2
    assert [r.location for r in report.rows] == ["Old Mill"]  # results are never judged


def test_stories_you_rated_are_re_scored_from_their_saved_evidence(
    engine: Engine, archive: Path
) -> None:
    story = engine.select(request(archive, "triptych", limit=2))
    liked, rejected = story.groups
    for group, rating in ((liked, 1), (rejected, -1)):
        for pick in group.picks:
            engine.ratings.rate_result(story.run_id, pick.content_key, -rating)
            engine.ratings.rate_result(story.run_id, pick.content_key, rating)
    best = engine.select(request(archive, "best"))
    pick = best.groups[0].picks[0]
    engine.ratings.rate_result(best.run_id, pick.content_key, -1)  # not a story

    reports = Sessions(engine).evaluate_stories([(0.8, 0.5)])
    report = reports.results
    assert reports.sessions.runs == 0  # thumbs are never counted as a story session's
    assert (report.liked, report.rejected, report.mixed, report.runs) == (1, 1, 0, 1)
    today, varied = report.rows
    assert (today.too_alike, varied.too_alike) == (None, 0.8)
    assert today.auc == 1.0  # the first story outscored the second when it was told


def test_a_story_session_tells_stories_todays_way_and_the_way_under_test(
    engine: Engine, archive: Path
) -> None:
    """The second way is the variant under test (``STORY_VARIANT``): anchors tried by the
    anchor prior, after 1,000 candidates (3 November 2026) and telling by roles (28 October
    to 2 November)."""
    made = Sessions(engine).label_stories([archive])
    assert (made.kind, made.items, made.text) == ("stories", (), "Stories: archive")
    assert len(made.stories) >= 3
    assert all(len(s.photos) == 3 and s.label is None for s in made.stories)
    told = [frozenset(p.content_key for p in s.photos) for s in made.stories]
    assert len(told) == len(set(told))  # a story told two ways is rated once
    ways = {(a.template, a.variety, a.way) for s in made.stories for a in s.arms}
    assert ways <= {("general", True, "today"), ("general", True, "likely_anchors")}
    asked = [engine.runs.load(r.run_id).request.options for r in engine.runs.recent()]
    assert {"anchor_by": "likely"} in asked
    assert made.scorer == "triptych: general, general likely_anchors"  # Yours not trained
    assert {s.run_id for s in made.stories} <= {r.run_id for r in engine.runs.recent()}
    assert engine.labels.load(made.set_id) == made
    again = Sessions(engine).label_stories([archive])
    assert [s.photos for s in again.stories] == [s.photos for s in made.stories]


def test_a_story_session_tells_them_your_way_too_once_yours_is_trained(
    engine: Engine, archive: Path
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    Sessions(engine).train_taste(force=True)
    made = Sessions(engine).label_stories([archive])
    assert {a.template for s in made.stories for a in s.arms} == {"general", "yours"}


CANDIDATE = StoryVariant(name="sy-vw55", template="general", options={"variety_weight": 0.55})


def test_a_story_session_can_set_a_named_candidate_beside_todays_way(
    engine: Engine, archive: Path
) -> None:
    made = Sessions(engine).label_stories([archive], variant=CANDIDATE)
    ways = {(a.template, a.variety, a.way) for s in made.stories for a in s.arms}
    assert ("general", True, "candidate:sy-vw55") in ways
    assert ways <= {("general", True, "today"), ("general", True, "candidate:sy-vw55")}
    asked = [engine.runs.load(r.run_id).request.options for r in engine.runs.recent()]
    assert {"variety_weight": 0.55} in asked
    assert {"anchor_by": "likely"} not in asked
    assert made.scorer == 'triptych: general, general candidate:sy-vw55 {"variety_weight": 0.55}'
    assert engine.labels.load(made.set_id) == made
    told = [frozenset(p.content_key for p in s.photos) for s in made.stories]
    assert len(told) == len(set(told))
    limits = {engine.runs.load(r.run_id).request.limit for r in engine.runs.recent()}
    assert limits == {evaluation.CANDIDATE_SESSION_STORIES}


def test_a_candidate_way_is_measured_against_todays_way(engine: Engine, archive: Path) -> None:
    made = Sessions(engine).label_stories([archive], variant=CANDIDATE)
    for index in range(len(made.stories)):
        engine.ratings.rate_story(made.set_id, index, 2 if index % 2 else 0, doesnt_fit=[])
    report = Sessions(engine).evaluate_stories([(0.8, 0.5)]).sessions
    assert {a.way for a in report.arms} == {"today", "candidate:sy-vw55"}


def test_a_candidate_for_yours_needs_yours_trained(engine: Engine, archive: Path) -> None:
    yours = dataclasses.replace(CANDIDATE, template="yours")
    with pytest.raises(InvalidRequestError, match="calico train taste"):
        Sessions(engine).label_stories([archive], variant=yours)


@pytest.mark.parametrize(
    ("name", "options"),
    [("", {"x": 1}), ("two words", {"x": 1}), ("sy-vw55", {})],
)
def test_a_candidate_needs_a_name_of_its_own_and_options(
    name: str, options: dict[str, object]
) -> None:
    with pytest.raises(InvalidRequestError):
        StoryVariant(name=name, template="general", options=options)


def test_stories_rated_in_a_session_are_measured_with_how_they_were_told(
    engine: Engine, archive: Path
) -> None:
    made = Sessions(engine).label_stories([archive])
    first = made.stories[0]
    engine.ratings.rate_story(made.set_id, 0, 2, doesnt_fit=[first.photos[0].content_key])
    engine.ratings.rate_story(made.set_id, 1, 0, doesnt_fit=[])
    engine.ratings.rate_story(made.set_id, 2, 1, doesnt_fit=[])
    reports = Sessions(engine).evaluate_stories([(0.8, 0.5)])
    report = reports.sessions
    assert reports.results.runs == 0  # a fit mark is never a thumb
    assert (report.liked, report.rejected, report.unsure, report.runs) == (1, 1, 1, 1)
    assert report.doesnt_fit == 1
    assert sum(a.works for a in report.arms) >= 1
    assert sum(a.doesnt for a in report.arms) >= 1
    assert all(not math.isnan(row.auc) for row in report.rows)


def test_a_story_session_needs_photos_to_tell_stories_from(engine: Engine, archive: Path) -> None:
    lone = archive / "Lone - May 2023"
    f.save_jpeg(f.scene(30), lone / "only.jpg", captured_at=T0)
    with pytest.raises(InvalidRequestError, match="No stories"):
        Sessions(engine).label_stories([lone])


def test_an_anchor_that_is_no_longer_uploaded_is_still_measured(
    engine: Engine, archive: Path, tmp_path: Path
) -> None:
    upload = f.save_jpeg(f.scene(7), tmp_path / "uploads" / "abc" / "mine.jpg")
    story = engine.select(request(archive, "triptych", limit=1, references=[upload]))
    upload.unlink()
    for pick in story.groups[0].picks:
        engine.ratings.rate_result(story.run_id, pick.content_key, 1)
    assert Sessions(engine).evaluate_stories([]).results.liked == 1


def finished_stories(tmp_path: Path, stories: dict[str, list[Path]]) -> Path:
    path = tmp_path / "finished.json"
    data = {"stories": {name: {"paths": [str(p) for p in ps]} for name, ps in stories.items()}}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_finished_stories_are_scored_against_swaps_of_each_photo(
    engine: Engine, archive: Path, tmp_path: Path
) -> None:
    engine.indexer.index([archive])
    mill = archive / "Old Mill - March 2021"
    story = [
        mill / "view2.jpg",
        archive / "Old Mill - May 2024" / "revisit1.jpg",
        mill / "view3.jpg",
    ]
    path = finished_stories(tmp_path, {"A": story, "W": [*story[:2], tmp_path / "gone.jpg"]})
    reports = Sessions(engine).evaluate_finished_stories(
        path, templates=["general"], variants=[{}, {"variety_weight": 0}]
    )
    assert [(r.template, r.variant) for r in reports] == [
        ("general", "defaults"),
        ("general", "variety_weight=0"),
    ]
    (scored,) = reports[0].stories
    assert reports[0].missing == ["W"]
    assert scored.name == "A"
    assert len(scored.swaps) == len(scored.shortlisted) == 3
    assert all(scored.swaps)  # the mill's other photos can take each place
    assert 0.0 <= scored.win <= 1.0
    assert scored.anchor_tried  # a pool this small is tried whole


def test_a_finished_photo_also_filed_at_another_place_is_still_scored(
    engine: Engine, archive: Path, tmp_path: Path
) -> None:
    mill = archive / "Old Mill - March 2021"
    copy = archive / "Harbor - June 2022" / "view2-copy.jpg"  # same bytes, catalogued first
    copy.write_bytes((mill / "view2.jpg").read_bytes())
    engine.indexer.index([archive])
    path = finished_stories(
        tmp_path, {"A": [mill / "view2.jpg", mill / "view3.jpg", mill / "view4.jpg"]}
    )
    (report,) = Sessions(engine).evaluate_finished_stories(path, templates=["general"])
    assert [s.name for s in report.stories] == ["A"]
    assert report.missing == []


def test_finished_stories_need_a_file_that_lists_three_photos_each(
    engine: Engine, tmp_path: Path
) -> None:
    with pytest.raises(InvalidRequestError, match="paths"):
        Sessions(engine).evaluate_finished_stories(tmp_path / "nope.json")
    bad = finished_stories(tmp_path, {"A": [tmp_path / "a.jpg"]})
    with pytest.raises(InvalidRequestError, match="three"):
        Sessions(engine).evaluate_finished_stories(bad)


def rate_stories(engine: Engine, made: LabelSet) -> None:
    for i, _ in enumerate(made.stories):
        engine.ratings.rate_story(made.set_id, i, (2, 0, 1)[i % 3], doesnt_fit=[])


def test_a_scorer_learned_from_story_answers_is_judged_session_by_session(
    engine: Engine, archive: Path, tmp_path: Path
) -> None:
    mill = Sessions(engine).label_stories(old_mill(archive))
    harbor = Sessions(engine).label_stories([archive / "Harbor - June 2022"])
    rate_stories(engine, mill)
    rate_stories(engine, harbor)
    may = archive / "Old Mill - May 2024"
    march = archive / "Old Mill - March 2021"
    path = finished_stories(
        tmp_path,
        {"A": [march / "view2.jpg", may / "revisit1.jpg", march / "view3.jpg"]},
    )
    report = Sessions(engine).learn_stories(path)
    assert report.finished == 1
    (judged,) = [s for s in report.sessions if s.session == mill.set_id]
    rated = [i for i in range(len(mill.stories)) if i % 3 != 2]  # not sure is left out
    assert judged.works + judged.doesnt == len(rated)
    assert judged.name == mill.text
    assert "today" in judged.auc
    assert {"strength", "theme", "variety", "bond", "belong"} <= set(report.signals)


def test_swapped_stories_are_made_from_finished_stories_only_when_asked(
    engine: Engine, archive: Path, tmp_path: Path
) -> None:
    rate_stories(engine, Sessions(engine).label_stories(old_mill(archive)))
    march = archive / "Old Mill - March 2021"
    path = finished_stories(
        tmp_path, {"A": [march / "view2.jpg", march / "view3.jpg", march / "view4.jpg"]}
    )
    assert Sessions(engine).learn_stories(path).swaps == 0
    swapped = Sessions(engine).learn_stories(path, swaps=True)
    assert 1 <= swapped.swaps <= 3  # one at each side the pool has a candidate for
    assert swapped.finished == 1
    assert Sessions(engine).learn_stories(path, swaps=True).swaps == swapped.swaps
    with pytest.raises(InvalidRequestError, match="finished stories"):
        Sessions(engine).learn_stories(swaps=True)


def test_session_stories_are_also_scored_today_s_way_with_other_options(
    engine: Engine, archive: Path
) -> None:
    mill = Sessions(engine).label_stories(old_mill(archive))
    rate_stories(engine, mill)
    report = Sessions(engine).learn_stories(variants=[{"distinct_weight": 0.3}, {}])
    assert report.variants == ("distinct_weight=0.3",)  # today itself is not a variant
    (judged,) = report.sessions
    assert "distinct_weight=0.3" in judged.auc


def test_stories_are_measured_with_the_view_embedder_a_search_uses(
    engine: Engine, archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rate_stories(engine, Sessions(engine).label_stories(old_mill(archive)))
    march = archive / "Old Mill - March 2021"
    path = finished_stories(
        tmp_path, {"A": [march / "view2.jpg", march / "view3.jpg", march / "view4.jpg"]}
    )
    seen: list[str] = []
    scorer = TriptychTask.scorer

    def spy(self: TriptychTask, ctx: TaskContext, *args: Any, **kwargs: Any) -> StoryScorer:
        seen.append(ctx.view_feature)
        return scorer(self, ctx, *args, **kwargs)

    monkeypatch.setattr(TriptychTask, "scorer", spy)
    monkeypatch.setattr(engine, "view_feature", lambda: "dinov2")
    Sessions(engine).evaluate_finished_stories(path, templates=["general"])
    Sessions(engine).learn_stories(path)
    assert seen
    assert set(seen) == {"dinov2"}  # not the main embedding, as a search would not


def test_held_out_story_sessions_are_never_learned_from_or_judged(
    engine: Engine, archive: Path
) -> None:
    mill = Sessions(engine).label_stories(old_mill(archive))
    rate_stories(engine, mill)
    write_holdout(
        engine.settings.holdout_path,
        HoldOut(places=(), references=(), story_sessions=(mill.set_id,), seed=1, drawn_at=T0),
    )
    report = Sessions(engine).learn_stories()
    assert report.sessions == []
    assert report.finished == 0


def test_without_a_trained_taste_yours_falls_back_to_general(engine: Engine, archive: Path) -> None:
    result = engine.select(request(archive, "best", template="yours"))
    assert result.request.template == "general"
    assert any("Yours" in n and "calico train taste" in n for n in result.notes)


def test_a_trained_taste_ranks_photos_when_you_choose_yours(engine: Engine, archive: Path) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    trained = Sessions(engine).train_taste(force=True)
    assert trained.saved
    yours = engine.select(request(archive, "best", template="yours"))
    assert yours.request.template == "yours"
    assert yours.versions["model:aesthetic"].startswith("taste@")
    general = engine.select(request(archive, "best", template="general"))
    assert not general.versions.get("model:aesthetic", "").startswith("taste@")
    with Engine.open(engine.settings.model_copy(update={"template": "yours"})) as again:
        assert again.select(request(archive, "best")).request.template == "yours"


def test_a_candidate_sharpness_is_measured_on_the_rated_photos_alone(
    engine: Engine, archive: Path
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    reports = Sessions(engine).evaluate_taste(quality_weights=[0.2, 0.5], technical="3")
    assert [r.quality_weight for r in reports] == [0.2, 0.5]
    for report in reports:
        assert report.technical == "technical@3"
        assert report.candidate is not None
        assert report.candidate.b.value == report.general_strength
        assert report.unread == 0
    rated = list(taste_labels(engine.ratings.current()).liked)
    assert engine.catalog.has_features("technical@3", rated) == set(rated)  # cached apart
    everything = [p.content_key for p in engine.catalog.photos_under([archive])]
    assert len(engine.catalog.has_features("technical@3", everything)) == len(rated)
    (today,) = Sessions(engine).evaluate_taste()
    assert (today.candidate, today.technical) == (None, "")


def test_a_rated_photo_that_cannot_be_read_is_left_out_of_the_candidate_and_counted(
    engine: Engine, archive: Path
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    (archive / "Harbor - June 2022" / "h11.jpg").unlink()  # still catalogued, gone from disk
    (report,) = Sessions(engine).evaluate_taste(technical="3")
    assert report.unread == 1


def test_centred_yours_is_measured_on_each_place_mean_over_all_its_photos(
    engine: Engine, archive: Path
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    (report,) = Sessions(engine).evaluate_taste(fit="centred")
    assert report.fit == "centred"
    assert report.fitted is not None
    assert report.fitted.b.value == report.yours_strength
    labels = taste_labels(engine.ratings.current())
    photos = Sessions(engine)._taste_photos(engine.settings.embedder, labels, centred=True)
    mill = [p for p in photos if p.location == "Old Mill"]
    everything = [p for p in engine.catalog.photos_under([archive]) if p.location == "Old Mill"]
    assert len(everything) > len(mill)  # the mean is over every photo there, rated or not
    _, view = engine.feature_view(everything, Trace())
    assert mill[0].place_mean is not None
    np.testing.assert_allclose(mill[0].place_mean, view.embeddings.mean(axis=0), atol=1e-6)


def test_measuring_taste_says_where_it_compares_views_and_does_it_once(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strongest's first picks compare a place's views part by part (one pick per view): at
    a place no search has compared, that is minutes of work a run must not do silently."""
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    monkeypatch.setattr(engine.catalog, "pair_shares_among", lambda key, keys: {})  # as if new
    seen: list[tuple[str, int, int]] = []
    Sessions(engine).evaluate_taste(
        progress=lambda place, done, total: seen.append((place, done, total))
    )
    assert seen
    assert {place for place, _, _ in seen} <= {"Old Mill", "Harbor"}
    assert all(0 <= done <= total for _, done, total in seen)
    monkeypatch.undo()
    seen.clear()
    Sessions(engine).evaluate_taste(
        progress=lambda place, done, total: seen.append((place, done, total))
    )
    assert seen == []  # remembered: nothing left to compare


def test_each_rated_photo_carries_its_collection_so_a_fit_is_judged_collection_by_collection(
    engine: Engine, archive: Path
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    labels = taste_labels(engine.ratings.current())
    photos = Sessions(engine)._taste_photos(engine.settings.embedder, labels)
    catalogued = {p.content_key: p.collection for p in engine.catalog.photos_under([archive])}
    assert photos
    assert all(p.collection and p.collection == catalogued[p.content_key] for p in photos)
    (report,) = Sessions(engine).evaluate_taste(fit="leaning")
    assert report.fit == "leaning"
    assert report.fitted is not None


class Brightness:
    """A one-number aesthetic, registered but not configured: built only when measured."""

    name = "bright"
    version = "1"
    dim = 1
    columns = ("score",)

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        return np.asarray(
            [[np.asarray(im, dtype=np.float32).mean()] for im in images], dtype=np.float32
        )


def with_brightness(monkeypatch: pytest.MonkeyPatch) -> None:
    real = extractor_registry()
    monkeypatch.setattr(
        "calico.features.extractor_registry",
        lambda: Registry.from_mapping(
            "extractor", {**{n: real.get(n) for n in real.names()}, "bright": Brightness}
        ),
    )


def test_a_candidate_aesthetic_for_general_is_measured_on_the_rated_photos_alone(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_brightness(monkeypatch)
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    (archive / "Harbor - June 2022" / "h11.jpg").unlink()  # still catalogued, gone from disk
    (report,) = Sessions(engine).evaluate_taste(aesthetic="bright")
    assert report.aesthetic_candidate == "bright@1"
    assert report.candidate_general is not None
    assert report.candidate_general.b.value == report.general_strength
    assert report.candidate_general_alone is not None
    assert report.unread == 1
    rated = list(taste_labels(engine.ratings.current()).liked)
    everything = [p.content_key for p in engine.catalog.photos_under([archive])]
    assert len(engine.catalog.has_features("bright@1", everything)) == len(rated) - 1
    (today,) = Sessions(engine).evaluate_taste()
    assert (today.candidate_general, today.aesthetic_candidate) == (None, "")


def score_file(path: Path, scores: dict[str, float]) -> Path:
    path.write_text("".join(json.dumps({"key": k, "score": v}) + "\n" for k, v in scores.items()))
    return path


def test_a_file_of_scores_is_measured_on_the_photos_it_covers(
    engine: Engine, archive: Path, tmp_path: Path
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    liked = taste_labels(engine.ratings.current()).liked
    always_right = score_file(tmp_path / "judge.jsonl", {k: float(v) for k, v in liked.items()})
    (report,) = Sessions(engine).evaluate_taste(scores=always_right)
    assert report.scores is not None
    assert report.scores.name.startswith("judge@")
    assert report.scores.scored == len(liked)
    assert report.scores.whole.versus_general.a.value == 1.0
    assert report.first_at == ()  # Strongest's first picks do not read a file's photos
    assert any("scores file" in note for note in report.notes)
    (today,) = Sessions(engine).evaluate_taste()
    assert today.scores is None


def test_a_photo_a_file_does_not_score_still_teaches_and_the_report_counts_what_matched(
    engine: Engine, archive: Path, tmp_path: Path
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    liked = taste_labels(engine.ratings.current()).liked
    mill = {
        p.content_key
        for p in engine.catalog.photos_with_keys(list(liked))
        if p.location == "Old Mill"
    }
    path = score_file(
        tmp_path / "judge.jsonl", {**{k: float(liked[k]) for k in mill}, "f" * 32: 1.0}
    )
    (report,) = Sessions(engine).evaluate_taste(scores=path)
    (today,) = Sessions(engine).evaluate_taste()
    assert [r.location for r in report.rows] == ["Old Mill"]  # the harbor has no score
    assert report.trained_on == today.trained_on  # but its ratings still teach
    assert report.scores is not None
    assert report.scores.scored == len(mill)
    (note,) = [n for n in report.notes if "scores file" in n]
    assert f"{len(mill)} of {len(liked)} rated photos" in note
    assert "1 photo in the file is not one you rated" in note


def test_a_file_that_scores_no_rated_photo_is_refused(
    engine: Engine, archive: Path, tmp_path: Path
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    path = score_file(tmp_path / "judge.jsonl", {"f" * 32: 1.0})
    with pytest.raises(InvalidRequestError, match="content key"):
        Sessions(engine).evaluate_taste(scores=path)


@pytest.mark.parametrize(
    "other", [{"fit": "centred"}, {"technical": "3"}, {"aesthetic": "bright"}, {"blend": True}]
)
def test_a_file_of_scores_is_the_one_candidate_measured_at_a_time(
    engine: Engine, archive: Path, tmp_path: Path, other: dict[str, Any]
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    liked = taste_labels(engine.ratings.current()).liked
    path = score_file(tmp_path / "judge.jsonl", {k: float(v) for k, v in liked.items()})
    with pytest.raises(InvalidRequestError, match="one candidate"):
        Sessions(engine).evaluate_taste(scores=path, **other)


class Larger(TinyEmbedder):
    """An embedding registered but not computed for every photo, as clip_large is."""

    name = "tiny_large"
    version = "1"


def with_larger(monkeypatch: pytest.MonkeyPatch) -> None:
    real = extractor_registry()
    monkeypatch.setattr(
        "calico.features.extractor_registry",
        lambda: Registry.from_mapping(
            "extractor", {**{n: real.get(n) for n in real.names()}, "tiny_large": Larger}
        ),
    )


def test_yours_is_measured_on_an_embedding_computed_for_the_rated_photos_alone(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A larger CLIP is judged on the rated photos before any hour goes into embedding the
    archive. What Strongest would show first needs every photo at a place, so those counts
    are left out, and the report says so (rule 5)."""
    with_larger(monkeypatch)
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    (archive / "Harbor - June 2022" / "h11.jpg").unlink()  # still catalogued, gone from disk
    (report,) = Sessions(engine).evaluate_taste(["tiny_large"])
    assert report.feature == "tiny_large@1"
    assert report.unread == 1
    assert report.rows
    assert (report.first_at, report.general_first, report.yours_first) == ((), (), ())
    assert any("First picks" in n and "tiny_large" in n for n in report.notes)
    rated = list(taste_labels(engine.ratings.current()).liked)
    everything = [p.content_key for p in engine.catalog.photos_under([archive])]
    assert len(engine.catalog.has_features("tiny_large@1", everything)) == len(rated) - 1
    (today,) = Sessions(engine).evaluate_taste()
    assert today.first_at == TASTE_FIRST
    assert not any("First picks" in n for n in today.notes)


def test_yours_on_another_embedding_is_measured_against_todays_yours_paired(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ViT-L/14 bar is set against today's Yours, so the two are judged on the same
    photos in one report."""
    with_larger(monkeypatch)
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    (archive / "Harbor - June 2022" / "h11.jpg").unlink()  # still catalogued, gone from disk
    (report,) = Sessions(engine).evaluate_taste(yours_embedder="tiny_large")
    assert report.feature.startswith("tiny@")  # today's Yours, against which it is judged
    assert (report.fit, report.yours_candidate) == ("embedding", "tiny_large@1")
    assert report.fitted is not None
    assert report.fitted.b.value == report.yours_strength
    assert report.unread == 1
    assert report.first_at == TASTE_FIRST  # today's Yours still runs Strongest
    with pytest.raises(InvalidRequestError, match="already"):
        Sessions(engine).evaluate_taste(yours_embedder=engine.settings.embedder)
    with pytest.raises(InvalidRequestError, match="one way"):
        Sessions(engine).evaluate_taste(yours_embedder="tiny_large", fit="offset")


def test_an_embedding_for_the_rated_photos_alone_is_refused_where_it_needs_every_photo(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Yours saved on it could not run in a search, and a centred fit needs each place's
    mean over every photo there."""
    with_larger(monkeypatch)
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    with pytest.raises(InvalidRequestError, match=r"tiny_large.*every photo"):
        Sessions(engine).train_taste("tiny_large", force=True)
    with pytest.raises(InvalidRequestError, match=r"centred.*tiny_large"):
        Sessions(engine).evaluate_taste(["tiny_large"], fit="centred")
    with pytest.raises(InvalidRequestError, match=r"layout.*embedding"):
        Sessions(engine).evaluate_taste(["layout"])


class LargerReader(Larger):
    """A larger embedding that reads text: "reads" any text as the first axis."""

    name = "tiny_large_reader"
    supports_text = True

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.eye(1, self.dim, dtype=np.float32).repeat(len(texts), axis=0)


def with_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    real = extractor_registry()
    monkeypatch.setattr(
        "calico.features.extractor_registry",
        lambda: Registry.from_mapping(
            "extractor",
            {
                **{n: real.get(n) for n in real.names()},
                "tiny_large": Larger,
                "tiny_large_reader": LargerReader,
            },
        ),
    )


def test_general_read_by_another_embedder_is_measured_on_the_rated_photos_alone(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Today's wording for General, read by a larger CLIP: no search uses it."""
    with_reader(monkeypatch)
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    (archive / "Harbor - June 2022" / "h11.jpg").unlink()  # still catalogued, gone from disk
    (report,) = Sessions(engine).evaluate_taste(general_embedder="tiny_large_reader")
    assert report.aesthetic_candidate.startswith("zs")
    assert report.aesthetic_candidate.endswith("-tiny_large_reader@1")
    assert report.candidate_general is not None
    assert report.candidate_general.b.value == report.general_strength
    assert report.unread == 1
    rated = list(taste_labels(engine.ratings.current()).liked)
    everything = [p.content_key for p in engine.catalog.photos_under([archive])]
    cached = engine.catalog.has_features("tiny_large_reader@1", everything)
    assert len(cached) == len(rated) - 1
    (blended,) = Sessions(engine).evaluate_taste(general_embedder="tiny_large_reader", blend=True)
    assert blended.general_blend


def test_general_read_by_another_embedder_is_measured_beside_todays_yours_alone(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate General does not depend on what Yours is learned on, so it is measured
    once, not once per embedding (each would count as a setting tried)."""
    with_reader(monkeypatch)
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    monkeypatch.setattr(Sessions, "_taste_candidates", lambda self: ["tiny", "tiny_large"])
    assert len(Sessions(engine).evaluate_taste()) == 2
    (report,) = Sessions(engine).evaluate_taste(general_embedder="tiny_large_reader")
    assert report.feature.startswith("tiny@")


def test_general_read_by_another_embedder_is_refused_where_it_cannot_be_measured(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_reader(monkeypatch)
    rate(engine, old_mill(archive))
    with pytest.raises(InvalidRequestError, match="already"):
        Sessions(engine).evaluate_taste(general_embedder=engine.settings.embedder)
    with pytest.raises(InvalidRequestError, match=r"tiny_large.*text"):
        Sessions(engine).evaluate_taste(general_embedder="tiny_large")
    with pytest.raises(InvalidRequestError, match="one candidate"):
        Sessions(engine).evaluate_taste(general_embedder="tiny_large_reader", aesthetic="bright")


CANDIDATE_PROMPTS = """
version = "4-candidate"
[labels.strong]
prompts = ["a beautifully composed photograph"]
[labels.weak]
prompts = ["a casual phone snapshot"]
[labels.document]
prompts = ["a reference photo of an object in a museum display case"]
"""


def test_candidate_wording_for_general_is_measured_beside_todays_on_the_rated_photos(
    engine: Engine, archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S5: try a wording for General's aesthetic without changing what any search uses."""
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    candidate = tmp_path / "phone.toml"
    candidate.write_text(CANDIDATE_PROMPTS)
    monkeypatch.setattr(engine, "embedder", f.FirstAxisReader(engine.embedder))
    (report,) = Sessions(engine).evaluate_taste(prompts=candidate)
    assert report.aesthetic_candidate.startswith("phone@")
    assert report.candidate_general is not None
    assert report.candidate_general.b.value == report.general_strength
    assert report.unread == 0  # read from the stored embeddings: no photo is opened
    candidate.write_text(CANDIDATE_PROMPTS.replace("phone snapshot", "phone photo"))
    (other,) = Sessions(engine).evaluate_taste(prompts=candidate)
    assert other.aesthetic_candidate != report.aesthetic_candidate  # each wording is a try


def test_yours_fitted_on_pairs_among_the_first_picks_learns_from_photos_a_taste_chose(
    engine: Engine, archive: Path
) -> None:
    trained_taste(engine, archive)
    with pytest.raises(InvalidRequestError, match="Compare Strongest"):
        Sessions(engine).evaluate_taste(fit="top_pairs")  # no first picks rated yet
    for folders, k in ((old_mill(archive), 4), ([archive / "Harbor - June 2022"], 2)):
        made = Sessions(engine).label_compare(folders, per_template=k)
        for n, item in enumerate(made.items):
            engine.ratings.rate_photo(made.set_id, item.content_key, 2 if n % 2 else 0)
    (report,) = Sessions(engine).evaluate_taste(fit="top_pairs")
    assert report.fit == "top_pairs"
    assert report.fitted is not None


def test_yours_fitted_by_scale_reads_each_rated_photo_s_scale(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    with pytest.raises(CapabilityError, match="clip"):
        Sessions(engine).evaluate_taste(fit="by_scale")  # the test embedder reads no text
    monkeypatch.setattr(engine, "embedder", f.FirstAxisReader(engine.embedder))
    (report,) = Sessions(engine).evaluate_taste(fit="by_scale")
    assert report.fit == "by_scale"
    assert report.fitted is not None
    labels = taste_labels(engine.ratings.current())
    photos = Sessions(engine)._taste_photos(engine.settings.embedder, labels, scaled=True)
    assert {p.scale for p in photos} <= {"aerial", "wide", "medium", "close", "macro"}
    assert all(p.scale for p in photos)


def test_candidate_wording_is_refused_where_it_cannot_be_measured(
    engine: Engine, archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rate(engine, old_mill(archive))
    candidate = tmp_path / "phone.toml"
    candidate.write_text(CANDIDATE_PROMPTS)
    with pytest.raises(CapabilityError, match="clip"):
        Sessions(engine).evaluate_taste(prompts=candidate)  # the test embedder reads no text
    monkeypatch.setattr(engine, "embedder", f.FirstAxisReader(engine.embedder))
    no_strong = tmp_path / "plain.toml"
    no_strong.write_text(CANDIDATE_PROMPTS.replace("labels.strong", "labels.fine"))
    with pytest.raises(InvalidRequestError, match="strong"):
        Sessions(engine).evaluate_taste(prompts=no_strong)
    with pytest.raises(InvalidRequestError, match="one candidate"):
        Sessions(engine).evaluate_taste(prompts=candidate, aesthetic="laion_aesthetic")


def test_a_candidate_aesthetic_must_be_one_number_per_photo(engine: Engine, archive: Path) -> None:
    rate(engine, old_mill(archive))
    with pytest.raises(InvalidRequestError, match=r"layout.*one number"):
        Sessions(engine).evaluate_taste(aesthetic="layout")


def test_the_candidate_must_differ_from_what_calico_uses(engine: Engine, archive: Path) -> None:
    rate(engine, old_mill(archive))
    with pytest.raises(InvalidRequestError, match="already"):
        Sessions(engine).evaluate_taste(technical="2")


def hold_out(engine: Engine, *places: str, references: tuple[str, ...] = ()) -> None:
    write_holdout(
        engine.settings.holdout_path,
        HoldOut(places=places, references=references, story_sessions=(), seed=1, drawn_at=T0),
    )


def test_held_out_places_are_left_out_of_taste_unless_asked_for_alone(
    engine: Engine, archive: Path
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    hold_out(engine, "Harbor")
    (report,) = Sessions(engine).evaluate_taste()
    assert [r.location for r in report.rows] + report.one_sided == ["Old Mill"]
    assert report.trained_on == 6  # the harbor's ratings are not learned from either
    (alone,) = Sessions(engine).evaluate_taste(only_holdout=True)
    assert [r.location for r in alone.rows] == ["Harbor"]
    assert alone.trained_on == 6


def test_the_saved_taste_never_learns_from_the_hold_out(engine: Engine, archive: Path) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    hold_out(engine, "Harbor")
    assert Sessions(engine).train_taste(force=True).saved
    taste = engine.trained.load_taste()
    assert taste is not None
    assert taste.places == ("Old Mill",)


def leaky_archive(engine: Engine, archive: Path) -> None:
    """A harbor photo saved again elsewhere, and a second name for the harbor, all rated."""
    exports = archive / "Exports - 2023"
    exports.mkdir()
    with PILImage.open(archive / "Harbor - June 2022" / "h10.jpg") as image:
        image.save(exports / "h10.png")  # saved again: another file, the same frame
    north = archive / "Harbor (north) - June 2022"
    f.save_jpeg(f.scene(13), north / "n13.jpg", captured_at=datetime(2022, 6, 2, 9))
    engine.indexer.index([archive])
    rate(engine, old_mill(archive))
    for folder in (archive / "Harbor - June 2022", north, exports):
        rate(engine, [folder])


def test_copies_and_grouped_names_of_a_judged_place_are_never_learned_from(
    engine: Engine, archive: Path
) -> None:
    leaky_archive(engine, archive)
    (report,) = Sessions(engine).evaluate_taste()
    assert report.same_places == (("Harbor", "Harbor (north)"),)
    # Judging the harbor: neither its re-export nor the photo filed under its other name.
    assert (report.copies_left_out, report.same_place_left_out) == (1, 1)


def test_the_saved_taste_never_learns_from_copies_of_the_hold_out(
    engine: Engine, archive: Path
) -> None:
    leaky_archive(engine, archive)
    hold_out(engine, "Harbor")
    assert Sessions(engine).train_taste(force=True).saved
    taste = engine.trained.load_taste()
    assert taste is not None
    assert taste.places == ("Old Mill",)


def filed_twice(engine: Engine, archive: Path) -> str:
    """The harbor's h10.jpg filed again in a mill folder: one file at two locations. Its
    content key."""
    shutil.copy(
        archive / "Harbor - June 2022" / "h10.jpg", archive / "Old Mill - May 2024" / "h10.jpg"
    )
    engine.indexer.index([archive])
    (photo,) = [
        p
        for p in engine.catalog.photos_under([archive / "Harbor - June 2022"])
        if p.path.name == "h10.jpg"
    ]
    return photo.content_key


def test_a_photo_filed_at_a_held_out_location_is_held_out_with_it(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The hold-out looked only at a photo's first path, so a file also filed under a held-out
    # folder was learned from (10 November 2026 audit, B-1: 133 photos of one trip).
    monkeypatch.setattr(holdout_module, "NEAR_FILES", 99)  # only the file is shared, nothing else
    shared = filed_twice(engine, archive)
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    hold_out(engine, "Old Mill")
    photos = Sessions(engine)._taste_photos(
        engine.settings.embedder, taste_labels(engine.ratings.current())
    )
    assert {p.content_key: p.location for p in photos}[shared] == "Old Mill"
    kept = {p.content_key for p in taste_apart(photos, ("Old Mill",)).photos}
    assert shared not in kept
    assert {p.location for p in photos if p.content_key in kept} == {"Harbor"}  # h11, h12 stay


def test_a_location_sharing_files_with_a_held_out_one_is_left_out_with_it(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(holdout_module, "NEAR_FILES", 1)  # one shared file is enough here
    filed_twice(engine, archive)
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    hold_out(engine, "Old Mill")
    photos = Sessions(engine)._taste_photos(
        engine.settings.embedder, taste_labels(engine.ratings.current())
    )
    left = taste_apart(photos, ("Old Mill",))
    assert left.photos == []  # the harbor shares the mill's file: the same trip, left out with it
    assert left.same_place  # and said so, as a place grouped with a held-out one


def test_a_compare_session_next_to_a_location_yours_learned_from_is_not_fresh(
    engine: Engine, archive: Path
) -> None:
    trained_taste(engine, archive)  # Old Mill and Harbor
    annex = archive / "Harbor annex - June 2022"
    annex.mkdir()
    for name in ("h10", "h11", "h12"):  # the harbor's frames, filed under another location
        shutil.copy(archive / "Harbor - June 2022" / f"{name}.jpg", annex / f"{name}.jpg")
    quarry = archive / "Quarry - July 2023"
    for seed in (20, 21, 22):
        f.save_jpeg(f.scene(seed), quarry / f"q{seed}.jpg", captured_at=T0 + timedelta(days=400))
    engine.indexer.index([archive])
    next_to = Sessions(engine).label_compare([annex], per_template=2)
    assert next_to.taste is not None
    assert next_to.taste.seen is True  # it had learned the harbor's own frames
    unrelated = Sessions(engine).label_compare([quarry], per_template=2)
    assert unrelated.taste is not None
    assert unrelated.taste.seen is False


def test_a_new_hold_out_can_leave_out_locations_and_the_ones_next_to_them(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(holdout_module, "HOLDOUT_SHARE", 1.0)  # two judged locations: draw both
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    drawn = Sessions(engine).draw_holdout(leave_out=("Harbor",))
    assert drawn.places == ("Old Mill",)  # the harbor was designed against: never drawn
    engine.settings.holdout_path.unlink()
    monkeypatch.setattr(holdout_module, "NEAR_FILES", 1)
    filed_twice(engine, archive)  # now the mill is next to the harbor
    assert Sessions(engine).draw_holdout(leave_out=("Harbor",)).places == ()


def test_a_left_out_name_the_catalog_does_not_have_is_refused(
    engine: Engine, archive: Path
) -> None:
    # A misspelt name would leave the place it meant in the draw, and a hold-out is drawn once.
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    with pytest.raises(HoldoutError, match="no location named 'Old Mil'") as refused:
        Sessions(engine).draw_holdout(leave_out=("Harbor", "Old Mil"))
    assert "did you mean 'Old Mill'" in str(refused.value)
    assert "'Harbor'" not in str(refused.value)  # a name the catalog has is not complained about
    assert Sessions(engine).holdout() is None  # and nothing was drawn


def test_a_hold_out_can_be_a_chosen_size_and_previewed_before_it_is_drawn(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(holdout_module, "HOLDOUT_SHARE", 1.0)  # two judged locations
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    previewed = Sessions(engine).draw_holdout(places=1, write=False)
    assert len(previewed.places) == 1  # one place, where the usual size would draw both
    assert Sessions(engine).holdout() is None  # a preview writes nothing
    drawn = Sessions(engine).draw_holdout(places=1)
    assert drawn.places == previewed.places  # and the draw is what it showed
    assert Sessions(engine).holdout() == drawn


def test_the_hold_out_alone_needs_a_draw_first(engine: Engine) -> None:
    with pytest.raises(HoldoutError, match="calico holdout draw"):
        Sessions(engine).evaluate_taste(only_holdout=True)


def test_the_hold_out_is_drawn_once_from_what_you_rated(engine: Engine, archive: Path) -> None:
    rate(engine, old_mill(archive))
    assert Sessions(engine).holdout() is None
    drawn = Sessions(engine).draw_holdout()
    assert Sessions(engine).holdout() == drawn
    assert drawn.places == ()  # one judged place: holding it out would leave nothing
    with pytest.raises(HoldoutError, match="already drawn"):
        Sessions(engine).draw_holdout()


def test_taste_is_not_saved_unless_shown_to_beat_general(engine: Engine, archive: Path) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    trained = Sessions(engine).train_taste()
    assert not trained.report.yours_wins  # two places cannot show it
    assert not trained.saved
    assert engine.trained.load_taste() is None


def test_training_calibrates_yours_on_what_it_forecast_at_places_it_never_learned(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real calibration needs hundreds of judged photos at five places or more; two places and
    # a handful of photos stand in with the thresholds lowered (the fit itself is tested apart).
    monkeypatch.setattr(calibration_module, "MIN_FORECASTS", 4)
    monkeypatch.setattr(calibration_module, "MIN_PLACES", 2)
    monkeypatch.setattr(calibration_module, "SURE", -1e9)
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    assert Sessions(engine).train_taste(force=True).calibration is None  # no first picks rated yet
    for place in (old_mill(archive)[0], archive / "Harbor - June 2022"):
        made = Sessions(engine).label_compare([place], per_template=3)
        for n, item in enumerate(made.items):
            engine.ratings.rate_photo(made.set_id, item.content_key, 2 if n % 2 else 0)
    trained = Sessions(engine).train_taste(force=True)
    assert trained.saved
    assert trained.calibration is not None
    first = [f for f in trained.report.forecasts if f.first_pick]
    assert trained.calibration.photos == len(first)  # read from what Strongest showed first
    assert trained.calibration.places == 2
    saved = engine.trained.load_taste()
    assert saved is not None
    assert saved.calibration == trained.calibration


def test_training_with_too_little_to_calibrate_saves_the_taste_without_one(
    engine: Engine, archive: Path
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    trained = Sessions(engine).train_taste(force=True)
    assert trained.saved
    assert trained.calibration is None  # two places and a few photos
    saved = engine.trained.load_taste()
    assert saved is not None
    assert saved.calibration is None


def test_comparing_strongest_needs_a_trained_yours(engine: Engine, archive: Path) -> None:
    with pytest.raises(InvalidRequestError, match="calico train taste"):
        Sessions(engine).label_compare([archive])
    assert engine.labels.all() == []


def trained_taste(engine: Engine, archive: Path) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    assert Sessions(engine).train_taste(force=True).saved


def test_compare_strongest_shows_each_templates_first_picks_once_without_saying_whose(
    engine: Engine, archive: Path
) -> None:
    trained_taste(engine, archive)
    made = Sessions(engine).label_compare(old_mill(archive), per_template=4)
    assert made.kind == "compare"
    assert made.text == "Compare Strongest: Old Mill - March 2021, Old Mill - May 2024"
    runs = {}
    for template in ("general", "yours"):
        (run_id,) = {s.run_id for i in made.items for s in i.shown_by if s.template == template}
        runs[template] = engine.runs.load(run_id)  # each template's picks are a saved run
        assert runs[template].request.template == template
        assert runs[template].request.limit == 4
    keys = [i.content_key for i in made.items]
    for template, run in runs.items():
        picks = [p.content_key for p in run.groups[0].picks]
        ranks = {
            i.content_key: s.rank for i in made.items for s in i.shown_by if s.template == template
        }
        assert ranks == {k: n for n, k in enumerate(picks)}
    assert sorted(keys) == sorted({p.content_key for r in runs.values() for p in r.groups[0].picks})
    again = Sessions(engine).label_compare(old_mill(archive), per_template=4)
    assert [i.content_key for i in again.items] == keys  # the same place, the same order


def test_compare_ratings_train_yours_but_do_not_judge_a_candidate_over_the_whole_range(
    engine: Engine, archive: Path
) -> None:
    """A Compare Strongest photo was rated because today's General or Yours showed it first:
    judging a candidate on it favours the tastes that chose it, as a photo drawn for its
    value would (owner's decision, 7 November 2026). It is still learned from and still
    counts among the first picks."""
    trained_taste(engine, archive)
    made = Sessions(engine).label_compare([archive / "Harbor - June 2022"], per_template=2)
    for item in made.items:
        engine.ratings.rate_photo(made.set_id, item.content_key, 2 if item.shown_by[0].rank else 0)
    labels = taste_labels(engine.ratings.current())
    photos = {
        p.content_key: p for p in Sessions(engine)._taste_photos(engine.settings.embedder, labels)
    }
    compared = {k for k in photos if labels.session.get(k) == made.set_id}  # newest answer
    assert compared
    for key in compared:
        assert not photos[key].whole_range
        assert photos[key].from_session  # still learned from
    assert any(p.whole_range for k, p in photos.items() if k not in compared)


def harbor_and_mill(archive: Path) -> list[Path]:
    return [
        *sorted((archive / "Harbor - June 2022").glob("*.jpg")),
        archive / "Old Mill - March 2021" / "view3.jpg",
    ]


def test_a_list_session_rates_the_photos_given_once_each_in_an_order_of_its_own(
    engine: Engine, archive: Path
) -> None:
    """Rate this list: the photos are given, not drawn (two lists' picks to judge blind,
    13 November 2026). The session shows each once, shuffled the same way whatever order
    the list came in, and keeps nothing about where a photo came from."""
    engine.indexer.index([archive])
    given = harbor_and_mill(archive)
    made = Sessions(engine).label_list([*given, given[0]], "picks to judge")
    assert made.kind == "taste"
    assert made.text == "Your taste: picks to judge"
    assert sorted(i.path for i in made.items) == sorted(given)
    assert {i.stratum for i in made.items} == {evaluation.GIVEN}
    assert all(not i.shown_by and i.inclusion is None and not i.control for i in made.items)
    assert set(made.folders) == {p.parent for p in given}
    again = Sessions(engine).label_list(given[::-1], "picks to judge")
    assert [i.content_key for i in again.items] == [i.content_key for i in made.items]
    assert engine.labels.load(made.set_id).items == made.items
    lists = [[i.path for i in Sessions(engine).label_list(given, name).items] for name in "abcdef"]
    assert any(order != sorted(order) for order in lists)  # not the order of the file


def test_a_list_session_refuses_a_photo_the_catalog_does_not_hold(
    engine: Engine, archive: Path
) -> None:
    engine.indexer.index([archive])
    given = harbor_and_mill(archive)
    with pytest.raises(InvalidRequestError, match=r"nowhere\.jpg.*calico index"):
        Sessions(engine).label_list([*given, archive / "Harbor - June 2022" / "nowhere.jpg"], "x")
    with pytest.raises(InvalidRequestError, match="no photos"):
        Sessions(engine).label_list([], "x")
    assert not engine.labels.all()


def test_list_ratings_train_yours_but_do_not_judge_a_candidate_over_the_whole_range(
    engine: Engine, archive: Path
) -> None:
    """A photo in a list was put there by some ranking (a champion's picks, a candidate's),
    so it is read as a Compare Strongest photo is: learned from, and left out of the
    whole-range measures."""
    trained_taste(engine, archive)
    made = Sessions(engine).label_list(harbor_and_mill(archive), "picks to judge")
    for n, item in enumerate(made.items):
        engine.ratings.rate_photo(made.set_id, item.content_key, 2 if n % 2 else 0)
    labels = taste_labels(engine.ratings.current())
    photos = {
        p.content_key: p for p in Sessions(engine)._taste_photos(engine.settings.embedder, labels)
    }
    listed = {k for k in photos if labels.session.get(k) == made.set_id}
    assert len(listed) == len(made.items)
    for key in listed:
        assert not photos[key].whole_range
        assert photos[key].from_session
    assert any(p.whole_range for k, p in photos.items() if k not in listed)


def test_a_photo_rated_in_two_compare_sessions_counts_in_each_by_what_the_taste_had_learned(
    engine: Engine, archive: Path
) -> None:
    # "A photo rated in two sessions counts once, in the newer" moved photos from a session the
    # taste had not learned (out of sample) to a later one it had (10 November 2026 audit, B-6:
    # 33 of one place's 57 photos).
    trained_taste(engine, archive)
    quarry = archive / "Quarry - July 2023"
    for seed in (20, 21, 22):
        f.save_jpeg(f.scene(seed), quarry / f"q{seed}.jpg", captured_at=T0 + timedelta(days=400))
    engine.indexer.index([archive])
    first = Sessions(engine).label_compare([quarry], per_template=2)
    assert first.taste is not None
    assert first.taste.seen is False  # Yours had not learned the quarry
    for n, item in enumerate(first.items):
        engine.ratings.rate_photo(first.set_id, item.content_key, 2 if n % 2 else 0)
    assert Sessions(engine).train_taste(force=True).saved  # it learns the quarry from those ratings
    second = Sessions(engine).label_compare([quarry], per_template=2)
    assert second.taste is not None
    assert second.taste.seen is True
    assert {i.content_key for i in first.items} & {i.content_key for i in second.items}
    for n, item in enumerate(second.items):
        engine.ratings.rate_photo(second.set_id, item.content_key, 2 if n % 2 else 0)

    report = Sessions(engine).evaluate_compare()
    (fresh,) = report.rows  # the first session keeps every photo it showed
    assert fresh.rated == len(first.items)
    assert report.in_sample is not None
    (learned,) = report.in_sample.rows
    assert learned.rated == len(second.items)


def test_compare_strongest_asks_control_photos_blind_and_keeps_their_answers_apart(
    engine: Engine, archive: Path
) -> None:
    trained_taste(engine, archive)
    harbor = Sessions(engine).label_compare([archive / "Harbor - June 2022"], per_template=2)
    assert not any(i.control for i in harbor.items)  # no pool drawn yet: nothing is asked again
    with pytest.raises(InvalidRequestError, match="Compare Strongest"):
        Sessions(engine).draw_controls()  # nothing rated in one yet
    for n, item in enumerate(harbor.items):
        engine.ratings.rate_photo(harbor.set_id, item.content_key, 2 if n % 2 else 0)
    pool = Sessions(engine).draw_controls(size=4)
    assert {c.key for c in pool.photos} <= {i.content_key for i in harbor.items}
    assert {c.liked for c in pool.photos} == {True, False}  # both directions
    assert read_controls(engine.settings.controls_path) == pool
    with pytest.raises(ControlsError, match="already drawn"):
        Sessions(engine).draw_controls()

    made = Sessions(engine).label_compare(old_mill(archive), per_template=3)
    controls = [i for i in made.items if i.control]
    picks = [i for i in made.items if not i.control]
    assert controls
    assert len(controls) <= CONTROLS_PER_SESSION
    assert {i.content_key for i in controls} <= {c.key for c in pool.photos}
    assert all(i.shown_by == () for i in controls)  # shown by neither taste
    assert not {i.content_key for i in controls} & {i.content_key for i in picks}

    baseline = {c.key: c.liked for c in pool.photos}
    flipped = controls[0].content_key
    before = taste_labels(engine.ratings.current())
    engine.ratings.rate_photo(made.set_id, flipped, 0 if baseline[flipped] else 2)
    assert taste_labels(engine.ratings.current()) == before  # asked again, never a new label
    for item in picks:
        engine.ratings.rate_photo(made.set_id, item.content_key, 2)
    report = Sessions(engine).evaluate_compare()
    rows = [*report.rows, *(report.in_sample.rows if report.in_sample else [])]
    assert sorted(r.rated for r in rows) == sorted([len(harbor.items), len(picks)])  # no controls
    retest = Sessions(engine).retest()
    assert (retest.asked, retest.against_baseline.kept) == (1, 0)  # the one flip


def test_a_control_pool_is_needed_to_read_a_retest(engine: Engine) -> None:
    with pytest.raises(InvalidRequestError, match="calico controls draw"):
        Sessions(engine).retest()


def compared_and_rated(engine: Engine, folders: list[Path]) -> set[str]:
    """Rate a Compare Strongest session over ``folders`` (alternately liked and not), and say
    the content keys it showed."""
    made = Sessions(engine).label_compare(folders, per_template=3)
    for n, item in enumerate(made.items):
        engine.ratings.rate_photo(made.set_id, item.content_key, 2 if n % 2 else 0)
    return {i.content_key for i in made.items}


def test_the_control_pool_leaves_out_what_the_hold_out_reaches(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A control photo is asked again in later sessions, so none may come from the hold-out:
    # not a held-out location, and not a frame also filed under one (10 November 2026 audit, B-1).
    monkeypatch.setattr(holdout_module, "NEAR_FILES", 99)  # only the file is shared
    monkeypatch.setattr(controls_module, "CONTROL_PER_LOCATION", 99)  # every candidate fits
    trained_taste(engine, archive)
    shared = filed_twice(engine, archive)  # the harbor's h10 is also filed at Old Mill
    harbor = compared_and_rated(engine, [archive / "Harbor - June 2022"])
    mill = compared_and_rated(engine, old_mill(archive))
    hold_out(engine, "Old Mill")
    pool = {c.key for c in Sessions(engine).draw_controls(size=40).photos}
    assert pool == harbor - {shared}  # the harbor's other frames; nothing from the mill
    assert shared in harbor | mill


def test_a_control_pool_is_refused_when_the_hold_out_reaches_every_photo_to_draw_from(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pool is drawn once and never replaced: an empty one would be there for good.
    monkeypatch.setattr(holdout_module, "NEAR_FILES", 1)  # the harbor is next to the mill
    trained_taste(engine, archive)
    filed_twice(engine, archive)
    compared_and_rated(engine, [archive / "Harbor - June 2022"])
    compared_and_rated(engine, old_mill(archive))
    hold_out(engine, "Old Mill")
    with pytest.raises(InvalidRequestError, match="outside the hold-out"):
        Sessions(engine).draw_controls()
    assert read_controls(engine.settings.controls_path) is None


def test_compare_reads_the_marker_yours_drew_in_each_session(engine: Engine, archive: Path) -> None:
    trained_taste(engine, archive)
    made = Sessions(engine).label_compare([archive / "Harbor - June 2022"], per_template=3)
    run_id = next(s.run_id for i in made.items for s in i.shown_by if s.template == "yours")
    result = engine.runs.load(run_id)  # a calibrated taste draws a marker; give this run one
    group = result.groups[0].model_copy(update={"marker": Marker(before=2, probability=0.8)})
    saved = result.model_copy(update={"groups": (group,)})
    (engine.runs.path(run_id) / "result.json").write_text(saved.model_dump_json(indent=2))
    for n, item in enumerate(made.items):
        engine.ratings.rate_photo(made.set_id, item.content_key, 2 if n % 2 else 0)
    report = Sessions(engine).evaluate_compare()
    seen = report.in_sample  # the taste had learned from the harbor
    assert seen is not None
    assert seen.marker is not None
    assert seen.marker.at == (2,)
    assert seen.marker.above.rated + seen.marker.below.rated == seen.yours.rated


def with_a_quarry(engine: Engine, archive: Path) -> Path:
    """A third place (three photos, none liked by ``rate``) for the taste to learn from."""
    quarry = archive / "Quarry - July 2023"
    for seed in (20, 21, 22):
        f.save_jpeg(f.scene(seed), quarry / f"q{seed}.jpg", captured_at=T0 + timedelta(days=400))
    engine.indexer.index([archive])
    return quarry


def test_a_search_runs_yours_with_a_taste_it_is_given_and_does_not_save_it(
    engine: Engine, archive: Path
) -> None:
    trained_taste(engine, archive)
    saved = engine.trained.load_taste()
    assert saved is not None
    given = dataclasses.replace(saved, bias=saved.bias + 1.0, places=("Old Mill",))
    request = SelectionRequest(
        task="best", folders=(archive / "Harbor - June 2022",), template="yours", limit=3
    )
    result = engine.select(request, taste=given)
    assert result.request.template == "yours"
    assert result.versions["model:aesthetic"] == f"taste@{given.version}"
    assert any("fitted for this search" in note for note in result.notes)
    kept = engine.trained.load_taste()
    assert kept is not None
    assert kept.version == saved.version  # the saved taste is as it was
    assert engine.select(request).versions["model:aesthetic"] == f"taste@{saved.version}"


def test_a_taste_given_to_a_search_that_runs_general_is_ignored_and_said_so(
    engine: Engine, archive: Path
) -> None:
    trained_taste(engine, archive)
    saved = engine.trained.load_taste()
    assert saved is not None
    request = SelectionRequest(
        task="best", folders=(archive / "Harbor - June 2022",), template="general", limit=3
    )
    result = engine.select(request, taste=saved)
    assert result.request.template == "general"
    assert any("does not read one" in note for note in result.notes)
    assert "taste@" not in result.versions.get("model:aesthetic", "")


def test_a_cross_fitted_taste_never_learned_the_place_it_is_fitted_without(
    engine: Engine, archive: Path
) -> None:
    trained_taste(engine, archive)
    saved = engine.trained.load_taste()
    assert saved is not None
    assert set(saved.places) == {"Old Mill", "Harbor"}
    weights = Sessions(engine).cross_fitted_taste([archive / "Harbor - June 2022"])
    assert weights.places == ("Old Mill",)
    assert weights.rated < saved.rated
    assert weights.version != saved.version
    assert weights.feature == saved.feature
    again = Sessions(engine).cross_fitted_taste([archive / "Harbor - June 2022"])
    assert again.version == weights.version  # deterministic
    kept = engine.trained.load_taste()
    assert kept is not None
    assert kept.version == saved.version  # never saved over Yours


def test_a_cross_fitted_taste_leaves_out_the_hold_out_as_training_does(
    engine: Engine, archive: Path
) -> None:
    quarry = with_a_quarry(engine, archive)
    trained_taste(engine, archive)
    rate(engine, [quarry])
    hold_out(engine, "Old Mill")
    weights = Sessions(engine).cross_fitted_taste([quarry])
    assert weights.places == (
        "Harbor",
    )  # not the quarry it is fitted without, nor the held-out mill


def test_a_cross_fitted_taste_leaves_out_the_places_next_to_the_one_asked_for(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The mill and the harbor share a file, so a trip filed under both is one place to a taste.
    monkeypatch.setattr(holdout_module, "NEAR_FILES", 1)
    trained_taste(engine, archive)
    filed_twice(engine, archive)
    with pytest.raises(InvalidRequestError, match="Rate more places"):
        Sessions(engine).cross_fitted_taste([archive / "Harbor - June 2022"])


def test_a_cross_fitted_taste_is_refused_when_nothing_else_has_both_likes_and_dislikes(
    engine: Engine, archive: Path
) -> None:
    rate(engine, [archive / "Harbor - June 2022"])
    with pytest.raises(InvalidRequestError, match="Rate more places"):
        Sessions(engine).cross_fitted_taste([archive / "Harbor - June 2022"])


def test_a_cross_fitted_compare_session_shows_yours_fitted_without_the_place(
    engine: Engine, archive: Path
) -> None:
    trained_taste(engine, archive)
    saved = engine.trained.load_taste()
    assert saved is not None
    harbor = archive / "Harbor - June 2022"
    plain = Sessions(engine).label_compare([harbor], per_template=2)
    assert plain.taste is not None
    assert (plain.taste.seen, plain.taste.cross_fit) == (True, False)  # the saved Yours learned it

    made = Sessions(engine).label_compare([harbor], per_template=2, cross_fit=True)
    weights = Sessions(engine).cross_fitted_taste([harbor])
    assert made.taste is not None
    assert (made.taste.seen, made.taste.cross_fit) == (False, True)
    assert made.taste.version == weights.version
    assert weights.version != saved.version
    run_id = next(s.run_id for i in made.items for s in i.shown_by if s.template == "yours")
    assert engine.runs.load(run_id).versions["model:aesthetic"] == f"taste@{weights.version}"
    kept = engine.trained.load_taste()
    assert kept is not None
    assert kept.version == saved.version


def test_a_cross_fitted_session_needs_ratings_not_a_saved_taste(
    engine: Engine, archive: Path
) -> None:
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    assert engine.trained.load_taste() is None
    with pytest.raises(InvalidRequestError, match="calico train taste"):
        Sessions(engine).label_compare([archive / "Harbor - June 2022"], per_template=2)
    made = Sessions(engine).label_compare(
        [archive / "Harbor - June 2022"], per_template=2, cross_fit=True
    )
    assert made.taste is not None
    assert made.taste.cross_fit
    assert engine.trained.load_taste() is None  # still nothing saved


def test_compare_reports_cross_fitted_sessions_apart_from_fresh_and_in_sample_ones(
    engine: Engine, archive: Path
) -> None:
    trained_taste(engine, archive)
    harbor = archive / "Harbor - June 2022"
    made = Sessions(engine).label_compare([harbor], per_template=2, cross_fit=True)
    for n, item in enumerate(made.items):
        engine.ratings.rate_photo(made.set_id, item.content_key, 2 if n % 2 else 0)
    report = Sessions(engine).evaluate_compare()
    assert report.rows == []  # not a place the saved Yours had not learned
    assert report.in_sample is None  # and not judged on what it learned
    assert report.cross_fitted is not None
    assert [r.place for r in report.cross_fitted.rows] == ["Harbor"]
    assert report.cross_fitted.rows[0].rated == len(made.items)


def test_cross_fitted_sessions_can_be_set_apart_by_size_like_fresh_ones(
    engine: Engine, archive: Path
) -> None:
    trained_taste(engine, archive)
    harbor = archive / "Harbor - June 2022"
    made = Sessions(engine).label_compare([harbor], per_template=2, cross_fit=True)
    for n, item in enumerate(made.items):
        engine.ratings.rate_photo(made.set_id, item.content_key, 2 if n % 2 else 0)
    held = Sessions(engine).evaluate_compare().cross_fitted
    assert held is not None
    (row,) = held.rows
    assert row.bursts is not None
    apart = Sessions(engine).evaluate_compare(min_bursts=row.bursts + 1).cross_fitted
    assert apart is not None
    assert apart.rows == []
    assert apart.small is not None
    assert [r.place for r in apart.small.rows] == ["Harbor"]


def test_compare_reads_how_many_bursts_each_sessions_pool_held_and_can_set_small_ones_apart(
    engine: Engine, archive: Path
) -> None:
    # 12 November 2026: four of the 18 places read for the bar had fewer than the 100 bursts the
    # roadmap asks of one, and nothing in the report said so.
    trained_taste(engine, archive)
    quarry = archive / "Quarry - July 2023"
    for seed in (20, 21, 22):
        f.save_jpeg(f.scene(seed), quarry / f"q{seed}.jpg", captured_at=T0 + timedelta(days=400))
    engine.indexer.index([archive])
    made = Sessions(engine).label_compare([quarry], per_template=2)
    run_id = next(s.run_id for i in made.items for s in i.shown_by if s.template == "yours")
    held = next(s.count_out for s in engine.runs.load(run_id).stages if s.name == "collapse bursts")
    for n, item in enumerate(made.items):
        engine.ratings.rate_photo(made.set_id, item.content_key, 2 if n % 2 else 0)

    report = Sessions(engine).evaluate_compare()
    assert [r.bursts for r in report.rows] == [held]
    assert report.small is None
    enough = Sessions(engine).evaluate_compare(min_bursts=held)
    assert [r.place for r in enough.rows] == ["Quarry"]
    few = Sessions(engine).evaluate_compare(min_bursts=held + 1)
    assert few.rows == []
    assert few.small is not None
    assert [(r.place, r.bursts) for r in few.small.rows] == [("Quarry", held)]


def test_what_a_taste_chose_to_show_is_read_apart_from_the_whole_range_in_the_report(
    engine: Engine, archive: Path
) -> None:
    trained_taste(engine, archive)
    (before,) = Sessions(engine).evaluate_taste()
    assert before.chosen is None  # no photo was rated that a taste chose
    compared_and_rated(engine, [archive / "Harbor - June 2022"])
    compared_and_rated(engine, old_mill(archive))
    (report,) = Sessions(engine).evaluate_taste()
    assert report.chosen is not None
    assert report.chosen.a.rated == sum(r.for_value for r in report.rows)
    assert report.chosen.a.places >= 1


def test_compare_reads_each_templates_list_in_tens_by_the_place_each_photo_had_in_it(
    engine: Engine, archive: Path
) -> None:
    trained_taste(engine, archive)
    made = Sessions(engine).label_compare(old_mill(archive), per_template=4)
    for item in made.items:
        engine.ratings.rate_photo(made.set_id, item.content_key, 2)
    shown = {
        t: [s.rank for i in made.items for s in i.shown_by if s.template == t]
        for t in ("general", "yours")
    }
    report = Sessions(engine).evaluate_compare()
    both = [report, report.in_sample] if report.in_sample else [report]
    curves = [r.curve for r in both if r is not None and r.curve]
    assert curves  # each session's photos carry where its templates had them
    for template in ("general", "yours"):
        assert sum(sum(f.rated for f in c[template]) for c in curves) == len(shown[template])


def test_compare_ratings_train_your_taste_and_say_whose_picks_you_liked(
    engine: Engine, archive: Path
) -> None:
    trained_taste(engine, archive)
    (before,) = Sessions(engine).evaluate_taste()
    made = Sessions(engine).label_compare([archive / "Harbor - June 2022"], per_template=2)
    for item in made.items:
        engine.ratings.rate_photo(made.set_id, item.content_key, 2 if item.shown_by[0].rank else 0)
    (after,) = Sessions(engine).evaluate_taste()
    taste_keys = {i.content_key for s in engine.labels.all() if s.kind == "taste" for i in s.items}
    new = {i.content_key for i in made.items} - taste_keys
    assert after.trained_on == before.trained_on + len(new)

    report = Sessions(engine).evaluate_compare()
    assert report.rows == []  # the taste had learned from Harbor: reported apart
    assert report.in_sample is not None
    (row,) = report.in_sample.rows
    assert row.place == "Harbor"
    assert row.rated == len(made.items)
    liked = {i.content_key for i in made.items if i.shown_by[0].rank}
    shown = {
        t: {i.content_key for i in made.items if t in {s.template for s in i.shown_by}}
        for t in ("general", "yours")
    }
    assert row.general == FirstPicks(
        rated=len(shown["general"]), liked=len(shown["general"] & liked)
    )
    assert row.yours == FirstPicks(rated=len(shown["yours"]), liked=len(shown["yours"] & liked))


def test_a_taste_records_the_places_it_learned_from(engine: Engine, archive: Path) -> None:
    trained_taste(engine, archive)
    weights = engine.trained.load_taste()
    assert weights is not None
    assert weights.places == ("Harbor", "Old Mill")


def test_a_compare_session_records_whether_its_taste_had_learned_from_the_place(
    engine: Engine, archive: Path
) -> None:
    rate(engine, old_mill(archive))
    assert Sessions(engine).train_taste(force=True).saved  # learned from Old Mill only
    weights = engine.trained.load_taste()
    assert weights is not None
    fresh = Sessions(engine).label_compare([archive / "Harbor - June 2022"], per_template=2)
    seen = Sessions(engine).label_compare(old_mill(archive), per_template=2)
    assert fresh.taste is not None
    assert seen.taste is not None
    assert (fresh.taste.version, fresh.taste.seen) == (weights.version, False)
    assert seen.taste.seen is True
    for made in (fresh, seen):
        for item in made.items:
            engine.ratings.rate_photo(made.set_id, item.content_key, 2)
    assert {a.taste for a in engine.journal.answers() if a.use == "compare"} == {weights.version}
    report = Sessions(engine).evaluate_compare()
    assert [r.place for r in report.rows] == ["Harbor"]
    assert report.in_sample is not None
    assert [r.place for r in report.in_sample.rows] == ["Old Mill"]


# -- recurring views -------------------------------------------------------------


def test_recurring_views_finds_a_view_shot_on_two_visits(engine: Engine, archive: Path) -> None:
    progress: list[tuple[int, int, str]] = []
    result = engine.select(
        request(archive, "recurring_views"), progress=lambda *a: progress.append(a)
    )
    (mill,) = [g for g in result.groups if "revisit1.jpg" in [p.path.name for p in g.picks]]
    assert mill.title.startswith("Old Mill")
    assert mill.title.endswith(": 2 occasions, Mar 2021 to May 2024")
    first, later = mill.picks
    assert later.path.name == "revisit1.jpg"
    assert {first.path.name, *(a.name for a in first.alternates)} >= {"burst1.jpg", "burst2.jpg"}
    assert any(step == "comparing" for *_, step in progress)
    assert not any("Harbor" in g.title for g in result.groups)  # one visit only


def test_pairs_compared_once_are_remembered_across_engines(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = engine.select(request(archive, "recurring_views"))
    assert first.groups
    extractor = engine.patch_extractor(Trace())
    assert extractor is not None
    calls: list[int] = []
    original = type(extractor).extract_patches

    def counted(self: Any, images: Any) -> Any:
        calls.append(len(images))
        return original(self, images)

    monkeypatch.setattr(type(extractor), "extract_patches", counted)
    with Engine.open(engine.settings) as again:
        second = again.select(request(archive, "recurring_views"))
    assert calls == []
    assert [g.title for g in second.groups] == [g.title for g in first.groups]


def test_recurring_views_says_so_when_photos_cannot_be_compared_part_by_part(
    engine: Engine, archive: Path
) -> None:
    settings = engine.settings.model_copy(update={"view_embedder": "palette"})
    with Engine.open(settings) as whole:
        result = whole.select(request(archive, "recurring_views"))
    assert result.groups == ()
    assert any("part by part" in n for n in result.notes)


def test_comparing_pairs_uses_scratch_space_named_by_the_process_and_removes_it(
    engine: Engine, archive: Path
) -> None:
    seen: list[list[str]] = []

    def progress(_done: int, _total: int, step: str) -> None:
        if step == "comparing":
            seen.append([p.name for p in scratch_folders(engine.settings.home)])

    engine.select(request(archive, "recurring_views"), progress=progress)
    assert any(names and names[0].startswith(f"patches-{os.getpid()}-") for names in seen)
    assert scratch_folders(engine.settings.home) == []


def test_opening_the_engine_sweeps_scratch_space_a_stopped_calico_left(
    engine: Engine,
) -> None:
    gone = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        check=True,
        capture_output=True,
        text=True,
    )
    stale = engine.settings.home / f"patches-{gone.stdout.strip()}-abc"
    in_use = engine.settings.home / f"patches-{os.getpid()}-def"
    for folder in (stale, in_use):
        folder.mkdir()
        (folder / "patches.npy").write_bytes(b"x")
    with Engine.open(engine.settings):
        pass
    assert not stale.exists()
    assert in_use.is_dir()


class Probe(SameViewTask):
    """Asks for patch matches of the March photos only, and keeps what it was given."""

    name = "probe"
    seen: ClassVar[list[TaskContext]] = []

    def patch_shortlist(self, ctx: TaskContext, options: SameViewOptions) -> NDArray[np.intp]:
        march = [i for i, p in enumerate(ctx.photos) if "March" in p.path.parent.name]
        return np.asarray(march, dtype=np.intp)

    def run(self, ctx: TaskContext, options: SameViewOptions) -> list[Group]:
        Probe.seen.append(ctx)
        return []


def test_a_task_gets_patch_matches_for_the_shortlist_it_asks_for(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine, "task_registry", Registry.from_mapping("task", {"probe": Probe}))
    reference = archive / "Old Mill - March 2021" / "view1.jpg"
    result = engine.select(request(archive, "probe", references=[reference]))
    ctx = Probe.seen[-1]
    assert ctx.patch_scores is not None
    march = np.array(["March" in p.path.parent.name for p in ctx.photos])
    assert np.isfinite(ctx.patch_scores[march]).all()
    assert np.isnan(ctx.patch_scores[~march]).all()
    itself = next(i for i, p in enumerate(ctx.photos) if p.path.name == "view1.jpg")
    assert ctx.patch_scores[itself] == 1.0
    assert any(s.name == "patch match" for s in result.stages)


def counting_patches(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """How many images the view embedder is asked to cut into patches, call by call."""
    extractor = engine.extractors[engine.settings.view_embedder]
    assert isinstance(extractor, PatchExtractor)
    real = extractor.extract_patches
    sizes: list[int] = []

    def extract_patches(images: Sequence[Image.Image]) -> NDArray[np.float16]:
        sizes.append(len(images))
        return real(images)

    monkeypatch.setattr(extractor, "extract_patches", extract_patches)
    return sizes


def test_labelling_right_after_a_search_reuses_its_patch_matches(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes = counting_patches(engine, monkeypatch)
    reference = archive / "Old Mill - March 2021" / "view1.jpg"
    engine.select(request(archive, "same_view", references=[reference]))
    compared = sum(sizes)
    assert compared > 1
    Sessions(engine).label_same_view(reference, [archive])
    engine.select(request(archive, "same_view", references=[reference]))
    assert sum(sizes) == compared
    other = archive / "Old Mill - March 2021" / "view2.jpg"
    engine.select(request(archive, "same_view", references=[other]))
    assert sum(sizes) > compared


def test_progress_counts_the_photos_compared_part_by_part(engine: Engine, archive: Path) -> None:
    seen: list[tuple[int, int, str]] = []
    reference = archive / "Old Mill - March 2021" / "view1.jpg"
    engine.select(
        request(archive, "same_view", references=[reference]),
        progress=lambda done, total, step: seen.append((done, total, step)),
    )
    steps = [step for _, _, step in seen]
    assert steps.index("comparing") > steps.index("reading")
    comparing = [(done, total) for done, total, step in seen if step == "comparing"]
    assert comparing[0][0] == 0
    assert comparing[-1][0] == comparing[-1][1] > 0


def test_labelled_photos_missing_a_new_feature_get_it_rather_than_being_dropped(
    engine: Engine, archive: Path, calico_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = label_old_mill_view(engine, archive / "Old Mill - March 2021" / "view1.jpg", archive)
    real = extractor_registry()
    monkeypatch.setattr(
        "calico.features.extractor_registry",
        lambda: Registry.from_mapping(
            "extractor", {**{n: real.get(n) for n in real.names()}, "mirror": Mirror}
        ),
    )
    settings = engine.settings.model_copy(update={"view_embedder": "mirror"})
    with Engine.open(settings) as later:
        (row,) = Sessions(later).evaluate_same_view().rows
    assert row.labelled == len(made.items)


def test_a_reference_photo_a_task_ignores_is_not_read(engine: Engine, archive: Path) -> None:
    """Audit A01-6: the note said the reference was ignored, yet an unreadable one failed
    the run."""
    broken = archive / "Harbor - June 2022" / "broken.jpg"
    broken.write_bytes(b"not a photo")
    result = engine.select(request(archive, "best", references=[broken]))
    assert result.groups
    assert any("does not use reference photos" in n for n in result.notes)


def test_when_the_examples_are_the_whole_pool_the_note_says_so(
    engine: Engine, archive: Path
) -> None:
    """Audit A01-14: an empty pool is not always unreadable photos."""
    harbor = archive / "Harbor - June 2022"
    examples = sorted(harbor.glob("*.jpg"))
    result = engine.select(request(archive, "best", folders=[harbor], examples=examples))
    assert not result.groups
    assert not any("No readable photos" in n for n in result.notes)
    assert any("All the photos in these folders are your examples" in n for n in result.notes)


def test_taste_says_when_general_could_not_run(engine: Engine, archive: Path) -> None:
    """Audit A01-7: with no text to read (tiny), General is a constant 0.5, and Yours was
    measured against it without a word."""
    rate(engine, old_mill(archive))
    rate(engine, [archive / "Harbor - June 2022"])
    (report,) = Sessions(engine).evaluate_taste()
    assert any("General's aesthetic could not run" in n for n in report.notes)


def test_anchor_sessions_whose_place_has_no_photos_now_are_left_out_and_named(
    engine: Engine, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audit A01-4: one session whose folder was emptied or unmounted stopped the report with
    an IndexError."""
    made = Sessions(engine).label_anchors([archive])
    for i, item in enumerate(made.items):
        engine.ratings.rate_photo(made.set_id, item.content_key, 2 if i % 2 else 0)
    monkeypatch.setattr(engine, "gather", lambda folders, trace: [])
    report = Sessions(engine).evaluate_anchors()
    assert (report.could, report.could_not) == (0, 0)
    assert report.skipped == [made.text]
