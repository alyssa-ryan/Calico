"""The improvement loop's harness, run end to end on a made-up archive.

The owner's liking is planted: a photo is liked when it lies along one direction of the
embedding. General cannot know that direction and a Yours fitted on the other places can,
so the harness must show Yours ahead, and a config against a copy of itself at zero.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from calico.engine import Engine
from calico.features import extractor_registry
from calico.features.tiny import TinyEmbedder
from calico.plugins import Registry
from calico.settings import CollectionConfig, Settings
from tests import factories as f

TOOLS = Path(__file__).resolve().parents[1] / "claude" / "improvement-loop" / "tools"
sys.path.insert(0, str(TOOLS))

import catalogue  # noqa: E402
import disagreement  # noqa: E402
import harness  # noqa: E402
import summarize  # noqa: E402
import verdict  # noqa: E402

TRACK = "strongest-yours"
PLACES = {
    "Mill": "validation",
    "Harbor": "validation",
    "Quarry": "validation",
    "Depot": "dev",
    "Mill Museum": "validation",
    "Depot museum": "dev",
}
LOW_PRIORITY = ("Museum",)
EACH = 30
GRID = harness.Grid(
    outputs=(10,), windows=(), group_min=10**9, min_labels=4, folds=3, draws=2, whole=False
)
PLAIN = {"diversity": 0.0, "one_per_view": False}
TASTE_ALONE = {**PLAIN, "quality_weight": 0.0}
GENERAL = harness.Config(
    track=TRACK, name="general", task="best", template="general", options=PLAIN
)
YOURS = harness.Config(
    track=TRACK,
    name="yours",
    task="best",
    template="yours",
    options=TASTE_ALONE,
    taste={"l2": 0.01},
)


class Reversed(TinyEmbedder):
    """A second embedding for the made-up archive: tiny's vectors with their columns
    reversed, so every similarity is tiny's but the vectors are not."""

    name = "dinov2"

    def extract(self, images: Any) -> Any:
        return np.ascontiguousarray(super().extract(images)[:, ::-1])


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> harness.World:
    root = tmp_path_factory.mktemp("loop")
    patch = pytest.MonkeyPatch()
    patch.setenv("LOOP_SNAPSHOT", "made-up")
    patch.setenv("LOOP_CACHE", str(root / "cache"))
    real = extractor_registry()
    plugins = {name: real.get(name) for name in real.names()} | {"dinov2": Reversed}
    patch.setattr(
        "calico.features.extractor_registry", lambda: Registry.from_mapping("extractor", plugins)
    )
    archive = root / "archive"
    for n, place in enumerate(PLACES):
        for i in range(EACH):
            f.save_jpeg(
                f.scene(1000 * n + i),
                archive / f"{place} - 2024" / f"{i:02d}.jpg",
                captured_at=datetime(2024, 1 + n, 1, 8) + timedelta(minutes=20 * i),
                camera=f.SANDBOX_CAMERA,
            )
    settings = Settings(
        home=root / "home",
        embedder="tiny",
        describe_embedder="tiny",
        view_embedder="dinov2",
        collections=(
            CollectionConfig(
                name="archive", root=archive, location_pattern=r"^(?P<location>.+?) - "
            ),
        ),
    )
    eng = Engine.open(settings)
    eng.indexer.index([archive])
    cat = catalogue.build(eng)
    splits = root / "splits"
    write_splits(splits, cat)
    made = harness.World(eng, cat, splits, catalogue.cache_dir(), GRID, LOW_PRIORITY)
    patch.undo()
    return made


def write_splits(splits: Path, cat: catalogue.Catalogue) -> None:
    direction = np.random.default_rng(7).standard_normal(cat.clip.shape[1])
    along = cat.clip @ direction
    ratings: dict[str, list[dict[str, Any]]] = {t: [] for t in harness.TIERS}
    middle = float(
        np.median(along[~np.char.endswith(np.char.lower(cat.location.astype(str)), "museum")])
    )
    for place, tier in PLACES.items():
        here = np.flatnonzero(cat.location == place)
        ratings[tier] += [
            {"key": str(cat.keys[i]), "liked": bool(along[i] > middle), "place": place}
            for i in here
        ]
    splits.mkdir()
    (splits / "places.json").write_text(
        json.dumps({"places": PLACES, "excluded_photos": [], "derived_photos": []})
    )
    for tier in harness.TIERS:
        (splits / tier).mkdir()
        files = {
            "strongest-yours": {"ratings": ratings[tier]},
            "strongest-general": {
                "kept": [{"original": r["key"], "place": r["place"]} for r in ratings[tier][::7]]
            },
            "story-yours": {
                "stories": [],
                "finished": [
                    [r["key"] for r in rows] for rows in (ratings[tier][:3], ratings[tier][-3:])
                ],
                "anchors": [{"key": r["key"], "could": True} for r in ratings[tier][::5]],
            },
            "legacy": {
                "columns": ["key", "stars", "place"],
                "legacy_stars": [[r["key"], 3, r["place"]] for r in ratings[tier][::9]],
            },
        }
        for name, data in files.items():
            (splits / tier / f"{name}.json").write_text(json.dumps(data))


@pytest.fixture(scope="module")
def planted(world: harness.World) -> dict[str, Any]:
    return harness.evaluate(world, TRACK, [GENERAL, YOURS], "validation")


def test_a_planted_gain_is_recovered(planted: dict[str, Any]) -> None:
    precision = summarize.paired(planted, "general", "yours")["precision"]
    assert precision["candidate"] > 0.7 > precision["champion"]
    assert precision["difference"] > 0.15
    assert precision["interval_90"][0] > 0


def test_every_validation_place_is_judged_by_a_yours_that_never_saw_it(
    world: harness.World, planted: dict[str, Any]
) -> None:
    """Three places in three folds: each is ranked by a taste fitted on the other two and
    on dev, so a perfect score cannot come from having seen the answers."""
    judged = {p["unit"] for p in planted["results"]["yours"]["place@10"]}
    assert judged == {"Mill", "Harbor", "Quarry"}
    folds = harness.fold_map("validation", sorted(judged), GRID.folds)
    assert sorted(folds.values()) == [0, 1, 2]
    fits = harness.Fits(world, YOURS, "validation", folds)
    assert "Mill" not in fits.without({folds["Mill"]}).places
    assert {"Depot", "Harbor", "Quarry"} <= set(fits.without({folds["Mill"]}).places)


def test_yours_can_be_fitted_on_another_embeddings_vectors(world: harness.World) -> None:
    """``[taste] embedder`` names the vectors Yours is fitted on and reads at search time;
    without it, the main embedding's."""
    folds = harness.fold_map("validation", ["Harbor", "Mill", "Quarry"], GRID.folds)
    on_clip = harness.Fits(world, YOURS, "validation", folds).without({0})
    viewed = dataclasses.replace(
        YOURS, name="yours-dinov2", taste={"l2": 0.01, "embedder": "dinov2"}
    )
    on_dino = harness.Fits(world, viewed, "validation", folds).without({0})
    assert (on_clip.extractor, on_dino.extractor) == ("", "dinov2")
    assert on_dino.feature == world.eng.taste_feature("dinov2") != on_clip.feature
    assert np.allclose(on_dino.coef, on_clip.coef[::-1], atol=1e-4)
    report = harness.evaluate(world, TRACK, [GENERAL, viewed], "validation")
    precision = summarize.paired(report, "general", "yours-dinov2")["precision"]
    assert precision["candidate"] > 0.7 > precision["champion"]


def test_an_embedding_the_catalogue_lacks_stops_the_fit(world: harness.World) -> None:
    lacking = dataclasses.replace(YOURS, taste={"l2": 0.01, "embedder": "siglip"})
    with pytest.raises(SystemExit, match="siglip"):
        harness.Fits(world, lacking, "validation", {"Mill": 0}).without({1})


def test_a_low_priority_place_neither_judges_nor_trains(
    world: harness.World, planted: dict[str, Any]
) -> None:
    """The owner's decision of 13 November: a place named "… Museum" is reference snapshots.
    Its photos stay in the pools as unjudged ones; none of its labels is read, in any tier."""
    low = planted["low_priority"]
    assert (low["words"], low["places"]) == (["Museum"], ["Depot museum", "Mill Museum"])
    assert low["labels_dropped"] == {
        "dev/strongest-yours:ratings": EACH,
        "validation/strongest-general:kept": 5,
        "validation/strongest-yours:ratings": EACH,
    }
    judged = {p["unit"] for p in planted["results"]["yours"]["place@10"]}
    assert not judged & {"Mill Museum", "Depot museum"}
    folds = harness.fold_map("validation", sorted(judged), GRID.folds)
    fitted = harness.Fits(world, YOURS, "validation", folds).without(set()).places
    assert set(fitted) == {"Mill", "Harbor", "Quarry", "Depot"}
    assert "Mill Museum" in world.by_place

    def place(key: str) -> str:
        return str(world.cat.location[world.cat.row[key]])

    for tier in ("dev", "validation"):
        stories = world.labels.read(tier, "story-yours")
        legacy = world.labels.read(tier, "legacy")
        read = [
            *(r["place"] for r in world.labels.ratings(tier)),
            *(k["place"] for k in world.labels.read(tier, "strongest-general")["kept"]),
            *(place(a["key"]) for a in stories["anchors"]),
            *(place(k) for trio in stories["finished"] for k in trio),
            *(row[2] for row in legacy["legacy_stars"]),
        ]
        assert read
        assert not [p for p in read if "museum" in p.lower()]
        assert legacy["columns"] == ["key", "stars", "place"]
    assert len(world.labels.read("validation", "story-yours")["finished"]) == 1


def test_the_share_of_picks_from_low_priority_places_is_reported(
    world: harness.World, planted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Museum snapshots once came out above stronger photos (owner, 14 November). Their
    labels are not read, so this is the one sign that a change brings them back: of the
    picks from pools that hold such photos, the share that are theirs."""
    assert summarize.summary(planted, "general")["low_priority_share"]["point"] is None
    monkeypatch.setattr(world, "grid", dataclasses.replace(GRID, whole=True))
    report = harness.evaluate(world, TRACK, [GENERAL, YOURS], "validation")
    (pool,) = [
        p
        for pools in report["results"]["general"].values()
        for p in pools
        if p["low_priority_pool"]
    ]
    assert pool["unit"] == "whole"
    assert pool["low_priority_pool"] == pytest.approx(2 / len(PLACES))
    museum = [str(world.cat.location[i]).lower().endswith("museum") for i in pool["pick_rows"]]
    assert pool["low_priority_picks"] == sum(museum)
    share = summarize.summary(report, "general")["low_priority_share"]["point"]
    assert share == pytest.approx(sum(museum) / len(museum))
    assert "low_priority_share" in summarize.paired(report, "general", "yours")
    assert "low_priority_share" not in [
        name for rule in verdict.RULES.values() for name, _ in (*rule.gains, *rule.guards)
    ]


def test_a_set_of_stories_shorter_than_asked_is_reported() -> None:
    """The first story attempt (14 November 2026) removed repetition by making 4 stories
    where 12 were asked for, and no measure saw it."""

    def made(n: int, k: int, unit: str) -> dict[str, Any]:
        same = {
            "reuse": 0.0,
            "anchor_repeat": 0.0,
            "pair_repeat": 0.0,
            "within": 0.0,
            "siglip": 0.0,
        }
        return {
            "pool": "place",
            "unit": unit,
            "k": k,
            "stories": [[0, 1, 2]] * n,
            "result": same,
            "random": same,
            "seconds": 1.0,
        }

    res = {"sessions": {}, "finished": [], "sets": [made(4, 12, "a"), made(50, 50, "b")]}
    filled = summarize.story_measures(res)["filled"]
    assert filled.fn(None) == pytest.approx((4 / 12 + 1) / 2)


def test_laion_says_where_the_picks_stand_in_their_pool(
    world: harness.World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strongest-General's second judge: the mean LAION percentile of the picks within
    their pool, 0.5 being chance. Here LAION is made to agree with the planted taste."""
    direction = np.random.default_rng(7).standard_normal(world.cat.clip.shape[1])
    monkeypatch.setattr(world, "laion", world.cat.clip @ direction)
    report = harness.evaluate(world, TRACK, [GENERAL, YOURS], "validation")
    laion = summarize.paired(report, "general", "yours")["laion_pct"]
    assert laion["candidate"] > 0.6 > laion["champion"]
    assert laion["interval_90"][0] > 0


class AlongLastAxis:
    """General's aesthetic for the made-up archive, which has no zero-shot model: strong
    along the last axis of the embedding, which the planted taste does not favour."""

    name = "aesthetic"
    labels = ("strong", "weak")
    version = "test"
    description = "test double"

    def predict_proba(self, view: Any) -> Any:
        strong = 1 / (1 + np.exp(-5 * np.asarray(view.embeddings)[:, -1]))
        return np.stack([strong, 1 - strong], axis=1).astype(np.float32)


class WithGeneralAesthetic:
    def __init__(self, base: Any) -> None:
        self._base = base

    def names(self) -> list[str]:
        return [*self._base.names(), "aesthetic"]

    def get(self, name: str) -> Any:
        return AlongLastAxis() if name == "aesthetic" else self._base.get(name)

    def versions(self) -> dict[str, str]:
        return {**self._base.versions(), "aesthetic": "test"}


def test_general_can_rank_by_laion_blended_into_its_aesthetic(
    world: harness.World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``[models] aesthetic = "laion"`` answers General's "aesthetic" with LAION mixed in,
    ``share`` of it; at 0 it is today's General. Here LAION agrees with the planted taste."""
    direction = np.random.default_rng(7).standard_normal(world.cat.clip.shape[1])
    monkeypatch.setattr(world, "laion", world.cat.clip @ direction)
    monkeypatch.setattr(world.eng, "models", WithGeneralAesthetic(world.eng.models))
    today = dataclasses.replace(GENERAL, options=TASTE_ALONE)
    laion = dataclasses.replace(
        today, name="general-laion", models={"aesthetic": "laion", "share": 1.0}
    )
    none = dataclasses.replace(
        laion, name="general-laion-0", models={"aesthetic": "laion", "share": 0.0}
    )
    report = harness.evaluate(world, TRACK, [today, laion, none], "validation")
    precision = summarize.paired(report, "general", "general-laion")["precision"]
    assert precision["candidate"] > 0.7 > precision["champion"]
    assert summarize.paired(report, "general", "general-laion-0")["overlap"] == 1.0


@pytest.mark.parametrize(
    ("models", "laion"), [({"aesthetic": "clip_large"}, True), ({"aesthetic": "laion"}, False)]
)
def test_an_aesthetic_the_harness_cannot_read_stops_the_run(
    world: harness.World, monkeypatch: pytest.MonkeyPatch, models: dict[str, Any], laion: bool
) -> None:
    monkeypatch.setattr(world, "laion", np.zeros(len(world.cat.keys)) if laion else None)
    asked = dataclasses.replace(GENERAL, models=models)
    with pytest.raises(SystemExit, match="aesthetic"):
        harness.Runner(world, asked).context(world.by_place["Mill"], 5, None)


def test_yours_can_read_laion_as_an_offset(
    world: harness.World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``[taste] offset = "laion"`` fits Yours with LAION's score as an input that is not
    held back, and reads it at search time. With the embedding all but held back, Yours
    alone knows less; with LAION, which here agrees with the planted taste, it knows."""
    direction = np.random.default_rng(7).standard_normal(world.cat.clip.shape[1])
    monkeypatch.setattr(world, "laion", world.cat.clip @ direction)
    held = dataclasses.replace(YOURS, name="held", taste={"l2": 1e6})
    offset = dataclasses.replace(YOURS, name="offset", taste={"l2": 1e6, "offset": "laion"})
    folds = harness.fold_map("validation", ["Harbor", "Mill", "Quarry"], GRID.folds)
    fitted = harness.Fits(world, offset, "validation", folds).without({0})
    assert fitted.general_coef > 0
    report = harness.evaluate(world, TRACK, [held, offset], "validation")
    precision = summarize.paired(report, "held", "offset")["precision"]
    assert precision["candidate"] > 0.9
    assert precision["difference"] > 0.2


@pytest.mark.parametrize(
    ("taste", "laion"), [({"offset": "general"}, True), ({"offset": "laion"}, False)]
)
def test_an_offset_the_harness_cannot_read_stops_the_fit(
    world: harness.World, monkeypatch: pytest.MonkeyPatch, taste: dict[str, Any], laion: bool
) -> None:
    monkeypatch.setattr(world, "laion", np.zeros(len(world.cat.keys)) if laion else None)
    asked = dataclasses.replace(YOURS, taste={"l2": 0.01, **taste})
    with pytest.raises(SystemExit, match="offset"):
        harness.Fits(world, asked, "validation", {"Mill": 0}).without({1})


def test_without_laion_scores_the_measure_is_absent(planted: dict[str, Any]) -> None:
    assert summarize.paired(planted, "general", "yours")["laion_pct"]["difference"] is None


def test_a_config_against_itself_shows_nothing_and_fails(world: harness.World) -> None:
    twin = dataclasses.replace(GENERAL, name="twin")
    report = harness.evaluate(world, TRACK, [GENERAL, twin], "validation")
    paired = summarize.paired(report, "general", "twin")
    assert paired["overlap"] == 1.0
    assert paired["precision"]["difference"] == 0
    assert paired["precision"]["interval_90"] == [0.0, 0.0]
    ladder = [{"sigma": 0.1, "overlap": 1.0, "sd": {"precision": 0.0, "repeat_excess": 0.0}}]
    assert verdict.decide(TRACK, paired, ladder)["decision"] == "fail"


def test_a_null_moves_picks_the_same_way_every_time_and_makes_a_ladder(
    world: harness.World,
) -> None:
    """A null moves what the taste says, so it is made from a config that has a taste: the
    tiny embedder reads no words, and General here ranks by technical quality alone."""
    nulls = harness.null_configs(YOURS, [1.0], 2)
    first = harness.evaluate(world, TRACK, [YOURS, *nulls], "validation")
    again = harness.evaluate(world, TRACK, [YOURS, *nulls], "validation")
    picks = [p["pick_rows"] for p in first["results"]["null-1-1"]["place@10"]]
    assert picks == [p["pick_rows"] for p in again["results"]["null-1-1"]["place@10"]]
    (rung,) = summarize.ladder(first)
    assert rung["nulls"] == 2
    assert rung["overlap"] < 1.0
    assert rung["sd"]["precision"] > 0


def test_the_planted_gain_passes_the_bar_its_own_nulls_set(
    world: harness.World, planted: dict[str, Any]
) -> None:
    nulls = harness.null_configs(YOURS, [0.3], 2)
    ladder = summarize.ladder(harness.evaluate(world, TRACK, [YOURS, *nulls], "validation"))
    found = verdict.decide(TRACK, summarize.paired(planted, "general", "yours"), ladder)
    assert found["decision"] == "pass", found["checks"]


def test_a_report_is_written_small_enough_to_commit_and_reads_back(tmp_path: Path) -> None:
    """A look's report holds every pick, and the repository refuses a file over 1 MB: it
    is written compressed, the same bytes each time so its recorded hash can be re-made."""
    report = {"champion": "a", "picks": list(range(50_000))}
    summarize.write_report(tmp_path / "r.json.gz", report)
    first = (tmp_path / "r.json.gz").read_bytes()
    summarize.write_report(tmp_path / "r.json.gz", report)
    assert (tmp_path / "r.json.gz").read_bytes() == first
    assert len(first) < len(json.dumps(report)) / 2
    assert summarize.read_report(tmp_path / "r.json.gz") == report
    summarize.write_report(tmp_path / "plain.json", report)
    assert summarize.read_report(tmp_path / "plain.json") == report


def test_a_tier_that_cannot_be_read_says_how_it_opens(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "strongest-yours.json").write_text("{}")
    (locked / "strongest-yours.json").chmod(0)
    with pytest.raises(SystemExit, match="loop look"):
        harness.Labels(tmp_path).ratings("locked")


def test_the_photos_to_rate_are_the_unrated_ones_only_one_list_shows() -> None:
    """Rated blind, they say which list is better with the fewest photos: what both lists
    show, and what already has an answer, tells the two apart no further."""

    def pool(rows: list[int], rated: list[bool]) -> dict[str, Any]:
        return {"unit": "Mill", "pick_rows": rows, "pick_rated": rated}

    report = {
        "champion": "a",
        "results": {
            "a": {"place@50": [pool([1, 2, 3], [False, False, True])]},
            "b": {"place@50": [pool([2, 4, 5], [False, False, True])]},
        },
    }
    rows = disagreement.only_one_list(report, "b", "place@50")
    assert sorted((r["row"], r["arm"]) for r in rows) == [(1, "champion"), (4, "candidate")]


def fitted_on(world: harness.World, taste: dict[str, Any]) -> Any:
    asked = dataclasses.replace(YOURS, taste={"l2": 1000.0, **taste})
    return harness.Fits(world, asked, "validation", {}).without({-1})


def test_yours_can_learn_from_a_share_of_its_photos(world: harness.World) -> None:
    """``[taste] keep`` fits Yours on a seeded share of its ratings, for a learning curve;
    at 1 it is today's fit."""
    whole = fitted_on(world, {})
    assert fitted_on(world, {"keep": 1.0}).rated == whole.rated
    half = fitted_on(world, {"keep": 0.5, "keep_seed": 3})
    assert half.rated == round(whole.rated / 2)
    assert fitted_on(world, {"keep": 0.5, "keep_seed": 3}).coef.tolist() == half.coef.tolist()


def test_yours_can_learn_from_a_share_of_its_places(world: harness.World) -> None:
    whole = fitted_on(world, {})
    some = fitted_on(world, {"keep": 0.5, "keep_by": "place", "keep_seed": 1})
    assert set(some.places) < set(whole.places)
    assert len(some.places) == round(len(whole.places) / 2)


@pytest.mark.parametrize("taste", [{"keep": 0.0}, {"keep": 1.5}, {"keep_by": "visit"}])
def test_a_share_the_harness_cannot_keep_stops_the_fit(
    world: harness.World, taste: dict[str, Any]
) -> None:
    with pytest.raises(SystemExit, match="keep"):
        fitted_on(world, taste)
