"""End-to-end with the real CLIP and SigLIP embedders: "Describe it" and zero-shot models.

Marked ``ml``: skipped unless the ml extra is installed. Downloads weights on
first run.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from calico.domain import SelectionRequest
from calico.engine import Engine
from calico.models.base import LabelModel
from calico.settings import CollectionConfig, Settings
from tests import factories as f

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.ml

COLOURS = {"red": (200, 30, 30), "blue": (30, 40, 200), "green": (30, 170, 40)}


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Engine]:
    base = tmp_path_factory.mktemp("clip")
    root = base / "archive"
    t0 = datetime(2024, 1, 1, 12)
    for i, (name, rgb) in enumerate(COLOURS.items()):
        f.save_jpeg(f.solid(rgb, (256, 192)), root / "Colours" / f"{name}.jpg", captured_at=t0)
        f.save_jpeg(
            f.scene(i), root / "Scenes" / f"scene{i}.jpg", captured_at=t0 + timedelta(days=i)
        )
    settings = Settings(
        home=base / "home",
        embedder="clip",
        collections=(CollectionConfig(name="archive", root=root),),
    )
    with Engine.open(settings) as eng:
        yield eng


def test_criteria_text_ranks_the_described_photo_first(engine: Engine) -> None:
    root = engine.settings.collections[0].root
    request = SelectionRequest(
        task="best",
        folders=(root,),
        criteria="a plain solid blue image",
        limit=3,
        describe_weight=0.85,
        options={"diversity": 0.0},
    )
    result = engine.select(request)
    assert result.groups[0].picks[0].path.name == "blue.jpg"
    assert not any("cannot read text" in n for n in result.notes)


def test_zero_shot_models_load_and_return_distributions(engine: Engine) -> None:
    model = engine.models.get("shot_kind")
    assert isinstance(model, LabelModel)
    assert "portrait" in model.labels


def test_triptych_with_text_models_runs_every_connection(engine: Engine) -> None:
    root = engine.settings.collections[0].root
    request = SelectionRequest(
        task="triptych",
        folders=(root,),
        limit=1,
        options={"connections": ["theme", "mood", "place_type", "color", "kind_mix"]},
    )
    result = engine.select(request)
    assert len(result.groups[0].picks) == 3
    assert not any("unavailable" in n.lower() for n in result.notes)


def test_versions_record_the_clip_model(engine: Engine, tmp_path: Path) -> None:
    root = engine.settings.collections[0].root
    result = engine.select(SelectionRequest(task="best", folders=(root,)))
    assert result.versions["clip"].startswith("clip@ViT-B-16")


def test_describe_it_is_read_by_siglip_which_is_not_fooled_by_absent_things(
    engine: Engine,
) -> None:
    root = engine.settings.collections[0].root
    request = SelectionRequest(
        task="best", folders=(root,), criteria="a bicycle", only_clear_matches=True, limit=6
    )
    result = engine.select(request)
    assert result.versions["criteria"] == "parts@1 (siglip)"
    assert any("isn't confident any photo here clearly matches" in n for n in result.notes)
    picks = [p for g in result.groups for p in g.picks]
    assert picks  # the closest, not nothing
    assert all(p.clear_match is False for p in picks)


def test_a_page_of_text_is_a_document_not_a_strong_photograph(
    engine: Engine, tmp_path: Path
) -> None:
    page = f.text_page(["Ashby Asylum, 1874", *["The wards opened in the spring."] * 12])
    path = f.save_jpeg(page, tmp_path / "page.jpg")
    _, matrices = engine.indexer.features_for_files([path])
    model = engine.models.get("aesthetic")
    assert isinstance(model, LabelModel)
    (probs,) = model.predict_proba(engine.as_view(matrices))
    assert model.labels[int(probs.argmax())] == "document"
