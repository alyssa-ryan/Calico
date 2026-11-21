from __future__ import annotations

import dataclasses
import json
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest

from calico.errors import TasteUnreadableError
from calico.models.calibration import Calibration
from calico.models.taste import TasteWeights, fit_taste
from calico.trained import TrainedModels
from tests.factories import unit

if TYPE_CHECKING:
    from pathlib import Path


def weights(n: int = 40, dim: int = 6, seed: int = 0) -> TasteWeights:
    rng = np.random.default_rng(seed)
    x = unit(rng.normal(size=(n, dim)))
    return fit_taste(x, (x[:, 0] > 0).astype(np.int8), feature="clip@1", extractor="")


def test_a_saved_taste_loads_as_it_was_fitted(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = unit(rng.normal(size=(40, 6)))
    weights = fit_taste(x, (x[:, 0] > 0).astype(np.int8), feature="clip@1", extractor="")
    store = TrainedModels(tmp_path / "models")
    assert store.load_taste() is None
    store.save_taste(weights)
    loaded = store.load_taste()
    assert loaded is not None
    assert loaded.version == weights.version
    assert (loaded.feature, loaded.extractor, loaded.rated, loaded.liked) == (
        "clip@1",
        "",
        40,
        weights.liked,
    )
    np.testing.assert_array_equal(loaded.coef, weights.coef)


def test_a_taste_keeps_the_places_it_learned_from(tmp_path: Path) -> None:
    store = TrainedModels(tmp_path / "models")
    learned = dataclasses.replace(weights(), places=("Harbor", "Old Mill"))
    store.save_taste(learned)
    loaded = store.load_taste()
    assert loaded is not None
    assert loaded.places == ("Harbor", "Old Mill")
    assert loaded.version == weights().version  # what it scores is unchanged


def test_a_taste_keeps_the_calibration_its_probabilities_were_read_for(tmp_path: Path) -> None:
    store = TrainedModels(tmp_path / "models")
    made = Calibration(slope=0.6, intercept=0.3, photos=719, liked=568, places=14)
    store.save_taste(dataclasses.replace(weights(), calibration=made))
    loaded = store.load_taste()
    assert loaded is not None
    assert loaded.calibration == made
    assert loaded.version == weights().version  # it changes nothing a search scores
    store.save_taste(weights())
    again = store.load_taste()
    assert again is not None
    assert again.calibration is None  # a taste trained without one shows no probability


def test_a_damaged_calibration_costs_the_taste_nothing(tmp_path: Path) -> None:
    store = TrainedModels(tmp_path / "models")
    made = Calibration(slope=0.6, intercept=0.3, photos=719, liked=568, places=14)
    store.save_taste(dataclasses.replace(weights(), calibration=made))
    meta = json.loads((tmp_path / "models" / "taste.json").read_text())
    meta["calibration"] = {"slope": "not a number"}
    (tmp_path / "models" / "taste.json").write_text(json.dumps(meta))
    (tmp_path / "models" / "taste.npz").unlink()  # the two-file form reads taste.json's meta
    np.savez(
        tmp_path / "models" / "taste.npz",
        mean=weights().mean,
        scale=weights().scale,
        coef=weights().coef,
    )
    loaded = store.load_taste()
    assert loaded is not None
    assert loaded.calibration is None


def test_a_taste_saved_by_an_older_calico_still_loads(tmp_path: Path) -> None:
    """Older Calicos saved the arrays in taste.npz and what they were fitted on in
    taste.json."""
    fitted = weights()
    models = tmp_path / "models"
    models.mkdir()
    np.savez(models / "taste.npz", mean=fitted.mean, scale=fitted.scale, coef=fitted.coef)
    meta = {
        "feature": fitted.feature,
        "extractor": fitted.extractor,
        "bias": fitted.bias,
        "rated": fitted.rated,
        "liked": fitted.liked,
        "version": fitted.version,
    }
    (models / "taste.json").write_text(json.dumps(meta))
    loaded = TrainedModels(models).load_taste()
    assert loaded is not None
    assert loaded.version == fitted.version


@pytest.mark.parametrize("damage", [b"", b"PK\x03\x04 not a zip", b"garbage"])
def test_a_damaged_taste_says_to_retrain(tmp_path: Path, damage: bytes) -> None:
    store = TrainedModels(tmp_path / "models")
    store.save_taste(weights())
    (tmp_path / "models" / "taste.npz").write_bytes(damage)
    with pytest.raises(TasteUnreadableError, match="calico train taste"):
        store.load_taste()


def test_retraining_while_a_search_reads_the_taste_is_never_seen_half_written(
    tmp_path: Path,
) -> None:
    """calico train taste runs in the CLI while calico serve reads the taste."""
    store = TrainedModels(tmp_path / "models")
    big = [weights(n=60, dim=4000, seed=s) for s in range(2)]
    store.save_taste(big[0])
    versions = {w.version for w in big}
    stop = threading.Event()
    torn: list[str] = []

    def write() -> None:
        i = 0
        while not stop.is_set():
            store.save_taste(big[i % 2])
            i += 1

    writer = threading.Thread(target=write)
    writer.start()
    try:
        for _ in range(300):
            try:
                loaded = store.load_taste()
            except TasteUnreadableError as exc:
                torn.append(str(exc))
                continue
            if loaded is None or loaded.version not in versions:
                torn.append(repr(loaded))
    finally:
        stop.set()
        writer.join()
    assert torn == []
