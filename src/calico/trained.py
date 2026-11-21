"""Models trained on your judgements, saved in the Calico home (``models/``).

Kept apart from the catalog, like labels: they are learned from ratings, not computed
from photos. The models themselves stay pure; this module only saves and loads their
weights.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import TYPE_CHECKING, Any

import numpy as np

from calico.errors import TasteUnreadableError
from calico.models.calibration import Calibration
from calico.models.taste import TasteWeights
from calico.storage import write_atomic

if TYPE_CHECKING:
    from pathlib import Path

_TASTE = "taste"
_META = "meta"
"""The array in ``taste.npz`` that holds what the weights were fitted on, as JSON."""


class TrainedModels:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir

    def save_taste(self, weights: TasteWeights) -> Path:
        """Replace the saved taste: one file, ``taste.npz``, written atomically, so a search
        reading it while you retrain sees the old taste or the new one, never half of one.
        ``taste.json`` is written too, for Calicos that read the two-file form."""
        meta = {
            "feature": weights.feature,
            "extractor": weights.extractor,
            "bias": weights.bias,
            "rated": weights.rated,
            "liked": weights.liked,
            "places": list(weights.places),
            "version": weights.version,
        }
        if weights.calibration is not None:
            meta["calibration"] = weights.calibration.to_json()
        buffer = io.BytesIO()
        np.savez(
            buffer,
            mean=weights.mean,
            scale=weights.scale,
            coef=weights.coef,
            meta=np.array(json.dumps(meta)),  # _META
        )
        path = self.models_dir / f"{_TASTE}.npz"
        write_atomic(path, buffer.getvalue())
        write_atomic(self.models_dir / f"{_TASTE}.json", json.dumps(meta, indent=2))
        return path

    def load_taste(self) -> TasteWeights | None:
        """The saved taste, or ``None`` if none was trained (or, saved in the two-file form,
        its files disagree). ``TasteUnreadableError`` if it is damaged."""
        arrays_path = self.models_dir / f"{_TASTE}.npz"
        if not arrays_path.is_file():
            return None
        try:
            with arrays_path.open("rb") as fh, np.load(fh) as arrays:  # numpy leaks a bad zip
                if _META in arrays:
                    meta = json.loads(str(arrays[_META]))
                else:  # the two-file form: what it was fitted on is in taste.json
                    meta_path = self.models_dir / f"{_TASTE}.json"
                    if not meta_path.is_file():
                        return None
                    meta = json.loads(meta_path.read_text())
                weights = _weights(arrays, meta)
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            raise TasteUnreadableError(
                f"Your saved taste ({arrays_path}) could not be read: {exc}. Retrain it with: "
                "calico train taste"
            ) from exc
        return weights if weights.version == meta.get("version") else None


def _weights(arrays: Any, meta: dict[str, Any]) -> TasteWeights:
    return TasteWeights(
        feature=str(meta["feature"]),
        extractor=str(meta["extractor"]),
        mean=arrays["mean"].astype(np.float32),
        scale=arrays["scale"].astype(np.float32),
        coef=arrays["coef"].astype(np.float32),
        bias=float(meta["bias"]),
        rated=int(meta["rated"]),
        liked=int(meta["liked"]),
        places=tuple(str(p) for p in meta.get("places", ())),
        calibration=Calibration.from_json(meta.get("calibration")),
    )
