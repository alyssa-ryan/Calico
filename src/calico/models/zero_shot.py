"""Zero-shot label model: softmax over similarity to each label's text prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from calico.features.base import Embedder
    from calico.models.taxonomy import Taxonomy
    from calico.selection.context import FeatureView


class ZeroShotLabelModel:
    """Each label is the normalised mean of its prompt embeddings (prompt ensembling
    averages out the wording of any single prompt). ``temperature`` sharpens the
    softmax: CLIP cosines sit in a narrow band, so 0.01 is needed to spread them."""

    def __init__(self, taxonomy: Taxonomy, embedder: Embedder, temperature: float = 0.01) -> None:
        self.name = taxonomy.name
        self.labels = taxonomy.labels
        self.description = taxonomy.description
        self.version = f"zs{taxonomy.version}-t{temperature:g}-{embedder.name}@{embedder.version}"
        self._temperature = temperature
        flat = [p for group in taxonomy.prompts for p in group]
        vectors = np.asarray(embedder.encode_texts(flat), dtype=np.float64)
        classes = np.zeros((len(taxonomy.labels), vectors.shape[1]))
        start = 0
        for i, group in enumerate(taxonomy.prompts):
            mean = vectors[start : start + len(group)].mean(axis=0)
            classes[i] = mean / max(float(np.linalg.norm(mean)), 1e-12)
            start += len(group)
        self._classes = classes

    def predict_proba(self, view: FeatureView) -> NDArray[np.float32]:
        logits = (view.embeddings.astype(np.float64) @ self._classes.T) / self._temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
