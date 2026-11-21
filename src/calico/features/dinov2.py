"""DINOv2 (timm): an image embedding for "this exact place", not "this kind of place".

CLIP and SigLIP learned to match pictures with captions, so two different
corridors are close because they would get the same caption. DINOv2 learned from
images alone, by recognising the same picture under crops and colour changes, which
is what same view asks: is this the spot in the reference, seen again? It reads no
text, so it can never read "Describe it".

Rows are the class token (``pool="token"``) or the mean of the patch tokens
(``pool="avg"``), L2-normalised. The default is the base model with registers
(``reg4``), whose patch tokens carry fewer artefacts, at 448 px with the patches
averaged: on the labelled same views it kept more of them, and fewer look-alikes,
than the class token or 224 px (``calico eval same-view --embedder dinov2``). It
costs about four times CLIP per photo indexed (~9 photos/s on an M3). The version
names all three, so a change to any is a new cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from calico.errors import CapabilityError
from calico.features.hub import auto_device, offline_first

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from PIL import Image

_HINT = "Install the ml extra with: uv sync --extra ml."
_POOLS = ("token", "avg")
NO_TEXT = "The 'dinov2' embedder reads images only; it cannot read text."


class Dinov2Embedder:
    name = "dinov2"
    supports_text = False
    columns: tuple[str, ...] | None = None

    def __init__(
        self,
        model: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        image_size: int = 448,
        pool: str = "avg",
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        if pool not in _POOLS:
            raise CapabilityError(
                f"Unknown pool {pool!r} for 'dinov2'; use one of {', '.join(_POOLS)} "
                "in extractor_options.dinov2."
            )
        try:
            import timm  # noqa: PLC0415 - lazy: PyTorch is optional
            import timm.data  # noqa: PLC0415
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise CapabilityError(f"The 'dinov2' embedder needs PyTorch and timm. {_HINT}") from exc
        if device == "auto":
            device = auto_device(torch)
        self._torch: Any = torch
        self.device = device
        self._batch_size = batch_size
        self._image_size = image_size
        # timm's helpers are loosely typed; treat them like torch, as untyped.
        data: Any = timm.data
        try:
            net: Any = offline_first(
                # fc_norm=False keeps the pretrained norm before pooling: timm would
                # otherwise add an untrained one after it for "avg".
                lambda: timm.create_model(
                    model,
                    pretrained=True,
                    num_classes=0,
                    img_size=image_size,
                    global_pool=pool,
                    fc_norm=False,
                ),
                f"DINOv2 weights {model}",
            )
        except Exception as exc:  # network, unknown model name
            raise CapabilityError(
                f"Could not load 'dinov2' model {model!r}: {exc}. Check the name in "
                "extractor_options.dinov2, and that the weights are downloaded or the "
                "network is reachable."
            ) from exc
        self._model: Any = net.to(device).eval()
        # The pretrained config still says 518 px; resize to the size the model now takes.
        config = {
            **data.resolve_model_data_config(net),
            "input_size": (3, image_size, image_size),
        }
        self._preprocess: Any = data.create_transform(**config, is_training=False)
        self.dim = int(net.num_features)
        self.version = f"{model}/{image_size}/{pool}"

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks: list[NDArray[np.float32]] = []
        with self._torch.no_grad():
            for start in range(0, len(images), self._batch_size):
                batch = images[start : start + self._batch_size]
                tensor = self._torch.stack([self._preprocess(im) for im in batch])
                out = self._model(tensor.to(self.device)).float().cpu().numpy()
                norms = np.linalg.norm(out, axis=1, keepdims=True)
                chunks.append((out / np.maximum(norms, 1e-12)).astype(np.float32))
        return np.concatenate(chunks, axis=0)

    def extract_patches(self, images: Sequence[Image.Image]) -> NDArray[np.float16]:
        """Every patch token (``(image_size / 14)²`` of them, 1,024 at 448 px), normalised
        and unit-length, without the class and register tokens. Half precision: a photo's
        patches are 1.5 MB as it is."""
        chunks: list[NDArray[np.float16]] = []
        prefix = int(self._model.num_prefix_tokens)
        with self._torch.no_grad():
            for start in range(0, len(images), self._batch_size):
                batch = images[start : start + self._batch_size]
                tensor = self._torch.stack([self._preprocess(im) for im in batch])
                tokens = self._model.forward_features(tensor.to(self.device))[:, prefix:]
                out = tokens.float().cpu().numpy()
                norms = np.linalg.norm(out, axis=2, keepdims=True)
                chunks.append((out / np.maximum(norms, 1e-12)).astype(np.float16))
        side = self._image_size // 14
        if not chunks:
            return np.zeros((0, side * side, self.dim), dtype=np.float16)
        return np.concatenate(chunks, axis=0)

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        raise CapabilityError(NO_TEXT)

    def readable_prefix(self, text: str) -> str:
        raise CapabilityError(NO_TEXT)

    def match_probability(self, similarities: NDArray[np.float32]) -> NDArray[np.float32] | None:
        return None
