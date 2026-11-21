"""CLIP and SigLIP embedders (open_clip): images and text in one space.

CLIP is the main embedding (same view, stories, kinds of shot). SigLIP reads
"Describe it": on labelled searches it ranked better (AUC 0.81 -> 0.86), and,
trained to say yes or no to each image and text, it can tell a clear match.
``clip_large`` (ViT-L/14) is a larger CLIP measured as a judge for the tastes
(roadmap, "ViT-L/14, planned"); nothing computes it for every photo yet.

torch/open_clip are imported inside ``__init__`` so that merely listing
extractors never loads PyTorch, and so the missing-extra error is actionable.

Weights are loaded **offline first** (``calico.features.hub``): a cached model
loads without any network call; only a cache miss goes online, once.
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

_HINT = 'Install the ml extra with: uv sync --extra ml (or set embedder = "tiny" in calico.toml).'


def _create(open_clip: Any, model: str, pretrained: str) -> Any:
    return offline_first(
        lambda: open_clip.create_model_and_transforms(model, pretrained=pretrained),
        f"CLIP weights {model}/{pretrained}",
    )


def _tokenizer(open_clip: Any, model: str) -> Any:
    """SigLIP's tokenizer is a Hugging Face one, fetched like the weights: offline first."""
    return offline_first(lambda: open_clip.get_tokenizer(model), f"Tokenizer for {model}")


class ClipEmbedder:
    name = "clip"
    supports_text = True
    columns: tuple[str, ...] | None = None

    def __init__(
        self,
        model: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        device: str = "auto",
        batch_size: int = 32,
        text_template: str = "{}",
    ) -> None:
        """``text_template`` wraps each description before it is read, e.g. "a photo of {}."."""
        try:
            import open_clip  # noqa: PLC0415 - lazy: PyTorch is optional
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise CapabilityError(
                f"The {self.name!r} embedder needs PyTorch and open_clip. {_HINT}"
            ) from exc
        if device == "auto":
            device = auto_device(torch)
        self._torch: Any = torch
        self.device = device
        self._batch_size = batch_size
        self._template = text_template
        try:
            net, _, preprocess = _create(open_clip, model, pretrained)
            self._model: Any = net.to(device).eval()
            self._preprocess: Any = preprocess
            self._tokenizer: Any = _tokenizer(open_clip, model)
        except Exception as exc:  # network, Hugging Face, unknown model or weights
            raise CapabilityError(
                f"Could not load {self.name!r} model {model!r} with weights {pretrained!r}: "
                f"{exc}. Check the name in extractor_options.{self.name}, and that the "
                "weights are downloaded or the network is reachable."
            ) from exc
        bias = getattr(net, "logit_bias", None)
        self._calibration = (
            None if bias is None else (float(net.logit_scale.exp().item()), float(bias.item()))
        )
        self.version = f"{model}/{pretrained}"
        output_dim = getattr(self._model.visual, "output_dim", None)  # SigLIP's (timm) has none
        self.dim = int(output_dim or self.encode_texts(["a"]).shape[1])

    @staticmethod
    def _normalise(features: Any) -> NDArray[np.float32]:
        arr: NDArray[np.float32] = features.detach().float().cpu().numpy()
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return (arr / np.maximum(norms, 1e-12)).astype(np.float32)

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks: list[NDArray[np.float32]] = []
        with self._torch.no_grad():
            for start in range(0, len(images), self._batch_size):
                batch = images[start : start + self._batch_size]
                tensor = self._torch.stack([self._preprocess(im) for im in batch])
                chunks.append(self._normalise(self._model.encode_image(tensor.to(self.device))))
        return np.concatenate(chunks, axis=0)

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        with self._torch.no_grad():
            captions = [self._template.format(t) for t in texts]
            tokens = self._tokenizer(captions).to(self.device)
            return self._normalise(self._model.encode_text(tokens))

    def match_probability(self, similarities: NDArray[np.float32]) -> NDArray[np.float32] | None:
        if self._calibration is None:
            return None
        scale, bias = self._calibration
        logits = np.clip(scale * similarities.astype(np.float64) + bias, -60, 60)
        return (1 / (1 + np.exp(-logits))).astype(np.float32)

    def _tokens(self, text: str) -> int:
        """Tokens ``text`` takes of the context, start and end markers included."""
        hf = getattr(self._tokenizer, "tokenizer", None)
        if hf is not None:  # Hugging Face (SigLIP): cleans first, adds only an end marker
            return len(hf(self._tokenizer.clean_fn(text)).input_ids)
        return len(self._tokenizer.encode(text)) + 2

    def readable_prefix(self, text: str) -> str:
        budget = int(self._tokenizer.context_length)
        words = text.split()

        def fits(n: int) -> bool:
            return self._tokens(self._template.format(" ".join(words[:n]))) <= budget

        if fits(len(words)):
            return text
        fitting, too_long = 0, len(words)
        while too_long - fitting > 1:
            middle = (fitting + too_long) // 2
            if fits(middle):
                fitting = middle
            else:
                too_long = middle
        return " ".join(words[:fitting])


class SiglipEmbedder(ClipEmbedder):
    """SigLIP for "Describe it", read as a caption: "a photo of {}." measured better."""

    name = "siglip"

    def __init__(
        self,
        model: str = "ViT-B-16-SigLIP",
        pretrained: str = "webli",
        device: str = "auto",
        batch_size: int = 32,
        text_template: str = "a photo of {}.",
    ) -> None:
        super().__init__(model, pretrained, device, batch_size, text_template)


class ClipLargeEmbedder(ClipEmbedder):
    """CLIP ViT-L/14, cached beside the main CLIP under its own key, so a taste can be
    measured on it without changing what any search reads. OpenAI's weights, not
    LAION-2B's: LAION's aesthetic head was trained on this embedding, so once every photo
    has it, LAION's score needs no second pass (``features.laion``). OpenAI's CLIPs use
    QuickGELU, which open_clip names apart."""

    name = "clip_large"

    def __init__(
        self,
        model: str = "ViT-L-14-quickgelu",
        pretrained: str = "openai",
        device: str = "auto",
        batch_size: int = 16,
        text_template: str = "{}",
    ) -> None:
        super().__init__(model, pretrained, device, batch_size, text_template)
