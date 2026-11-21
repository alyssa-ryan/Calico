"""LAION's aesthetic predictor: how good a photo looks to many people, as one number.

A head trained on about 240,000 human ratings (SAC, LOGOS and AVA) reads OpenAI's
CLIP ViT-L/14 image embedding and gives a score from about 1 to 10. Today's General
compares a ViT-B/16 embedding with a few text prompts, so its score is nearly a
direction Yours can already learn (audit A10-4). This one is a different, larger
embedding and a direction learned from ratings, a view of photos Calico does not have
yet (audit A10-6). Measured with ``calico eval taste --aesthetic laion_aesthetic`` on
542 ratings at 11 places (30 October 2026), General with it minus today's, by strength:
+0.02 [-0.05, 0.12] at quality weight 0.35, and +0.05 [-0.03, 0.17] by the aesthetic
alone. Again on 975 ratings at 20 places: +0.00 [-0.06, 0.07] at 0.35, and +0.03 [-0.05,
0.10] by the aesthetic alone. Each range reaches below zero, and the gain shrank as the
ratings grew, so nothing uses it; it stays for measuring.

The head's layers have no activations between them (they are commented out in LAION's
reference), so the head is one straight line through the embedding: applied here in
numpy, layer by layer.

Its weights are on GitHub (Apache-2.0), not Hugging Face: fetched once from a fixed
commit into torch's hub cache, checked by SHA-256, and loaded without running pickled
code. ViT-L/14 itself loads offline first like every CLIP (``features.clip``).
"""

from __future__ import annotations

import hashlib
import http.client
import shutil
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from calico.errors import CapabilityError
from calico.features.clip import ClipLargeEmbedder

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray
    from PIL import Image

HEAD_FILE = "sac+logos+ava1-l14-linearMSE.pth"
HEAD_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/"
    f"6934dd81792f086e613a121dbce43082cb8be85e/{HEAD_FILE}"
)
HEAD_SHA256 = "21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0"
FETCH_TRIES = 3
"""Tries at fetching the head: one read timing out failed a CI run (8 November 2026; 10
November 2026 audit, B-11), and a second or third try usually succeeds."""
FETCH_WAIT = 5.0
"""Seconds to wait after the first failed try, twice as long after the second."""
_LINEAR = (0, 2, 4, 6, 7)
"""The head's linear layers, by index in its ``nn.Sequential`` (dropouts between)."""

type Layers = tuple[tuple[NDArray[np.float32], NDArray[np.float32]], ...]
"""Each linear layer's weight ``(out, in)`` and bias ``(out,)``, in order."""


def head_score(embeddings: NDArray[np.float32], layers: Layers) -> NDArray[np.float32]:
    """The head's score for each unit-norm embedding row."""
    x = embeddings.astype(np.float32)
    for weight, bias in layers:
        x = x @ weight.T + bias
    return x[:, 0].astype(np.float32)


def load_head(torch: Any, *, url: str = HEAD_URL, sha256: str = HEAD_SHA256) -> Layers:
    """The head's layers, from torch's hub cache; on a miss (or a cached file that is not
    the published one), fetched once from ``url`` and checked before it is kept."""
    path = Path(torch.hub.get_dir()) / "checkpoints" / HEAD_FILE
    if not (path.is_file() and _sha256(path) == sha256):
        _fetch(url, path, sha256)
    state = torch.load(path, map_location="cpu", weights_only=True)
    return tuple(
        (
            np.asarray(state[f"layers.{i}.weight"].numpy(), dtype=np.float32),
            np.asarray(state[f"layers.{i}.bias"].numpy(), dtype=np.float32),
        )
        for i in _LINEAR
    )


def _worth_trying_again(exc: OSError | http.client.HTTPException) -> bool:
    """Whether another try can help: a timeout, a dropped connection, a name that did not
    resolve, a transfer cut short or a server error. A missing file or a refusal will not mend."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    return isinstance(
        reason, TimeoutError | ConnectionError | socket.gaierror | http.client.HTTPException
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch(
    url: str, path: Path, sha256: str, *, sleep: Callable[[float], None] = time.sleep
) -> None:
    """``url`` to ``path`` through a partial file, kept only if its SHA-256 is ``sha256``:
    a cut-off or altered download is never loaded. A timeout or a dropped connection is tried
    again (``FETCH_TRIES``, each after a longer wait); a missing file or a refusal is not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    for attempt in range(1, FETCH_TRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:  # noqa: S310 - a fixed https URL
                shutil.copyfileobj(response, out)
            break
        except (OSError, http.client.HTTPException) as exc:
            partial.unlink(missing_ok=True)
            if attempt < FETCH_TRIES and _worth_trying_again(exc):
                sleep(FETCH_WAIT * attempt)
                continue
            raise CapabilityError(
                f"Could not download LAION's aesthetic predictor from {url}: {exc}. Check the "
                f"network, or download that file by hand to {path}."
            ) from exc
    found = _sha256(partial)
    if found != sha256:
        partial.unlink()
        raise CapabilityError(
            f"The file downloaded from {url} is not LAION's aesthetic predictor (SHA-256 "
            f"{found}, expected {sha256}), so it was not kept. Try again later, or download "
            f"it by hand to {path}."
        )
    partial.replace(path)


class LaionAestheticExtractor:
    """One column, ``score``: LAION's aesthetic score (about 1 to 10, higher looks
    better). Built only when measured; nothing computes it for every photo."""

    name = "laion_aesthetic"
    dim = 1
    columns: tuple[str, ...] | None = ("score",)

    def __init__(self, device: str = "auto", batch_size: int = 16) -> None:
        try:
            import torch  # noqa: PLC0415 - lazy: PyTorch is optional
        except ImportError as exc:
            raise CapabilityError(
                f"The {self.name!r} extractor needs PyTorch and open_clip. Install the ml "
                "extra with: uv sync --extra ml"
            ) from exc
        # The head was trained on this embedding alone, so the model is not an option.
        self._clip = ClipLargeEmbedder(device=device, batch_size=batch_size)
        self._layers = load_head(torch)
        self.version = f"{self._clip.version}+{HEAD_FILE.removesuffix('.pth')}"

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        if not images:
            return np.zeros((0, 1), dtype=np.float32)
        return head_score(self._clip.extract(images), self._layers)[:, None]
