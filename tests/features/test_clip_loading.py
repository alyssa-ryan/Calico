"""How ClipEmbedder loads weights (runs without torch, via fake modules).

Cached weights must load with Hugging Face in offline mode: no network call,
so no server warnings and no dependence on connectivity. Only a cache miss
goes online, and then the hub's log chatter is silenced.
"""

from __future__ import annotations

import logging
import math
import sys
import types
from dataclasses import dataclass, field

import numpy as np
import pytest

from calico.errors import CapabilityError
from calico.features import extractor_registry
from calico.features.clip import ClipEmbedder, ClipLargeEmbedder, SiglipEmbedder


@dataclass
class Hub:
    """Records whether each load attempt ran with the hub offline."""

    cached: bool = True
    online_fails: bool = False
    calibrated: bool = False
    """A model trained to say yes or no to each image and text (SigLIP)."""
    attempts: list[bool] = field(default_factory=list)
    loaded: list[tuple[str, str]] = field(default_factory=list)
    tokenizer_offline: list[bool] = field(default_factory=list)


@dataclass
class Scalar:
    """Stands in for a one-element tensor."""

    value: float

    def exp(self) -> Scalar:
        return Scalar(math.exp(self.value))

    def item(self) -> float:
        return self.value


def fake_modules(monkeypatch: pytest.MonkeyPatch, hub: Hub) -> types.ModuleType:
    constants = types.ModuleType("huggingface_hub.constants")
    constants.HF_HUB_OFFLINE = False  # type: ignore[attr-defined]
    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.constants = constants  # type: ignore[attr-defined]

    net = types.SimpleNamespace(visual=types.SimpleNamespace(output_dim=4))
    net.to = lambda device: net
    net.eval = lambda: net
    net.logit_scale = Scalar(math.log(10))
    net.logit_bias = Scalar(-5) if hub.calibrated else None

    def create(model: str, pretrained: str) -> tuple[object, None, object]:
        offline = bool(constants.HF_HUB_OFFLINE)
        hub.attempts.append(offline)
        hub.loaded.append((model, pretrained))
        logging.getLogger("huggingface_hub.utils._http").warning("unauthenticated requests")
        if offline and not hub.cached:
            raise FileNotFoundError("not in cache")
        if not offline and hub.online_fails:
            raise RuntimeError("no network")
        return net, None, lambda im: im

    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    open_clip = types.ModuleType("open_clip")
    open_clip.create_model_and_transforms = create  # type: ignore[attr-defined]
    open_clip.get_tokenizer = lambda model: hub.tokenizer_offline.append(  # type: ignore[attr-defined]
        bool(constants.HF_HUB_OFFLINE)
    )
    for name, module in {
        "torch": torch,
        "open_clip": open_clip,
        "huggingface_hub": huggingface_hub,
        "huggingface_hub.constants": constants,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return constants


def test_cached_weights_load_offline_in_one_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = Hub(cached=True)
    constants = fake_modules(monkeypatch, hub)
    ClipEmbedder()
    assert hub.attempts == [True]
    assert constants.HF_HUB_OFFLINE is False  # restored


def test_cache_miss_downloads_once_online(monkeypatch: pytest.MonkeyPatch) -> None:
    hub = Hub(cached=False)
    constants = fake_modules(monkeypatch, hub)
    ClipEmbedder()
    assert hub.attempts == [True, False]
    assert constants.HF_HUB_OFFLINE is False


def test_hub_log_chatter_is_silenced_during_loading(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_modules(monkeypatch, Hub(cached=False))
    before = logging.getLogger("huggingface_hub").level
    with caplog.at_level(logging.WARNING):
        ClipEmbedder()
    assert "unauthenticated" not in caplog.text
    assert logging.getLogger("huggingface_hub").level == before  # restored


def test_load_failure_is_capability_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_modules(monkeypatch, Hub(cached=False, online_fails=True))
    with pytest.raises(CapabilityError, match=r"ViT-B-16.*laion2b.*no network"):
        ClipEmbedder()


def test_siglip_is_its_own_extractor_and_reads_its_tokenizer_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SigLIP's tokenizer comes from the hub too, so it must load offline like the weights."""
    hub = Hub(calibrated=True)
    fake_modules(monkeypatch, hub)
    siglip = SiglipEmbedder()
    assert hub.loaded == [("ViT-B-16-SigLIP", "webli")]
    assert hub.tokenizer_offline == [True]
    assert (siglip.name, siglip.version) == ("siglip", "ViT-B-16-SigLIP/webli")


def test_a_model_trained_to_say_yes_or_no_gives_match_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_modules(monkeypatch, Hub(calibrated=True))
    sims = np.array([[0.5, 0.9], [0.1, 0.5]], dtype=np.float32)
    p = SiglipEmbedder().match_probability(sims)
    assert p is not None
    np.testing.assert_allclose(p, 1 / (1 + np.exp(-(10 * sims - 5))), rtol=1e-5)


def test_clip_only_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_modules(monkeypatch, Hub())
    assert ClipEmbedder().match_probability(np.zeros((1, 1), dtype=np.float32)) is None


def test_clip_large_is_its_own_extractor_on_openais_vit_l14(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A larger CLIP cached beside today's, under its own key, so a taste can be measured
    on it without changing what any search reads. OpenAI's weights, because LAION's
    aesthetic head was trained on that embedding."""
    hub = Hub()
    fake_modules(monkeypatch, hub)
    large = ClipLargeEmbedder()
    assert hub.loaded == [("ViT-L-14-quickgelu", "openai")]
    assert (large.name, large.version) == ("clip_large", "ViT-L-14-quickgelu/openai")
    assert large.supports_text
    assert extractor_registry().get("clip_large") is ClipLargeEmbedder
