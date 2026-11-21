"""LAION's aesthetic predictor: its head, how its weights are fetched, and (ml) the
real model against the reference network.

The head's weights come from GitHub, not Hugging Face, so they are fetched and checked
here rather than by ``features.hub``: once, then read from the cache.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import sys
import types
import urllib.error
import urllib.request
from email.message import Message
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from calico.errors import CapabilityError
from calico.features.clip import ClipLargeEmbedder
from calico.features.laion import (
    FETCH_TRIES,
    HEAD_FILE,
    LaionAestheticExtractor,
    Layers,
    _fetch,
    head_score,
    load_head,
)
from tests.factories import scene

if TYPE_CHECKING:
    from pathlib import Path


def layers(seed: int = 0) -> Layers:
    rng = np.random.default_rng(seed)
    sizes = [8, 5, 3, 1]
    return tuple(
        (
            rng.normal(size=(out, inp)).astype(np.float32),
            rng.normal(size=out).astype(np.float32),
        )
        for inp, out in itertools.pairwise(sizes)
    )


def test_the_head_is_its_linear_layers_applied_in_turn() -> None:
    """LAION's head has no activations between its layers (they are commented out in the
    reference), so it is one straight line through the embedding."""
    x = np.random.default_rng(1).normal(size=(4, 8)).astype(np.float32)
    head = layers()
    expected = x
    for weight, bias in head:
        expected = expected @ weight.T + bias
    out = head_score(x, head)
    assert out.shape == (4,)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected[:, 0], rtol=1e-5)


class Tensor:
    def __init__(self, value: np.ndarray[Any, Any]) -> None:
        self.value = value

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.value


def fake_torch(monkeypatch: pytest.MonkeyPatch, cache: Path) -> list[Path]:
    """A torch whose hub cache is ``cache`` and whose ``load`` reads any file as a head
    of five layers, as LAION's is laid out; returns the paths it loaded."""
    loaded: list[Path] = []
    state = {
        f"layers.{i}.{part}": Tensor(np.zeros((1, 1) if part == "weight" else 1, np.float32))
        for i in (0, 2, 4, 6, 7)
        for part in ("weight", "bias")
    }

    def load(path: Path, *, map_location: str, weights_only: bool) -> dict[str, Tensor]:
        assert (map_location, weights_only) == ("cpu", True)  # never runs pickled code
        loaded.append(path)
        return state

    torch = types.ModuleType("torch")
    torch.hub = types.SimpleNamespace(get_dir=lambda: str(cache))  # type: ignore[attr-defined]
    torch.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    return loaded


def published(tmp_path: Path, content: bytes = b"laion head") -> tuple[str, str]:
    """A head "on GitHub": a file:// URL and its SHA-256."""
    source = tmp_path / "remote" / HEAD_FILE
    source.parent.mkdir()
    source.write_bytes(content)
    return source.as_uri(), hashlib.sha256(content).hexdigest()


def test_the_head_is_downloaded_once_then_read_from_the_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = fake_torch(monkeypatch, tmp_path / "hub")
    url, sha256 = published(tmp_path)
    head = load_head(sys.modules["torch"], url=url, sha256=sha256)
    assert [w.shape for w, _ in head] == [(1, 1)] * 5
    cached = tmp_path / "hub" / "checkpoints" / HEAD_FILE
    assert loaded == [cached]
    (tmp_path / "remote" / HEAD_FILE).unlink()  # offline from now on
    load_head(sys.modules["torch"], url=url, sha256=sha256)
    assert loaded == [cached, cached]


def test_a_download_that_is_not_laions_head_is_refused_and_not_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = fake_torch(monkeypatch, tmp_path / "hub")
    url, _ = published(tmp_path, b"something else")
    with pytest.raises(CapabilityError, match="SHA-256"):
        load_head(sys.modules["torch"], url=url, sha256="0" * 64)
    assert not list((tmp_path / "hub").rglob("*.pth*"))
    assert loaded == []


def test_a_cached_head_that_changed_is_fetched_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_torch(monkeypatch, tmp_path / "hub")
    url, sha256 = published(tmp_path)
    cached = tmp_path / "hub" / "checkpoints" / HEAD_FILE
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"half a download")
    load_head(sys.modules["torch"], url=url, sha256=sha256)
    assert cached.read_bytes() == b"laion head"


def test_a_head_that_cannot_be_fetched_says_where_to_put_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_torch(monkeypatch, tmp_path / "hub")
    url = (tmp_path / "nowhere.pth").as_uri()
    with pytest.raises(CapabilityError, match=r"network.*checkpoints"):
        load_head(sys.modules["torch"], url=url, sha256="0" * 64)


HEAD = b"laion head"


def serving(monkeypatch: pytest.MonkeyPatch, failures: list[Exception]) -> list[str]:
    """``urlopen`` failing with each of ``failures`` in turn, then serving ``HEAD``; the URLs
    it was asked for."""
    asked: list[str] = []

    def urlopen(url: str, timeout: float) -> io.BytesIO:
        asked.append(url)
        if failures:
            raise failures.pop(0)
        return io.BytesIO(HEAD)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return asked


def test_a_download_that_times_out_is_tried_again_after_a_longer_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # One read timing out failed a CI run on 8 November 2026 (10 November 2026 audit, B-11).
    asked = serving(
        monkeypatch,
        [TimeoutError("The read operation timed out"), urllib.error.URLError(TimeoutError())],
    )
    waits: list[float] = []
    path = tmp_path / "hub" / "checkpoints" / HEAD_FILE
    _fetch("https://example.test/head", path, hashlib.sha256(HEAD).hexdigest(), sleep=waits.append)
    assert path.read_bytes() == HEAD
    assert len(asked) == 3
    assert len(waits) == 2
    assert 0 < waits[0] < waits[1]  # a longer wait each time
    assert not list(path.parent.glob("*.partial"))


def test_a_download_that_keeps_timing_out_gives_up_after_its_tries_and_says_where_to_put_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asked = serving(monkeypatch, [TimeoutError("timed out")] * 10)
    waits: list[float] = []
    path = tmp_path / "hub" / "checkpoints" / HEAD_FILE
    with pytest.raises(CapabilityError, match=r"network.*checkpoints"):
        _fetch("https://example.test/head", path, "0" * 64, sleep=waits.append)
    assert len(asked) == FETCH_TRIES
    assert len(waits) == FETCH_TRIES - 1  # no wait after the last try
    assert not path.exists()
    assert not list(path.parent.glob("*.partial"))


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("no such file"),
        urllib.error.HTTPError("https://example.test/head", 404, "Not Found", Message(), None),
        urllib.error.URLError("unknown url type"),
    ],
)
def test_a_missing_file_or_a_refusal_is_not_tried_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: Exception
) -> None:
    asked = serving(monkeypatch, [failure] * 10)
    waits: list[float] = []
    with pytest.raises(CapabilityError):
        _fetch("https://example.test/head", tmp_path / HEAD_FILE, "0" * 64, sleep=waits.append)
    assert len(asked) == 1
    assert waits == []


def test_a_server_error_or_a_cut_connection_is_tried_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = urllib.error.HTTPError(
        "https://example.test/head", 503, "Unavailable", Message(), None
    )
    asked = serving(monkeypatch, [server, ConnectionResetError("reset by peer")])
    path = tmp_path / HEAD_FILE
    _fetch(
        "https://example.test/head", path, hashlib.sha256(HEAD).hexdigest(), sleep=lambda _: None
    )
    assert len(asked) == 3
    assert path.read_bytes() == HEAD


def test_without_torch_the_extractor_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(CapabilityError, match="ml"):
        LaionAestheticExtractor()


@pytest.mark.ml
def test_the_real_predictor_matches_laions_reference_network() -> None:
    import torch  # noqa: PLC0415 - ml only

    extractor = LaionAestheticExtractor(device="cpu")
    assert extractor.version == "ViT-L-14-quickgelu/openai+sac+logos+ava1-l14-linearMSE"
    assert isinstance(extractor._clip, ClipLargeEmbedder)  # the embedding clip_large caches
    images = [scene(1), scene(2)]
    out = extractor.extract(images)
    assert out.shape == (2, 1)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, extractor.extract(images))  # deterministic
    assert np.all((out > 0) & (out < 10))  # LAION's scores run from 1 to 10
    # The reference: nn.Sequential with dropout between, in eval mode.
    reference = torch.nn.Sequential(
        torch.nn.Linear(768, 1024),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(1024, 128),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 64),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(64, 16),
        torch.nn.Linear(16, 1),
    )
    path = f"{torch.hub.get_dir()}/checkpoints/{HEAD_FILE}"
    state = torch.load(path, map_location="cpu", weights_only=True)
    reference.load_state_dict({k.removeprefix("layers."): v for k, v in state.items()})
    reference.eval()
    embedded = extractor._clip.extract(images)
    with torch.no_grad():
        expected = reference(torch.from_numpy(embedded)).numpy()
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)
