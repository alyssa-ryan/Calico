from __future__ import annotations

from types import SimpleNamespace

import pytest

from calico.errors import CalicoError
from calico.features.hub import DEVICE_ENV, auto_device


def fake_torch(*, mps: bool, cuda: bool) -> SimpleNamespace:
    return SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        cuda=SimpleNamespace(is_available=lambda: cuda),
    )


@pytest.mark.parametrize(
    ("mps", "cuda", "device"),
    [(True, False, "mps"), (False, True, "cuda"), (True, True, "mps"), (False, False, "cpu")],
)
def test_auto_takes_the_gpu_pytorch_finds(
    monkeypatch: pytest.MonkeyPatch, mps: bool, cuda: bool, device: str
) -> None:
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    assert auto_device(fake_torch(mps=mps, cuda=cuda)) == device


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "cuda:1"])
def test_the_environment_names_the_device(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    monkeypatch.setenv(DEVICE_ENV, device)
    assert auto_device(fake_torch(mps=True, cuda=True)) == device


def test_an_unknown_device_says_which_ones_there_are(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEVICE_ENV, "gpu")
    with pytest.raises(CalicoError, match="cpu, mps or cuda"):
        auto_device(fake_torch(mps=False, cuda=False))
