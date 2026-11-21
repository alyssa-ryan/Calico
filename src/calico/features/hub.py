"""Loading model weights from Hugging Face's hub: offline first, quietly.

With Hugging Face's offline switch on, a cached model loads without any network
call (faster, works without internet, and the hub has no chance to answer with
"unauthenticated request" warnings). Only a cache miss goes online, once, with the
hub's log chatter silenced. Shared by every extractor that loads hub weights.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from calico.errors import CalicoError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

DEVICE_ENV = "CALICO_DEVICE"


def auto_device(torch: Any) -> str:
    """The device an extractor set to ``"auto"`` runs on: ``$CALICO_DEVICE`` when set, else
    the GPU PyTorch finds (MPS, then CUDA), else the CPU."""
    named = os.environ.get(DEVICE_ENV, "").strip()
    if named:
        if not re.fullmatch(r"cpu|mps|cuda(:\d+)?", named):
            raise CalicoError(f"{DEVICE_ENV}={named!r}: name cpu, mps or cuda (or cuda:N).")
        return named
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@contextmanager
def hub_offline() -> Iterator[None]:
    """Turn on huggingface_hub's offline mode for the duration (it is read at call time)."""
    try:
        constants: Any = importlib.import_module("huggingface_hub.constants")
    except ImportError:  # pragma: no cover - installed without the hub
        yield
        return
    previous = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        constants.HF_HUB_OFFLINE = previous


@contextmanager
def quiet_hub() -> Iterator[None]:
    """Silence huggingface_hub's warnings (server notices, progress) while loading."""
    hub_logger = logging.getLogger("huggingface_hub")
    previous = hub_logger.level
    hub_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        hub_logger.setLevel(previous)


def offline_first[T](load: Callable[[], T], what: str) -> T:
    """``load()`` from the cache; on a miss (or an unreadable cache), once from the hub."""
    with quiet_hub():
        try:
            with hub_offline():
                return load()
        except Exception:  # not cached: fetch it once
            logger.info("%s not cached; downloading once", what)
        return load()
