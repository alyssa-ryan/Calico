"""Builders for test data. Use these instead of hand-rolling fixtures in each test file.

Two families:

* **Images on disk** -- ``scene`` draws a deterministic synthetic photograph
  from a seed; its keyword arguments simulate what changes between visits to
  the same view (light, tint, a small shift of camera angle). ``save_jpeg``
  writes it with real EXIF so the catalog's metadata path is exercised.
* **In-memory selection inputs** -- ``photo``, ``view`` and ``context`` build
  what a task receives, so task tests need no images, catalog or models.
  Fake models read their answers from feature columns (``FeatureScore``,
  ``FeatureLabels``) so they stay correct when a task subsets the view.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from calico.domain import Photo, SelectionRequest
from calico.errors import CapabilityError
from calico.selection.context import FeatureView, TaskContext

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from numpy.typing import ArrayLike, NDArray

    from calico.features.base import Embedder
    from calico.models.base import Model

# images on disk: re-exported from calico.sandbox, which `calico sandbox` also uses
from calico.sandbox import (
    EXIF_IFD,
    EXIF_MAKE,
    EXIF_MODEL,
    SANDBOX_CAMERA,
    save_jpeg,
    scene,
    solid,
)

__all__ = [
    "EXIF_IFD",
    "EXIF_MAKE",
    "EXIF_MODEL",
    "SANDBOX_CAMERA",
    "FakeModels",
    "FeatureLabels",
    "FeatureScore",
    "FirstAxisReader",
    "context",
    "fake_rawpy",
    "photo",
    "save_jpeg",
    "scene",
    "solid",
    "text_page",
    "unit",
    "view",
    "with_exif",
]


def text_page(lines: Sequence[str], size: tuple[int, int] = (480, 640)) -> Image.Image:
    """Dark lines of text on a pale page: a museum label or a photographed document."""
    page = Image.new("RGB", size, (236, 232, 222))
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=22)
    for i, line in enumerate(lines):
        draw.text((32, 40 + i * 38), line, fill=(30, 30, 30), font=font)
    return page


# --------------------------------------------------------------------------
# in-memory selection inputs
# --------------------------------------------------------------------------


def photo(
    name: str = "a.jpg",
    *,
    location: str = "Old Mill",
    captured_at: datetime | None = None,
    content_key: str | None = None,
    collection: str = "test",
    folder: str = "/archive",
    camera: str | None = None,
    utc_offset: int | None = None,
) -> Photo:
    path = Path(folder) / location / name
    return Photo(
        path=path,
        content_key=content_key or f"key-{location}-{name}",
        collection=collection,
        location=location,
        captured_at=captured_at,
        width=320,
        height=240,
        camera=camera,
        utc_offset=utc_offset,
    )


def unit(vectors: ArrayLike) -> NDArray[np.float32]:
    """Rows scaled to unit length (the contract for embeddings)."""
    arr = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.where(norms == 0, 1, norms)).astype(np.float32)


def view(
    embeddings: ArrayLike,
    features: Mapping[str, ArrayLike] | None = None,
    columns: Mapping[str, tuple[str, ...]] | None = None,
) -> FeatureView:
    """A FeatureView. 1-D feature values become a single column (one value per photo)."""
    feats = {k: _as_matrix(v) for k, v in (features or {}).items()}
    return FeatureView(embeddings=unit(embeddings), features=feats, columns=dict(columns or {}))


def _as_matrix(values: ArrayLike) -> NDArray[np.float32]:
    arr = np.asarray(values, dtype=np.float32)
    return arr[:, None] if arr.ndim == 1 else arr


@dataclass
class FeatureScore:
    """A ScoreModel whose answer is feature column 0 of ``feature``."""

    name: str
    feature: str
    version: str = "test"
    description: str = "test double"

    def score(self, v: FeatureView) -> NDArray[np.float32]:
        return v.features[self.feature][:, 0].astype(np.float32)


@dataclass
class FeatureLabels:
    """A LabelModel whose probabilities are the rows of ``feature``."""

    name: str
    feature: str
    labels: tuple[str, ...]
    version: str = "test"
    description: str = "test double"

    def predict_proba(self, v: FeatureView) -> NDArray[np.float32]:
        return v.features[self.feature].astype(np.float32)


class FakeModels:
    """A ModelProvider over a fixed dict. Missing names raise ``CapabilityError``,
    exactly as the real provider does when an embedder cannot read text."""

    def __init__(self, models: Mapping[str, Model] | None = None) -> None:
        self._models = dict(models or {})
        self._used: dict[str, str] = {}

    def names(self) -> list[str]:
        return sorted(self._models)

    def get(self, name: str) -> Model:
        if name not in self._models:
            raise CapabilityError(f"model {name!r} unavailable in this test")
        model = self._models[name]
        self._used[name] = model.version
        return model

    def versions(self) -> dict[str, str]:
        return dict(self._used)


def context(
    photos: Sequence[Photo],
    v: FeatureView,
    *,
    task: str = "test",
    criteria: ArrayLike | None = None,
    references: FeatureView | None = None,
    reference_keys: Sequence[str] | None = None,
    models: FakeModels | None = None,
    limit: int = 12,
    criteria_text: str = "",
    describe_weight: float | None = None,
    view_feature: str = "",
) -> TaskContext:
    n_refs = 0 if references is None else len(references)
    request = SelectionRequest(
        task=task,
        folders=(Path("/archive"),),
        criteria=criteria_text,
        limit=limit,
        describe_weight=describe_weight,
    )
    return TaskContext(
        request=request,
        photos=tuple(photos),
        view=v,
        models=models or FakeModels(),
        references=references,
        reference_paths=tuple(Path(f"/refs/{i}.jpg") for i in range(n_refs)),
        reference_keys=tuple(reference_keys or (f"ref-{i}" for i in range(n_refs))),
        criteria_scores=None if criteria is None else v.embeddings @ unit(criteria)[0],
        view_feature=view_feature,
    )


class FirstAxisReader:
    """Wraps an embedder that cannot read text; "reads" any text as the first axis.

    So every description prefers photos whose embedding leans on that axis:
    enough to drive labelling and evaluation end to end without PyTorch.
    """

    supports_text = True

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.name, self.version, self.dim = inner.name, inner.version, inner.dim
        self.columns = inner.columns

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        return self.inner.extract(images)

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.eye(1, self.dim, dtype=np.float32).repeat(len(texts), axis=0)

    def readable_prefix(self, text: str) -> str:
        return text

    def match_probability(self, similarities: NDArray[np.float32]) -> NDArray[np.float32] | None:
        return self.inner.match_probability(similarities)


# --------------------------------------------------------------------------
# RAW files (a stand-in for rawpy: a test cannot make a real RAW file)
# --------------------------------------------------------------------------


class _NoThumbnailError(Exception):
    pass


def fake_rawpy(
    preview: Image.Image | None = None,
    *,
    developed: Image.Image | None = None,
    timestamp: datetime | None = None,
) -> ModuleType:
    """A module that answers like ``rawpy`` for any file: ``preview`` is the JPEG it
    embeds (with whatever EXIF it carries), ``developed`` what developing it gives, and
    ``timestamp`` LibRaw's capture time. Install with ``monkeypatch.setitem(sys.modules,
    "rawpy", fake_rawpy(...))``."""
    module = ModuleType("rawpy")
    module.libraw_version = (0, 22, 1)  # type: ignore[attr-defined]
    module.LibRawError = Exception  # type: ignore[attr-defined]
    module.LibRawNoThumbnailError = _NoThumbnailError  # type: ignore[attr-defined]
    module.ThumbFormat = SimpleNamespace(JPEG="jpeg", BITMAP="bitmap")  # type: ignore[attr-defined]

    class Raw:
        other = SimpleNamespace(timestamp=timestamp)

        def __enter__(self) -> Raw:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def extract_thumb(self) -> SimpleNamespace:
            if preview is None:
                raise _NoThumbnailError("no thumbnail")
            buf = BytesIO()
            preview.save(buf, "JPEG", exif=preview.getexif().tobytes())
            return SimpleNamespace(format="jpeg", data=buf.getvalue())

        def postprocess(self, **_: object) -> NDArray[np.uint8]:
            if developed is None:
                raise AssertionError("this RAW was not expected to be developed")
            return np.asarray(developed.convert("RGB"))

    module.imread = lambda _path: Raw()  # type: ignore[attr-defined]
    return module


def with_exif(
    image: Image.Image,
    *,
    camera: tuple[str, str] | None = None,
    captured_at: datetime | None = None,
    orientation: int | None = None,
) -> Image.Image:
    """``image`` carrying these EXIF fields, as LibRaw writes them into a preview."""
    exif = image.getexif()
    if camera is not None:
        exif[EXIF_MAKE], exif[EXIF_MODEL] = camera
    if captured_at is not None:
        exif[0x0132] = captured_at.strftime("%Y:%m:%d %H:%M:%S")
    if orientation is not None:
        exif[0x0112] = orientation
    return image


# -- the improvement loop's frozen splits, made up ----------------------------------------------

LOOP_PLACES = {
    "Old Mill": "dev",
    "Tilden School": "dev",
    "Quarry": "validation",
    "Harbor Terminal": "validation",
    "Kestrel Plant": "validation",
    "Signal Hill": "locked",
    "Grain Mill": "locked",
}


def loop_splits(root: Path, *, photos_per_place: int = 6) -> Path:
    """A frozen ``splits/`` folder shaped like the loop's real one, with made-up places and
    keys, and its ``SHA256SUMS``."""

    def key(place: str, n: int) -> str:
        return hashlib.blake2b(f"{place}/{n}".encode(), digest_size=16).hexdigest()

    root.mkdir(parents=True)
    tiers = sorted(set(LOOP_PLACES.values()))
    places: dict[str, object] = {
        "places": LOOP_PLACES,
        "counts_by_tier": {t: sum(v == t for v in LOOP_PLACES.values()) for t in tiers},
        "derived_photos": [],
        "excluded_photos": [],
        "look_alike_pairs_across_tiers": 0,
        "rules": {},
        "seed": 1,
        "share_by_tier": {},
        "snapshot": "made-up",
        "target": {},
    }
    files: dict[str, dict[str, object]] = {"places.json": places}
    for tier in tiers:
        here = [p for p, t in LOOP_PLACES.items() if t == tier]
        keys = [(p, key(p, n)) for p in here for n in range(photos_per_place)]
        head = {"snapshot": "made-up", "tier": tier}
        story = {"place": here[0], "session": "s1", "templates": ["t"], "ways": ["w"]}
        files[f"{tier}/strongest-yours.json"] = {
            **head,
            "ratings": [
                {"key": k, "liked": n % 2 == 0, "place": p, "session": "s1", "use": "taste"}
                for n, (p, k) in enumerate(keys)
            ],
        }
        files[f"{tier}/strongest-general.json"] = {
            **head,
            "kept": [{"edit": k, "how": "copy", "original": k, "place": p} for p, k in keys[::3]],
        }
        files[f"{tier}/story-yours.json"] = {
            **head,
            "anchors": [{"could": True, "key": k, "session": "s1"} for _, k in keys[::2]],
            "finished": [[k for _, k in keys[:3]]],
            "stories": [{**story, "keys": [k for _, k in keys[:3]], "unfit": [], "works": True}],
        }
        files[f"{tier}/story-general.json"] = {
            **head,
            "unfit_marks": [
                {**story, "keys": [k for _, k in keys[:3]], "unfit": ["k"], "works": False}
            ],
        }
        files[f"{tier}/same-view.json"] = {
            **head,
            "same_view": [
                {"key": k, "place": p, "reference": keys[0][1], "value": 1} for p, k in keys[1:4]
            ],
        }
        files[f"{tier}/legacy.json"] = {
            **head,
            "columns": ["key", "stars", "place", "session", "use", "at"],
            "legacy_stars": [[k, "3", p, "s1", "legacy", "made-up"] for p, k in keys[::2]],
        }
    sums = []
    for name, data in sorted(files.items()):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1) + "\n")
        sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return root
