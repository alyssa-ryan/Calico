"""Synthetic photographs and throwaway archives, for tests and experiments.

``scene`` draws a deterministic fake photograph from a seed; its keyword
arguments simulate what changes between visits to the same view (light, tint,
a small shift of camera angle). ``build_sandbox`` writes a small archive of
them plus an isolated calico home, which is what ``calico sandbox`` uses, so
any change can be tried end to end without a real archive.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from PIL import Image, ImageDraw

from calico.errors import CalicoError
from calico.settings import CollectionConfig, Settings, load_settings, save_settings

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

MARKER = ".calico-sandbox"
"""Written into every sandbox, so ``calico sandbox --force`` replaces only a folder it made."""
EXIF_DATETIME_ORIGINAL = 0x9003
EXIF_OFFSET_TIME_ORIGINAL = 0x9011
EXIF_IFD = 0x8769
EXIF_MAKE = 0x010F
EXIF_MODEL = 0x0110


def scene(
    seed: int,
    size: tuple[int, int] = (320, 240),
    *,
    brightness: float = 1.0,
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
    shift: tuple[int, int] = (0, 0),
) -> Image.Image:
    """A synthetic 'photograph': background gradient plus seeded shapes.

    Same ``seed`` = same view. ``brightness``/``tint`` mimic different light or
    season; ``shift`` mimics a slightly different camera position.
    """
    rng = random.Random(seed)  # noqa: S311 - seeded for reproducible images, not security
    w, h = size
    base = tuple(rng.randrange(40, 200) for _ in range(3))
    img = Image.new("RGB", size, base)
    draw = ImageDraw.Draw(img)
    for y in range(0, h, 4):  # vertical gradient, so the image is not flat
        shade = int(60 * y / h)
        draw.rectangle((0, y, w, y + 4), fill=tuple(max(0, c - shade) for c in base))
    for _ in range(12):
        x0, y0 = rng.randrange(-20, w), rng.randrange(-20, h)
        x1, y1 = x0 + rng.randrange(20, w // 2), y0 + rng.randrange(20, h // 2)
        color = tuple(rng.randrange(0, 256) for _ in range(3))
        box = (x0 + shift[0], y0 + shift[1], x1 + shift[0], y1 + shift[1])
        if rng.random() < 0.5:
            draw.rectangle(box, fill=color)
        else:
            draw.ellipse(box, fill=color)
    arr = np.asarray(img, dtype=np.float32) * brightness * np.asarray(tint, dtype=np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def solid(color: tuple[int, int, int], size: tuple[int, int] = (64, 48)) -> Image.Image:
    return Image.new("RGB", size, color)


SANDBOX_CAMERA = ("Calico", "Sandbox Camera")
"""Make and model written on synthetic photos: a photo from a camera has them, and one
without them is taken for an edited export (``selection.copies.edited_from``)."""


def save_jpeg(
    image: Image.Image,
    path: Path,
    *,
    captured_at: datetime | None = None,
    camera: tuple[str, str] | None = None,
    orientation: int | None = None,
    utc_offset: str | None = None,
) -> Path:
    """Write ``image`` as a JPEG with the given EXIF fields (``utc_offset`` as cameras
    write it, e.g. "+01:00"); returns ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    if captured_at is not None:
        exif.get_ifd(EXIF_IFD)[EXIF_DATETIME_ORIGINAL] = captured_at.strftime("%Y:%m:%d %H:%M:%S")
    if utc_offset is not None:
        exif.get_ifd(EXIF_IFD)[EXIF_OFFSET_TIME_ORIGINAL] = utc_offset
    if camera is not None:
        exif[EXIF_MAKE], exif[EXIF_MODEL] = camera
    if orientation is not None:
        exif[0x0112] = orientation
    image.save(path, "JPEG", quality=92, exif=exif.tobytes())
    return path


class Visit(NamedTuple):
    folder: str
    seeds: tuple[int, ...]
    light: float
    when: datetime


VISITS = (
    Visit("Old Mill - March 2021", (1, 2, 3, 4, 5), 1.0, datetime(2021, 3, 6, 9)),
    Visit("Old Mill - May 2024", (1, 2, 3), 0.7, datetime(2024, 5, 1, 9)),
    Visit("Harbor - June 2022", (10, 11, 12, 13), 1.1, datetime(2022, 6, 2, 9)),
)


def build_sandbox(target: Path, embedder: str = "tiny") -> Settings:
    """Write a synthetic archive under ``target/archive`` and settings under ``target/home``.

    With the ``tiny`` embedder (no PyTorch), "Describe it" uses it too: SigLIP needs
    PyTorch as well. Otherwise "Describe it" keeps its own default.
    """
    archive = target / "archive"
    for visit in VISITS:
        for i, seed in enumerate(visit.seeds):
            name = f"{visit.folder[:3].lower()}{seed}.jpg"
            image = scene(seed, (800, 600), brightness=visit.light)
            save_jpeg(
                image,
                archive / visit.folder / name,
                captured_at=visit.when + timedelta(hours=i),
                camera=SANDBOX_CAMERA,
            )
    first = VISITS[0]
    for i in range(3):  # a burst: three near-identical frames seconds apart
        save_jpeg(
            scene(1, (800, 600), shift=(i, 0)),
            archive / first.folder / f"burst{i}.jpg",
            captured_at=first.when + timedelta(minutes=30, seconds=2 * i),
            camera=SANDBOX_CAMERA,
        )
    settings = Settings(
        home=target / "home",
        embedder=embedder,
        **({"describe_embedder": "tiny", "view_embedder": "tiny"} if embedder == "tiny" else {}),
        collections=(
            CollectionConfig(
                name="sandbox", root=archive, location_pattern=r"^(?P<location>.+?) - "
            ),
        ),
    )
    save_settings(settings)
    (target / MARKER).write_text("Made by calico sandbox; calico sandbox --force may replace it.\n")
    return settings


def refuse_to_replace(target: Path, home: Path, roots: Sequence[Path]) -> str | None:
    """Why ``calico sandbox --force`` must not delete ``target``, or None if it may.

    It may replace a sandbox it made (marked, or, from before the mark, a home whose only
    collection is the sandbox's own archive) or an empty folder. Never one that holds the
    calico ``home`` in use or a collection ``roots`` of it, links followed, unless that
    home is the sandbox's own: then they are its own synthetic archive and home.
    """
    if not target.is_dir():
        return f"{target} is not a folder, so calico sandbox will not replace it."
    resolved = target.resolve()
    sandbox = (target / MARKER).is_file() or _older_sandbox(target)
    if not (sandbox and home.resolve() == resolved / "home"):
        for kept in (home, *roots):
            if kept.resolve().is_relative_to(resolved):
                return (
                    f"{target} holds {kept}, which is in use (your calico home, or a "
                    "collection), so calico sandbox --force will not delete it. Choose "
                    "another TARGET."
                )
    if sandbox or not any(target.iterdir()):
        return None
    return (
        f"{target} is a folder calico sandbox did not make (it has no {MARKER}), so --force "
        "will not delete it. Choose another TARGET, or remove the folder yourself."
    )


def _older_sandbox(target: Path) -> bool:
    """A sandbox made before sandboxes were marked: its home's one collection is its own
    archive."""
    if not (target / "home" / "calico.toml").is_file():
        return False
    try:
        settings = load_settings(target / "home")
    except CalicoError:
        return False
    roots = [c.root for c in settings.collections]
    return [c.name for c in settings.collections] == ["sandbox"] and roots == [
        (target / "archive").resolve()
    ]
