from __future__ import annotations

import json
import os
import re
import sqlite3
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import platformdirs

if TYPE_CHECKING:
    from calico.engine import Engine

REPO = Path(__file__).resolve().parents[3]
LOOP = REPO / "claude" / "improvement-loop"
RESERVED_KEYS = REPO / "out" / "2026-11-12-judge-new-eight" / "keys.json"
CURATION = re.compile(
    r"^(edits|edited|lightroom.*|(lightroom )?reedit|prints.*|to print|the best)$", re.IGNORECASE
)


@dataclass
class Catalogue:
    snapshot: str
    keys: np.ndarray
    location: np.ndarray
    collection: np.ndarray
    captured: np.ndarray
    camera: np.ndarray
    clip: np.ndarray
    dino: np.ndarray
    siglip: np.ndarray
    technical: np.ndarray
    technical_columns: tuple[str, ...]
    curated: np.ndarray

    def __post_init__(self) -> None:
        self.row = {str(k): i for i, k in enumerate(self.keys)}


def engine() -> Engine:
    from calico.engine import Engine

    eng = Engine.open()
    home = eng.settings.home.resolve()
    if home == platformdirs.user_data_path("calico").resolve():
        raise SystemExit(f"refusing the real calico home: {home}")
    return eng


def snapshot_name() -> str:
    name = os.environ.get("LOOP_SNAPSHOT")
    if not name:
        raise SystemExit("run through `claude/improvement-loop/loop py`, which sets the copy")
    return name


def cache_dir() -> Path:
    given = os.environ.get("LOOP_CACHE")
    path = Path(given) if given else REPO / ".loop" / "cache" / snapshot_name()
    path.mkdir(parents=True, exist_ok=True)
    return path


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def curated_paths(home: Path) -> dict[str, list[str]]:
    con = sqlite3.connect(f"file:{home}/catalog.sqlite?mode=ro", uri=True)
    out: dict[str, list[str]] = {}
    for path, key in con.execute("select path, content_key from photos"):
        folders = Path(path).parts[:-1]
        if any(CURATION.match(f) for f in folders):
            out.setdefault(key, []).append(nfc(path))
    return out


def all_paths(home: Path) -> dict[str, str]:
    con = sqlite3.connect(f"file:{home}/catalog.sqlite?mode=ro", uri=True)
    return {nfc(p): k for p, k in con.execute("select path, content_key from photos")}


def build(eng: Engine) -> Catalogue:
    from calico.selection.context import Trace

    roots = tuple(c.root for c in eng.settings.collections)
    photos, view = eng.feature_view(eng.gather(roots, Trace()), Trace())
    feats = view.features
    dino_name = eng.view_feature()
    siglip = next((k for k in feats if k.startswith("siglip")), None)
    curated = curated_paths(eng.settings.home)
    captured = np.array(
        [p.captured_at.timestamp() if p.captured_at else np.nan for p in photos], dtype=np.float64
    )
    return Catalogue(
        snapshot=snapshot_name(),
        keys=np.array([p.content_key for p in photos]),
        location=np.array([p.location for p in photos]),
        collection=np.array([p.collection for p in photos]),
        captured=captured,
        camera=np.array([p.camera or "" for p in photos]),
        clip=np.asarray(view.embeddings, dtype=np.float32),
        dino=np.asarray(feats[dino_name] if dino_name else view.embeddings, dtype=np.float32),
        siglip=np.asarray(feats[siglip], dtype=np.float32)
        if siglip
        else np.zeros((len(photos), 0), np.float32),
        technical=np.asarray(feats["technical"], dtype=np.float32),
        technical_columns=tuple(view.columns.get("technical", ())),
        curated=np.array([p.content_key in curated for p in photos]),
    )


FIELDS = (
    "keys",
    "location",
    "collection",
    "captured",
    "camera",
    "clip",
    "dino",
    "siglip",
    "technical",
    "curated",
)


def load(eng: Engine | None = None) -> Catalogue:
    path = cache_dir() / "catalogue.npz"
    meta = cache_dir() / "catalogue.json"
    if path.exists() and meta.exists():
        data = np.load(path, allow_pickle=False)
        info = json.loads(meta.read_text())
        return Catalogue(
            snapshot=snapshot_name(),
            technical_columns=tuple(info["technical_columns"]),
            **{f: data[f] for f in FIELDS},
        )
    cat = build(eng or engine())
    np.savez(path, **{f: getattr(cat, f) for f in FIELDS})
    meta.write_text(json.dumps({"technical_columns": list(cat.technical_columns)}))
    return cat


def reserved_keys() -> set[str]:
    return set(json.loads(RESERVED_KEYS.read_text()))


def holdout(home: Path) -> dict[str, Any]:
    path = home / "holdout.toml"
    return tomllib.loads(path.read_text()) if path.exists() else {}


if __name__ == "__main__":
    import sys
    import time

    t0 = time.time()
    cat = load()
    sys.stdout.write(
        json.dumps(
            {
                "snapshot": cat.snapshot,
                "photos": len(cat.keys),
                "locations": len(set(cat.location.tolist())),
                "curated": int(cat.curated.sum()),
                "curated_locations": len(set(cat.location[cat.curated].tolist())),
                "siglip_dim": cat.siglip.shape[1],
                "seconds": round(time.time() - t0, 1),
            }
        )
        + "\n"
    )
