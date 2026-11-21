from __future__ import annotations

import json
import sys
import time

import catalogue
import numpy as np
from PIL import Image

from calico.catalog.indexer import thumbnail_path
from calico.features.clip import ClipLargeEmbedder
from calico.features.laion import head_score, load_head

VERSION = "l14-openai-from-thumb480@1"
CHUNK = 2048


def main() -> None:
    eng = catalogue.engine()
    cat = catalogue.load(eng)
    out = catalogue.cache_dir() / "thumb_l14"
    out.mkdir(exist_ok=True)
    (out / "version.json").write_text(json.dumps({"version": VERSION, "rows": len(cat.keys)}))
    embedder = ClipLargeEmbedder()
    thumbs = eng.settings.thumbs_dir
    n = len(cat.keys)
    for start in range(0, n, CHUNK):
        part = out / f"{start:06d}.npy"
        if part.exists():
            continue
        t0 = time.time()
        images, ok = [], []
        for key in cat.keys[start : start + CHUNK]:
            path = thumbnail_path(thumbs, str(key))
            if path.is_file():
                with Image.open(path) as im:
                    images.append(im.convert("RGB").copy())
                ok.append(True)
            else:
                ok.append(False)
        vectors = np.full((len(ok), 768), np.nan, dtype=np.float32)
        if images:
            vectors[np.array(ok)] = np.asarray(embedder.extract(images), dtype=np.float32)
        np.save(part, vectors)
        sys.stdout.write(f"{start + len(ok)}/{n} in {time.time() - t0:.0f}s\n")
        sys.stdout.flush()
    parts = sorted(out.glob("[0-9]*.npy"))
    l14 = np.concatenate([np.load(p) for p in parts])
    import torch

    layers = load_head(torch)
    finite = np.isfinite(l14).all(axis=1)
    laion = np.full(n, np.nan, dtype=np.float32)
    unit = l14[finite] / np.linalg.norm(l14[finite], axis=1, keepdims=True)
    laion[finite] = head_score(unit, layers)
    np.save(catalogue.cache_dir() / "l14_thumb.npy", l14)
    np.save(catalogue.cache_dir() / "laion_thumb.npy", laion)
    sys.stdout.write(json.dumps({"rows": n, "scored": int(finite.sum())}) + "\n")


if __name__ == "__main__":
    main()
