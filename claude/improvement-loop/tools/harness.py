from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import hashlib
import json
import os
import random
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import catalogue
import numpy as np
import summarize

from calico.domain import SelectionRequest
from calico.errors import CapabilityError
from calico.models.registry import WithTaste
from calico.models.taste import fit_taste, taste_probability
from calico.selection.context import FeatureView, TaskContext, Trace

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

SEED = 20261113
REPEAT_VIEW = 0.871
NEAR_DUP = 0.95
TIERS = ("dev", "validation", "locked")
TRAINS_WITH = {"dev": ("validation",), "validation": ("dev",), "locked": ("dev", "validation")}
REPORTS = catalogue.LOOP / "reports"
TRACKS = catalogue.LOOP / "tracks"
PRIORITIES = Path(__file__).with_name("priorities.toml")


@dataclass(frozen=True)
class Grid:
    outputs: tuple[int, ...] = (12, 50, 200)
    windows: tuple[int, ...] = (50, 500)
    group: int = 5000
    group_min: int = 2000
    max_share: float = 0.4
    min_labels: int = 8
    folds: int = 5
    draws: int = 20
    whole: bool = True
    swaps: int = 400


@dataclass
class Config:
    track: str
    name: str
    task: str
    template: str
    options: dict[str, Any] = field(default_factory=dict)
    taste: dict[str, Any] = field(default_factory=dict)
    null: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    source: str = ""


def load_config(path: Path) -> Config:
    raw = tomllib.loads(path.read_text())
    return Config(
        track=raw["track"],
        name=raw["name"],
        task=raw["task"],
        template=raw["template"],
        options=raw.get("options", {}),
        taste=raw.get("taste", {}),
        null=raw.get("null", {}),
        models=raw.get("models", {}),
        source=hashlib.sha256(path.read_bytes()).hexdigest()[:12],
    )


def null_configs(champion: Config, sigmas: list[float], seeds: int) -> list[Config]:
    return [
        dataclasses.replace(
            champion, name=f"null-{sigma:g}-{seed}", null={"sigma": sigma, "seed": seed}
        )
        for sigma in sigmas
        for seed in range(1, seeds + 1)
    ]


def progress(text: str) -> None:
    sys.stderr.write(f"{dt.datetime.now():%H:%M:%S} {text}\n")
    sys.stderr.flush()


class Jittered:
    description = "a null: the model's view of strong, moved along a random direction"

    def __init__(self, base: Any, sigma: float, seed: int) -> None:
        self._base = base
        self._sigma = sigma
        self._seed = seed
        self.name = base.name
        self.labels = tuple(base.labels)
        self.version = f"{base.version}+null{seed}@{sigma:g}"

    def predict_proba(self, view: FeatureView) -> np.ndarray:
        proba = np.asarray(self._base.predict_proba(view), dtype=np.float64)
        col = self.labels.index("strong")
        strong = np.clip(proba[:, col], 1e-6, 1 - 1e-6)
        logit = np.log(strong / (1 - strong))
        vectors = np.asarray(view.embeddings, dtype=np.float64)
        direction = np.random.default_rng(SEED + self._seed).standard_normal(vectors.shape[1])
        along = vectors @ direction
        along = (along - along.mean()) / (along.std() or 1.0)
        moved = 1 / (1 + np.exp(-(logit + self._sigma * (logit.std() or 1.0) * along)))
        out = proba * ((1 - moved) / (1 - strong))[:, None]
        out[:, col] = moved
        moved_all: np.ndarray = out.astype(np.float32)
        return moved_all


class Nulled:
    def __init__(self, base: Any, sigma: float, seed: int) -> None:
        self._base = base
        self._sigma = sigma
        self._seed = seed

    def names(self) -> list[str]:
        return list(self._base.names())

    def get(self, name: str) -> Any:
        model = self._base.get(name)
        return Jittered(model, self._sigma, self._seed) if name == "aesthetic" else model

    def versions(self) -> dict[str, str]:
        return dict(self._base.versions())


LAION = "laion_thumb"
"""The view's column of LAION scores, added for a candidate whose aesthetic reads them."""


def standardised(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() or 1.0)


class Blended:
    description = "General's view of strong with LAION's score mixed in"

    def __init__(self, base: Any, share: float) -> None:
        self._base = base
        self._share = share
        self.name = base.name
        self.labels = tuple(base.labels)
        self.version = f"{base.version}+laion@{share:g}"

    def predict_proba(self, view: FeatureView) -> np.ndarray:
        proba = np.asarray(self._base.predict_proba(view), dtype=np.float64)
        col = self.labels.index("strong")
        strong = np.clip(proba[:, col], 1e-6, 1 - 1e-6)
        logit = np.log(strong / (1 - strong))
        laion = np.asarray(view.features[LAION], dtype=np.float64).reshape(len(view))
        laion = np.where(np.isfinite(laion), laion, np.nanmean(laion) if len(laion) else 0.0)
        mixed = (1 - self._share) * standardised(logit) + self._share * standardised(laion)
        moved = 1 / (1 + np.exp(-mixed))
        out = proba * ((1 - moved) / (1 - strong))[:, None]
        out[:, col] = moved
        blended: np.ndarray = out.astype(np.float32)
        return blended


def as_offset(scores: np.ndarray) -> np.ndarray:
    """Scores as the probabilities ``fit_taste(general=)`` takes, whose logit they are."""
    return np.asarray(1 / (1 + np.exp(-np.asarray(scores, dtype=np.float64))))


class OffsetTaste:
    name = "taste"
    description = "Your taste, with LAION's score as an input that is not held back"
    labels: tuple[str, ...] = ("weak", "strong")

    def __init__(self, weights: Any) -> None:
        self.weights = weights
        self.version = f"taste@{weights.version}+laion"

    def predict_proba(self, view: FeatureView) -> NDArray[np.float32]:
        w = self.weights
        x = view.features[w.extractor] if w.extractor else view.embeddings
        laion = np.asarray(view.features[LAION], dtype=np.float64).reshape(len(view))
        strong = taste_probability(w, x, general=as_offset(laion))
        out: NDArray[np.float32] = np.stack([1.0 - strong, strong], axis=1).astype(np.float32)
        return out


class WithAesthetic:
    def __init__(self, base: Any, share: float) -> None:
        self._base = base
        self._share = share

    def names(self) -> list[str]:
        return list(self._base.names())

    def get(self, name: str) -> Any:
        model = self._base.get(name)
        return Blended(model, self._share) if name == "aesthetic" else model

    def versions(self) -> dict[str, str]:
        versions = dict(self._base.versions())
        versions["aesthetic"] = f"{versions.get('aesthetic', '')}+laion@{self._share:g}"
        return versions


def low_priority_words() -> tuple[str, ...]:
    return tuple(tomllib.loads(PRIORITIES.read_text())["low_priority"])


def is_low_priority(place: str, words: tuple[str, ...]) -> bool:
    return any(w.casefold() in place.casefold() for w in words)


class Labels:
    def __init__(
        self,
        splits: Path,
        low_priority: tuple[str, ...] = (),
        place_of: Callable[[str], str] = lambda key: "",
    ) -> None:
        self.splits = splits
        self.low_priority = low_priority
        self.place_of = place_of
        self.dropped: dict[str, int] = {}

    def read(self, tier: str, name: str) -> dict[str, Any]:
        path = self.splits / tier / f"{name}.json"
        try:
            data = dict(json.loads(path.read_text()))
        except PermissionError:
            raise SystemExit(
                f"{path.name} of the {tier} tier is closed here: it opens only through "
                "`loop look` / `loop locked`"
            ) from None
        columns = data.get("columns")
        for name_, rows in data.items():
            if name_ != "columns" and isinstance(rows, list):
                data[name_] = [r for r in rows if not self._low(r, columns)]
                if len(data[name_]) < len(rows):
                    self.dropped[f"{tier}/{name}:{name_}"] = len(rows) - len(data[name_])
        return data

    def _low(self, row: Any, columns: list[str] | None) -> bool:
        if isinstance(row, dict):
            if "place" in row:
                places = [str(row["place"])]
            else:
                places = [self.place_of(str(row[k])) for k in ("key", "original") if k in row]
        elif isinstance(row, list) and columns and "place" in columns:
            places = [str(row[columns.index("place")])]
        elif isinstance(row, list):
            places = [self.place_of(str(k)) for k in row]
        else:
            places = []
        return any(is_low_priority(p, self.low_priority) for p in places)

    def ratings(self, tier: str) -> list[dict[str, Any]]:
        return list(self.read(tier, "strongest-yours")["ratings"])


class World:
    def __init__(
        self,
        eng: Any,
        cat: catalogue.Catalogue,
        splits: Path,
        cache: Path,
        grid: Grid,
        low_priority: tuple[str, ...] = (),
    ) -> None:
        self.eng = eng
        self.cat = cat
        self.grid = grid
        self.cache = cache
        places = json.loads((splits / "places.json").read_text())
        self.tier_of: dict[str, str] = places["places"]
        self.low_priority = low_priority
        self.low_places = frozenset(p for p in self.tier_of if is_low_priority(p, low_priority))
        self.labels = Labels(
            splits,
            low_priority,
            lambda key: str(cat.location[cat.row[key]]) if key in cat.row else "",
        )
        left_out = set(places["excluded_photos"]) | set(places["derived_photos"])
        usable = np.array(
            [
                self.tier_of.get(str(p), "reserved") != "reserved" and str(k) not in left_out
                for k, p in zip(cat.keys, cat.location, strict=True)
            ]
        )
        self.usable = usable
        self.low_rows = np.array([str(p) in self.low_places for p in cat.location])
        lists: dict[str, list[int]] = {}
        order = np.lexsort((cat.keys, np.nan_to_num(cat.captured, nan=np.inf), cat.location))
        for i in order:
            if usable[i]:
                lists.setdefault(str(cat.location[i]), []).append(int(i))
        self.by_place = {p: np.asarray(v) for p, v in lists.items()}
        self.roots = tuple(c.root for c in eng.settings.collections)
        self.photo_of = {p.content_key: p for p in eng.gather(self.roots, Trace())}
        laion = cache / "laion_thumb.npy"
        self.laion = np.load(laion) if laion.exists() else None
        self.kinds = self._kinds()
        self._viewed: tuple[Any, Any] | None = None

    def _kinds(self) -> dict[str, np.ndarray]:
        path = self.cache / "kinds.npz"
        if path.exists():
            data = np.load(path)
            return {k: data[k] for k in data.files}
        view = FeatureView(
            embeddings=self.cat.clip, features={"technical": self.cat.technical}, columns={}
        )
        out = {}
        try:
            for model, wanted in (("scale", ("close", "macro")), ("shot_kind", ("portrait",))):
                m = self.eng.models.get(model)
                top = np.asarray(m.labels)[np.argmax(m.predict_proba(view), axis=1)]
                out["close_up" if model == "scale" else "portrait"] = np.isin(top, wanted)
        except CapabilityError:
            return {}
        np.savez(path, **out)  # type: ignore[arg-type]
        return out

    def keys(self, rows: np.ndarray) -> list[str]:
        return [str(self.cat.keys[i]) for i in rows]

    def features(self, rows: np.ndarray) -> tuple[list[Any], FeatureView]:
        mark = (len(rows), hashlib.sha256(np.asarray(rows).tobytes()).hexdigest())
        if self._viewed is None or self._viewed[0] != mark:
            photos = [self.photo_of[k] for k in self.keys(rows) if k in self.photo_of]
            self._viewed = (mark, self.eng.feature_view(photos, Trace()))
        kept, view = self._viewed[1]
        return kept, view


def open_world(grid: Grid) -> World:
    eng = catalogue.engine()
    splits = Path(os.environ.get("LOOP_SPLITS", catalogue.LOOP / "splits"))
    return World(
        eng, catalogue.load(eng), splits, catalogue.cache_dir(), grid, low_priority_words()
    )


def fold_map(tier: str, places: list[str], folds: int) -> dict[str, int]:
    if tier == "locked":
        return {}
    order = sorted(set(places))
    random.Random(SEED).shuffle(order)  # noqa: S311
    return {p: i % folds for i, p in enumerate(order)}


class Fits:
    def __init__(self, world: World, config: Config, tier: str, folds: dict[str, int]) -> None:
        self.world = world
        self.config = config
        self.folds = folds
        self.locked = tier == "locked"
        self.base: list[dict[str, Any]] = []
        self.own: list[dict[str, Any]] = []
        if config.template == "yours":
            for other in TRAINS_WITH[tier]:
                self.base += world.labels.ratings(other)
            if not self.locked:
                self.own = world.labels.ratings(tier)
        self._made: dict[frozenset[int], Any] = {}

    def fold(self, place: str) -> int:
        return -1 if self.locked else self.folds.get(place, 0)

    def without(self, folds: set[int]) -> Any:
        if self.config.template != "yours":
            return None
        mark = frozenset(folds)
        if mark not in self._made:
            train = self.base + [r for r in self.own if self.fold(r["place"]) not in mark]
            self._made[mark] = taste_for(self.world, self.config, train)
        return self._made[mark]


def taste_vectors(world: World, embedder: str) -> tuple[np.ndarray, str]:
    """The catalogue's vectors for ``[taste] embedder``, and the ``FeatureView.features``
    entry a fit on them reads ("" = the main embedding)."""
    eng, cat = world.eng, world.cat
    if embedder in ("", eng.settings.embedder):
        return cat.clip, ""
    if embedder == eng.view_feature():
        return cat.dino, embedder
    if embedder == "siglip" and cat.siglip.shape[1]:
        return cat.siglip, embedder
    have = [eng.settings.embedder, eng.view_feature(), "siglip" if cat.siglip.shape[1] else ""]
    raise SystemExit(
        f"[taste] embedder = {embedder!r}: the catalogue has no such vectors; it has "
        f"{', '.join(h for h in have if h)}"
    )


def laion_scores(world: World, config: Config, what: str) -> np.ndarray:
    if world.laion is None:
        raise SystemExit(
            f"{config.name}: {what} reads LAION, but the cache has no LAION scores "
            "(laion_thumb.npy)"
        )
    scores: np.ndarray = world.laion
    return scores


def taste_for(world: World, config: Config, ratings: list[dict[str, Any]]) -> Any:
    cat = world.cat
    vectors, extractor = taste_vectors(world, str(config.taste.get("embedder", "")))
    rows = [r for r in ratings if r["key"] in cat.row]
    at = [cat.row[r["key"]] for r in rows]
    x = vectors[at]
    y = np.array([int(r["liked"]) for r in rows])
    offset = config.taste.get("offset")
    general = None
    if offset is not None:
        if offset != "laion":
            raise SystemExit(f"{config.name}: [taste] offset = {offset!r}; only 'laion' is carried")
        general = as_offset(laion_scores(world, config, "[taste] offset")[at])
    weights = fit_taste(
        x,
        y,
        feature=world.eng.taste_feature(extractor),
        extractor=extractor,
        l2=float(config.taste.get("l2", 1000.0)),
        general=general,
    )
    return dataclasses.replace(weights, places=tuple(sorted({r["place"] for r in rows})))


def window(rows: np.ndarray, size: int, labelled: np.ndarray, least: int) -> np.ndarray | None:
    if len(rows) < size * 2:
        return None
    hits = np.convolve(labelled[rows].astype(int), np.ones(size, dtype=int), mode="valid")
    start = int(np.argmax(hits))
    return rows[start : start + size] if hits[start] >= least else None


def pack(places: list[str], world: World, seed: int) -> list[list[str]]:
    order = sorted(places)
    random.Random(seed).shuffle(order)  # noqa: S311
    groups: list[list[str]] = []
    size = 0
    for p in order:
        n = len(world.by_place.get(p, ()))
        if not groups or size + n > world.grid.group * 1.2:
            groups.append([])
            size = 0
        groups[-1].append(p)
        size += n
    return groups


def same_subject(cat: catalogue.Catalogue, a: int, b: int) -> bool:
    return bool(
        a == b
        or float(cat.clip[a] @ cat.clip[b]) >= NEAR_DUP
        or (cat.location[a] == cat.location[b] and float(cat.dino[a] @ cat.dino[b]) >= REPEAT_VIEW)
    )


def repeat_share(world: World, rows: list[int]) -> float:
    flags = [
        any(same_subject(world.cat, i, j) for j in rows[:n] if j != i) for n, i in enumerate(rows)
    ]
    return float(np.mean(flags)) if flags else 0.0


def random_repeat(world: World, pool: np.ndarray, k: int, rng: np.random.Generator) -> float:
    draws = [
        repeat_share(
            world, [int(i) for i in rng.choice(pool, size=min(k, len(pool)), replace=False)]
        )
        for _ in range(world.grid.draws)
    ]
    return float(np.mean(draws))


def auc(pos: list[float] | np.ndarray, neg: list[float] | np.ndarray) -> tuple[float, float]:
    p, n = np.asarray(pos, dtype=np.float64), np.sort(np.asarray(neg, dtype=np.float64))
    if not len(p) or not len(n):
        return 0.0, 0.0
    below = np.searchsorted(n, p, side="left")
    ties = np.searchsorted(n, p, side="right") - below
    return float(below.sum() + 0.5 * ties.sum()), float(len(p) * len(n))


def burst_mates(world: World, is_kept: np.ndarray, seconds: float, similarity: float) -> np.ndarray:
    cat = world.cat
    out = np.zeros(len(cat.keys), dtype=bool)
    for rows in world.by_place.values():
        kept = rows[is_kept[rows]]
        if not len(kept):
            continue
        for start in range(0, len(rows), 4096):
            part = rows[start : start + 4096]
            close = np.abs(cat.captured[part][:, None] - cat.captured[kept][None, :]) <= seconds
            alike = cat.clip[part] @ cat.clip[kept].T >= similarity
            out[part] = (close & alike).any(axis=1)
    mates: np.ndarray = out & ~is_kept
    return mates


class Runner:
    def __init__(self, world: World, config: Config) -> None:
        self.world = world
        self.config = config
        self.task = world.eng.task(config.task)
        self.options = self.task.parse_options(dict(config.options))

    def with_laion(self, kept: list[Any], view: FeatureView, what: str) -> FeatureView:
        scores = laion_scores(self.world, self.config, what)
        rows = [self.world.cat.row[p.content_key] for p in kept]
        return FeatureView(view.embeddings, {**view.features, LAION: scores[rows]}, view.columns)

    def context(self, rows: np.ndarray, limit: int, taste: Any) -> TaskContext:
        eng = self.world.eng
        trace = Trace()
        kept, view = self.world.features(rows)
        offset = taste is not None and bool(taste.general_coef)
        asked = dataclasses.replace(taste, general_coef=0.0) if offset else taste
        which, models = eng.template(self.config.template, trace, taste=asked)
        if which != self.config.template:
            raise SystemExit(f"{self.config.name}: asked for {self.config.template}, ran {which}")
        if offset:
            view = self.with_laion(kept, view, "[taste] offset")
            models = WithTaste(eng.models, OffsetTaste(taste))
        if self.config.models:
            aesthetic = self.config.models.get("aesthetic")
            if aesthetic != "laion":
                raise SystemExit(
                    f"{self.config.name}: [models] aesthetic = {aesthetic!r}; "
                    "only 'laion' is carried"
                )
            if LAION not in view.features:
                view = self.with_laion(kept, view, "[models] aesthetic")
            models = WithAesthetic(models, float(self.config.models.get("share", 1.0)))
        if self.config.null:
            models = Nulled(models, float(self.config.null["sigma"]), int(self.config.null["seed"]))
        return TaskContext(
            request=SelectionRequest(
                task=self.config.task, folders=self.world.roots, limit=limit, template=which
            ),
            photos=tuple(kept),
            view=view,
            models=models,
            trace=trace,
            limit=limit,
            view_feature=eng.view_feature(),
        )

    def strongest(
        self, rows: np.ndarray, k: int, taste: Any
    ) -> tuple[list[int], dict[int, float], float, float]:
        ctx = self.context(rows, k, taste)
        t0 = time.time()
        ctx = self.world.eng.compare_parts(self.task, self.options, ctx)
        t1 = time.time()
        groups = self.task.run(ctx, self.options)
        t2 = time.time()
        row = self.world.cat.row
        picks = [row[p.content_key] for g in groups for p in g.picks]
        score = ctx.trace.columns.get("score")
        scores = (
            {
                row[p.content_key]: float(score[i])
                for i, p in enumerate(ctx.photos)
                if np.isfinite(score[i])
            }
            if score is not None
            else {}
        )
        return picks, scores, t2 - t1, t1 - t0

    def stories(self, rows: np.ndarray, k: int, taste: Any) -> tuple[list[list[int]], float]:
        ctx = self.context(rows, k, taste)
        t0 = time.time()
        groups = self.task.run(ctx, self.options)
        seconds = time.time() - t0
        row = self.world.cat.row
        return [[row[p.content_key] for p in g.picks] for g in groups], seconds

    def scorer(self, rows: np.ndarray, include: list[str], taste: Any) -> Any:
        ctx = self.context(rows, 6, taste)
        return self.task.scorer(ctx, self.options, include=include)


def strongest_eval(world: World, configs: list[Config], tier: str) -> dict[str, Any]:
    grid, cat = world.grid, world.cat
    ratings = world.labels.ratings(tier)
    kept = world.labels.read(tier, "strongest-general")["kept"]
    count = len(cat.keys)
    rated, good, is_kept = (np.zeros(count, dtype=bool) for _ in range(3))
    for r in ratings:
        if r["key"] in cat.row:
            rated[cat.row[r["key"]]] = True
            good[cat.row[r["key"]]] = bool(r["liked"])
    for k_ in kept:
        if k_["original"] in cat.row:
            is_kept[cat.row[k_["original"]]] = True
    burst = world.eng.task("best").parse_options({})
    mates = burst_mates(world, is_kept, burst.burst_window_seconds, burst.burst_similarity)
    credited = is_kept | mates
    labelled = rated | is_kept
    tier_places = sorted(
        p
        for p, t in world.tier_of.items()
        if t == tier
        and p in world.by_place
        and p not in world.low_places
        and labelled[world.by_place[p]].sum() >= grid.min_labels
    )
    folds = fold_map(tier, sorted({*tier_places, *(r["place"] for r in ratings)}), grid.folds)
    fits = {c.name: Fits(world, c, tier, folds) for c in configs}
    fold = fits[configs[0].name].fold

    pools: list[tuple[str, str, np.ndarray, int]] = []
    for p in tier_places:
        rows = world.by_place[p]
        pools.append(("place", p, rows, fold(p)))
        for size in grid.windows:
            w = window(rows, size, labelled, grid.min_labels)
            if w is not None:
                pools.append((str(size), p, w, fold(p)))
    by_fold: dict[int, list[str]] = collections.defaultdict(list)
    for p in tier_places:
        by_fold[fold(p)].append(p)
    for f, ps in sorted(by_fold.items()):
        for n, group in enumerate(pack(ps, world, SEED + f)):
            rows = np.concatenate([world.by_place[p] for p in group])
            if len(rows) >= grid.group_min:
                pools.append((str(grid.group), f"fold{f}#{n}", rows, f))
    if grid.whole and by_fold:
        pools.append(("whole", "whole", np.flatnonzero(world.usable), min(by_fold)))

    rng = np.random.default_rng(SEED)
    cells: dict[str, dict[str, list[dict[str, Any]]]] = {
        c.name: collections.defaultdict(list) for c in configs
    }
    runners = {c.name: Runner(world, c) for c in configs}
    for n, (kind, unit, rows, f) in enumerate(pools):
        progress(f"pool {n + 1}/{len(pools)}: {kind}, {len(rows)} photos, fold {f}")
        judged = {
            int(i) for i in rows if labelled[i] and (f == -1 or fold(str(cat.location[i])) == f)
        }
        if not judged:
            continue
        base = float(is_kept[rows].mean())
        others = rows[~credited[rows]]
        first = True
        for k in grid.outputs:
            if k > grid.max_share * len(rows):
                continue
            chance = random_repeat(world, rows, k, rng)
            for c in configs:
                picks, scores, seconds, parts = runners[c.name].strongest(
                    rows, k, fits[c.name].without({f})
                )
                by_place: dict[str, dict[str, float]] = collections.defaultdict(
                    lambda: collections.defaultdict(float)
                )
                for i in picks:
                    if i not in judged:
                        continue
                    place = by_place[str(cat.location[i])]
                    if rated[i]:
                        place["rated"] += 1
                        place["liked"] += good[i]
                    place["kept"] += is_kept[i]
                for i in judged:
                    by_place[str(cat.location[i])]["kept_in_pool"] += is_kept[i]
                laion = None
                if world.laion is not None and picks:
                    vals = world.laion[rows]
                    vals = vals[np.isfinite(vals)]
                    pct = [
                        float((vals < world.laion[r]).mean())
                        for r in picks
                        if np.isfinite(world.laion[r])
                    ]
                    laion = float(np.mean(pct)) if pct else None
                aucs = {}
                mine = [i for i in judged if i in scores and rated[i]]
                for name, mask in (("all", None), *world.kinds.items()):
                    chosen = [i for i in mine if mask is None or mask[i]]
                    aucs[name] = auc(
                        [scores[i] for i in chosen if good[i]],
                        [scores[i] for i in chosen if not good[i]],
                    )
                entry: dict[str, Any] = {
                    "unit": unit,
                    "pool": len(rows),
                    "fold": f,
                    "places": {p: dict(v) for p, v in by_place.items()},
                    "picks": len(picks),
                    "pick_rows": picks,
                    "pick_rated": [bool(rated[i]) for i in picks],
                    "kept_base": base,
                    "kept_picks": int(credited[picks].sum()) if picks else 0,
                    "laion_pct": laion,
                    "low_priority_pool": float(world.low_rows[rows].mean()),
                    "low_priority_picks": int(world.low_rows[picks].sum()) if picks else 0,
                    "repeat": repeat_share(world, picks),
                    "repeat_random": chance,
                    "auc": aucs,
                    "seconds": round(seconds, 2),
                    "seconds_parts": round(parts, 2),
                }
                if kind == "place" and first:
                    entry["kept_auc"] = auc(
                        [scores[i] for i in rows[is_kept[rows]] if i in scores],
                        [scores[i] for i in others if i in scores],
                    )
                cells[c.name][f"{kind}@{k}"].append(entry)
            first = False
    return {name: dict(v) for name, v in cells.items()}


def unfit_located(scorer: Any, trio: list[int], marks: list[int], swaps: int) -> bool:
    base = scorer.score(*trio)
    beaten = []
    for side in range(3):
        better = []
        for r in np.flatnonzero(scorer.shortlisted)[:swaps]:
            if int(r) in trio:
                continue
            t = list(trio)
            t[side] = int(r)
            if scorer.allowed(*t):
                better.append(scorer.score(*t) > base)
        beaten.append(float(np.mean(better)) if better else 0.0)
    if len(marks) == 1:
        return int(np.argmax(beaten)) in marks
    return int(np.argmin(beaten)) not in marks


def story_eval(world: World, configs: list[Config], tier: str) -> dict[str, Any]:
    from calico.sessions import _finished_story

    grid, cat = world.grid, world.cat
    stories = world.labels.read(tier, "story-yours")

    def place_of(key: str) -> str:
        return str(cat.location[cat.row[key]])

    story_places = {s["place"] for s in stories["stories"]} | {
        place_of(k) for f in stories["finished"] for k in f
    }
    tier_places = sorted(
        p for p in story_places if p in world.by_place and p not in world.low_places
    )
    yours = any(c.template == "yours" for c in configs)
    rated_places = {r["place"] for r in world.labels.ratings(tier)} if yours else set()
    folds = fold_map(tier, sorted({*tier_places, *rated_places}), grid.folds)

    sets_grid: list[tuple[str, str, np.ndarray, str]] = []
    rng = np.random.default_rng(SEED)
    for p in tier_places:
        rows = world.by_place[p]
        sets_grid.append(("place", p, rows, p))
        for size in grid.windows:
            if len(rows) >= size * 2:
                start = int(rng.integers(0, len(rows) - size))
                sets_grid.append((str(size), p, rows[start : start + size], p))
    for n, group in enumerate(pack(tier_places, world, SEED)):
        rows = np.concatenate([world.by_place[p] for p in group])
        if len(rows) >= grid.group_min:
            sets_grid.append((str(grid.group), f"group#{n}", rows, group[0]))
    if grid.whole and tier_places:
        sets_grid.append(("whole", "whole", np.flatnonzero(world.usable), tier_places[0]))
    chance: dict[tuple[str, str, int], dict[str, float]] = {}

    out: dict[str, Any] = {}
    for c in configs:
        fits = Fits(world, c, tier, folds)
        runner = Runner(world, c)
        sessions: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for s in stories["stories"]:
            sessions[s["session"]].append(s)
        per_session = {}
        for sid, rated in sorted(sessions.items()):
            progress(f"{c.name}: session {sid}, {len(rated)} stories")
            places = sorted({place_of(k) for s in rated for k in s["keys"] if k in cat.row})
            rows = np.concatenate([world.by_place[p] for p in places if p in world.by_place])
            include = sorted({k for s in rated for k in s["keys"]})
            scorer = runner.scorer(rows, include, fits.without({fits.fold(p) for p in places}))
            pos: list[float] = []
            neg: list[float] = []
            hits, tried = 0, 0
            for s in rated:
                if not all(k in scorer.keys for k in s["keys"]):
                    continue
                trio = [scorer.row(k) for k in s["keys"]]
                (pos if s["works"] else neg).append(scorer.score(*trio))
                marks = [n for n, k in enumerate(s["keys"]) if k in set(s["unfit"])]
                if len(marks) in (1, 2):
                    tried += 1
                    hits += unfit_located(scorer, trio, marks, grid.swaps)
            concordant, pairs = auc(pos, neg)
            per_session[sid] = {
                "works": len(pos),
                "doesnt": len(neg),
                "concordant": concordant,
                "pairs": pairs,
                "unfit_hits": hits,
                "unfit_n": tried,
            }
        finished = []
        for keys in stories["finished"]:
            places = sorted({place_of(k) for k in keys})
            rows = np.concatenate([world.by_place[p] for p in places if p in world.by_place])
            rows = np.union1d(rows, [cat.row[k] for k in keys])
            ctx = runner.context(
                rows, runner.task.default_limit, fits.without({fits.fold(p) for p in places})
            )
            photos = [world.photo_of[k] for k in keys]
            fs = _finished_story(runner.task, ctx, runner.options, "-".join(keys)[:20], photos)
            swaps = [s for side in fs.swaps for s in side]
            if swaps:
                finished.append(float(np.mean([fs.score > s for s in swaps])))
        sets = []
        for kind, unit, rows, p in sets_grid:
            progress(f"{c.name}: made stories over {kind}, {len(rows)} photos")
            for k in grid.outputs:
                if 3 * k > grid.max_share * len(rows):
                    continue
                made, seconds = runner.stories(rows, k, fits.without({fits.fold(p)}))
                mark = (kind, unit, k)
                if mark not in chance:
                    chance[mark] = random_sets(world, rows, k, rng)
                sets.append(
                    {
                        "pool": kind,
                        "unit": unit,
                        "size": len(rows),
                        "k": k,
                        "seconds": round(seconds, 2),
                        "stories": made,
                        "low_priority_pool": float(world.low_rows[rows].mean()),
                        "low_priority_photos": int(
                            sum(world.low_rows[i] for story in made for i in story)
                        ),
                        "result": set_measures(world, made),
                        "random": chance[mark],
                    }
                )
        pairs = sum(v["pairs"] for v in per_session.values())
        marked = sum(v["unfit_n"] for v in per_session.values())
        out[c.name] = {
            "rated_auc": sum(v["concordant"] for v in per_session.values()) / pairs
            if pairs
            else None,
            "rated_pairs": pairs,
            "sessions": per_session,
            "unfit_located": sum(v["unfit_hits"] for v in per_session.values()) / marked
            if marked
            else None,
            "unfit_stories": marked,
            "finished_beats_swaps": float(np.mean(finished)) if finished else None,
            "finished": finished,
            "sets": sets,
        }
    return out


def set_measures(world: World, stories: list[list[int]]) -> dict[str, float]:
    cat = world.cat
    n = len(stories)
    if not n:
        return {}

    def same(a: int, b: int) -> bool:
        return same_subject(cat, a, b)

    reuse = anchor = pair = within = 0
    siglip = []
    for i, s in enumerate(stories):
        earlier = [x for t in stories[:i] for x in t]
        reuse += any(same(x, y) for x in s for y in earlier)
        anchor += any(same(t[1], s[1]) for t in stories[:i])
        pairs = [(s[0], s[1]), (s[1], s[2]), (s[0], s[2])]
        pair += any(
            (same(a, c) and same(b, d)) or (same(a, d) and same(b, c))
            for t in stories[:i]
            for a, b in pairs
            for c, d in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])]
        )
        within += any(same(a, b) for a, b in pairs)
        if cat.siglip.shape[1]:
            v = cat.siglip[s]
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            siglip.append(float(np.mean([v[0] @ v[1], v[1] @ v[2], v[0] @ v[2]])))
    return {
        "stories": n,
        "reuse": reuse / n,
        "anchor_repeat": anchor / n,
        "pair_repeat": pair / n,
        "within": within / n,
        "siglip": float(np.mean(siglip)) if siglip else 0.0,
    }


def random_sets(
    world: World, rows: np.ndarray, k: int, rng: np.random.Generator
) -> dict[str, float]:
    m = min(k * 3, len(rows) - len(rows) % 3)
    if m < 3:
        return {}
    draws = [
        set_measures(
            world,
            [[int(i) for i in x] for x in rng.choice(rows, size=m, replace=False).reshape(-1, 3)],
        )
        for _ in range(world.grid.draws)
    ]
    return {key: float(np.mean([d[key] for d in draws])) for key in draws[0]}


def evaluate(world: World, track: str, configs: list[Config], tier: str) -> dict[str, Any]:
    if any(c.track != track for c in configs):
        raise SystemExit("every config must be for the track evaluated")
    if len({c.name for c in configs}) != len(configs):
        raise SystemExit("two configs share a name")
    t0 = time.time()
    if track.startswith("strongest"):
        results = strongest_eval(world, configs, tier)
    else:
        results = story_eval(world, configs, tier)
    return {
        "track": track,
        "tier": tier,
        "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "snapshot": world.cat.snapshot,
        "head": os.environ.get("LOOP_HEAD", ""),
        "dirty": os.environ.get("LOOP_DIRTY", ""),
        "tools_sha256": os.environ.get("LOOP_TOOLS_SHA256", ""),
        "grid": dataclasses.asdict(world.grid),
        "low_priority": {
            "words": list(world.low_priority),
            "places": sorted(world.low_places),
            "labels_dropped": dict(sorted(world.labels.dropped.items())),
        },
        "champion": configs[0].name,
        "configs": {c.name: {"sha": c.source, "null": c.null} for c in configs},
        "seconds": round(time.time() - t0),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("track")
    parser.add_argument("--tier", default="dev", choices=TIERS)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--nulls", default="")
    parser.add_argument("--null-seeds", type=int, default=2)
    parser.add_argument("--out")
    args = parser.parse_args()
    champion = TRACKS / args.track / "champion.toml"
    paths = [champion] + [Path(p) for p in args.config if Path(p).resolve() != champion.resolve()]
    configs = [load_config(p) for p in paths]
    if args.nulls:
        configs += null_configs(
            configs[0], [float(s) for s in args.nulls.split(",")], args.null_seeds
        )
    world = open_world(Grid(whole=not args.quick))
    report = evaluate(world, args.track, configs, args.tier)
    report["quick"] = args.quick
    now = dt.datetime.now(dt.UTC)
    name = "-".join(c.name for c in configs if not c.null) + ("-nulls" if args.nulls else "")
    out = (
        Path(args.out)
        if args.out
        else REPORTS / args.track / args.tier / f"{now:%Y-%m-%d-%H%M}-{name}.json.gz"
    )
    summarize.write_report(out, report)
    sys.stdout.write(f"{out}\n")


if __name__ == "__main__":
    main()
