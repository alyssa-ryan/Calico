from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

SEED = 20261113
BOOT = 2000
MIN_JUDGED = 20
MIN_PAIRS = 50
SET_PARTS = ("reuse", "anchor_repeat", "pair_repeat")
PRIMARY = {
    "strongest-yours": "precision",
    "strongest-general": "kept_auc",
    "story-yours": "rated_auc",
    "story-general": "unfit_located",
}

Pick = dict[str, int] | None
Results = dict[str, Any]


@dataclass(frozen=True)
class Measure:
    units: tuple[str, ...]
    fn: Callable[[Pick], float | None]


def read_report(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    text = gzip.decompress(raw) if path.suffix == ".gz" else raw
    return dict(json.loads(text))


def write_report(path: Path, report: dict[str, Any]) -> None:
    text = (json.dumps(report, default=float) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(text, mtime=0) if path.suffix == ".gz" else text)


def times(pick: Pick, unit: str) -> float:
    return 1.0 if pick is None else float(pick.get(unit, 0))


def spread(pick: Pick, units: list[str]) -> float:
    if pick is None or not units:
        return 1.0
    return sum(pick.get(u, 0) for u in units) / len(units)


def graded(cell: str) -> bool:
    return not cell.startswith("whole")


def judged_cells(res: Results) -> set[str]:
    out = set()
    for cell, pools in res.items():
        rated = sum(v.get("rated", 0) for p in pools for v in p["places"].values())
        if graded(cell) and rated >= MIN_JUDGED:
            out.add(cell)
    return out


def ratio(parts: list[tuple[str, float, float]]) -> Callable[[Pick], float | None]:
    def fn(pick: Pick) -> float | None:
        top = sum(times(pick, u) * a for u, a, _ in parts)
        bottom = sum(times(pick, u) * b for u, _, b in parts)
        return top / bottom if bottom else None

    return fn


def mean_of(fns: list[Callable[[Pick], float | None]]) -> Callable[[Pick], float | None]:
    def fn(pick: Pick) -> float | None:
        vals = [v for f in fns if (v := f(pick)) is not None]
        return float(np.mean(vals)) if vals else None

    return fn


def weighted(parts: list[tuple[list[str], float]]) -> Callable[[Pick], float | None]:
    def fn(pick: Pick) -> float | None:
        total = sum(spread(pick, units) for units, _ in parts)
        if not total:
            return None
        return sum(spread(pick, units) * v for units, v in parts) / total

    return fn


def lift(pool: dict[str, Any]) -> float | None:
    if not pool["picks"] or not pool["kept_base"]:
        return None
    return float((pool["kept_picks"] / pool["picks"]) / pool["kept_base"])


def strongest_measures(res: Results, cells: set[str]) -> dict[str, Measure]:
    places = tuple(sorted({pl for pools in res.values() for p in pools for pl in p["places"]}))
    out: dict[str, Measure] = {}
    by_cell = {}
    for cell in sorted(cells):
        parts = [
            (place, v.get("liked", 0.0), v.get("rated", 0.0))
            for p in res.get(cell, [])
            for place, v in p["places"].items()
        ]
        by_cell[cell] = ratio(parts)
        out[f"precision:{cell}"] = Measure(places, by_cell[cell])
    out["precision"] = Measure(places, mean_of(list(by_cell.values())))
    lifts = {}
    for cell, pools in sorted(res.items()):
        parts2 = [(sorted(p["places"]), v) for p in pools if (v := lift(p)) is not None]
        if graded(cell) and parts2:
            lifts[cell] = weighted(parts2)
            out[f"kept_lift:{cell}"] = Measure(places, lifts[cell])
    out["kept_lift"] = Measure(places, mean_of(list(lifts.values())))
    laion = {}
    for cell, pools in sorted(res.items()):
        parts3 = [
            (sorted(p["places"]), float(v)) for p in pools if (v := p.get("laion_pct")) is not None
        ]
        if graded(cell) and parts3:
            laion[cell] = weighted(parts3)
    out["laion_pct"] = Measure(places, mean_of(list(laion.values())))
    low = [
        (sorted(p["places"]), p["low_priority_picks"] / p["picks"])
        for pools in res.values()
        for p in pools
        if p.get("low_priority_pool") and p["picks"]
    ]
    out["low_priority_share"] = Measure(places, weighted(low))
    seen: dict[str, tuple[float, float]] = {}
    kinds: dict[str, dict[str, tuple[float, float]]] = collections.defaultdict(dict)
    for cell, pools in sorted(res.items()):
        if not cell.startswith("place@"):
            continue
        for p in pools:
            if "kept_auc" in p and p["unit"] not in seen and p["kept_auc"][1] >= MIN_PAIRS:
                seen[p["unit"]] = (p["kept_auc"][0], p["kept_auc"][1])
            for kind, (conc, pairs) in p["auc"].items():
                kinds[kind].setdefault(p["unit"], (conc, pairs))
    each = [(([u]), c / n) for u, (c, n) in sorted(seen.items())]
    out["kept_auc"] = Measure(tuple(sorted(seen)), weighted(each))
    for kind, got in kinds.items():
        out[f"auc_{kind}"] = Measure(
            tuple(sorted(got)), ratio([(u, c, n) for u, (c, n) in got.items()])
        )
    everything = [
        (sorted(p["places"]), p["repeat"] - p["repeat_random"])
        for pools in res.values()
        for p in pools
    ]
    out["repeat_excess"] = Measure(places, weighted(everything))
    return out


def set_key(s: dict[str, Any]) -> str:
    return f"{s['pool']}|{s['unit']}"


def story_measures(res: Results) -> dict[str, Measure]:
    sessions = res["sessions"]
    ids = tuple(sorted(sessions))
    out = {
        "rated_auc": Measure(
            ids, ratio([(s, v["concordant"], v["pairs"]) for s, v in sessions.items()])
        ),
        "unfit_located": Measure(
            ids, ratio([(s, v["unfit_hits"], v["unfit_n"]) for s, v in sessions.items()])
        ),
    }
    sets = [s for s in res["sets"] if s["result"] and s["random"]]
    units = tuple(sorted({set_key(s) for s in sets}))

    def excess(names: tuple[str, ...]) -> Callable[[Pick], float | None]:
        return weighted(
            [
                (
                    [set_key(s)],
                    float(np.mean([s["result"][n] - s["random"][n] for n in names])),
                )
                for s in sets
            ]
        )

    out["set_repetition"] = Measure(units, excess(SET_PARTS))
    out["within"] = Measure(units, excess(("within",)))
    out["siglip"] = Measure(units, excess(("siglip",)))
    out["low_priority_share"] = Measure(
        units,
        weighted(
            [
                ([set_key(s)], s["low_priority_photos"] / (3 * len(s["stories"])))
                for s in sets
                if s.get("low_priority_pool") and s["stories"]
            ]
        ),
    )
    out["filled"] = Measure(
        units, weighted([([set_key(s)], min(1.0, len(s["stories"]) / s["k"])) for s in sets])
    )
    done = res["finished"]
    out["finished_beats_swaps"] = Measure(
        tuple(str(i) for i in range(len(done))),
        weighted([([str(i)], v) for i, v in enumerate(done)]),
    )
    return out


def measures(track: str, res: Results, cells: set[str] | None = None) -> dict[str, Measure]:
    if track.startswith("strongest"):
        return strongest_measures(res, judged_cells(res) if cells is None else cells)
    return story_measures(res)


def draws(units: tuple[str, ...], fn: Callable[[Pick], float | None]) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(BOOT if units else 0):
        drawn = rng.choice(len(units), size=len(units), replace=True)
        v = fn(dict(collections.Counter(units[i] for i in drawn)))
        if v is not None:
            vals.append(v)
    return np.asarray(vals)


def describe(units: tuple[str, ...], fn: Callable[[Pick], float | None]) -> dict[str, Any]:
    point = fn(None)
    vals = draws(units, fn)
    if point is None or not len(vals):
        return {"point": point, "units": len(units)}
    return {
        "point": point,
        "interval_90": [float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))],
        "interval_99": [float(np.quantile(vals, 0.005)), float(np.quantile(vals, 0.995))],
        "sd": float(vals.std()),
        "units": len(units),
    }


def speed(track: str, res: Results) -> float:
    if track.startswith("strongest"):
        return max((p["seconds"] for pools in res.values() for p in pools), default=0.0)
    return max((s["seconds"] for s in res["sets"]), default=0.0)


def overlap(track: str, a: Results, b: Results) -> float | None:
    shares = []
    if track.startswith("strongest"):
        for cell, pools in a.items():
            theirs = {p["unit"]: p for p in b.get(cell, [])}
            for p in pools:
                q = theirs.get(p["unit"])
                if q is not None and p["pick_rows"]:
                    both = set(p["pick_rows"]) & set(q["pick_rows"])
                    shares.append(len(both) / len(p["pick_rows"]))
    else:
        theirs = {(set_key(s), s["k"]): s for s in b["sets"]}
        for s in a["sets"]:
            q = theirs.get((set_key(s), s["k"]))
            mine = {i for story in s["stories"] for i in story}
            if q is not None and mine:
                other = {i for story in q["stories"] for i in story}
                shares.append(len(mine & other) / len(mine))
    return float(np.mean(shares)) if shares else None


def summary(report: dict[str, Any], name: str) -> dict[str, Any]:
    res = report["results"][name]
    out = {k: describe(m.units, m.fn) for k, m in measures(report["track"], res).items()}
    out["seconds"] = {"point": speed(report["track"], res)}
    return out


def paired(report: dict[str, Any], champion: str, candidate: str) -> dict[str, Any]:
    track = report["track"]
    a, b = report["results"][champion], report["results"][candidate]
    cells = judged_cells(a) & judged_cells(b) if track.startswith("strongest") else None
    ma, mb = measures(track, a, cells), measures(track, b, cells)
    out: dict[str, Any] = {}
    for key in sorted(set(ma) & set(mb)):
        fa, fb = ma[key].fn, mb[key].fn

        def diff(
            pick: Pick,
            fa: Callable[[Pick], float | None] = fa,
            fb: Callable[[Pick], float | None] = fb,
        ) -> float | None:
            x, y = fa(pick), fb(pick)
            return None if x is None or y is None else y - x

        got = describe(tuple(sorted(set(ma[key].units) | set(mb[key].units))), diff)
        out[key] = {
            "difference": got.pop("point"),
            "champion": fa(None),
            "candidate": fb(None),
            **got,
        }
    out["seconds"] = {"champion": speed(track, a), "candidate": speed(track, b)}
    out["overlap"] = overlap(track, a, b)
    return out


def ladder(report: dict[str, Any]) -> list[dict[str, Any]]:
    champion = report["champion"]
    rungs: dict[float, list[dict[str, Any]]] = collections.defaultdict(list)
    for name, info in report["configs"].items():
        if info.get("null"):
            rungs[float(info["null"]["sigma"])].append(paired(report, champion, name))
    out = []
    for sigma, runs in sorted(rungs.items()):
        sd = {}
        for key in runs[0]:
            if key in ("seconds", "overlap"):
                continue
            moved = [r[key]["difference"] for r in runs if r[key].get("difference") is not None]
            boots = [r[key]["sd"] for r in runs if "sd" in r[key]]
            if moved and boots:
                sd[key] = max(float(np.sqrt(np.mean(np.square(moved)))), float(np.mean(boots)))
        out.append(
            {
                "sigma": sigma,
                "nulls": len(runs),
                "overlap": float(np.mean([r["overlap"] for r in runs])),
                "sd": sd,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--candidate")
    parser.add_argument("--ladder", action="store_true")
    parser.add_argument("--cells", action="store_true")
    args = parser.parse_args()
    report = read_report(Path(args.report))
    champion = report["champion"]
    out: dict[str, Any] = {"track": report["track"], "tier": report["tier"], "report": args.report}
    out["champion"] = {champion: summary(report, champion)}
    if args.candidate:
        out["paired"] = paired(report, champion, args.candidate)
    if args.ladder:
        out["ladder"] = ladder(report)
    if not args.cells:
        for block in (out["champion"][champion], out.get("paired", {})):
            for key in [k for k in block if ":" in k]:
                del block[key]
    sys.stdout.write(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
