from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import catalogue
import numpy as np
import summarize

LOOP = Path(__file__).resolve().parents[1]
RATINGS = LOOP / "splits" / "dev" / "strongest-yours.json"
BURST = (0.90, 120.0)
LOOK_ALIKE = (0.95, 3600.0)
SAME_IMAGE = 0.99
BOOT = 2000
SEED = 20261113


def twin_of(
    row: int, rated: np.ndarray, unit: np.ndarray, captured: np.ndarray
) -> tuple[int, float] | None:
    if not len(rated):
        return None
    cos = unit[rated] @ unit[row]
    dt = np.abs(captured[rated] - captured[row])
    near = (cos >= SAME_IMAGE) | ((cos >= BURST[0]) & (dt <= BURST[1]))
    near |= (cos >= LOOK_ALIKE[0]) & (dt <= LOOK_ALIKE[1])
    if not near.any():
        return None
    best = int(np.argmax(np.where(near, cos, -np.inf)))
    return int(rated[best]), float(cos[best])


def arms(report: dict[str, Any], candidate: str, cell: str) -> dict[str, list[tuple[str, int]]]:
    mine = {p["unit"]: p for p in report["results"][report["champion"]][cell]}
    out: dict[str, list[tuple[str, int]]] = {"champion": [], "candidate": []}
    for pool in report["results"][candidate][cell]:
        other = mine.get(pool["unit"])
        if other is None:
            continue
        a, b = set(other["pick_rows"]), set(pool["pick_rows"])
        out["champion"] += [(pool["unit"], r) for r in other["pick_rows"] if r not in b]
        out["candidate"] += [(pool["unit"], r) for r in pool["pick_rows"] if r not in a]
    return out


def interval(per_place: dict[str, dict[str, list[int]]]) -> tuple[float, float, float]:
    places = sorted(per_place)
    rng = np.random.default_rng(SEED)

    def share(sample: list[str], arm: str) -> float:
        got = [v for p in sample for v in per_place[p][arm]]
        return float(np.mean(got)) if got else float("nan")

    point = share(places, "candidate") - share(places, "champion")
    draws = []
    for _ in range(BOOT):
        sample = [places[i] for i in rng.integers(0, len(places), len(places))]
        draws.append(share(sample, "candidate") - share(sample, "champion"))
    low, high = np.nanpercentile(draws, [5, 95])
    return point, float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--cell", default="place@12")
    args = parser.parse_args()
    report = summarize.read_report(LOOP.parents[1] / args.report)
    if report["tier"] != "dev":
        raise SystemExit("reads dev reports only")
    cat = catalogue.load()
    unit = cat.clip / np.linalg.norm(cat.clip, axis=1, keepdims=True)
    captured = np.nan_to_num(cat.captured.astype(float), nan=-1e12)
    row_of = {str(k): i for i, k in enumerate(cat.keys)}
    liked: dict[int, bool] = {}
    for r in json.loads(RATINGS.read_text())["ratings"]:
        if r["key"] in row_of:
            liked[row_of[r["key"]]] = bool(r["liked"])
    rated_at: dict[str, np.ndarray] = {}
    for row in liked:
        rated_at.setdefault(str(cat.location[row]), np.array([], dtype=int))
        rated_at[str(cat.location[row])] = np.append(rated_at[str(cat.location[row])], row)
    out = {}
    for name in args.candidate:
        sides = arms(report, name, args.cell)
        counts: dict[str, dict[str, int]] = {}
        direct: dict[str, dict[str, list[int]]] = {}
        spread: dict[str, dict[str, list[int]]] = {}
        for arm, picks in sides.items():
            c = {"swapped": len(picks), "rated": 0, "by_twin": 0}
            for place, row in picks:
                direct.setdefault(place, {"champion": [], "candidate": []})
                spread.setdefault(place, {"champion": [], "candidate": []})
                if row in liked:
                    c["rated"] += 1
                    direct[place][arm].append(int(liked[row]))
                    spread[place][arm].append(int(liked[row]))
                    continue
                twin = twin_of(row, rated_at.get(place, np.array([], dtype=int)), unit, captured)
                if twin is not None:
                    c["by_twin"] += 1
                    spread[place][arm].append(int(liked[twin[0]]))
            counts[arm] = c
        out[name] = {
            "cell": args.cell,
            "counts": counts,
            "liked_rated_only": interval(direct),
            "liked_with_twins": interval(spread),
        }
    sys.stdout.write(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
