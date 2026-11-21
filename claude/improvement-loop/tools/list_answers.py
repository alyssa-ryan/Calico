from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SEED = 20261113
BOOT = 2000
GIVEN = "given"
ARMS = ("champion", "candidate")


def share(liked: float, rated: float) -> float | None:
    return liked / rated if rated else None


def as_roles(arms: dict[str, str], champion: str | None) -> dict[str, str]:
    if champion is None:
        return arms
    return {key: "champion" if arm == champion else "candidate" for key, arm in arms.items()}


def read_answers(
    arms: dict[str, str], answers: dict[str, int], place_of: dict[str, str]
) -> dict[str, Any]:
    counts: dict[str, dict[str, list[int]]] = collections.defaultdict(
        lambda: {arm: [0, 0] for arm in ARMS}
    )
    unsure = 0
    for key, arm in sorted(arms.items()):
        label = answers.get(key)
        if label == 1:
            unsure += 1
        if label not in (0, 2) or arm not in ARMS:
            continue
        cell = counts[place_of.get(key, "")][arm]
        cell[0] += label == 2
        cell[1] += 1
    places = sorted(counts)
    table = np.array([[*counts[p]["champion"], *counts[p]["candidate"]] for p in places])

    def difference(rows: np.ndarray) -> float | None:
        if not len(rows):
            return None
        a_liked, a_rated, b_liked, b_rated = rows.sum(axis=0)
        a, b = share(a_liked, a_rated), share(b_liked, b_rated)
        return None if a is None or b is None else b - a

    rng = np.random.default_rng(SEED)
    drawn = [
        d
        for _ in range(BOOT if len(places) > 1 else 0)
        if (d := difference(table[rng.integers(0, len(places), len(places))])) is not None
    ]
    totals = table.sum(axis=0) if places else np.zeros(4)
    out: dict[str, Any] = {
        "listed": len(arms),
        "rated": int(totals[1] + totals[3]),
        "not_sure": unsure,
        "places": len(places),
        "difference": difference(table),
    }
    for n, arm in enumerate(ARMS):
        liked, rated = int(totals[2 * n]), int(totals[2 * n + 1])
        out[arm] = {"liked": liked, "rated": rated, "share": share(liked, rated)}
    if drawn:
        out["interval_90"] = [float(np.quantile(drawn, 0.05)), float(np.quantile(drawn, 0.95))]
    return out


def main() -> None:
    import catalogue

    from calico.reviews import labels_of

    parser = argparse.ArgumentParser()
    parser.add_argument("arms")
    parser.add_argument("--champion")
    args = parser.parse_args()
    arms = as_roles(dict(json.loads(Path(args.arms).read_text())), args.champion)
    eng = catalogue.engine()
    cat = catalogue.load(eng)
    current = eng.ratings.current()
    answers: dict[str, int] = {}
    sessions = []
    for s in eng.labels.all():
        if s.kind != "taste" or not s.items or any(i.stratum != GIVEN for i in s.items):
            continue
        found = {k: int(v) for k, v in labels_of(s, current, controls=False).items() if k in arms}
        if found:
            sessions.append(s.set_id)
            answers.update(found)
    place_of = {k: str(cat.location[cat.row[k]]) for k in arms if k in cat.row}
    result = {"arms": args.arms, "sessions": sessions, **read_answers(arms, answers, place_of)}
    sys.stdout.write(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
