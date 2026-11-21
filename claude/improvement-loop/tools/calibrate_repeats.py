from __future__ import annotations

import collections
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import catalogue
import harness
import numpy as np

TIERS = ("dev", "validation")
BANDS = [0.0, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
KINDS = {2: "same", 1: "nearly", 0: "no"}
BOOT = 1000
Row = dict[str, Any]


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    p, n = np.asarray(pos), np.asarray(neg)
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def rates(pos: list[float], neg: list[float], cut: float) -> tuple[float, float]:
    return float(np.mean(np.asarray(pos) >= cut)), float(np.mean(np.asarray(neg) >= cut))


def youden(pos: list[float], neg: list[float]) -> dict[str, float] | None:
    if not pos or not neg:
        return None
    cuts = np.unique(np.concatenate([pos, neg]))
    p, n = np.sort(pos), np.sort(neg)
    tpr = 1 - np.searchsorted(p, cuts, side="left") / len(p)
    fpr = 1 - np.searchsorted(n, cuts, side="left") / len(n)
    best = int(np.argmax(tpr - fpr))
    return {
        "j": float(tpr[best] - fpr[best]),
        "cut": float(cuts[best]),
        "tpr": float(tpr[best]),
        "fpr": float(fpr[best]),
    }


def sides(rows: list[Row], field: str, pos: set[str], neg: set[str]) -> tuple[list[float], ...]:
    return (
        [float(r[field]) for r in rows if r["kind"] in pos],
        [float(r[field]) for r in rows if r["kind"] in neg],
    )


def cut_interval(rows: list[Row], field: str, pos: set[str], neg: set[str]) -> list[float] | None:
    groups: dict[str, list[Row]] = collections.defaultdict(list)
    for r in rows:
        groups[r["reference"]].append(r)
    names = sorted(groups)
    rng = np.random.default_rng(harness.SEED)
    cuts = []
    for _ in range(BOOT if names else 0):
        drawn = [r for i in rng.integers(0, len(names), len(names)) for r in groups[names[i]]]
        got = youden(*sides(drawn, field, pos, neg))
        if got:
            cuts.append(got["cut"])
    if not cuts:
        return None
    return [float(np.quantile(cuts, 0.05)), float(np.quantile(cuts, 0.95))]


def split(rows: list[Row], field: str, pos: set[str], neg: set[str]) -> dict[str, Any]:
    p, q = sides(rows, field, pos, neg)
    quart = [0.25, 0.5, 0.75]
    out: dict[str, Any] = {
        "n_pos": len(p),
        "n_neg": len(q),
        "references": len({r["reference"] for r in rows}),
        "auc": auc(p, q),
        "youden": youden(p, q),
        "cut_interval_90_over_references": cut_interval(rows, field, pos, neg),
        "pos_quartiles": np.quantile(p, quart).round(3).tolist() if p else None,
        "neg_quartiles": np.quantile(q, quart).round(3).tolist() if q else None,
    }
    if field == "dino" and p and q:
        tpr, fpr = rates(p, q, harness.REPEAT_VIEW)
        out["at_the_line_in_use"] = {
            "cut": harness.REPEAT_VIEW,
            "tpr": tpr,
            "fpr": fpr,
            "j": tpr - fpr,
        }
    return out


def by_band(story_rows: list[Row]) -> list[Row]:
    out = []
    for lo, hi in itertools.pairwise(BANDS):
        rows = [r for r in story_rows if lo <= r["dino_max"] < hi]
        out.append(
            {
                "dino_max": [lo, hi],
                "stories": len(rows),
                "works": round(float(np.mean([r["works"] for r in rows])), 3) if rows else None,
            }
        )
    return out


def main() -> None:
    out = Path(sys.argv[1])
    eng = catalogue.engine()
    cat = catalogue.load(eng)
    home = eng.settings.home
    world_splits = Path(os.environ.get("LOOP_SPLITS", catalogue.LOOP / "splits"))
    labels = harness.Labels(
        world_splits,
        harness.low_priority_words(),
        lambda key: str(cat.location[cat.row[key]]) if key in cat.row else "",
    )
    views = {t: labels.read(t, "same-view")["same_view"] for t in TIERS}
    stories = {t: labels.read(t, "story-yours")["stories"] for t in TIERS}

    real = str(catalogue.platformdirs.user_data_path("calico"))

    def local(reference: str) -> Path:
        return Path(reference.replace(real, str(home)))

    refs = sorted({local(v["reference"]) for t in TIERS for v in views[t]})
    found = eng.references([p for p in refs if p.exists()])
    view_pairs: list[Row] = []
    missing = 0
    if found:
        rv = found["references"]
        ref_row = {str(p): i for i, p in enumerate(found["reference_paths"])}
        dino_name = eng.view_feature()
        for t in TIERS:
            for v in views[t]:
                r = ref_row.get(str(local(v["reference"])))
                if r is None or v["key"] not in cat.row:
                    missing += 1
                    continue
                i = cat.row[v["key"]]
                view_pairs.append(
                    {
                        "tier": t,
                        "reference": v["reference"],
                        "kind": KINDS[int(v["value"])],
                        "dino": float(rv.features[dino_name][r] @ cat.dino[i]),
                        "clip": float(rv.embeddings[r] @ cat.clip[i]),
                    }
                )

    story_rows: list[Row] = []
    for t in TIERS:
        for s in stories[t]:
            if not all(k in cat.row for k in s["keys"]):
                continue
            rows = [cat.row[k] for k in s["keys"]]
            closest = max(
                float(cat.dino[a] @ cat.dino[b]) for a, b in itertools.combinations(rows, 2)
            )
            story_rows.append({"tier": t, "works": bool(s["works"]), "dino_max": closest})

    result: dict[str, Any] = {
        "snapshot": cat.snapshot,
        "tiers": list(TIERS),
        "line_in_use": harness.REPEAT_VIEW,
        "low_priority_labels_dropped": dict(sorted(labels.dropped.items())),
        "same_view_labels": {t: len(views[t]) for t in TIERS},
        "same_view_labels_without_reference_or_photo": missing,
        "same_view_pairs": dict(collections.Counter(p["kind"] for p in view_pairs)),
        "same_view_same_or_nearly_vs_no": {
            f: split(view_pairs, f, {"same", "nearly"}, {"no"}) for f in ("dino", "clip")
        },
        "same_view_same_vs_nearly": {
            f: split(view_pairs, f, {"same"}, {"nearly"}) for f in ("dino", "clip")
        },
        "same_view_same_or_nearly_vs_no_by_tier": {
            t: split([p for p in view_pairs if p["tier"] == t], "dino", {"same", "nearly"}, {"no"})
            for t in TIERS
        },
        "stories": {t: sum(r["tier"] == t for r in story_rows) for t in TIERS},
        "stories_by_closest_pair": by_band(story_rows),
        "stories_works_auc_by_dino_max_lower_is_better": auc(
            [-r["dino_max"] for r in story_rows if r["works"]],
            [-r["dino_max"] for r in story_rows if not r["works"]],
        ),
    }
    text = json.dumps(result, indent=1) + "\n"
    out.write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
